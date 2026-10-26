"""Polariseq arms: in-RAM, and disk-backed (out-of-core).

Two arms rather than one, because they are different claims:

- ``polariseq-ram`` is the "we are at least as fast as a tuned scanpy on a
  machine where the data fits" claim.
- ``polariseq-spill`` is the "and we still run when it does not fit, at a
  bounded footprint you chose in advance" claim, which scanpy has no
  equivalent of. Its interesting number is not wall time — spilling to SSD
  is *expected* to cost time — but the shape of the memory curve and
  whether it honours ``ps.plan()``'s prediction.

Presenting the spill arm's wall time next to an in-RAM scanpy without that
caveat would be the sort of comparison this harness exists to prevent, so
the analysis labels it `out-of-core` and reports its predicted-vs-actual
footprint alongside.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import write_artifacts


def _apply_budget(ps: Any, params: Any) -> int:
    """Pin the budget the campaign asks for, and prove the pool took it.

    Returns the worker threads Polariseq actually runs with. Checked rather
    than assumed: until 2026-09-19 `set_budget(threads=4)` could leave the
    pool at the machine's detected count, so the Windows and Linux runs of
    the first cross-platform campaign used 12 and 48 threads against the
    other arms' 4, and nothing in their records showed it.
    """
    kw: dict[str, Any] = {}
    if params.threads:
        kw["threads"] = params.threads
    if getattr(params, "ram_gb", 0):
        # Pin the RAM ceiling: the planner sizes spill blocks from the
        # budget, and a budget that floats with ambient free memory makes two
        # reps of the same run disagree by exactly the HVG boundary genes
        # (measured; chapter 09).
        kw["ram_gb"] = params.ram_gb
    # Point the library at the per-run spill directory the runner created and
    # is watching. Without this the library spills to its own default while
    # `SpillWatcher` watches a different path, and `spill_peak_bytes` is
    # silently 0 — measured 2026-09-09, which made the out-of-core arm look as
    # though it used no disk at all.
    _sd = os.environ.get("POLARISEQ_SPILL_DIR")
    if _sd:
        kw["spill_dir"] = _sd
    if kw:
        ps.set_budget(**kw)
    # The in-memory arm is run past the planner's recommendation on purpose
    # (the benchmark measures what that costs), so the warning that says so would
    # only repeat, run after run, what the campaign already knows.
    import warnings

    warnings.simplefilter("ignore", ps.BudgetWarning)
    from polariseq import _polariseq

    used = int(_polariseq.thread_count())
    if params.threads and used != params.threads:
        raise RuntimeError(f"the worker pool runs {used} threads; this campaign pins "
                           f"{params.threads}, and a run at another count is not comparable")
    return used


def _finish(adata: Any, params: Any, timeline: Any, artifacts: Path, ps: Any) -> dict[str, Any]:
    """Neighbours, Leiden and UMAP — shared by both Polariseq arms."""
    with timeline.step("neighbors"):
        ps.pp.neighbors(adata, n_neighbors=params.n_neighbors, seed=params.seed)
    with timeline.step("leiden"):
        ps.tl.leiden(adata, resolution=params.resolution, seed=params.seed)
    umap = None
    with timeline.step("umap"):
        if params.with_umap:
            ps.tl.umap(adata, seed=params.seed)
            umap = np.asarray(adata.obsm.get("X_umap"))
    return {"umap": umap}


class PolariseqRam:
    name = "polariseq-ram"
    label = "Polariseq (in-RAM)"
    tuning_notes = (
        "Defaults throughout. Threads come from ps.budget(), which prefers "
        "performance cores on Apple Silicon; the harness records the number "
        "and gives every arm the same one."
    )

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        # Timed: importing a library costs wall time and resident memory
        # before any data is touched, and the two arms' import graphs are not
        # remotely the same size. Folding that into the first data step would
        # charge it to whichever step happened to run first.
        with timeline.step("import"):
            import polariseq as ps

        threads_used = _apply_budget(ps, params)

        # In memory whatever the planner would choose: this arm measures what the
        # in-memory analysis costs, also past the size the planner sends out of core.
        with timeline.step("load"):
            adata = ps.read_h5ad(path, lazy=True, mode="memory")
            adata = ps.pp._ensure_materialized(adata)
        with timeline.step("qc"):
            ps.pp.calculate_qc_metrics(adata)
        with timeline.step("filter_cells"):
            ps.pp.filter_cells(adata, min_genes=params.min_genes)
        with timeline.step("normalize"):
            ps.pp.normalize_total(adata, target_sum=params.target_sum)
        with timeline.step("log1p"):
            ps.pp.log1p(adata)
        # Gene selection, normalization and scaling as the workflow defines them
        # (ps.set_workflow; POLARISEQ_WORKFLOW=scanpy for the published steps).
        recipe = ps.workflow() == "polariseq"
        with timeline.step("hvg"):
            ps.pp.highly_variable_genes(adata, n_top_genes=params.n_hvg, subset=True)
        if recipe:
            with timeline.step("renormalize"):
                ps.pp.renormalize(adata, target_sum=1000)
        with timeline.step("pca"):
            ps.pp.pca(adata, params.n_pcs, seed=params.seed, scale=recipe)

        extra = _finish(adata, params, timeline, artifacts, ps)

        labels = np.asarray(adata.obs["leiden"]).astype(np.int32)
        files = write_artifacts(
            artifacts,
            embedding=np.asarray(adata.obsm["X_pca"]),
            labels=labels,
            hvg_names=list(map(str, adata.var_names)),
            obs_names=list(map(str, adata.obs_names)),
            umap=extra["umap"],
        )
        return {
            "n_obs_after": int(adata.n_obs),
            "n_vars_after": int(adata.n_vars),
            "n_clusters": int(len(set(labels.tolist()))),
            "storage": "ram",
            "workflow": ps.workflow(),
            "threads_used": threads_used,
            "artifacts": files,
        }


class PolariseqSpill:
    name = "polariseq-spill"
    label = "Polariseq (out-of-core)"
    tuning_notes = (
        "process_diskbacked: the matrix is streamed to SSD and read back in "
        "budget-sized blocks (FileCsr, explicit positional reads). Block size "
        "is chosen by ps.plan() from the budget, not hand-tuned per dataset. "
        "Wall time is expected to be worse than the in-RAM arm; the claim "
        "under test is the footprint and that it matches the prediction."
    )

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        # Timed: importing a library costs wall time and resident memory
        # before any data is touched, and the two arms' import graphs are not
        # remotely the same size. Folding that into the first data step would
        # charge it to whichever step happened to run first.
        with timeline.step("import"):
            import polariseq as ps

        threads_used = _apply_budget(ps, params)

        plan = ps.plan(
            path,
            n_hvg=params.n_hvg,
            n_pcs=params.n_pcs,
            storage="spill",
            with_umap=params.with_umap,
            verbose=False,
        )
        # `process_diskbacked` fuses load/qc/normalize/log1p/hvg/pca into one
        # streaming pass, which is the entire point of it — so it emits one
        # `preprocess` marker instead of six. The analysis knows this arm has
        # a fused block and renders it as a single span rather than pretending
        # the six steps happened separately.
        with timeline.step("preprocess"):
            adata = ps.process_diskbacked(
                path,
                n_hvg=params.n_hvg,
                n_pcs=params.n_pcs,
                n_neighbors=params.n_neighbors,
                target_sum=params.target_sum,
                resolution=params.resolution,
                seed=params.seed,
                min_genes=params.min_genes,
            )
        # `process_diskbacked` already ran neighbours and Leiden internally;
        # timing them again would double-count. UMAP is not part of it.
        umap = None
        with timeline.step("umap"):
            if params.with_umap:
                ps.tl.umap(adata, seed=params.seed)
                umap = np.asarray(adata.obsm.get("X_umap"))

        labels = np.asarray(adata.obs["leiden"]).astype(np.int32)
        files = write_artifacts(
            artifacts,
            embedding=np.asarray(adata.obsm["X_pca"]),
            labels=labels,
            # The disk path filters cells, so its row i is not the file's
            # row i. Emitting the barcodes it kept lets the equivalence
            # check align by name instead of assuming the two orders match
            # — the assumption that hid the missing-filter bug the first
            # time this arm was compared.
            hvg_names=[str(g) for g in adata.uns.get("hvg_names", [])] or None,
            obs_names=list(map(str, adata.obs_names)),
            umap=umap,
        )
        return {
            "n_obs_after": int(adata.n_obs),
            "n_vars_after": int(adata.uns.get("hvg", {}).get("n_top_genes", 0)),
            "n_clusters": int(len(set(labels.tolist()))),
            "storage": "spill",
            "workflow": ps.workflow(),
            "threads_used": threads_used,
            "predicted_peak_bytes": int(plan.peak_bytes),
            "rows_per_block": int(plan.rows_per_block),
            "artifacts": files,
        }
