"""Scanpy arms: as-shipped defaults, and a best-faith tuned configuration.

Two arms, because "Polariseq beats scanpy" is a claim with two very
different meanings and only one of them is interesting:

- ``scanpy-default`` is what a user actually gets by following the tutorial.
  It is the honest baseline for "what will my day be like", and nothing
  more. Beating it is not, by itself, an engineering result.
- ``scanpy-tuned`` is scanpy configured as well as we know how to configure
  it, by someone who wants it to win. This is the arm the performance claim
  must be made against. If Polariseq only beats the default, the results say
  so.

Every tuning choice is listed in ``tuning_notes`` and justified below, so a
reviewer who knows scanpy better than we do can point at a specific line and
say "you left this on the table". That is the intended failure mode: a
tuning we missed is a bug report against the comparison, not against
Polariseq.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import write_artifacts


def _pipeline(
    sc: Any, path: str, params: Any, timeline: Any, artifacts: Path, *, tuned: bool,
    repair_indices: bool = False,
) -> dict[str, Any]:
    leiden_kw: dict[str, Any] = {}
    if tuned:
        # scanpy itself recommends the igraph flavour and warns that the
        # leidenalg default will change; using it is what an informed user
        # does, and it is substantially faster. `n_iterations=2` is what
        # scanpy's own docs pair with it.
        leiden_kw = {"flavor": "igraph", "n_iterations": 2, "directed": False}

    with timeline.step("load"):
        adata = sc.read_h5ad(path)
        if repair_indices and adata.X.indptr.dtype != adata.X.indices.dtype:
            # Past 2**31 stored values the index pointer must be int64, and
            # scipy then refuses int32 column indices alongside it. Widening
            # them is the repair a user would make; it is timed, and its
            # memory counts.
            adata.X.indices = adata.X.indices.astype(adata.X.indptr.dtype)
    with timeline.step("qc"):
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    with timeline.step("filter_cells"):
        sc.pp.filter_cells(adata, min_genes=params.min_genes)
    with timeline.step("normalize"):
        sc.pp.normalize_total(adata, target_sum=params.target_sum)
    with timeline.step("log1p"):
        sc.pp.log1p(adata)
    with timeline.step("hvg"):
        sc.pp.highly_variable_genes(
            adata, n_top_genes=params.n_hvg, flavor="seurat", subset=True
        )
    with timeline.step("pca"):
        sc.pp.pca(
            adata,
            n_comps=params.n_pcs,
            zero_center=True,
            svd_solver="arpack" if tuned else None,
            random_state=params.seed,
        )
    with timeline.step("neighbors"):
        sc.pp.neighbors(
            adata, n_neighbors=params.n_neighbors, n_pcs=params.n_pcs,
            random_state=params.seed,
        )
    with timeline.step("leiden"):
        sc.tl.leiden(
            adata, resolution=params.resolution, random_state=params.seed, **leiden_kw
        )
    umap = None
    with timeline.step("umap"):
        if params.with_umap:
            sc.tl.umap(adata, random_state=params.seed)
            umap = np.asarray(adata.obsm["X_umap"])

    labels = np.asarray(adata.obs["leiden"]).astype(np.int32)
    files = write_artifacts(
        artifacts,
        embedding=np.asarray(adata.obsm["X_pca"]),
        labels=labels,
        hvg_names=list(map(str, adata.var_names)),
        obs_names=list(map(str, adata.obs_names)),
        umap=umap,
    )
    return {
        "n_obs_after": int(adata.n_obs),
        "n_vars_after": int(adata.n_vars),
        "n_clusters": int(len(set(labels.tolist()))),
        "storage": "ram",
        "artifacts": files,
    }


def _warm(sc: Any, params: Any) -> None:
    """Run the whole timed pipeline once, untimed, on synthetic counts.

    Scanpy compiles Numba code on first use, in normalisation, gene
    selection and the neighbour search. The earlier warm-up (neighbours on a
    5,000 x 20 random matrix) left a near-constant compile cost inside the
    timed steps, measured in the public-datasets stage: `neighbors` took 11.5 s on
    10,000 cells and 11.6 s on 50,000 against 0.7 s once compiled, and
    `normalize` and `hvg` about 1 s each on 2,700 cells. Two reasons: it
    skipped every step but one, and scanpy computes exact distances below
    8,192 cells (not 4,096, as the old comment assumed) and switches to
    pynndescent only above, so a 5,000-cell warm-up compiled the wrong path.
    This runs every step once on sparse counts of the real data's dtype below
    the threshold, then the neighbour search once above it, so the timed run
    measures computation, as it does for Polariseq's precompiled code.
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    x = sp.random(6000, 3000, density=0.08, format="csr", random_state=rng,
                  dtype=np.float32)
    x.data = np.ceil(x.data * 10.0).astype(np.float32)
    a = sc.AnnData(x)
    sc.pp.calculate_qc_metrics(a, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(a, min_genes=10)
    sc.pp.normalize_total(a, target_sum=params.target_sum)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1000, flavor="seurat", subset=True)
    sc.pp.pca(a, n_comps=min(params.n_pcs, 50), zero_center=True, svd_solver="arpack",
              random_state=0)
    sc.pp.neighbors(a, n_neighbors=params.n_neighbors, n_pcs=min(params.n_pcs, 50),
                    random_state=0)
    sc.tl.leiden(a, resolution=params.resolution, random_state=0, flavor="igraph",
                 n_iterations=2, directed=False)
    if params.with_umap:
        sc.tl.umap(a, random_state=0)
    b = sc.AnnData(np.zeros((12_000, 1), dtype=np.float32))
    b.obsm["X_pca"] = rng.random((12_000, min(params.n_pcs, 50))).astype(np.float32)
    sc.pp.neighbors(b, n_neighbors=params.n_neighbors, use_rep="X_pca", random_state=0)


class ScanpyDefault:
    name = "scanpy-default"
    label = "scanpy (defaults)"
    tuning_notes = (
        "Library defaults, deliberately untouched: sc.settings.n_jobs is left "
        "at 1 (which pins numba inside pynndescent), no JIT warm-up, "
        "sc.tl.leiden uses its default leidenalg flavour, and svd_solver is "
        "left to scanpy's own choice. This arm answers 'what does the "
        "tutorial give me', not 'how fast can scanpy be'."
    )

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        # Timed for the same reason as the Polariseq arm: scanpy's import
        # graph is large, it costs wall time and resident memory before any
        # data is touched, and burying it in the first data step would charge
        # it to whichever step happened to run first.
        with timeline.step("import"):
            import scanpy as sc

        return _pipeline(sc, path, params, timeline, artifacts, tuned=False)


class ScanpyTuned:
    name = "scanpy-tuned"
    label = "scanpy (tuned)"
    tuning_notes = (
        "Best-faith configuration: (1) sc.settings.n_jobs set to the pinned "
        "thread budget, which un-pins numba threading inside pynndescent — "
        "worth ~10x on the neighbours step at 100k cells in the earlier "
        "fairness audit. It is set to the budget rather than -1 because "
        "joblib resolves -1 from os.cpu_count() and ignores the thread "
        "environment variables entirely (measured: NUMBA_NUM_THREADS=4 is "
        "honoured, effective_n_jobs(-1) still returns 8), so -1 would give "
        "scanpy twice the workers Polariseq gets and the comparison would "
        "measure the launcher; (2) an untimed run of the whole pipeline on a "
        "6,000-cell synthetic count matrix before timing starts, so numba's "
        "JIT compilation (normalisation, gene selection, neighbours) is not "
        "charged to the timed steps "
        "(Polariseq is precompiled Rust and pays no equivalent, so charging "
        "scanpy for JIT would be unfair); "
        "(3) sc.tl.leiden(flavor='igraph', n_iterations=2, directed=False), "
        "the fast path scanpy's own docs recommend; (4) svd_solver='arpack' "
        "pinned for determinism. The pipeline itself is identical — no "
        "reduced n_pcs, no loosened convergence, no skipped steps."
    )

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        with timeline.step("import"):
            import numpy as np
            import scanpy as sc

        # `-1` only when no budget was pinned. See `tuning_notes`: joblib
        # reads os.cpu_count() for -1 and ignores THREAD_ENV_VARS, so on a
        # pinned run it would silently hand scanpy more workers than every
        # other arm.
        sc.settings.n_jobs = params.threads if getattr(params, "threads", 0) else -1
        with timeline.step("warmup"):
            _warm(sc, params)

        return _pipeline(sc, path, params, timeline, artifacts, tuned=True)


def second_seed(primary: int) -> int:
    """A seed guaranteed different from `primary`, for the floor arm."""
    return 1 if primary != 1 else 2


class ScanpyTunedSeed1(ScanpyTuned):
    """`scanpy-tuned` at a second seed — the floor, not a contender.

    `umap_knn_overlap` between two implementations means nothing until you
    know what *the same implementation* scores against itself at a different
    seed: UMAP's SGD makes layout point-identity noise (chapter 00, pitfall
    11), so the seed-to-seed overlap is the floor below which a
    cross-implementation number is unremarkable. This arm is that floor,
    measured with the same pipeline, same tuning, one different seed.

    It is not in `DEFAULT_ARMS`; opt in with `--arms ...,
    scanpy-tuned-seed1` on a with-UMAP campaign. A no-UMAP campaign gains
    nothing from it (the overlap check needs the layout).
    """

    name = "scanpy-tuned-seed1"
    label = "scanpy (tuned, second seed)"
    tuning_notes = (
        "Byte-for-byte the scanpy-tuned configuration except the random seed, "
        "forced to a value different from the campaign seed (see "
        "second_seed). This arm exists to measure the seed-to-seed floor of "
        "umap_knn_overlap — the same implementation against itself at two "
        "seeds — against which every cross-implementation overlap is judged. "
        "Its timings duplicate scanpy-tuned's and should not be quoted as a "
        "separate result."
    )

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        from dataclasses import replace

        forced = second_seed(int(params.seed))
        out = super().run(path, replace(params, seed=forced), timeline, artifacts)
        # The record's params still show the campaign seed; make the arm's
        # actual seed inspectable on the record itself.
        out["seed_forced"] = forced
        return out


class ScanpyTunedInt64(ScanpyTuned):
    """`scanpy-tuned` with the index-width repair, for the 4M-cell file.

    The 4.06M-cell fetal atlas holds 2.63 billion values, so its index
    pointer is int64 and anndata returns int32 column indices beside it,
    which scipy refuses ("failed: file format" in the earlier campaign). This
    arm widens the indices on load, so the run reaches the question that
    matters: whether the analysis fits, not whether the file opens.
    """

    name = "scanpy-tuned-int64"
    label = "scanpy (tuned, int64 indices)"
    tuning_notes = (
        "Byte-for-byte scanpy-tuned, plus one step inside the timed load: "
        "when the matrix's index pointer is int64 and its indices int32 (a "
        "matrix past 2**31 values), the indices are widened to int64, which "
        "scipy requires. Only the 4.06M-cell fetal atlas triggers it."
    )

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        with timeline.step("import"):
            import numpy as np
            import scanpy as sc

        sc.settings.n_jobs = params.threads if getattr(params, "threads", 0) else -1
        with timeline.step("warmup"):
            _warm(sc, params)
        return _pipeline(sc, path, params, timeline, artifacts, tuned=True, repair_indices=True)
