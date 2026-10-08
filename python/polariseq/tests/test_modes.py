"""One object, two modes: the planning step decides where an analysis runs.

`ps.read_h5ad` plans from the file's shape and the budget, and every step on
the object then runs in that mode: in memory, or out of core over a copy of
the counts kept in the project folder. The out-of-core steps are the kernels
of `process_diskbacked`, so on the same settings the two must agree bit for
bit, and both must agree with the in-memory mode.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import polariseq._project as P
import pytest

FIXTURE = "tests/data/medium_10kx2k.h5ad"
N_HVG, N_PCS = 200, 10
MITO = "gene_19"  # stands in for "MT-" on this fixture


def _qc(ad, qc: bool):
    if qc:
        ps.pp.calculate_qc_metrics(ad, mito_prefix=MITO)
        ps.pp.filter_cells(ad, min_genes=150, max_pct_mt=8.0)
        ps.pp.filter_genes(ad, min_cells=700)
    else:
        ps.pp.calculate_qc_metrics(ad)
        ps.pp.filter_cells(ad, min_genes=160)


def steps(ad, *, qc: bool = True):
    """The analysis step by step, as the active workflow defines it (process_diskbacked's fused
    pipeline follows the same workflow): in 'polariseq', each cell normalized again over the selected
    genes and the genes standardized before PCA (in memory in place, from each cell's recorded total)."""
    recipe = ps.workflow() == "polariseq"
    _qc(ad, qc)
    ps.pp.normalize_total(ad, target_sum=1e4)
    ps.pp.log1p(ad)
    ps.pp.highly_variable_genes(ad, n_top_genes=N_HVG, subset=True)
    if recipe:
        ps.pp.renormalize(ad, target_sum=1000)
    ps.pp.pca(ad, N_PCS, seed=42, scale=recipe)
    return ad


def test_a_small_file_is_planned_in_memory() -> None:
    assert ps.read_h5ad(FIXTURE).mode == "memory"


def test_the_planning_step_switches_modes_by_size() -> None:
    """At a 6 GB budget a 10,000-cell matrix is analyzed in memory and the
    4.06-million-cell atlas (2.63 billion values) out of core: the matrix
    fits in half the budget, or it does not. Below the out-of-core floor
    nothing fits, and the analysis stays in memory with a warning."""
    before = ps.budget()
    try:
        ps.set_budget(ram_gb=6)
        small = ps._choose_mode("auto", 10_000, 2_000, 1_700_000, "h5ad", "small.h5ad")
        atlas = ps._choose_mode("auto", 4_062_980, 63_561, 2_630_000_000, "h5ad", "atlas.h5ad")
        ps.set_budget(ram_gb=0.5)
        below_floor = ps._choose_mode("auto", 4_062_980, 63_561, 2_630_000_000, "h5ad", "a.h5ad")
    finally:
        ps.set_budget(ram_gb=before.ram_gb)
    assert (small, atlas, below_floor) == ("memory", "out-of-core", "memory")


def test_a_matrix_past_the_in_memory_limit_is_analyzed_out_of_core() -> None:
    """More stored values than the in-memory matrix's 32-bit row pointers
    address: out of core whatever the budget, and never in memory."""
    n = ps.MAX_IN_MEMORY_VALUES + 1
    assert ps._choose_mode("auto", 44_000_000, 60_000, n, "h5ad", "census.h5ad") == "out-of-core"
    with pytest.raises(ValueError, match="out of core"):
        ps._choose_mode("memory", 44_000_000, 60_000, n, "h5ad", "census.h5ad")


def test_the_object_runs_where_the_plan_says(proj, monkeypatch) -> None:
    """Read with the plan saying out of core, the object's steps run out of core.
    (This fixture is too small for any budget to plan it out of core: its peak
    is fixed overhead, not the matrix.)"""
    class Spill:
        storage = "spill"
        min_budget_bytes = 0

    with monkeypatch.context() as m:
        m.setattr(ps, "_estimate", lambda *a, **k: Spill())
        ad = ps.read_h5ad(FIXTURE)
    assert ad.mode == "out-of-core"
    ps.pp.calculate_qc_metrics(ad)
    assert isinstance(ad.X, ps.DiskMatrix)


def test_out_of_core_needs_a_project() -> None:
    P._CURRENT = None
    ad = ps.read_h5ad(FIXTURE, mode="out-of-core")
    with pytest.raises(ps.ProjectNotSet, match="ps.project"):
        ps.pp.calculate_qc_metrics(ad)


def test_out_of_core_steps_equal_process_diskbacked_bit_for_bit(proj) -> None:
    """The steps, one by one on the object, are process_diskbacked's fused
    pipeline: the same kernels on the same blocks in the same order."""
    ad = steps(ps.read_h5ad(FIXTURE, mode="out-of-core", name="fx"), qc=False)
    ps.pp.neighbors(ad, n_neighbors=15, seed=42)
    ps.tl.leiden(ad, resolution=1.0, seed=42)
    ref = ps.process_diskbacked(FIXTURE, n_hvg=N_HVG, n_pcs=N_PCS, n_neighbors=15, seed=42,
                                min_genes=160)
    assert ad.n_obs == ref.n_obs < 10_000
    np.testing.assert_array_equal(ad.obsm["X_pca"], ref.obsm["X_pca"])
    np.testing.assert_array_equal(np.asarray(ad.obs["leiden"]), np.asarray(ref.obs["leiden"]))


def test_out_of_core_steps_match_the_in_memory_mode(proj) -> None:
    mem = steps(ps.read_h5ad(FIXTURE, mode="memory"))
    ooc = steps(ps.read_h5ad(FIXTURE, mode="out-of-core", name="fx"))
    assert ooc.n_obs == mem.n_obs < 10_000
    for k in ("n_genes", "n_counts", "pct_counts_mt"):
        np.testing.assert_array_equal(np.asarray(ooc.obs[k]), np.asarray(mem.obs[k]))
    assert list(ooc.var_names) == list(mem.var_names)  # the same genes selected
    qa, _ = np.linalg.qr(np.asarray(ooc.obsm["X_pca"], dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(mem.obsm["X_pca"], dtype=np.float64))
    angles = np.degrees(np.arccos(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1, 1)))
    assert angles.max() < 1.0, f"principal angles: {np.round(angles, 4)}"
    # markers of the same clusters, over the same genes
    labels = (np.arange(ooc.n_obs) % 4).astype(np.int32)
    for a in (mem, ooc):
        a.obs["grp"] = labels
        ps.tl.rank_genes_groups(a, groupby="grp", n_genes=15)
    for g in ("0", "1", "2", "3"):
        assert ooc.uns["rank_genes_groups"][g]["names"] == mem.uns["rank_genes_groups"][g]["names"]


def test_the_counts_copy_lives_in_the_project_and_is_reused(proj, capsys) -> None:
    ad = ps.read_h5ad(FIXTURE, mode="out-of-core", name="fx")
    ps.pp.calculate_qc_metrics(ad)
    matrix = proj.run_dir("fx") / "matrix"
    assert any((matrix / f"counts.{e}").exists() for e in ("data", "cz"))
    assert (matrix / "source.json").exists()
    assert ad.out_dir == proj.run_dir("fx")
    assert (proj.path / "README.md").exists() and (proj.path / "project.json").exists()
    assert "copied" in capsys.readouterr().out
    again = ps.read_h5ad(FIXTURE, mode="out-of-core", name="fx")
    ps.pp.calculate_qc_metrics(again)
    assert "reused" in capsys.readouterr().out
    assert proj.status()["runs/fx/matrix"] > 0


def test_gene_values_read_by_column_out_of_core(proj) -> None:
    mem = ps.read_h5ad(FIXTURE, mode="memory")
    ooc = ps.read_h5ad(FIXTURE, mode="out-of-core", name="fx")
    for a in (mem, ooc):
        ps.pp.normalize_total(a, target_sum=1e4)
        ps.pp.log1p(a)
    want = np.asarray(mem.X.to_scipy()[:, [3, 17]].toarray())
    np.testing.assert_array_equal(ooc.X[:, 3], want[:, 0])
    np.testing.assert_array_equal(ooc.X[:, [3, 17]], want)


def test_steps_that_need_the_whole_matrix_refuse_out_of_core(proj) -> None:
    ad = ps.read_h5ad(FIXTURE, mode="out-of-core", name="fx")
    with pytest.raises(NotImplementedError, match="out of core"):
        ps.tl.score_genes(ad, ["gene_1", "gene_2"])


def test_an_out_of_core_object_exports_without_its_matrix(proj) -> None:
    pytest.importorskip("anndata")
    ad = steps(ps.read_h5ad(FIXTURE, mode="out-of-core", name="fx"), qc=False)
    out = ad.to_anndata()
    assert out.X is None and out.n_obs == ad.n_obs and "X_pca" in out.obsm
