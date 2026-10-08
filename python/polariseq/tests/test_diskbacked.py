"""The disk-backed pipeline: same answers as the in-memory one.

`process_diskbacked` streams the matrix onto SSD and processes it in place.
Its whole justification is that it runs the *same* kernels as the in-memory
path, so on a file small enough to do both, the results must agree.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

FIXTURE = "tests/data/medium_10kx2k.h5ad"
N_HVG = 200
N_PCS = 10


@pytest.fixture(scope="module", params=["polariseq", "scanpy"])
def wf(request):
    """Both workflows: the two modes must agree under each (see ps.set_workflow)."""
    before = ps.workflow()
    ps.set_workflow(request.param)
    yield request.param
    ps.set_workflow(before)


@pytest.fixture(scope="module")
def disk_result(wf):
    return ps.process_diskbacked(
        FIXTURE, n_hvg=N_HVG, n_pcs=N_PCS, n_neighbors=15, seed=42
    )


@pytest.fixture(scope="module")
def memory_result(wf):
    adata = ps.read_h5ad(FIXTURE)
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=True)
    if wf == "polariseq":   # the one-call pipeline normalizes again over the genes and scales them
        ps.pp.renormalize(adata, target_sum=1000)
    ps.pp.pca(adata, N_PCS, seed=42, scale=(wf == "polariseq"))
    return adata


class TestDiskBacked:
    def test_shapes_and_annotations(self, disk_result) -> None:
        a = disk_result
        assert a.n_obs == 10_000
        assert a.obsm["X_pca"].shape == (10_000, N_PCS)
        assert a.uns["pca"]["disk_backed"] is True
        for key in ("n_counts", "n_genes"):
            assert len(a.obs[key]) == 10_000
        for key in ("n_cells", "total_counts", "highly_variable"):
            assert len(a.var[key]) == 2_000
        assert a.var["highly_variable"].sum() == N_HVG
        assert "leiden" in a.obs and a.uns["leiden"]["n_communities"] >= 1

    def test_qc_matches_in_memory(self, disk_result, memory_result) -> None:
        # QC is computed on the raw counts in both paths.
        ref = ps.read_h5ad(FIXTURE)
        ps.pp.calculate_qc_metrics(ref)
        np.testing.assert_allclose(
            disk_result.obs["n_counts"], ref.obs["n_counts"], rtol=1e-5
        )
        np.testing.assert_array_equal(disk_result.obs["n_genes"], ref.obs["n_genes"])
        np.testing.assert_array_equal(disk_result.var["n_cells"], ref.var["n_cells"])

    def test_hvg_selection_matches_in_memory(self, disk_result, memory_result) -> None:
        # the workflow's default selection in memory (fixture memory_result sets the workflow)
        ref = ps.read_h5ad(FIXTURE)
        ps.pp.normalize_total(ref, target_sum=1e4)
        ps.pp.log1p(ref)
        ps.pp.highly_variable_genes(ref, n_top_genes=N_HVG)
        got = np.asarray(disk_result.var["highly_variable"], dtype=bool)
        want = np.asarray(ref.var["highly_variable"], dtype=bool)
        overlap = (got & want).sum() / N_HVG
        assert overlap >= 0.99, f"HVG overlap {overlap:.3f}"

    def test_pca_matches_in_memory(self, disk_result, memory_result) -> None:
        """Same subspace as the in-memory PCA (signs are arbitrary)."""
        a = np.asarray(disk_result.obsm["X_pca"], dtype=np.float64)
        b = np.asarray(memory_result.obsm["X_pca"], dtype=np.float64)
        assert a.shape == b.shape

        qa, _ = np.linalg.qr(a)
        qb, _ = np.linalg.qr(b)
        angles = np.degrees(
            np.arccos(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0))
        )
        assert angles.max() < 1.0, f"principal angles: {np.round(angles, 4)}"

        np.testing.assert_allclose(
            disk_result.uns["pca"]["variance_ratio"],
            memory_result.uns["pca"]["variance_ratio"],
            rtol=5e-2,
        )

    def test_min_genes_filters_like_the_in_memory_pipeline(self) -> None:
        """`min_genes` is the disk path's `filter_cells`.

        Before it existed, the out-of-core pipeline kept every cell while the
        in-memory one filtered — same parameters, different matrix, different
        answer (the telemetry harness caught it: PCA 90 deg off from PC2,
        ARI 0.07, because the two arms weren't decomposing the same cells).
        With the filter threaded through, the two paths must agree again.
        """
        MIN_GENES = 160  # fixture n_genes spans 122..223; this drops a tail

        disk = ps.process_diskbacked(
            FIXTURE, n_hvg=N_HVG, n_pcs=N_PCS, n_neighbors=15, seed=42,
            min_genes=MIN_GENES,
        )
        assert disk.n_obs < 10_000, "min_genes=160 must drop cells on this fixture"

        mem = ps.read_h5ad(FIXTURE)
        ps.pp.calculate_qc_metrics(mem)
        ps.pp.filter_cells(mem, min_genes=MIN_GENES)
        ps.pp.normalize_total(mem, target_sum=1e4)
        ps.pp.log1p(mem)
        ps.pp.highly_variable_genes(mem, n_top_genes=N_HVG, subset=True)
        ps.pp.pca(mem, N_PCS, seed=42)

        assert disk.n_obs == mem.n_obs
        a = np.asarray(disk.obsm["X_pca"], dtype=np.float64)
        b = np.asarray(mem.obsm["X_pca"], dtype=np.float64)
        assert a.shape == b.shape
        qa, _ = np.linalg.qr(a)
        qb, _ = np.linalg.qr(b)
        angles = np.degrees(
            np.arccos(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0))
        )
        assert angles.max() < 1.0, f"principal angles: {np.round(angles, 4)}"
        np.testing.assert_allclose(
            disk.obs["n_genes"], mem.obs["n_genes"][: disk.n_obs], rtol=0
        )


def test_the_run_deletes_its_spill(disk_result) -> None:
    """Until 2026-09-29 the raw spill was never deleted: 8 bytes per stored
    value left behind by every run."""
    import os

    left = [f for f in os.listdir(ps.budget().spill_dir)
            if f.startswith((f"raw_{os.getpid()}.", f"hvg_{os.getpid()}."))]
    assert left == []


def test_orphan_spills_of_dead_processes_are_swept(tmp_path) -> None:
    import os
    import subprocess
    import sys

    from polariseq import _sweep_orphan_spills

    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True)
    dead_pid = int(dead.stdout)
    for ext in ("data", "indices", "indptr"):
        (tmp_path / f"raw_{dead_pid}.{ext}").write_bytes(b"x" * 100)
        (tmp_path / f"raw_{os.getpid()}.{ext}").write_bytes(b"x")  # this process: kept
    (tmp_path / "unrelated.txt").write_text("keep")
    assert _sweep_orphan_spills(str(tmp_path)) == 300
    assert sorted(f.name for f in tmp_path.iterdir()) == sorted(
        [f"raw_{os.getpid()}.{e}" for e in ("data", "indices", "indptr")] + ["unrelated.txt"])


# ── Quality control out of core: every limit of the in-memory functions ──

MITO = "gene_19"  # stands in for "MT-": 111 of the fixture's genes, 0-15% of counts
QC = dict(min_genes=150, max_genes=210, min_counts=320, max_counts=500, max_pct_mt=8.0)
MIN_CELLS = 700  # genes are in 766-962 of all cells, about 13% fewer after the cell filter


def _in_memory_qc():
    """The in-memory steps a user writes: QC, filter_cells, filter_genes."""
    mem = ps.read_h5ad(FIXTURE)
    ps.pp.calculate_qc_metrics(mem, mito_prefix=MITO)
    ps.pp.filter_cells(mem, **QC)
    ps.pp.filter_genes(mem, min_cells=MIN_CELLS)
    return mem


@pytest.fixture(scope="module")
def disk_qc():
    return ps.process_diskbacked(
        FIXTURE, n_hvg=N_HVG, n_pcs=N_PCS, n_neighbors=15, seed=42,
        mito_prefix=MITO, min_cells=MIN_CELLS, **QC,
    )


def test_qc_limits_keep_the_same_cells_and_genes_as_in_memory(disk_qc) -> None:
    mem = _in_memory_qc()
    assert disk_qc.n_obs == mem.n_obs < 10_000, "the limits must remove cells"
    np.testing.assert_array_equal(disk_qc.obs["n_genes"], mem.obs["n_genes"])
    np.testing.assert_array_equal(disk_qc.obs["n_counts"], mem.obs["n_counts"])
    np.testing.assert_array_equal(disk_qc.obs["pct_counts_mt"], mem.obs["pct_counts_mt"])
    kept = np.asarray(disk_qc.var["kept"], dtype=bool)
    assert kept.sum() == mem.n_vars < 2_000, "min_cells must remove genes"
    assert [str(g) for g in np.asarray(disk_qc.var["name"])[kept]] == list(mem.var_names)
    assert not np.asarray(disk_qc.var["highly_variable"], dtype=bool)[~kept].any()


def test_qc_limits_give_the_in_memory_genes_and_components(disk_qc) -> None:
    mem = _in_memory_qc()
    ps.pp.normalize_total(mem, target_sum=1e4)
    ps.pp.log1p(mem)
    ps.pp.highly_variable_genes(mem, n_top_genes=N_HVG)
    kept = np.asarray(disk_qc.var["kept"], dtype=bool)
    got = np.asarray(disk_qc.var["highly_variable"], dtype=bool)[kept]
    want = np.asarray(mem.var["highly_variable"], dtype=bool)
    assert (got & want).sum() / N_HVG >= 0.99
    ps.pp.highly_variable_genes(mem, n_top_genes=N_HVG, subset=True)
    ps.pp.pca(mem, N_PCS, seed=42)
    qa, _ = np.linalg.qr(np.asarray(disk_qc.obsm["X_pca"], dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(mem.obsm["X_pca"], dtype=np.float64))
    angles = np.degrees(np.arccos(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1, 1)))
    assert angles.max() < 1.0, f"principal angles: {np.round(angles, 4)}"


def test_max_pct_mt_without_mitochondrial_genes_is_refused() -> None:
    with pytest.raises(ValueError, match="mitochondrial"):
        ps.process_diskbacked(FIXTURE, n_hvg=N_HVG, n_pcs=N_PCS, max_pct_mt=5.0,
                              mito_prefix="MT-")


def test_filter_genes_counts_the_cells_left_after_filter_cells() -> None:
    """Scanpy counts a gene's cells on the matrix as it is when filter_genes
    runs. Counting from an earlier calculate_qc_metrics included the cells
    filter_cells had removed, and so kept genes Scanpy drops."""
    a = ps.read_h5ad(FIXTURE)
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_cells(a, min_genes=180)
    now = np.asarray((a.X.to_scipy() > 0).sum(axis=0)).ravel()
    ps.pp.filter_genes(a, min_cells=400)
    assert a.n_vars == int((now >= 400).sum()) < 2_000


# ── Marker genes of an out-of-core result ──

def test_rank_genes_groups_out_of_core_matches_in_memory(disk_qc) -> None:
    """The matrix of an out-of-core result stays in its file; markers are
    ranked from statistics streamed with the analysis's normalization and
    gene filter, and must match the in-memory ranking of the same cells,
    genes and clusters."""
    ps.tl.rank_genes_groups(disk_qc, groupby="leiden", n_genes=25)
    mem = _in_memory_qc()
    ps.pp.normalize_total(mem, target_sum=1e4)
    ps.pp.log1p(mem)
    mem.obs["leiden"] = np.asarray(disk_qc.obs["leiden"])
    ps.tl.rank_genes_groups(mem, groupby="leiden", n_genes=25)
    got, want = disk_qc.uns["rank_genes_groups"], mem.uns["rank_genes_groups"]
    assert set(got) == set(want) and len(got) >= 2
    for g in want:
        assert got[g]["names"] == want[g]["names"], g
        np.testing.assert_allclose(got[g]["scores"], want[g]["scores"], rtol=1e-6)
        np.testing.assert_allclose(got[g]["pvals_adj"], want[g]["pvals_adj"], rtol=1e-6)


# ── Export to Scanpy ──

def test_to_anndata_runs_scanpy_paga_directly(disk_result) -> None:
    """Scanpy's tools must run on the exported object without setup:
    clusters as categoricals, neighbor keys where Scanpy looks for them."""
    pytest.importorskip("igraph")   # Scanpy's PAGA needs python-igraph
    sc = pytest.importorskip("scanpy")
    ad = disk_result.to_anndata()
    assert ad.obs["leiden"].dtype.name == "category"
    assert ad.uns["neighbors"]["connectivities_key"] == "connectivities"
    sc.tl.paga(ad, groups="leiden")
    k = len(ad.obs["leiden"].cat.categories)
    assert ad.uns["paga"]["connectivities"].shape == (k, k)
