"""Several datasets, one object: files read as one dataset over one set of genes.

`ps.read_h5ad([...])` reads several `.h5ad` files the way `anndata.concat`
merges them, without the merge: in memory each row is remapped as it is read,
and out of core the rows are copied once, the same way, into one block store
in the project folder. So the matrix must be the one `anndata.concat` builds,
in canonical form (each row's entries sorted by gene; anndata leaves a row
unsorted when a file lists its genes in another order), and analyzing the
files must equal, bit for bit, analyzing that merged file.
"""

from __future__ import annotations

import os

import numpy as np
import polariseq as ps
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")

FIXTURE = "tests/data/medium_10kx2k.h5ad"


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> list[str]:
    """The fixture's 10,000 cells in three files that disagree on genes: the
    second lacks 100 genes and lists the rest in reverse; the third lacks
    another 100 and has 10 of its own."""
    full = ad.read_h5ad(FIXTURE)
    full.X = sp.csr_matrix(full.X)
    a = full[:4000].copy()
    b = full[4000:7000, [f"gene_{i}" for i in range(1999, 99, -1)]].copy()
    c = full[7000:, [f"gene_{i}" for i in range(1900)]].copy()
    own = sp.random(c.n_obs, 10, density=0.3, format="csr", dtype=np.float32, random_state=1)
    own.data = np.ceil(own.data * 5)
    c = ad.concat([c, ad.AnnData(own, obs=c.obs[[]],
                                 var=pd.DataFrame(index=[f"own_{i}" for i in range(10)]))],
                  axis=1)
    c.X = sp.csr_matrix(c.X)
    folder = tmp_path_factory.mktemp("datasets")
    paths = [str(folder / f"{k}.h5ad") for k in "abc"]
    for x, p in zip((a, b, c), paths, strict=True):
        x.write_h5ad(p)
    return paths


def concat(paths: list[str], join: str):
    """What anndata.concat makes of the files, and its matrix."""
    want = ad.concat([ad.read_h5ad(p) for p in paths], join=join, label="dataset",
                     keys=[os.path.basename(p)[: -len(".h5ad")] for p in paths])
    x = sp.csr_matrix(want.X, dtype=np.float32)
    x.sort_indices()
    return want, x


def analyze(adata):
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=120)
    ps.pp.filter_genes(adata, min_cells=300)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=200, subset=True)
    ps.pp.pca(adata, 10, seed=42)
    return adata


JOINS = [("intersection", "inner"), ("union", "outer")]


@pytest.mark.parametrize(("genes", "join"), JOINS)
def test_the_files_read_as_the_matrix_anndata_concat_writes(files, genes, join) -> None:
    want, wx = concat(files, join)
    got = ps.read_h5ad(files, genes=genes, mode="memory")
    assert list(got.var_names) == list(want.var_names)
    assert list(got.obs_names) == list(want.obs_names)
    assert list(got.obs["dataset"]) == list(want.obs["dataset"])
    gx = got.X.to_scipy()
    for k in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(getattr(gx, k), getattr(wx, k))


@pytest.mark.parametrize(("genes", "join"), JOINS)
def test_out_of_core_the_copy_is_that_matrix(files, proj, genes, join, monkeypatch) -> None:
    # This checks the copy's bytes, so it writes the plain layout; that the
    # compact layout decodes to the same matrix is test_store_layouts.py.
    monkeypatch.setenv("POLARISEQ_STORE", "plain")
    _, wx = concat(files, join)
    got = ps.read_h5ad(files, genes=genes, mode="out-of-core")
    got._materialize()
    prefix = got.out_dir / "matrix" / "counts"
    for k, dtype in (("indptr", "<u8"), ("indices", "<u4"), ("data", "<f4")):
        np.testing.assert_array_equal(np.fromfile(f"{prefix}.{k}", dtype=dtype), getattr(wx, k))
    assert got.X.shape == wx.shape and got.X.nnz == wx.nnz


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_analyzing_the_files_equals_analyzing_their_merged_file(files, proj, tmp_path,
                                                               mode) -> None:
    """Bit for bit, in either mode: the same matrix and, out of core, the same
    blocks, though the files were planned from an upper bound on entries."""
    want, wx = concat(files, "inner")
    want.X = wx
    merged = str(tmp_path / "merged.h5ad")
    want.write_h5ad(merged)
    one = analyze(ps.read_h5ad(merged, mode=mode, name="merged"))
    many = analyze(ps.read_h5ad(files, mode=mode))
    assert list(many.var_names) == list(one.var_names)
    for k in ("n_genes", "n_counts"):
        np.testing.assert_array_equal(np.asarray(many.obs[k]), np.asarray(one.obs[k]))
    np.testing.assert_array_equal(many.obsm["X_pca"], one.obsm["X_pca"])


def test_planning_several_files(files) -> None:
    p = ps.plan(files, verbose=False)
    assert (p.n_cells, p.n_genes) == (10_000, 1_800)
    assert [d["genes_kept"] for d in p.datasets] == [1_800, 1_800, 1_800]
    assert p.nnz == sum(d["nnz"] for d in p.datasets)
    union = ps.plan(files, genes="union", storage="spill", verbose=False)
    assert union.n_genes == 2_010
    text = union.report()
    assert "3 datasets" in text and "copy of the counts" in text


def test_each_cell_keeps_its_dataset(files, proj) -> None:
    got = analyze(ps.read_h5ad(dict(zip(["d1", "d2", "d3"], files, strict=True)),
                               mode="out-of-core"))
    assert got.out_dir == proj.run_dir("d1+d2+d3")
    assert set(np.asarray(got.obs["dataset"])) == {"d1", "d2", "d3"}
    ps.pp.harmony_integrate(got, "dataset", seed=0)
    assert got.uns["harmony"]["n_batches"] == 3
    corrected = got.obsm["X_pca_harmony"]  # written by the core into the project
    assert isinstance(corrected, np.memmap) and "results/harmony" in str(corrected.filename)
    out = got.to_anndata()
    assert list(out.obs["dataset"].cat.categories) == ["d1", "d2", "d3"]


def test_the_copy_is_reused_until_a_file_changes(files, proj, capsys) -> None:
    ps.read_h5ad(files, mode="out-of-core")._materialize()
    assert "copied" in capsys.readouterr().out
    ps.read_h5ad(files, mode="out-of-core")._materialize()
    assert "reused" in capsys.readouterr().out
    st = os.stat(files[1])
    os.utime(files[1], (st.st_atime, st.st_mtime + 5))
    try:
        ps.read_h5ad(files, mode="out-of-core")._materialize()
        assert "copied" in capsys.readouterr().out
    finally:
        os.utime(files[1], (st.st_atime, st.st_mtime))


def test_rows_read_across_files(files) -> None:
    _, wx = concat(files, "inner")
    got = ps.read_h5ad(files)
    rows, want = got._read_rows(3995, 4005), wx[3995:4005]
    for k in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(rows[k], getattr(want, k))
    assert got.head(3)["n_cells"] == 3


def test_files_that_cannot_be_read_as_one_are_refused(files, tmp_path) -> None:
    with pytest.raises(ValueError, match="dict"):
        ps.read_h5ad([files[0], files[0]])
    with pytest.raises(ValueError, match="intersection"):
        ps.read_h5ad(files, genes="all")
    repeats = ad.read_h5ad(files[0])
    repeats.var_names = ["g", "g", *repeats.var_names[2:]]
    path = str(tmp_path / "repeats.h5ad")
    repeats.write_h5ad(path)
    with pytest.raises(ValueError, match="repeat"):
        ps.read_h5ad([files[1], path])


def test_planning_passes_the_densest_file_to_the_estimate(files, monkeypatch) -> None:
    """Gene selection is charged at the densest file's values per cell, which
    only the scan of several files knows; one file is planned at its average."""
    seen: list = []
    real = ps._estimate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("densest_nnz_per_row"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ps, "_estimate", spy)
    p = ps.plan(files, verbose=False)
    want = int(max(d["nnz"] / d["n_cells"] for d in p.datasets))
    assert seen == [want] and want > 0
    ps.plan(files[0], verbose=False)
    assert seen[-1] is None


def test_the_peak_announced_when_the_copy_opens_is_the_plans(files, proj, monkeypatch) -> None:
    """The copy's log line used to say 30.1 GB of a run the plan put at 39.2:
    it re-estimates from the store's own count and left out the densest file."""
    seen: list = []
    real = ps._estimate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("densest_nnz_per_row"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ps, "_estimate", spy)
    adata = ps.read_h5ad(files, mode="out-of-core")
    ps.pp.calculate_qc_metrics(adata)
    assert len(seen) >= 2 and None not in seen and len(set(seen)) == 1
