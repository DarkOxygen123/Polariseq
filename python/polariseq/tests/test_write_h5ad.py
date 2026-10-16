"""``adata.write_h5ad(path)``: the analysis as an ``.h5ad`` other tools open.

Read back with anndata, the file holds X as the analysis left it (in memory and
streamed out of core, blocks in any order), the raw counts when asked, every
annotation, embedding, graph and result, and a pointer to the source files.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def analyzed(mode: str):
    a = ps.read_h5ad(FIXTURE, mode=mode)
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_cells(a, min_genes=120)
    ps.pp.filter_genes(a, min_cells=50)
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=200, subset=True)
    ps.pp.pca(a, 10, seed=0)
    ps.pp.neighbors(a, n_neighbors=10, seed=0)
    ps.tl.leiden(a, seed=0)
    ps.tl.umap(a)
    a.obs["donor"] = pd.Categorical(np.arange(a.n_obs) % 3)
    a.obs["flag"] = np.arange(a.n_obs) % 2 == 0
    a.obs["note"] = np.array(["x", "y"] * (a.n_obs // 2) + ["x"] * (a.n_obs % 2), dtype=object)
    return a


def current_x(a) -> np.ndarray:
    """The analysis's X as a dense array."""
    if isinstance(a.X, ps.DiskMatrix):
        return np.asarray(a.X[:, list(range(a.n_vars))])
    return a.X.to_scipy().toarray()


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_the_file_holds_the_analysis(proj, tmp_path, mode, monkeypatch) -> None:
    import polariseq._h5ad as h5

    monkeypatch.setattr(h5, "BLOCK_ROWS", 777)  # many blocks, finishing in any order
    a = analyzed(mode)
    path = a.write_h5ad(tmp_path / "out.h5ad")
    b = ad.read_h5ad(path)
    assert b.shape == a.shape
    assert list(b.obs_names) == list(a.obs_names) and list(b.var_names) == list(a.var_names)
    assert sp.issparse(b.X)
    np.testing.assert_array_equal(b.X.toarray(), current_x(a))  # the values as the analysis left them
    for k in ("X_pca", "X_umap"):
        np.testing.assert_array_equal(b.obsm[k], np.asarray(a.obsm[k]))
    assert (b.obsp["connectivities"] != a.obsp["connectivities"]).nnz == 0
    assert b.obs["leiden"].dtype.name == "category"  # Scanpy's convention
    np.testing.assert_array_equal(b.obs["leiden"].astype(int).to_numpy(), np.asarray(a.obs["leiden"]))
    assert b.obs["donor"].cat.categories.tolist() == [0, 1, 2]
    assert b.obs["flag"].dtype == bool and b.obs["flag"].tolist() == list(a.obs["flag"])
    assert b.obs["note"].tolist()[:2] == ["x", "y"]
    np.testing.assert_array_equal(b.obs["n_counts"].to_numpy(), np.asarray(a.obs["n_counts"]))
    assert b.uns["neighbors"]["connectivities_key"] == "connectivities"
    assert b.uns["leiden"]["n_communities"] == a.uns["leiden"]["n_communities"]
    assert "counts" not in b.layers


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_counts_and_the_pointer_to_the_source(proj, tmp_path, mode) -> None:
    a = analyzed(mode)
    b = ad.read_h5ad(a.write_h5ad(tmp_path / "with_counts.h5ad", counts=True))
    raw = ad.read_h5ad(FIXTURE)
    want = raw[list(a.obs_names), list(a.var_names)].X
    want = want.toarray() if sp.issparse(want) else np.asarray(want)
    np.testing.assert_array_equal(b.layers["counts"].toarray(), want)
    p = b.uns["polariseq"]
    assert p["source_paths"][0].endswith("medium_10kx2k.h5ad") and p["counts_in_file"]
    # rows and cols say which of the file's cells and genes the file holds
    names = raw.obs_names[np.asarray(p["rows"])].tolist()
    assert names == list(a.obs_names)
    assert raw.var_names[np.asarray(p["cols"])].tolist() == list(a.var_names)
    assert list(p["x_steps"])[:2] in (["normalize_total(target_sum=10000)", "log1p"],
                                      ["normalize_total", "log1p"])


def test_tables_and_nested_results(proj, tmp_path) -> None:
    a = analyzed("memory")
    a.obs["cond"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "ctrl", "stim"))
    res = ps.tl.compare_conditions(a, "cond", "ctrl", groupby="leiden", sample_key="donor")
    ps.tl.rank_genes_groups(a, groupby="leiden")
    a.uns["mine"] = {"list": [1, 2, 3], "words": ["a", "b"], "mixed": [1, "a"], "flag": True,
                     "nothing": None, "nested": {"x": np.arange(3)}}
    b = ad.read_h5ad(a.write_h5ad(tmp_path / "t.h5ad"))
    t = b.uns["compare_conditions"]["table"]
    assert list(t.columns) == list(res.columns) and len(t) == len(res)
    np.testing.assert_allclose(t["p_value"].to_numpy(), res["p_value"].to_numpy())
    assert "rank_genes_groups" in b.uns
    m = b.uns["mine"]
    assert list(m["list"]) == [1, 2, 3] and list(m["words"]) == ["a", "b"]
    assert m["mixed"] == {"0": 1, "1": "a"} and bool(m["flag"]) is True and "nothing" not in m
    np.testing.assert_array_equal(m["nested"]["x"], np.arange(3))


def test_the_one_call_pipeline_writes_without_x(proj, tmp_path) -> None:
    ps.set_budget(ram_gb=2)
    a = ps.process_diskbacked(FIXTURE, n_hvg=200, n_pcs=10, n_neighbors=10)
    b = ad.read_h5ad(a.write_h5ad(tmp_path / "pd.h5ad"))
    assert b.X is None and b.n_obs == a.n_obs and "X_pca" in b.obsm


def test_an_interrupted_write_leaves_nothing(proj, tmp_path, monkeypatch) -> None:
    import polariseq._h5ad as h5

    a = analyzed("memory")
    target = tmp_path / "x.h5ad"
    target.write_text("the old file")

    def boom(self, path, df):
        raise RuntimeError("disk full")

    monkeypatch.setattr(h5._Encoder, "frame", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        a.write_h5ad(target)
    assert target.read_text() == "the old file"  # untouched
    assert not list(tmp_path.glob(".x.h5ad.partial-*"))


def test_mistakes_are_clear(proj, tmp_path) -> None:
    a = analyzed("memory")
    with pytest.raises(FileNotFoundError, match="does not exist; create it first"):
        a.write_h5ad(tmp_path / "no" / "x.h5ad")
    with pytest.raises(ValueError, match="is a folder"):
        a.write_h5ad(tmp_path)
    hand = ps.AnnData(X=np.ones((5, 3), dtype=np.float32))
    ps.pp.log1p(hand)
    with pytest.raises(ValueError, match="needs the raw counts"):
        hand.write_h5ad(tmp_path / "h.h5ad", counts=True)


def test_a_named_analysis_and_a_loaded_one_write_too(proj, tmp_path) -> None:
    a = analyzed("out-of-core")
    b = a.from_counts("again")
    ps.pp.normalize_total(b)
    ps.pp.log1p(b)
    a.save("s")
    c = ps.load(proj.path, "s")
    for obj, name in ((b, "b.h5ad"), (c, "c.h5ad")):
        r = ad.read_h5ad(obj.write_h5ad(tmp_path / name))
        np.testing.assert_array_equal(r.X.toarray(), current_x(obj))
    assert ad.read_h5ad(tmp_path / "b.h5ad").uns["polariseq"]["analysis"] == "again"
