"""Several analyses of one dataset: ``adata.from_counts(name)``.

A new analysis starts again from the raw counts of the same cells, keeps the
cells' own annotations and drops what the first analysis derived; it keeps its
results in its own folder, so two analyses never replace each other's files;
and run with the same settings it gives the same result as the first.
"""

from __future__ import annotations

import gc

import numpy as np
import polariseq as ps
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def qc(a):
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_cells(a, min_genes=120)
    a.obs["donor"] = pd.Categorical(np.arange(a.n_obs) % 3)  # a column the user added


def rest(a, n_top=200):
    ps.pp.filter_genes(a, min_cells=50)
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=n_top, subset=True)
    ps.pp.renormalize(a)
    ps.pp.pca(a, 10, seed=0)
    ps.pp.neighbors(a, n_neighbors=10, seed=0)
    ps.tl.leiden(a, seed=0)


def first(mode: str):
    a = ps.read_h5ad(FIXTURE, mode=mode)
    qc(a)
    rest(a)
    ps.tl.annotate(a, {"0": "first"}, key_added="s5", unassigned="other")  # a derived label
    return a


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_the_same_settings_give_the_same_result(proj, mode) -> None:
    a = first(mode)
    b = a.from_counts("again")
    rest(b)
    np.testing.assert_array_equal(np.asarray(a.obsm["X_pca"]), np.asarray(b.obsm["X_pca"]))
    np.testing.assert_array_equal(np.asarray(a.obs["leiden"]), np.asarray(b.obs["leiden"]))


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_a_new_analysis_starts_from_the_counts_of_the_same_cells(proj, mode) -> None:
    a = first(mode)
    b = a.from_counts("more_genes")
    assert b.analysis == "more_genes" and a.analysis is None
    assert list(b.obs_names) == list(a.obs_names)
    assert b.n_vars == 2000  # every gene of the file, to filter again
    # the cells' own annotations come along; what the analysis derived does not
    assert {"n_counts", "n_genes", "donor"} <= set(b.obs)
    assert not {"leiden", "s5"} & set(b.obs)
    assert b.obsm == {} and b.obsp == {}
    # the raw counts of those cells
    raw = ad.read_h5ad(FIXTURE)
    want = np.asarray(raw[list(a.obs_names)].X.sum(axis=1)).ravel()
    if mode == "memory":
        got = np.asarray(b.X.to_scipy().sum(axis=1)).ravel()
    else:
        assert b.X.transforms() == (False, False)
        got = np.asarray(b.X[:, list(range(b.n_vars))].sum(axis=1)).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-6)
    # the first analysis is unchanged
    assert "leiden" in a.obs and a.n_vars == 200


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_analyses_keep_their_results_apart(proj, mode) -> None:
    a = first(mode)
    b = a.from_counts("hvg400")
    rest(b, n_top=400)
    run = proj.path / "runs" / "medium_10kx2k"
    own = list((run / "results" / "pca").glob("X_pca.*.npy"))
    other = list((run / "analyses" / "hvg400" / "results" / "pca").glob("X_pca.*.npy"))
    assert len(own) == 1 and len(other) == 1
    assert str(a.obsm["X_pca"].filename) == str(own[0])  # not replaced by b's run
    assert np.asarray(b.obsm["X_pca"]).shape == (b.n_obs, 10) and b.n_vars == 400
    assert "runs/medium_10kx2k/analyses/hvg400" in proj.status()


def test_current_genes(proj) -> None:
    a = first("memory")
    b = a.from_counts("same_genes", genes="current")
    assert list(b.var_names) == list(a.var_names)
    raw = ad.read_h5ad(FIXTURE)
    want = raw[list(a.obs_names), list(a.var_names)].X
    np.testing.assert_array_equal(b.X.to_scipy().toarray(), np.asarray(want.todense()) if hasattr(want, "todense") else want)


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_a_named_analysis_saves_and_loads(proj, mode) -> None:
    a = first(mode)
    b = a.from_counts("hvg300")
    rest(b, n_top=300)
    b.save("hvg300")
    c = ps.load(proj.path, "hvg300")
    assert c.analysis == "hvg300"
    np.testing.assert_array_equal(np.asarray(c.obs["leiden"]), np.asarray(b.obs["leiden"]))
    # a further step on the loaded analysis writes to its own folder
    ps.tl.leiden(c, resolution=0.5, seed=1, key_added="fine")
    assert (proj.path / "runs" / "medium_10kx2k" / "analyses" / "hvg300" / "results").exists()


def test_clean_results_includes_the_analyses(proj) -> None:
    a = first("out-of-core")
    b = a.from_counts("x")
    rest(b)
    del a, b   # Windows deletes no file an analysis still maps
    gc.collect()
    assert proj.clean("results") > 0
    assert not list((proj.path / "runs").glob("*/analyses/*/results"))


def test_mistakes_are_clear(proj) -> None:
    a = first("memory")
    with pytest.raises(ValueError, match="letters, digits"):
        a.from_counts("a/b")
    with pytest.raises(ValueError, match="genes must be 'all' or 'current'"):
        a.from_counts("x", genes="some")
    b = a.from_counts("x")
    with pytest.raises(ValueError, match="this analysis's own name"):
        b.from_counts("x")
    hand = ps.AnnData(X=np.abs(np.random.default_rng(0).normal(size=(50, 20))).astype(np.float32))
    ps.pp.normalize_total(hand)
    with pytest.raises(ValueError, match="no source .h5ad"):
        hand.from_counts("y")
