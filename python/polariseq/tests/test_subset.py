"""Subsetting: ``adata[cells]``, ``adata[cells, genes]`` and ``copy()``.

In memory a subset must hold what anndata's does (matrix, annotations, names)
and be independent of the original. Out of core it is a second view over the
same on-disk counts: nothing copied, the same values read, and each view's
later filters its own.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")


@pytest.fixture(scope="module")
def path(tmp_path_factory) -> str:
    rng = np.random.default_rng(3)
    n, g = 300, 40
    x = sp.random(n, g, density=0.35, format="csr", dtype=np.float32, random_state=3)
    x.data = np.ceil(x.data * 7)
    obs = pd.DataFrame({
        "condition": pd.Categorical(rng.choice(["control", "treated"], n)),
        "donor": pd.Categorical(rng.choice(["d1", "d2", "d3"], n)),
        "score": rng.normal(size=n).astype(np.float32),
    }, index=[f"cell{i}" for i in range(n)])
    var = pd.DataFrame({"gene_ids": [f"E{j}" for j in range(g)]},
                       index=[f"g{j}" for j in range(g)])
    p = tmp_path_factory.mktemp("subset") / "data.h5ad"
    ad.AnnData(X=x, obs=obs, var=var).write_h5ad(p)
    return str(p)


def _check(sub, want) -> None:
    assert sub.shape == want.shape
    np.testing.assert_array_equal(sub.X.to_scipy().toarray(), want.X.toarray())
    assert list(sub.obs_names) == list(want.obs_names)
    assert list(sub.var_names) == list(want.var_names)
    for c in want.obs.columns:
        assert list(np.asarray(sub.obs[c], dtype=object)) == list(want.obs[c].astype(object))
    assert list(np.asarray(sub.var["gene_ids"], dtype=object)) == list(want.var["gene_ids"])


def test_every_kind_of_key_selects_as_anndata(path):
    a = ps.read_h5ad(path, mode="memory")
    ref = ad.read_h5ad(path)
    ctrl = np.asarray(a.obs["condition"] == "control")
    keys = [
        ctrl,
        np.array([5, 0, 17, 299]),
        ["cell3", "cell1"],
        slice(10, 50),
        (ctrl, ["g3", "g0", "g39"]),
        (slice(None), np.arange(40) % 3 == 0),
        (np.array([2, 4]), slice(5, 9)),
    ]
    for key in keys:
        _check(a[key], ref[key])
    # Categories survive, codes stay compact.
    assert isinstance(a[ctrl].obs["donor"], pd.Categorical)


def test_a_subset_is_independent_of_the_original(path):
    a = ps.read_h5ad(path, lazy=False)
    b = a[np.arange(a.n_obs) < 100]
    b.obs["score"][:] = 0
    ps.pp.normalize_total(b)
    assert np.asarray(a.obs["score"]).any()
    np.testing.assert_array_equal(a.X.to_scipy()[:100].toarray(),
                                  ad.read_h5ad(path).X[:100].toarray())
    c = a.copy()
    ps.pp.log1p(c)
    assert a.X.to_scipy().max() == ad.read_h5ad(path).X.max()


def test_embeddings_and_graph_follow_the_cells(path):
    a = ps.read_h5ad(path, lazy=False)
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    ps.pp.pca(a, n_components=5)
    ps.pp.neighbors(a, n_neighbors=5)
    keep = np.asarray(a.obs["donor"] != "d2")
    b = a[keep]
    np.testing.assert_array_equal(b.obsm["X_pca"], a.obsm["X_pca"][keep])
    for k, v in a.obsp.items():
        np.testing.assert_array_equal(b.obsp[k].toarray(), v[keep][:, keep].toarray())


def test_out_of_core_subset_is_a_view_over_the_same_counts(path, proj):
    a = ps.read_h5ad(path, mode="out-of-core")
    ref = ad.read_h5ad(path)
    treated = np.asarray(a.obs["condition"] == "treated")
    genes = ["g1", "g7", "g30"]
    b = a[treated, genes]
    assert b.shape == (int(treated.sum()), 3)
    assert a.shape == ref.shape  # the original is untouched
    assert b.X.prefix == a.X.prefix  # the same store, nothing copied
    want = ref[treated, genes].X.toarray()
    np.testing.assert_array_equal(b.X[:, [0, 1, 2]], want)
    # Each view filters on its own.
    ps.pp.normalize_total(b)
    np.testing.assert_array_equal(a.X[:, [1]].ravel(), ref.X[:, 1].toarray().ravel())
    with pytest.raises(IndexError, match="stored order"):
        a[np.array([5, 2])]
