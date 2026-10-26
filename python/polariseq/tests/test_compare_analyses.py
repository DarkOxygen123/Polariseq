"""``ps.tl.compare_analyses``: agreement between analyses of the same cells."""

from __future__ import annotations

import warnings

import numpy as np
import polariseq as ps
import pytest
from polariseq._analyses import ari, nmi

pd = pytest.importorskip("pandas")
metrics = pytest.importorskip("sklearn.metrics")

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def test_the_scores_match_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.integers(0, 7, 500), rng.integers(0, 4, 500)
        b[:300] = a[:300] % 4  # partly agreeing
        assert ari(a, b) == pytest.approx(metrics.adjusted_rand_score(a, b), abs=1e-12)
        assert nmi(a, b) == pytest.approx(metrics.normalized_mutual_info_score(a, b), abs=1e-12)
    labels = np.array(["x", "y", "y", "z"])
    assert ari(labels, np.array([3, 1, 1, 2])) == 1.0  # names do not matter, only the grouping


def run(a, n_top=200, resolution=1.0):
    ps.pp.filter_genes(a, min_cells=50)
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=n_top, subset=True)
    ps.pp.pca(a, 10, seed=0)
    ps.pp.neighbors(a, n_neighbors=10, seed=0)
    ps.tl.leiden(a, resolution=resolution, seed=0)
    ps.tl.umap(a)
    ps.tl.rank_genes_groups(a, groupby="leiden")
    return a


@pytest.fixture
def three(proj):
    a = ps.read_h5ad(FIXTURE, mode="memory")
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_cells(a, min_genes=120)
    a.obs["truth"] = pd.Categorical(np.arange(a.n_obs) % 4)
    same = a.from_counts("same")
    other = a.from_counts("other")
    run(a)
    run(same)
    run(other, n_top=400, resolution=0.4)
    return {"first": a, "same": same, "other": other}


def test_settings_that_differ_agree_less(three) -> None:
    before = {n: (dict(a.obs), dict(a.uns["leiden"])) for n, a in three.items()}
    cmp = ps.tl.compare_analyses(three, labels="truth", seeds=2)
    assert cmp.n_cells == three["first"].n_obs
    assert cmp.ari.loc["first", "same"] == 1.0 and cmp.nmi.loc["first", "same"] == pytest.approx(1.0)
    assert cmp.ari.loc["first", "other"] < 1.0
    assert cmp.n_clusters["other"] < cmp.n_clusters["first"]  # lower resolution
    # which cluster became which, cells conserved
    m = cmp.matches[("first", "other")]
    assert m["cells"].sum() == cmp.n_cells and set(m.columns) >= {"match", "jaccard", "markers_jaccard"}
    assert cmp.crossing[("first", "other")].to_numpy().sum() == cmp.n_cells
    assert m["markers_jaccard"].between(0, 1).all()
    assert list(cmp.labels.index) == ["first", "same", "other"]
    assert cmp.noise.between(0, 1).all()
    assert "adjusted Rand index" in repr(cmp)
    # the analyses are left as they were
    for n, a in three.items():
        obs, leiden = before[n]
        assert set(a.obs) == set(obs) and a.uns["leiden"] == leiden


def test_different_cells_are_compared_on_those_shared(three) -> None:
    sub = three["other"][np.arange(three["other"].n_obs) % 2 == 0]
    with pytest.warns(UserWarning, match="compared on the"):
        cmp = ps.tl.compare_analyses({"first": three["first"], "half": sub})
    assert cmp.n_cells == sub.n_obs


def test_the_plot_draws(three, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    axes = ps.pl.compare_analyses(three, show=False, save=str(tmp_path / "cmp.png"))
    assert (tmp_path / "cmp.png").stat().st_size > 0 and len(axes) >= 3


def test_mistakes_are_clear(three) -> None:
    with pytest.raises(ValueError, match="two analyses or more"):
        ps.tl.compare_analyses({"first": three["first"]})
    with pytest.raises(KeyError, match="Did you mean 'leiden'"):
        ps.tl.compare_analyses(three, key="leidn")
    with pytest.raises(KeyError, match="labels='truths'"):
        ps.tl.compare_analyses(three, labels="truths")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lonely = ps.read_h5ad(FIXTURE, mode="memory")
    lonely.obs_names = [f"other_{i}" for i in range(lonely.n_obs)]
    lonely.obs["leiden"] = np.zeros(lonely.n_obs, dtype=np.int32)
    with pytest.raises(ValueError, match="share no cells"):
        ps.tl.compare_analyses({"first": three["first"], "lonely": lonely})
