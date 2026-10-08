"""Curating clusters into cell types: annotate, merge_clusters, subcluster, get.markers, and the
labels of an embedding plot (numbers, names, callouts).

The curation steps rewrite a label column and nothing else, so they must give the same result in
memory and out of core; splitting a cluster must find the structure inside it; and the markers of a
curated (named, joined or split) labelling must be the ones of its cells.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

pd = pytest.importorskip("pandas")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def _ari(a, b) -> float:
    a = pd.factorize(np.asarray(a))[0]
    b = pd.factorize(np.asarray(b))[0]
    ct = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(ct, (a, b), 1)

    def c2(x):
        return (x * (x - 1) // 2).sum()

    n = len(a)
    si, sj, sij = c2(ct.sum(1)), c2(ct.sum(0)), c2(ct)
    exp = si * sj / (n * (n - 1) // 2)
    return float((sij - exp) / (0.5 * (si + sj) - exp))


@pytest.fixture(scope="module")
def blobs():
    """Six well-separated populations in 20 dimensions; the last two lie close together, so a
    coarse labelling can hold them as one cluster that subcluster must split again."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(6, 20)) * 8.0
    centres[5] = centres[4] + 3.5          # two neighbours, still separable
    sizes = [900, 700, 500, 400, 300, 300]
    truth = np.repeat(np.arange(6), sizes)
    X = (centres[truth] + rng.normal(size=(truth.size, 20))).astype(np.float32)
    ad = ps.AnnData(X=np.zeros((truth.size, 1), dtype=np.float32))
    ad.obsm["X_pca"] = X
    ad.obsm["X_umap"] = X[:, :2] * 1.0
    ps.pp.neighbors(ad, n_neighbors=15, seed=0)
    ad.obs["truth"] = truth
    coarse = truth.copy()
    coarse[coarse == 5] = 4                # the two neighbours as one cluster
    ad.obs["coarse"] = coarse
    return ad


# ── annotate ──

def test_annotate_names_every_cell_and_keeps_the_clustering(blobs) -> None:
    ad = blobs
    ps.tl.annotate(ad, {"0": "T cell", "1": "B cell", 2: "NK cell"}, groupby="coarse", key_added="ct1")
    got = ad.obs["ct1"]
    assert isinstance(got, pd.Categorical)
    truth = np.asarray(ad.obs["coarse"])
    assert (np.asarray(got)[truth == 0] == "T cell").all()
    assert (np.asarray(got)[truth == 2] == "NK cell").all()
    assert (np.asarray(got)[truth == 3] == "3").all(), "an unnamed cluster keeps its own label"
    assert np.array_equal(np.asarray(ad.obs["coarse"]), truth), "the source column is untouched"
    assert ad.uns["curation"][-1]["step"] == "annotate"


def test_annotate_joins_clusters_given_one_name_and_marks_the_rest(blobs) -> None:
    ad = blobs
    ps.tl.annotate(ad, {"lymphocyte": ["0", "1"], "other": [2]}, groupby="coarse", key_added="ct2",
                   unassigned="unknown")
    got, truth = np.asarray(ad.obs["ct2"]), np.asarray(ad.obs["coarse"])
    assert (got[np.isin(truth, [0, 1])] == "lymphocyte").all()
    assert (got[np.isin(truth, [3, 4])] == "unknown").all()
    assert set(ad.obs["ct2"].categories) == {"lymphocyte", "other", "unknown"}


def test_annotate_refuses_a_cluster_that_does_not_exist(blobs) -> None:
    with pytest.raises(KeyError, match="not in obs"):
        ps.tl.annotate(blobs, {"99": "x"}, groupby="coarse", key_added="bad")


# ── merge_clusters ──

def test_merge_clusters_joins_and_names_them(blobs) -> None:
    ad = blobs
    ps.tl.merge_clusters(ad, [["0", "1"], ["2", "3"]], groupby="truth")
    got, truth = np.asarray(ad.obs["truth_curated"]), np.asarray(ad.obs["truth"])
    assert (got[np.isin(truth, [0, 1])] == "0+1").all()
    assert (got[np.isin(truth, [2, 3])] == "2+3").all()
    assert (got[truth == 4] == "4").all()
    ps.tl.merge_clusters(ad, ["0", "1"], groupby="truth", key_added="m2", names=["lymphoid"])
    assert (np.asarray(ad.obs["m2"])[np.isin(truth, [0, 1])] == "lymphoid").all()


@pytest.mark.parametrize("groups, kwargs, error", [
    ([["0", "1"], ["1", "2"]], {}, ValueError),         # a cluster in two groups
    ([["0"]], {}, ValueError),                          # nothing to join
    ([["0", "1"]], {"names": ["2"]}, ValueError),       # the new name is another cluster
    ([["0", "1"]], {"names": ["a", "b"]}, ValueError),  # names do not match groups
    ([["0", "77"]], {}, KeyError),                      # no such cluster
])
def test_merge_clusters_refuses_ambiguous_requests(blobs, groups, kwargs, error) -> None:
    with pytest.raises(error):
        ps.tl.merge_clusters(blobs, groups, groupby="truth", key_added="bad", **kwargs)


# ── subcluster ──

def test_subcluster_splits_the_cluster_and_leaves_the_others(blobs) -> None:
    ad = blobs
    n = ps.tl.subcluster(ad, "4", groupby="coarse", resolution=0.3, key_added="split")
    got, coarse, truth = np.asarray(ad.obs["split"]), np.asarray(ad.obs["coarse"]), np.asarray(ad.obs["truth"])
    inside = coarse == 4
    assert n >= 2
    assert all(str(v).startswith("4.") for v in got[inside])
    assert np.array_equal(got[~inside], coarse[~inside].astype(str)), "other cells keep their labels"
    assert _ari(got[inside], truth[inside]) >= 0.9, "the two populations inside are found again"
    sizes = pd.Series(got[inside]).value_counts()
    assert sizes.index[0] == "4.0", "parts are numbered from the largest"
    assert ad.uns["curation"][-1] == {"step": "subcluster", "groupby": "coarse", "key_added": "split",
                                      "cluster": "4", "resolution": 0.3, "seed": 0, "parts": n}


def test_subcluster_is_reproducible(blobs) -> None:
    ps.tl.subcluster(blobs, "4", groupby="coarse", resolution=0.3, seed=3, key_added="s1")
    ps.tl.subcluster(blobs, "4", groupby="coarse", resolution=0.3, seed=3, key_added="s2")
    assert np.array_equal(np.asarray(blobs.obs["s1"]), np.asarray(blobs.obs["s2"]))


def test_subcluster_needs_the_graph_and_enough_cells(blobs) -> None:
    ad = ps.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    ad.obs["c"] = np.zeros(10, dtype=np.int32)
    with pytest.raises(KeyError, match="neighbors"):
        ps.tl.subcluster(ad, "0", groupby="c")
    tiny = np.asarray(blobs.obs["coarse"]).copy()
    tiny[:2] = 9
    blobs.obs["tiny"] = tiny
    with pytest.raises(ValueError, match="too few"):
        ps.tl.subcluster(blobs, "9", groupby="tiny")


def test_merge_then_split_then_name(blobs) -> None:
    """The curation a researcher does: join two clusters, split another, name the result."""
    ad = blobs
    ps.tl.merge_clusters(ad, [["0", "1"]], groupby="coarse", key_added="cur")
    ps.tl.subcluster(ad, "4", groupby="cur", resolution=0.3, key_added="cur")
    ps.tl.annotate(ad, {"0+1": "lymphocyte", "4.0": "type A", "4.1": "type B"}, groupby="cur",
                   key_added="final")
    final, truth = np.asarray(ad.obs["final"]), np.asarray(ad.obs["truth"])
    assert (final[np.isin(truth, [0, 1])] == "lymphocyte").all()
    assert _ari(final[np.isin(truth, [4, 5])], truth[np.isin(truth, [4, 5])]) >= 0.9
    assert [s["step"] for s in ad.uns["curation"][-3:]] == ["merge_clusters", "subcluster", "annotate"]


# ── markers ──

@pytest.fixture(scope="module")
def counts():
    """Counts with one planted marker gene per group: gene k is high in group k only."""
    rng = np.random.default_rng(1)
    n_groups, per, n_genes = 4, 250, 40
    labels = np.repeat(np.arange(n_groups), per)
    lam = np.full((labels.size, n_genes), 2.0)
    for k in range(n_groups):
        lam[labels == k, k] = 25.0
    X = rng.poisson(lam).astype(np.float32)
    ad = ps.AnnData(X=X, var_names=[f"g{j}" for j in range(n_genes)])
    ps.pp.normalize_total(ad, target_sum=1e4)
    ps.pp.log1p(ad)
    ad.obs["leiden"] = labels.astype(np.int32)
    return ad


def test_markers_of_a_curated_labelling(counts) -> None:
    ad = counts
    ps.tl.annotate(ad, {"0": "alpha", "1": "beta", "2": "gamma", "3": "delta"}, key_added="ct")
    ps.tl.rank_genes_groups(ad, groupby="ct", n_genes=10)
    long = ps.get.markers(ad, n=3)
    assert list(long.columns) == ["group", "rank", "gene", "score", "logfoldchange", "pval_adj", "pct"]
    assert len(long) == 12
    top = long[long["rank"] == 1].set_index("group")["gene"].to_dict()
    assert top == {"alpha": "g0", "beta": "g1", "gamma": "g2", "delta": "g3"}
    wide = ps.get.markers(ad, n=5, wide=True)
    assert wide.shape == (5, 4) and wide["beta"].iloc[0] == "g1"
    with pytest.raises(KeyError, match="were not ranked"):
        ps.get.markers(ad, groups=["epsilon"])


def test_markers_need_a_ranking() -> None:
    with pytest.raises(KeyError, match="rank_genes_groups"):
        ps.get.markers(ps.AnnData(X=np.zeros((3, 2), dtype=np.float32)))


# ── out of core: the same steps on a result whose matrix stays in its file ──

@pytest.fixture(scope="module")
def disk():
    return ps.process_diskbacked(FIXTURE, n_hvg=200, n_pcs=10, n_neighbors=15, seed=42)


def test_curation_out_of_core(disk) -> None:
    leiden = np.asarray(disk.obs["leiden"])
    big = pd.Series(leiden).value_counts().index[:3].astype(str).tolist()
    ps.tl.merge_clusters(disk, [big[1:3]], key_added="cur")
    parts = ps.tl.subcluster(disk, big[0], groupby="cur", resolution=0.5, key_added="cur")
    assert parts >= 1
    cur = np.asarray(disk.obs["cur"])
    assert (cur[np.isin(leiden.astype(str), big[1:3])] == "+".join(big[1:3])).all()
    assert all(str(v).startswith(f"{big[0]}.") for v in cur[leiden.astype(str) == big[0]])


def test_markers_out_of_core_on_curated_labels_match_in_memory(disk) -> None:
    """Out of core, markers of a named or joined labelling (strings, not cluster numbers) are ranked
    from the file and must match the in-memory ranking of the same cells and labels."""
    leiden = np.asarray(disk.obs["leiden"])
    first = str(pd.Series(leiden).value_counts().index[0])
    ps.tl.annotate(disk, {first: "big one"}, key_added="named")
    ps.tl.rank_genes_groups(disk, groupby="named", n_genes=20)
    got = disk.uns["rank_genes_groups"]
    mem = ps.read_h5ad(FIXTURE)
    ps.pp.normalize_total(mem, target_sum=1e4)
    ps.pp.log1p(mem)
    mem.obs["named"] = np.asarray(disk.obs["named"])
    ps.tl.rank_genes_groups(mem, groupby="named", n_genes=20)
    want = mem.uns["rank_genes_groups"]
    assert "big one" in got and set(got) == set(want)
    for g in want:
        assert got[g]["names"] == want[g]["names"], g
        np.testing.assert_allclose(got[g]["scores"], want[g]["scores"], rtol=1e-6)


# ── labels on an embedding ──

def _labelled(ad, **kw):
    ax = ps.pl.umap(ad, color="ct1", show=False, **kw)
    ax.figure.canvas.draw()
    return ax


def _text_boxes(ax):
    """The extents of the label texts (names and numbers), not the plot's heading."""
    r = ax.figure.canvas.get_renderer()
    labels = {n for v in ax.polariseq_labels.values() for n in v}
    return [t.get_window_extent(r) for t in ax.texts
            if t.get_text().strip() in labels or t.get_text().strip().isdigit()]


@pytest.mark.parametrize("render", ["points", "density"])
def test_labels_number_name_callout(blobs, render) -> None:
    ad = blobs
    ps.tl.annotate(ad, {"0": "T cell", "1": "B cell", "2": "NK cell", "3": "monocyte",
                        "4": "rare type"}, groupby="coarse", key_added="ct1")
    ax = _labelled(ad, labels="number", render=render)
    got = ax.polariseq_labels
    assert len(got["number"]) == 5 and not got["callout"]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend[0] == "1  T cell", "numbered from the largest group"
    plt.close(ax.figure)

    ax = _labelled(ad, labels="callout", render=render)
    assert sorted(ax.polariseq_labels["callout"]) == sorted(["T cell", "B cell", "NK cell", "monocyte",
                                                              "rare type"])
    boxes = _text_boxes(ax)
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), "callouts overlap"
    frame = ax.get_window_extent(ax.figure.canvas.get_renderer())
    assert all(b.x1 <= frame.x0 + 2 or b.x0 >= frame.x1 - 2 for b in boxes), "callouts sit in the margins"
    plt.close(ax.figure)


def test_no_two_labels_overlap_with_every_style_at_once(blobs) -> None:
    """Numbers, names and callouts together, with a legend: no text lies on another, and the legend
    stays clear of the callouts in the right margin."""
    ax = _labelled(blobs, labels="name", label_style={"T cell": "number", "rare type": "callout",
                                                      "monocyte": "callout"})
    r = ax.figure.canvas.get_renderer()
    boxes = _text_boxes(ax)
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    if ax.get_legend() is not None:
        lb = ax.get_legend().get_window_extent(r)
        assert not any(lb.overlaps(b) for b in boxes)
    plt.close(ax.figure)


def test_labels_per_group_choice_and_subset(blobs) -> None:
    ad = blobs
    ax = _labelled(ad, labels="number", label_style={"rare type": "callout", "B cell": "name"},
                   label_groups=["T cell", "B cell", "rare type"])
    got = ax.polariseq_labels
    assert got["number"] == ["T cell"]
    assert got["name"] == ["B cell"]
    assert got["callout"] == ["rare type"]
    plt.close(ax.figure)


def test_labels_auto_uses_callouts_for_small_groups(blobs) -> None:
    ax = _labelled(blobs, labels="auto", callout_below=0.12)
    got = ax.polariseq_labels
    # groups of 900, 700, 500, 400 (of 3,100 cells) are numbered; the joined 600 is above 12% too
    assert "T cell" in got["number"]
    ax2 = _labelled(blobs, labels="auto", callout_below=0.25)
    assert set(ax2.polariseq_labels["callout"]) >= {"NK cell", "monocyte"}
    plt.close(ax.figure)
    plt.close(ax2.figure)


def test_labels_refuse_unknown_styles_and_groups(blobs) -> None:
    with pytest.raises(ValueError, match="use one of"):
        ps.pl.umap(blobs, color="ct1", labels="arrow", show=False)
    with pytest.raises(ValueError, match="use one of"):
        ps.pl.umap(blobs, color="ct1", labels="number", label_style={"T cell": "arrow"}, show=False)
    with pytest.raises(KeyError, match="not groups"):
        ps.pl.umap(blobs, color="ct1", labels="number", label_groups=["dragon"], show=False)
    plt.close("all")


def test_a_label_sits_on_its_own_cells_even_when_the_group_lies_in_two_places() -> None:
    """A group in two separate places has its median in the empty space between them; the label
    must be placed on one of the group's cells, not there."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(300, 2)) + [0, 0]
    split = np.vstack([rng.normal(size=(20, 2)) * 0.3 + [-8, 6], rng.normal(size=(20, 2)) * 0.3 + [8, 6]])
    ad = ps.AnnData(X=np.zeros((340, 1), dtype=np.float32))
    ad.obsm["X_umap"] = np.vstack([a, split]).astype(np.float32)
    ad.obs["g"] = pd.Categorical(["big"] * 300 + ["split"] * 40)
    ax = ps.pl.umap(ad, color="g", labels="number", show=False)
    disc = [ln for ln in ax.lines if ln.get_marker() == "o"]
    xs = sorted(float(ln.get_xdata()[0]) for ln in disc)
    assert any(abs(x) > 6 for x in xs), "the split group's label must sit on one of its two parts"
    plt.close(ax.figure)
