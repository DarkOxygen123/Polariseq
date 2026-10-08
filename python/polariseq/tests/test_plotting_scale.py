"""The figures that adapt to the size of the data: ``ps.pl.qc_distributions``,
``cluster_sizes``, ``overview``, and embeddings drawn as density.

A run of 22 million cells came out with quality-control histograms of a few
wide blocks (linear bins on a log axis), a UMAP of a 200,000-cell sample and,
had it been drawn whole, a legend of 182 entries. These tests check the
choices that fix that; they do not check what the figures look like.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polariseq as ps
import pytest
from polariseq import plotting as plmod

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _toy(n: int = 30_000, k: int = 60, skew: float = 0.93, seed: int = 0):
    """Cells in ``k`` groups of very different sizes, with an embedding and
    quality-control columns of the usual log-normal shape."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    p = skew ** np.arange(k)
    lab = rng.choice(k, size=n, p=p / p.sum())
    centres = rng.normal(0, 10, (k, 2))
    a = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    a.obsm["X_umap"] = (centres[lab] + rng.normal(0, 0.6, (n, 2))).astype(np.float32)
    a.obs["leiden"] = pd.Categorical(lab.astype(str))
    a.obs["n_genes"] = np.exp(rng.normal(7.0, 0.7, n)).round()
    a.obs["n_counts"] = np.exp(rng.normal(8.4, 0.9, n)).round()
    a.obs["score"] = rng.normal(size=n)
    return a


def test_qc_bins_are_log_spaced_and_every_cell_counts() -> None:
    a = _toy()
    ax = ps.pl.qc_distributions(a, thresholds={"n_genes": 200}, show=False)[0]
    assert ax.get_xscale() == "log"
    stairs = ax.patches[0].get_data()
    ratios = np.asarray(stairs.edges[1:]) / np.asarray(stairs.edges[:-1])
    assert np.allclose(ratios, ratios[0]), "bins must be evenly spaced on the log axis"
    assert np.asarray(stairs.values).sum() == int((a.obs["n_genes"] > 0).sum()), "nothing sampled"
    assert any(np.allclose(line.get_xdata(), 200) for line in ax.lines), "threshold not drawn"


def test_cluster_sizes_turn_logarithmic_and_drop_names_when_many() -> None:
    a = _toy(k=60)
    ax = ps.pl.cluster_sizes(a, show=False)
    assert ax.get_yscale() == "log"
    assert len(ax.get_xticks()) == 0
    assert ax.get_title(loc="left") == "60 clusters"
    few = ps.pl.cluster_sizes(_toy(k=8, skew=0.9), show=False)
    assert len(few.get_xticks()) == 8, "a handful of clusters keeps its names"


def test_density_render_is_one_image_of_every_cell() -> None:
    ax = ps.pl.umap(_toy(), color="leiden", render="density", show=False)
    assert len(ax.images) == 1 and len(ax.collections) == 0


def test_auto_render_switches_above_the_threshold(monkeypatch) -> None:
    a = _toy(n=5000)
    ax = ps.pl.umap(a, color="leiden", show=False)
    assert len(ax.images) == 0 and len(ax.collections) > 0
    monkeypatch.setattr(plmod, "DENSITY_ABOVE", 1000)
    ax = ps.pl.umap(a, color="leiden", show=False)
    assert len(ax.images) == 1


@pytest.mark.parametrize("render", ["points", "density"])
def test_many_groups_are_named_on_the_data_not_in_a_legend(render) -> None:
    a = _toy(k=60)
    ax = ps.pl.umap(a, color="leiden", render=render, show=False)
    assert ax.get_legend() is None
    groups = set(a.obs["leiden"].astype(str))
    named = [t for t in ax.texts if t.get_text() in groups]
    assert 0 < len(named) <= plmod.MAX_NAMED


def test_a_continuous_colour_as_density_keeps_its_colourbar() -> None:
    ax = ps.pl.umap(_toy(), color="score", render="density", show=False)
    assert len(ax.get_figure().axes) == 2


def test_render_must_be_a_known_value() -> None:
    with pytest.raises(ValueError, match="render must be"):
        ps.pl.umap(_toy(n=500), color="leiden", render="dots", show=False)


def test_overview_writes_the_standard_figures(tmp_path) -> None:
    a = _toy()
    a.uns["pca"] = {"variance_ratio": np.linspace(0.2, 0.001, 30)}
    files = ps.pl.overview(a, str(tmp_path), color_by=["leiden"], thresholds={"n_genes": 200})
    assert set(files) == {"qc", "pca", "clusters", "umap", "umap_leiden", "figure_data"}
    for p in files.values():
        assert Path(p).stat().st_size > 0


def test_integer_cluster_labels_are_groups_not_a_gradient() -> None:
    """Out of core, Leiden labels come back as plain integers; 66 of them were
    drawn as a colourbar from 0 to 65 on the 10-file scTab run."""
    a = _toy(n=20_000, k=66)
    a.obs["leiden"] = np.asarray(a.obs["leiden"]).astype(np.int64)
    for render in ("points", "density"):
        ax = ps.pl.umap(a, color="leiden", render=render, show=False)
        assert len(ax.get_figure().axes) == 1, f"no colourbar for cluster labels ({render})"


def test_long_names_on_the_data_never_overlap() -> None:
    a = _toy(n=20_000, k=60)
    long = {c: f"{c} effector memory CD8-positive, alpha-beta T cell" for c in a.obs["leiden"].cat.categories}
    a.obs["leiden"] = a.obs["leiden"].cat.rename_categories(long)
    ax = ps.pl.umap(a, color="leiden", render="density", show=False)
    renderer = ax.figure.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.texts if "T cell" in t.get_text()]
    assert boxes
    assert not any(b1.overlaps(b2) for i, b1 in enumerate(boxes) for b2 in boxes[i + 1:])


def test_names_stay_clear_of_each_other_once_the_axes_has_its_final_shape() -> None:
    """An equal-aspect axes is shrunk to its box when the figure is drawn.
    Names judged before that looked clear and overlapped after it: the 22M-cell
    cell-type UMAP had "cardiac neuron" over "basal cell". Measured here after a
    real draw, on an embedding far from square."""
    a = _toy(n=40_000, k=164, skew=0.985)
    a.obsm["X_umap"] = a.obsm["X_umap"] * np.array([3.0, 0.7], dtype=np.float32)
    names = {c: f"{c} cardiac muscle cell of the left ventricle"
             for c in a.obs["leiden"].cat.categories}
    a.obs["leiden"] = a.obs["leiden"].cat.rename_categories(names)
    ax = ps.pl.umap(a, color="leiden", render="density", show=False)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.texts if "ventricle" in t.get_text()]
    assert len(boxes) >= 5, "some names must be placed"
    for i, b1 in enumerate(boxes):
        for b2 in boxes[i + 1:]:
            assert not b1.overlaps(b2)


def test_long_names_wrap_onto_lines_of_at_most_the_wrap_width() -> None:
    a = _toy(n=20_000, k=60)
    long = {c: f"{c} corticothalamic-projecting glutamatergic cortical neuron"
            for c in a.obs["leiden"].cat.categories}
    a.obs["leiden"] = a.obs["leiden"].cat.rename_categories(long)
    ax = ps.pl.umap(a, color="leiden", render="density", show=False)
    names = [t.get_text() for t in ax.texts if "neuron" in t.get_text()]
    assert names and all("\n" in n for n in names)
    assert all(len(line) <= plmod.LABEL_WRAP + 3 for n in names for line in n.split("\n"))


def test_many_groups_get_a_square_canvas_and_few_keep_the_small_one() -> None:
    many = ps.pl.umap(_toy(k=60), color="leiden", render="density", show=False)
    assert tuple(many.figure.get_size_inches()) == (6.6, 6.6)
    few = ps.pl.umap(_toy(k=8, skew=0.9), color="leiden", show=False)
    assert tuple(few.figure.get_size_inches()) == (4.9, 3.7)


def test_qc_shows_the_cells_a_threshold_removed() -> None:
    """The 22M run's quality-control figure drew only the kept cells, so the
    200-gene line sat at the left edge and cut nothing."""
    a = _toy()
    before = {k: np.asarray(a.obs[k], dtype=float).copy() for k in ("n_genes", "n_counts")}
    kept = a[np.asarray(a.obs["n_genes"]) >= 600].copy()
    removed = a.n_obs - kept.n_obs
    assert removed > 500
    ax = ps.pl.qc_distributions(kept, thresholds={"n_genes": 600}, before=before, show=False)[0]
    grey, blue = ax.patches[0].get_data(), ax.patches[1].get_data()
    assert np.asarray(grey.values).sum() == int((before["n_genes"] > 0).sum())
    assert np.asarray(blue.values).sum() == kept.n_obs
    assert ax.get_xlim()[0] < 600 < ax.get_xlim()[1], "cells below the line are on the axis"
    text = " ".join(t.get_text() for t in ax.texts)
    assert f"{removed:,} removed" in text and f"{kept.n_obs:,} kept" in text
    # Without the metrics from before, only the cells kept are drawn.
    plain = ps.pl.qc_distributions(kept, thresholds={"n_genes": 600}, show=False)[0]
    assert len(plain.patches) == 1


def test_overview_passes_the_metrics_from_before_filtering(tmp_path) -> None:
    a = _toy()
    before = {k: np.asarray(a.obs[k], dtype=float).copy() for k in ("n_genes", "n_counts")}
    kept = a[np.asarray(a.obs["n_genes"]) >= 600].copy()
    kept.uns["pca"] = {"variance_ratio": np.linspace(0.2, 0.001, 30)}
    files = ps.pl.overview(kept, str(tmp_path), thresholds={"n_genes": 600}, qc_before=before)
    assert Path(files["qc"]).stat().st_size > 0


def test_overview_writes_the_numbers_behind_its_figures(tmp_path) -> None:
    """Every figure of a run should carry its data: the quality-control counts per bin (with the cells
    a threshold removed), the cluster sizes and the variance ratios."""
    import json

    a = _toy()
    before = {k: np.asarray(a.obs[k], dtype=float).copy() for k in ("n_genes", "n_counts")}
    kept = a[np.asarray(a.obs["n_genes"]) >= 600].copy()
    kept.uns["pca"] = {"variance_ratio": np.linspace(0.2, 0.001, 30)}
    files = ps.pl.overview(kept, str(tmp_path), thresholds={"n_genes": 600}, qc_before=before)
    data = json.loads(Path(files["figure_data"]).read_text())
    q = data["qc"]["n_genes"]
    assert sum(q["cells_before_filtering"]) == int((before["n_genes"] > 0).sum())
    assert sum(q["cells_kept"]) == kept.n_obs
    assert q["cells_below_threshold"] == a.n_obs - kept.n_obs and q["threshold"] == 600
    assert len(q["bin_edges"]) == len(q["cells_kept"]) + 1
    assert sum(data["cluster_sizes"].values()) == kept.n_obs
    assert len(data["pca_variance_ratio"]) == 30
