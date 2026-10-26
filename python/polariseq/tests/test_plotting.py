"""Tests for the plotting namespace (`ps.pl`).

Figures are hard to assert on and easy to leave untested, which is how a
plotting module accumulates functions that raise on their most common
argument. These tests do not check what a plot *looks* like; they check that
each entry point runs on a realistic object, that the theme system actually
changes colours rather than only sizes, and that the arguments people
actually pass — a gene name, a list of colours, a subset of groups, scanpy's
spelling — are accepted.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest
from polariseq import plotting as plmod

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

FIXTURE = "tests/data/medium_10kx2k.h5ad"


@pytest.fixture(scope="module")
def adata():
    """A small but fully-processed object: embeddings, clusters, markers."""
    a = ps.pp._ensure_materialized(ps.read_h5ad(FIXTURE, lazy=True))
    ps.pp.calculate_qc_metrics(a)
    ps.pp.normalize_total(a, target_sum=1e4)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=200, subset=True)
    ps.pp.pca(a, n_components=10, seed=0)
    ps.pp.neighbors(a, n_neighbors=15, seed=0)
    ps.tl.leiden(a, seed=0)
    ps.tl.umap(a, seed=0)
    ps.tl.rank_genes_groups(a, groupby="leiden")
    return a


@pytest.fixture(autouse=True)
def _reset_theme():
    yield
    ps.pl.set_theme("paper")


class TestThemes:
    def test_every_preset_applies(self) -> None:
        for name in ps.pl.THEMES:
            ps.pl.set_theme(name)
            assert ps.pl.get_theme()["name"] == name

    def test_dark_changes_colours_not_only_sizes(self) -> None:
        """The original theme engine could only resize things — colours were
        module constants, so a dark theme was inexpressible."""
        ps.pl.set_theme("paper")
        light = dict(ps.pl.get_theme())
        ps.pl.set_theme("dark")
        dark = dict(ps.pl.get_theme())
        assert dark["text"] != light["text"]
        assert dark["figure_bg"] != light["figure_bg"]
        assert dark["axes_bg"] != light["axes_bg"]

    def test_palette_follows_the_theme(self) -> None:
        """Light themes share the validated palette, `dark` takes its dark steps,
        `classic` is the 0.2.0 palette (Tol colours washed toward grey)."""
        ps.pl.set_theme("paper")
        paper = ps.pl.categorical_palette(8)
        assert paper == plmod.PALETTE_LIGHT
        ps.pl.set_theme("print")
        assert ps.pl.categorical_palette(8) == paper
        ps.pl.set_theme("dark")
        assert ps.pl.categorical_palette(8) == plmod.PALETTE_DARK
        ps.pl.set_theme("classic")
        assert ps.pl.categorical_palette(4) == [plmod._wash(c) for c in plmod._TOL_RAW[:4]]

    def test_more_groups_than_hues_take_lighter_then_darker_steps(self) -> None:
        pal = ps.pl.categorical_palette(24)
        assert len(set(pal)) == 24 and pal[:8] == plmod.PALETTE_LIGHT

    def test_text_is_legible_on_every_theme(self) -> None:
        """Secondary and caption text reach WCAG 4.5:1 for small type on each background."""
        def lum(h):
            c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

        for name in (n for n in ps.pl.THEMES if n != "classic"):
            ps.pl.set_theme(name)
            bg = lum(plmod._c("figure_bg"))
            for key in ("text", "text_sub", "text_cap"):
                fg = lum(plmod._c(key))
                ratio = (max(fg, bg) + 0.05) / (min(fg, bg) + 0.05)
                assert ratio >= 4.5, (name, key, ratio)

    def test_figures_grow_with_the_type(self) -> None:
        ps.pl.set_theme("talk")
        fig, _ = plmod._subplots(plmod._plt(), figsize=(4.0, 3.0))
        assert fig.get_size_inches()[0] == pytest.approx(4.0 * 13.0 / 8.5)
        plmod._plt().close(fig)

    def test_switching_does_not_accumulate_overrides(self) -> None:
        ps.pl.set_theme("talk")
        big = ps.pl.get_theme()["base"]
        ps.pl.set_theme("paper")
        assert ps.pl.get_theme()["base"] != big

    def test_overrides_apply_and_unknown_keys_are_rejected(self) -> None:
        ps.pl.set_theme("paper", point_scale=3.0)
        assert ps.pl.get_theme()["point_scale"] == 3.0
        with pytest.raises(ValueError, match="unknown theme keys"):
            ps.pl.set_theme("paper", not_a_key=1)

    def test_context_manager_restores(self) -> None:
        ps.pl.set_theme("paper")
        with ps.pl.theme("dark"):
            assert ps.pl.get_theme()["name"] == "dark"
        assert ps.pl.get_theme()["name"] == "paper"

    def test_unknown_theme_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown theme"):
            ps.pl.set_theme("chartreuse")


class TestEmbeddings:
    def test_colour_by_obs_and_by_gene(self, adata) -> None:
        gene = str(adata.var_names[0])
        assert ps.pl.umap(adata, color="leiden", show=False) is not None
        assert ps.pl.umap(adata, color="n_genes", show=False) is not None
        assert ps.pl.umap(adata, color=gene, show=False) is not None

    def test_a_list_of_colours_draws_a_grid(self, adata) -> None:
        genes = [str(g) for g in adata.var_names[:3]]
        axes = ps.pl.umap(adata, color=["leiden", *genes], ncols=2, show=False)
        assert len(axes) == 4

    def test_a_list_with_an_explicit_axis_is_refused(self, adata) -> None:
        """Silently ignoring one of the two would produce a plot that is not
        what either argument asked for."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="single `color`"):
            ps.pl.umap(adata, color=["leiden", "n_genes"], ax=ax, show=False)
        plt.close("all")

    def test_groups_highlights_a_subset(self, adata) -> None:
        cats = sorted({str(v) for v in np.asarray(adata.obs["leiden"])})
        assert ps.pl.umap(adata, color="leiden", groups=cats[:1],
                          show=False) is not None

    def test_unknown_group_names_the_available_ones(self, adata) -> None:
        with pytest.raises(KeyError, match="available"):
            ps.pl.umap(adata, color="leiden", groups=["no-such-cluster"],
                       show=False)

    def test_unknown_colour_is_a_clear_error(self, adata) -> None:
        with pytest.raises(KeyError, match="neither a column of adata.obs"):
            ps.pl.umap(adata, color="definitely-not-a-gene", show=False)

    def test_generic_embedding_accepts_a_short_basis(self, adata) -> None:
        assert ps.pl.embedding(adata, basis="umap", color="leiden",
                               show=False) is not None
        assert ps.pl.embedding(adata, basis="X_umap", color="leiden",
                               show=False) is not None


class TestMarkerPlots:
    def test_rank_genes_groups_draws_a_panel_per_cluster(self, adata) -> None:
        axes = ps.pl.rank_genes_groups(adata, n_genes=5, show=False)
        assert len(axes) == len(adata.uns["rank_genes_groups"])

    def test_rank_genes_groups_needs_the_tool_to_have_run(self, adata) -> None:
        bare = ps.AnnData(X=adata.X)
        with pytest.raises(KeyError, match="run ps.tl.rank_genes_groups"):
            ps.pl.rank_genes_groups(bare, show=False)

    def test_dotplot_and_heatmap_accept_scanpy_spelling(self, adata) -> None:
        """`var_names=` is what scanpy calls it; a TypeError about an
        unexpected keyword is a poor way to learn the name differs."""
        genes = [str(g) for g in adata.var_names[:4]]
        assert ps.pl.dotplot(adata, var_names=genes, show=False) is not None
        assert ps.pl.heatmap(adata, var_names=genes, show=False) is not None
        assert ps.pl.dotplot(adata, genes, show=False) is not None

    def test_supplying_both_spellings_is_refused(self, adata) -> None:
        genes = [str(g) for g in adata.var_names[:3]]
        with pytest.raises(TypeError, match="use one"):
            ps.pl.dotplot(adata, markers=genes, var_names=genes, show=False)

    def test_supplying_neither_is_refused(self, adata) -> None:
        with pytest.raises(TypeError, match="needs the genes"):
            ps.pl.dotplot(adata, show=False)

    def test_violin_plots_genes_as_well_as_obs(self, adata) -> None:
        """The most common violin in scRNA-seq is marker expression per
        cluster; before this it could only read `.obs`."""
        gene = str(adata.var_names[0])
        assert ps.pl.violin(adata, [gene], groupby="leiden", show=False) is not None
        assert ps.pl.violin(adata, ["n_genes"], groupby="leiden",
                            show=False) is not None


class TestRemainingPlots:
    def test_they_all_run(self, adata) -> None:
        genes = [str(g) for g in adata.var_names[:4]]
        assert ps.pl.pca(adata, color="leiden", show=False) is not None
        assert ps.pl.pca_variance_ratio(adata, show=False) is not None
        assert ps.pl.highly_variable_genes(adata, show=False) is not None
        assert ps.pl.scatter_qc(adata, show=False) is not None
        assert ps.pl.heatmap(adata, genes, show=False) is not None

    def test_save_writes_a_file(self, adata, tmp_path) -> None:
        out = tmp_path / "u.png"
        ps.pl.umap(adata, color="leiden", show=False, save=str(out))
        assert out.exists() and out.stat().st_size > 1000

    def test_dark_theme_saves_a_dark_canvas(self, adata, tmp_path) -> None:
        """`_save` used to hardcode a white facecolor, which put dark-tuned
        colours and light text onto a white page."""
        out = tmp_path / "d.png"
        with ps.pl.theme("dark"):
            ps.pl.umap(adata, color="leiden", show=False, save=str(out))
        import matplotlib.pyplot as plt

        px = plt.imread(str(out))
        corner = px[2, 2, :3]
        assert corner.mean() < 0.4, f"corner is not dark: {corner}"


def test_pl_namespace_exposes_module_functions() -> None:
    """The namespace lists its surface explicitly for tab-completion, with a
    fallthrough so a new plotting function is not invisible until someone
    remembers to add it."""
    assert ps.pl.set_theme is plmod.set_theme
    assert callable(ps.pl.categorical_palette)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = ps.pl.no_such_plot


def test_many_clusters_stay_categorical():
    """A categorical column is categorical however many levels it has.

    The colour branch used to decide from the values alone, so cluster labels
    that cast to float cleanly became a continuous gradient the moment a
    dataset produced more than 30 of them. An 800K-cell run with 34 Leiden
    clusters rendered as a colourbar from 0 to 33, which is both wrong and
    exactly the case a large dataset hits.
    """
    import anndata as ad
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import polariseq as ps

    rng = np.random.default_rng(0)
    n, k = 4000, 34
    a = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    a.obsm["X_pca"] = rng.normal(size=(n, 2)).astype(np.float32)
    a.obs["leiden"] = pd.Categorical(rng.integers(0, k, n).astype(str))
    assert a.obs["leiden"].nunique() == k

    ax = ps.pl.embedding(a, basis="pca", color="leiden", show=False)
    fig = ax.get_figure()
    # A continuous rendering adds a colourbar axes; a categorical one does not.
    assert len(fig.axes) == 1, "categorical column rendered with a colourbar"
    # And the points must carry more than one distinct colour.
    colours = {tuple(c) for coll in ax.collections
               for c in coll.get_facecolors()}
    assert len(colours) > 2, f"expected many cluster colours, got {len(colours)}"


class TestThemedText:
    """Every text a plot draws takes the theme's colours (dark legends were black)."""

    def _texts(self, fig):
        out = []
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                out += [t.get_color() for t in leg.get_texts()]
            out += [t.get_color() for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()]
        return out

    def test_dark_legends_and_tick_labels_are_light(self, adata) -> None:
        import matplotlib.colors as mc

        ps.pl.set_theme("dark")
        adata.obs["half"] = np.where(np.arange(adata.n_obs) % 2, "a", "b").astype(object)
        for draw in (lambda: ps.pl.umap(adata, color="half", show=False),
                     lambda: ps.pl.dotplot(adata, list(adata.var_names[:3]), show=False),
                     lambda: ps.pl.heatmap(adata, list(adata.var_names[:3]), show=False)):
            ax = draw()
            fig = (ax[0] if isinstance(ax, (list, np.ndarray)) else ax).figure
            colours = self._texts(fig)
            assert colours, "nothing to check"
            assert all(sum(mc.to_rgb(c)) / 3 > 0.5 for c in colours), colours
            plmod._plt().close(fig)


class TestLabels:
    def test_on_data_names_the_groups(self, adata) -> None:
        adata.obs["half"] = np.where(np.arange(adata.n_obs) % 2, "first", "second").astype(object)
        ax = ps.pl.umap(adata, color="half", legend_loc="on data", show=False)
        assert ax.get_legend() is None
        assert {"first", "second"} <= {t.get_text() for t in ax.texts}
        plmod._plt().close(ax.figure)

    def test_many_clusters_are_numbered_with_their_own_labels(self, adata) -> None:
        assert adata.obs["leiden"].max() >= 8, "the fixture needs more clusters than hues"
        ax = ps.pl.umap(adata, color="leiden", show=False)
        drawn = {t.get_text() for t in ax.texts}
        assert drawn <= {str(c) for c in np.unique(adata.obs["leiden"])} and len(drawn) > 8
        plmod._plt().close(ax.figure)

    def test_classic_keeps_the_legend(self, adata) -> None:
        ps.pl.set_theme("classic")
        ax = ps.pl.umap(adata, color="leiden", show=False)
        assert ax.get_legend() is not None and not ax.texts
        plmod._plt().close(ax.figure)


class TestLayout:
    def test_a_top_y_label_lifts_the_heading_above_it(self, adata) -> None:
        ps.pl.set_theme("talk")
        ax = ps.pl.highly_variable_genes(adata, show=False)
        fig = ax.figure
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        label = ax.yaxis.label.get_window_extent(r)
        sub = ax._ps_subtitle.get_window_extent(r)
        assert sub.y0 >= label.y1 - 1   # the subtitle sits above the label, not on it
        plmod._plt().close(fig)

    def test_panels_line_up(self, adata) -> None:
        axes = ps.pl.umap(adata, color=["leiden", str(adata.var_names[0])], show=False)
        fig = axes[0].figure
        fig.canvas.draw()
        tops = [a.get_position().y1 for a in axes]
        assert max(tops) - min(tops) < 1e-6
        plmod._plt().close(fig)

    def test_log_axes_read_as_plain_numbers(self, adata) -> None:
        ax = ps.pl.highly_variable_genes(adata, show=False)
        ax.figure.canvas.draw()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert labels and not any("10^" in s or "times" in s for s in labels), labels
        plmod._plt().close(ax.figure)
