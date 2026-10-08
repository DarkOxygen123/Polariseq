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
        """`print` washes nothing, `dark` washes less than `paper`."""
        ps.pl.set_theme("paper")
        paper = ps.pl.categorical_palette(4)
        ps.pl.set_theme("print")
        printed = ps.pl.categorical_palette(4)
        assert paper != printed

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
        with pytest.raises(KeyError, match="neither a column of .obs"):
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
