"""4 - Figures.

Run:  python examples/04_plotting.py

Every plot in `ps.pl` follows the same three conventions:

    show=True     draw it now (default; set False in scripts)
    save="p.png"  write it to a file
    ax=<Axes>     draw into an axis you own, for multi-panel figures

and returns the axis, so you can keep customising with plain matplotlib.

Themes are global — `ps.pl.set_theme("dark")` — or scoped to a block with
`with ps.pl.theme("dark"): ...`. They carry colours as well as sizes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polariseq as ps

from _data import load

# ---- settings ------------------------------------------------------------
INPUT = ""   # your .h5ad / 10x .h5 / 10x folder; "" = PBMC 3k (downloaded once)
OUT = Path(__file__).resolve().parent / "output"
# --------------------------------------------------------------------------


def prepared():
    """Run the default analysis once, so every plot below has something to draw."""
    adata = load(INPUT)
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=200)
    ps.pp.filter_genes(adata, min_cells=3)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    ps.pp.renormalize(adata)
    ps.pp.pca(adata, n_components=50, scale=True, seed=0)
    ps.pp.neighbors(adata, n_neighbors=15, seed=0)
    ps.tl.leiden(adata, resolution=1.0, seed=0)
    ps.tl.umap(adata, seed=0)
    return adata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adata = prepared()

    ps.tl.rank_genes_groups(adata, groupby="leiden")
    de = adata.uns["rank_genes_groups"]
    top = [de[g]["names"][0] for g in sorted(de, key=int)]

    # --- one plot, one file ----------------------------------------------
    ps.pl.umap(adata, color="leiden", show=False, save=str(OUT / "p_umap.png"))

    # `color` takes an obs column, a gene name, or a continuous variable.
    ps.pl.umap(adata, color="n_genes", show=False, save=str(OUT / "p_umap_ngenes.png"))
    ps.pl.umap(adata, color=top[0], show=False, save=str(OUT / "p_umap_gene.png"))

    # --- a list of colours draws its own grid ------------------------------
    # The most common thing anyone wants from a UMAP: clusters beside a few
    # markers. `ncols` controls the shape; panels are sized to the data.
    ps.pl.umap(adata, color=["leiden", *top[:3]], ncols=2,
               show=False, save=str(OUT / "p_umap_panels.png"))

    # --- highlight a subset, keep the rest as context ----------------------
    # The un-selected cells stay on the plot in a flat neutral: a cluster
    # shown alone loses the shape of the manifold it sits in.
    ps.pl.umap(adata, color="leiden", groups=["0", "2"],
               show=False, save=str(OUT / "p_umap_groups.png"))

    # --- marker ranking ----------------------------------------------------
    ps.pl.rank_genes_groups(adata, n_genes=8, ncols=4,
                            show=False, save=str(OUT / "p_rank_genes.png"))

    # --- a multi-panel figure, built with matplotlib ----------------------
    # Pass `ax=` and the plot draws into your grid instead of its own figure.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ps.pl.umap(adata, color="leiden", ax=axes[0][0], show=False)
    ps.pl.pca(adata, color="leiden", ax=axes[0][1], show=False)
    ps.pl.pca_variance_ratio(adata, ax=axes[1][0], show=False)
    ps.pl.highly_variable_genes(adata, ax=axes[1][1], show=False)
    fig.suptitle("PBMC 3k — overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "p_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- marker-driven plots ----------------------------------------------
    # `markers=` or scanpy's `var_names=` — both are accepted.
    ps.pl.dotplot(adata, top, groupby="leiden",
                  show=False, save=str(OUT / "p_dotplot.png"))
    ps.pl.heatmap(adata, top, groupby="leiden",
                  show=False, save=str(OUT / "p_heatmap.png"))
    ps.pl.violin(adata, top[:4], groupby="leiden",
                 show=False, save=str(OUT / "p_violin.png"))

    # --- themes ------------------------------------------------------------
    # Presets: paper (default), notebook, talk (large type for slides),
    # dark, print. Themes carry colours as well as sizes, so `dark` is a
    # real inversion — text, spines, grid and the expression ramp all move.
    print(f"themes available: {', '.join(ps.pl.THEMES)}")

    # `theme` is a context manager: use one for a block, then restore.
    with ps.pl.theme("dark"):
        ps.pl.umap(adata, color="leiden",
                   show=False, save=str(OUT / "p_umap_dark.png"))
        ps.pl.umap(adata, color=top[0],
                   show=False, save=str(OUT / "p_umap_dark_gene.png"))

    # Any theme key can be overridden.
    ps.pl.set_theme("talk", point_scale=2.2)
    ps.pl.umap(adata, color="leiden", show=False, save=str(OUT / "p_umap_talk.png"))
    ps.pl.set_theme("paper")

    # --- the returned axis is a normal matplotlib Axes ---------------------
    ax = ps.pl.umap(adata, color="leiden", show=False)
    ax.set_title("annotated by hand")
    coords = np.asarray(adata.obsm["X_umap"])
    ax.annotate("largest cluster", xy=coords.mean(axis=0), fontsize=9,
                arrowprops={"arrowstyle": "->"}, xytext=(20, 20),
                textcoords="offset points")
    ax.figure.savefig(OUT / "p_custom.png", dpi=200, bbox_inches="tight")

    print(f"wrote {len(list(OUT.glob('p_*.png')))} figures to {OUT}/")


if __name__ == "__main__":
    main()
