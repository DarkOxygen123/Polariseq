"""2 - A complete analysis of one dataset, with figures and marker genes.

Run:  python examples/02_full_workflow.py

With the settings below unchanged it downloads the 10x Genomics PBMC 3k
dataset (about 8 MB, once). To analyze your own data, set INPUT to a .h5ad
file, a 10x .h5 file or a 10x matrix folder, and adjust the rest if needed.
Figures are written to OUTPUT.
"""

from pathlib import Path

import numpy as np
import polariseq as ps
from _data import load

# ---- settings ------------------------------------------------------------
INPUT = ""            # your .h5ad / 10x .h5 / 10x folder; "" = download PBMC 3k
OUTPUT = Path(__file__).resolve().parent / "output" / "full_workflow"
PROJECT = Path(__file__).resolve().parent / "output" / "project"   # Polariseq's working files
RAM_GB = 4            # memory Polariseq may use (GB)
THREADS = 4           # worker threads
MIN_GENES = 200       # fewest genes detected for a cell to be kept
MIN_CELLS = 3         # fewest cells a gene must be detected in
N_HVG = 2000          # highly variable genes used by PCA
N_PCS = 50            # principal components
RESOLUTION = 1.0      # Leiden resolution: higher gives more, smaller clusters
# --------------------------------------------------------------------------


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ps.project(PROJECT)
    ps.set_budget(ram_gb=RAM_GB, threads=THREADS)   # a pinned budget makes runs reproducible

    adata = load(INPUT)
    print(adata)

    # quality control
    ps.pp.calculate_qc_metrics(adata)
    print(f"median genes per cell {np.median(adata.obs['n_genes']):.0f}, "
          f"median counts per cell {np.median(adata.obs['n_counts']):.0f}")
    ps.pl.scatter_qc(adata, show=False, save=str(OUTPUT / "qc.png"))
    ps.pp.filter_cells(adata, min_genes=MIN_GENES)
    ps.pp.filter_genes(adata, min_cells=MIN_CELLS)
    print(f"after filtering: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # normalization and gene selection
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=False)
    ps.pl.highly_variable_genes(adata, show=False, save=str(OUTPUT / "hvg.png"))
    ps.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=True)
    ps.pp.renormalize(adata)              # each cell normalized again over the selected genes

    # PCA, neighbor graph, clusters, UMAP
    ps.pp.pca(adata, n_components=N_PCS, scale=True, seed=0)
    ps.pl.pca_variance_ratio(adata, show=False, save=str(OUTPUT / "pca_variance.png"))
    ps.pp.neighbors(adata, n_neighbors=15, seed=0)
    ps.tl.leiden(adata, resolution=RESOLUTION, seed=0)
    ps.tl.umap(adata, seed=0)
    n_clusters = len(set(np.asarray(adata.obs["leiden"]).tolist()))
    print(f"{n_clusters} clusters")
    ps.pl.umap(adata, color="leiden", show=False, save=str(OUTPUT / "umap_clusters.png"))

    # marker genes
    ps.tl.rank_genes_groups(adata, groupby="leiden")
    print("\ntop marker genes of each cluster:")
    print(ps.get.markers(adata, n=5))
    de = adata.uns["rank_genes_groups"]
    top_genes = list(dict.fromkeys(de[g]["names"][0] for g in sorted(de, key=int)))
    ps.pl.dotplot(adata, top_genes, groupby="leiden", show=False,
                  save=str(OUTPUT / "markers_dotplot.png"))

    print(f"\nfigures written to {OUTPUT}")


if __name__ == "__main__":
    main()
