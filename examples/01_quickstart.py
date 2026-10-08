"""1 - Quickstart: the whole pipeline in a dozen lines.

Run:  python examples/01_quickstart.py

Uses the small test fixture that ships with the package, so it needs no
download and finishes in a couple of seconds. Every later example builds on
this one.
"""

from pathlib import Path

import numpy as np
import polariseq as ps

# ---- settings ------------------------------------------------------------
# The data. Swap this for your own .h5ad and everything else works unchanged.
# The default is a small file in the repository's tests/data folder.
PATH = str(Path(__file__).resolve().parents[1] / "tests" / "data" / "medium_10kx2k.h5ad")
# --------------------------------------------------------------------------


def main() -> None:
    # 1. What is this machine, and what will this run cost?
    #    `plan()` reads only the file's metadata (milliseconds, no matrix
    #    data) and predicts the peak memory before committing to anything.
    print(ps.resources())
    # `plan()` prints its own report unless told otherwise, so don't print it
    # twice. Pass verbose=False when you only want the object back.
    plan = ps.plan(PATH, n_hvg=500, n_pcs=20)
    print(f"-> predicted peak {plan.peak_gb:.2f} GB, limited by '{plan.limiting_step}'\n")

    # 2. Load. `lazy=True` reads the header only; the matrix is pulled in
    #    when a step first needs it.
    adata = ps.read_h5ad(PATH, lazy=True)

    # 3. Preprocessing, in Scanpy's order. Polariseq's defaults rank genes
    #    among those detected in more than 1% of cells and normalize each cell
    #    again over the selected genes before PCA.
    ps.pp.calculate_qc_metrics(adata)

    # `min_genes` is a property of your data, not a universal constant. 200 is
    # the usual choice for a droplet dataset with ~30,000 genes; this fixture
    # has 2,000, so 200 would discard 98% of it. Look at the QC numbers before
    # picking a threshold — that is what `calculate_qc_metrics` is for.
    median_genes = float(np.median(adata.obs["n_genes"]))
    min_genes = max(10, int(0.2 * median_genes))
    print(f"median genes/cell = {median_genes:.0f} -> filtering at min_genes={min_genes}")
    ps.pp.filter_cells(adata, min_genes=min_genes)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=500, subset=True)
    ps.pp.renormalize(adata)   # each cell normalized again over the selected genes

    # 4. Dimensionality reduction and the neighbour graph.
    ps.pp.pca(adata, n_components=20, scale=True, seed=0)
    ps.pp.neighbors(adata, n_neighbors=15, seed=0)

    # 5. Clustering.
    ps.tl.leiden(adata, resolution=1.0, seed=0)

    n_clusters = len(set(adata.obs["leiden"].tolist()))
    print(f"\n{adata.n_obs:,} cells x {adata.n_vars:,} genes -> {n_clusters} clusters")
    print(f"PCA embedding: {adata.obsm['X_pca'].shape}")


if __name__ == "__main__":
    main()
