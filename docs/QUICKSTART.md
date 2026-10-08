# Polariseq in ten minutes

From install to annotated clusters on real data. Copy each block into a script
or a notebook.

## 0. Install and get data

```bash
pip install polariseq
mkdir -p data/pbmc3k && cd data/pbmc3k
curl -O https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz && cd ../..
```

## 1. Open the data

```python
import polariseq as ps

ps.project("pbmc3k_analysis")        # where Polariseq keeps what it writes
ps.set_budget(ram_gb=4, threads=4)   # the memory and threads it may use

adata = ps.read_10x_mtx("data/pbmc3k/filtered_gene_bc_matrices/hg19/")
# or ps.read_h5ad("my_dataset.h5ad") or ps.read_10x_h5("filtered_feature_bc_matrix.h5")
print(adata)            # shape and mode, without reading the matrix
```

Opening reads only the file's description. `adata.head(50)` reads the first
50 cells.

## 2. Quality control and normalization

```python
ps.pp.calculate_qc_metrics(adata)          # n_genes, n_counts, pct_counts_mt per cell
ps.pl.violin(adata, keys=["n_genes", "n_counts", "pct_counts_mt"])
ps.pp.filter_cells(adata, min_genes=200)
ps.pp.filter_genes(adata, min_cells=3)
ps.pp.normalize_total(adata, target_sum=1e4)
ps.pp.log1p(adata)
ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
ps.pp.renormalize(adata)                   # each cell normalized again over these genes
```

The genes are ranked among those detected in more than 1% of cells, so that
rarely detected genes do not crowd out the informative ones.

## 3. PCA, graph, clusters, UMAP

```python
ps.pp.pca(adata, n_components=50, scale=True)
ps.pp.neighbors(adata, n_neighbors=15)
ps.tl.leiden(adata, resolution=1.0)
ps.tl.umap(adata)

ps.pl.umap(adata, color="leiden")          # add save="umap.png" to write a file
ps.pl.pca_variance_ratio(adata)
```

`adata.obs["leiden"]` holds the clusters and `adata.obsm["X_umap"]` the
coordinates.

## 4. Marker genes and cell types

```python
ps.tl.rank_genes_groups(adata, groupby="leiden")
print(ps.get.markers(adata, n=5))                     # top genes of each cluster

ps.tl.auto_annotate(adata, tissue="blood")            # PanglaoDB markers, offline
ps.pl.umap(adata, color="cell_type")
ps.pl.dotplot(adata, markers=["CD3D", "CD8A", "MS4A1", "LYZ"], groupby="cell_type")
```

Clusters can be named by hand (`ps.tl.annotate(adata, {"0": "T cells", ...})`),
joined (`ps.tl.merge_clusters`) or split again (`ps.tl.subcluster`).

## 5. Large data: the same code

```python
plan = ps.plan("atlas.h5ad")       # predicted peak memory and disk, from the file's description
adata = ps.read_h5ad("atlas.h5ad") # out of core if the matrix does not fit in half the budget
print(adata.mode)
```

Every step above then runs unchanged. Out of core, the counts are copied once
into the project folder (about one to two bytes per stored value) and read in
blocks sized to the budget. `ps.project("...").status()` shows what the
folder holds.

Several files as one dataset, corrected for the file each cell came from:

```python
adata = ps.read_h5ad(["donor1.h5ad", "donor2.h5ad", "donor3.h5ad"])
# ... the steps of sections 2 and 3, up to PCA ...
ps.pp.harmony_integrate(adata, "dataset")              # adds obsm["X_pca_harmony"]
ps.pp.neighbors(adata, use_rep="X_pca_harmony")
```

## 6. Hand the results to other tools

```python
ad = adata.to_anndata()      # needs pip install anndata
ad.write_h5ad("pbmc3k_analyzed.h5ad")
```

## Next

- `examples/` in the repository: complete scripts with a settings block at
  the top, including a large-run script with a run report.
- `help(ps.pp)`, `help(ps.tl)`, `help(ps.pl)`, `help(ps.get)` for every
  function.
