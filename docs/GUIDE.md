# User guide

What Polariseq does, how to run a whole analysis in one call, and what is good to know before a large run.
For a first analysis step by step, see [QUICKSTART.md](QUICKSTART.md); for every function with its
arguments, [API.md](API.md).

## What it does

| area | functions |
|---|---|
| reading | `.h5ad` (dense, CSR, CSC), 10x `.h5`, MatrixMarket; several files as one dataset |
| planning | `ps.plan`, `ps.set_budget`, `ps.resources`, `ps.project` |
| one-call pipeline | `ps.process_diskbacked`: QC through Leiden, out of core |
| preprocessing | QC metrics, cell and gene filters, normalization, log, highly variable genes, renormalization, PCA |
| graph and clustering | kNN graph (exact or NN-Descent), Leiden, Paris hierarchy |
| embedding | UMAP from a multilevel spectral start |
| integration | Harmony (`ps.pp.harmony_integrate`) |
| doublets | Scrublet (`ps.pp.scrublet`) |
| biology | marker genes, gene scores, cell type annotation from marker databases (`ps.tl.auto_annotate`, `ps.marker_table`), merging and sub-clustering, PAGA, pseudotime |
| conditions | pseudobulk differential expression with donors as replicates (`ps.tl.compare_conditions`) |
| plots | UMAP, PCA, QC, dot plots, violins, heatmaps, volcano plots, overview figures ([PLOTTING.md](PLOTTING.md)) |
| settings | several analyses of the same cells (`adata.from_counts`), genes of your own in the selection (`include=`), agreement between analyses against the seed-to-seed noise (`ps.tl.compare_analyses`) |
| saving | `adata.save(name)` and `ps.load(project, name)`: the counts pointed to, results linked, in memory or out of core |
| export | `adata.write_h5ad(path)` (no `anndata` needed; streamed out of core; raw counts with `counts=True`) and `adata.to_anndata()` for Scanpy and the tools built on it |

`ps.set_workflow("scanpy")` switches every default to Scanpy's own steps (gene ranking over all genes, no
second normalization, UMAP from the exact spectral start), for a step-by-step comparison with Scanpy. How
each method works is in [METHODS.md](METHODS.md).

## Inputs

`ps.read_h5ad(path)` reads an `.h5ad` file, `ps.read_10x_h5("filtered_feature_bc_matrix.h5")` a 10x
`.h5` file and `ps.read_10x_mtx("filtered_feature_bc_matrix/")` a 10x matrix folder. A list of files,
`ps.read_h5ad(["donor1.h5ad", "donor2.h5ad"])`, is analyzed as one dataset, with each cell's file in
`adata.obs["dataset"]`. `mode="memory"` or `mode="out-of-core"` fixes where the analysis runs; by default
`ps.plan` decides from the file and the budget.

## The whole analysis in one call

`ps.process_diskbacked` runs quality control, the cell and gene filters, normalization, gene selection,
PCA, the neighbor graph and Leiden clustering in one call, always out of core. It streams the matrix from
the file into the block store in the project folder, so the matrix is never held in memory; only per-gene
vectors and the cells' principal components are.

```python
import polariseq as ps

ps.project("atlas_analysis")
ps.set_budget(ram_gb=6, threads=4)

adata = ps.process_diskbacked(
    "atlas.h5ad",
    min_genes=200, min_cells=3,       # the same filters as ps.pp.filter_cells / filter_genes
    n_hvg=2000, n_pcs=50, n_neighbors=15, resolution=1.0,
)
ps.tl.umap(adata)                     # UMAP runs on the result
ps.pl.umap(adata, color="leiden", show=False, save="umap.png")   # figures: the plot extra
```

It applies the same steps, filters and defaults as the step-by-step code (gene selection among genes
detected in more than 1% of cells, the second normalization and gene scaling before PCA) with the same
kernels, so both analyze the same cells and genes. Out of core the two give the same clusters for the same
seed (`process_diskbacked` defaults to `seed=42`, the step functions to `seed=0`); in memory the second
normalization rounds values at about 1e-7, which can move a few cells between neighboring clusters. The
filters default to keeping every cell and gene; set `min_genes` and `min_cells` as in the step-by-step
code. Further options: `target_sum`, `max_genes`, `min_counts`, `max_counts`, `max_pct_mt`,
`mito_prefix`, `seed`; `help(ps.process_diskbacked)` lists them.

## Good to know

- **Pin the budget for reproducible runs.** Without `ps.set_budget(ram_gb=...)` the budget follows the
  memory free at the moment, and the block size follows the budget.
- **`min_genes` depends on your data.** 200 suits droplet data with about 30,000 genes; look at
  `adata.obs["n_genes"]` after `ps.pp.calculate_qc_metrics` before choosing.
- **The project folder holds the block store.** Put it on a fast local disk. `ps.project(...).status()`
  shows what it holds, and its `README.md` says what may be deleted.
- **Save an analysis and continue it later.** `adata.save("clustered")` keeps it in the project folder and
  `ps.load("my_analysis", "clustered")` opens it again, in memory or out of core. The counts are not
  copied: the saved analysis points to the files they came from (keep them in the project folder and the
  two move together) and links the results the project already holds. If a file has moved, loading says
  where it was expected.
- **Other tools need an export.** `adata.to_anndata()` hands the results to Scanpy or writes an `.h5ad`.

## Reporting issues

Open an issue at <https://github.com/DarkOxygen123/Polariseq/issues> with what you ran, what happened,
and the output of `python -c "import polariseq as ps; print(ps.get_build_info())"`.
