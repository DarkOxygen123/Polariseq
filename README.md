# Polariseq

Fast, memory-aware single-cell RNA-seq analysis, from a laptop to atlas scale.

Polariseq is a Python package with a Rust core. It runs the standard
single-cell analysis (quality control, normalization, gene selection, PCA,
neighbor graph, Leiden clustering, UMAP, marker genes) through a Scanpy-shaped
API, and decides before reading any data whether the analysis fits in memory
or should run out of core. The same steps run in both modes.

- **Plans before it reads.** `ps.plan("data.h5ad")` reads only the file's
  description and predicts the peak memory of every step and the disk the run
  will use.
- **In memory or out of core, chosen for you.** A matrix that fits in half of
  the memory budget is analyzed in memory. A larger one is copied once into a
  compact block store on disk and read in blocks sized to the budget.
- **Defaults made for large atlases.** Genes are ranked among those detected
  in more than 1% of cells, and UMAP starts from a multilevel spectral layout
  that takes seconds on millions of cells.
- **No system dependencies.** HDF5 is built into the package. macOS, Linux and
  Windows.

| dataset | cells | machine | time | peak resident memory |
|---|---:|---|---:|---:|
| mouse brain (10x Genomics) | 1.3 million | laptop, 8 GB, 4 threads | 3.3 min | 3.1 GB |
| human fetal atlas (Cao et al. 2020) | 4.06 million | laptop, 8 GB, 4 threads | 5.3 min | 3.8 GB |

Quality control through clustering, out of core, default settings, 6 GB
budget, median of 3 runs. Peak resident memory is the memory the budget
governs. The macOS physical footprint, which also counts memory the system
compressed, peaked at 3.9 GB and 6.2 GB. The benchmark scripts and their results are on the
[`reproduce`](https://github.com/DarkOxygen123/Polariseq/tree/reproduce)
branch.

## Install

```bash
pip install polariseq
```

Python 3.10 to 3.14 on macOS (Apple Silicon or Intel), Linux (x86_64) or
Windows (x86_64). Check the install:

```bash
python -c "import polariseq as ps; print(ps.__version__); print(ps.resources())"
```

Optional extras:

```bash
pip install "polariseq[annotation]"    # export to AnnData (adata.to_anndata())
pip install "polariseq[celltypist]"    # CellTypist cell-type annotation
pip install "polariseq[interactive]"   # interactive plotly figures
```

Building from source, offline installs and cluster nodes: [INSTALL.md](INSTALL.md).

## Quick start

```python
import polariseq as ps

ps.project("my_analysis")           # folder for everything Polariseq writes
ps.set_budget(ram_gb=6, threads=4)  # the memory and threads it may use

ps.plan("data.h5ad")                # predicted memory and disk, before reading
adata = ps.read_h5ad("data.h5ad")   # adata.mode says where the analysis runs

ps.pp.calculate_qc_metrics(adata)
ps.pp.filter_cells(adata, min_genes=200)
ps.pp.filter_genes(adata, min_cells=3)
ps.pp.normalize_total(adata, target_sum=1e4)
ps.pp.log1p(adata)
ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
ps.pp.renormalize(adata)            # each cell normalized again over the selected genes
ps.pp.pca(adata, n_components=50, scale=True)
ps.pp.neighbors(adata, n_neighbors=15)
ps.tl.leiden(adata, resolution=1.0)
ps.tl.umap(adata)

ps.tl.rank_genes_groups(adata, groupby="leiden")
print(ps.get.markers(adata, n=5))   # top marker genes of each cluster
ps.pl.umap(adata, color="leiden", show=False, save="umap.png")
```

The same code runs in memory or out of core; `ps.read_h5ad(path,
mode="memory")` or `mode="out-of-core"` fixes the mode. Other inputs:
`ps.read_10x_h5("filtered_feature_bc_matrix.h5")` and
`ps.read_10x_mtx("filtered_feature_bc_matrix/")`. A list of files,
`ps.read_h5ad(["donor1.h5ad", "donor2.h5ad"])`, is analyzed as one dataset,
with each cell's file in `adata.obs["dataset"]`.

### The whole analysis in one call, out of core

`ps.process_diskbacked` runs quality control, the cell and gene filters,
normalization, gene selection, PCA, the neighbor graph and Leiden clustering
in one call, always out of core. It streams the matrix from the file into the
block store in the project folder, so the matrix is never held in memory;
only per-gene vectors and the cells' principal components are.

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
ps.pl.umap(adata, color="leiden", show=False, save="umap.png")
```

It applies the same steps, filters and defaults as the step-by-step code above
(gene selection among genes detected in more than 1% of cells, the second
normalization and gene scaling before PCA) with the same kernels, so both
analyze the same cells and genes. Out of core the two give the same clusters
for the same seed (`process_diskbacked` defaults to `seed=42`, the step
functions to `seed=0`); in memory the second normalization rounds values at
about 1e-7, which can move a few cells between neighboring clusters. The
filters default to keeping every cell and gene; set `min_genes` and
`min_cells` as in the step-by-step code. Further options: `target_sum`,
`max_genes`, `min_counts`, `max_counts`, `max_pct_mt`, `mito_prefix`, `seed`;
`help(ps.process_diskbacked)` lists them.

## Examples

The [`examples/`](examples) folder holds runnable scripts. Each starts with a
short block of settings (input, output folder, memory, threads): set your
paths there and run it.

| script | what it does |
|---|---|
| `01_quickstart.py` | the whole analysis on a small bundled file, in seconds |
| `02_full_workflow.py` | a real dataset (PBMC 3k, downloaded on first run) with figures and marker genes |
| `03_large_data_out_of_core.py` | memory budgets, `ps.plan()` and the out-of-core mode, on any `.h5ad` |
| `04_plotting.py` | every plot type and multi-panel figures |
| `05_comparing_conditions.py` | two conditions: doublets, integration and a per-cell-type comparison with donors as replicates |
| `06_large_run.py` | one large dataset (one or many `.h5ad` files) end to end: run report, figures and a per-cell table |

```bash
git clone https://github.com/DarkOxygen123/Polariseq.git polariseq
cd polariseq
pip install polariseq
python examples/01_quickstart.py
```

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
| biology | marker genes, gene scores, cluster annotation, merging and sub-clustering, PAGA, pseudotime |
| conditions | pseudobulk differential expression with donors as replicates (`ps.tl.compare_conditions`) |
| plots | UMAP, PCA, QC, dot plots, violins, heatmaps, volcano plots, overview figures |
| export | `adata.to_anndata()` for Scanpy and the tools built on it |

`ps.set_workflow("scanpy")` switches every default to Scanpy's own steps (gene
ranking over all genes, no second normalization, UMAP from the exact spectral
start), for a step-by-step comparison with Scanpy.

## Good to know

- **Pin the budget for reproducible runs.** Without `ps.set_budget(ram_gb=...)`
  the budget follows the memory free at the moment, and the block size
  follows the budget.
- **`min_genes` depends on your data.** 200 suits droplet data with about
  30,000 genes; look at `adata.obs["n_genes"]` after
  `ps.pp.calculate_qc_metrics` before choosing.
- **The project folder holds the block store.** Put it on a fast local disk.
  `ps.project(...).status()` shows what it holds, and its `README.md` says
  what may be deleted.
- **Results stay in Polariseq's object.** `adata.to_anndata()` hands them to
  Scanpy or writes an `.h5ad`.

## Reporting issues

Open an issue at <https://github.com/DarkOxygen123/Polariseq/issues> with what
you ran, what happened, and the output of
`python -c "import polariseq as ps; print(ps.get_build_info())"`.

## License

Polariseq is free software, released under the GNU General Public License,
version 3 or later ([LICENSE](LICENSE)). Some steps follow the source of
established tools so that they give the same answers; those tools and their
licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright (C) 2026 Uthamkumar M, Abhi Dutta and Trayambak Basak.
