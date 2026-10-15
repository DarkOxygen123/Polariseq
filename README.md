<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/banner-dark.svg">
    <img src="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/banner-light.svg" alt="Polariseq. Atlas-scale single-cell analysis on the computer you have." width="100%">
  </picture>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.14-3E7FC1?labelColor=2E3340" alt="Python 3.10 to 3.14">
  <img src="https://img.shields.io/badge/platforms-macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-3E7FC1?labelColor=2E3340" alt="macOS, Linux and Windows">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-3E7FC1?labelColor=2E3340" alt="PolyForm Noncommercial 1.0.0"></a>
  <a href="https://github.com/DarkOxygen123/Polariseq/actions"><img src="https://img.shields.io/github/actions/workflow/status/DarkOxygen123/Polariseq/ci.yml?branch=main&label=CI&labelColor=2E3340" alt="CI status"></a>
</p>

<p align="center">
  <a href="#install"><b>Install</b></a> &nbsp;·&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;·&nbsp;
  <a href="#examples"><b>Examples</b></a> &nbsp;·&nbsp;
  <a href="#what-it-does"><b>What it does</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/DarkOxygen123/Polariseq/tree/reproduce"><b>Reproduce</b></a>
</p>

Polariseq analyzes single-cell RNA-seq data from raw counts to clusters, UMAP, marker genes and comparisons between
conditions. It is a Python package with a Rust core that plans before it reads: from a file's description it predicts
the memory and disk each step needs, then runs in memory when the matrix fits and out of core when it does not, with
the same steps in both modes. A few million cells fit on an everyday computer and tens of millions on one server node.

## Optimizations

| Optimization | What it does |
|---|---|
| **Compact block store** | Out of core, the counts are copied once into a lossless layout of little more than one byte per stored value (a plain layout takes eight). Results are identical to those from the plain layout, and every chunk carries a checksum. |
| **Memory budget** | `ps.set_budget` fixes memory and threads, blocks are sized to it, and `ps.plan` reports the peak of every step before anything runs. |
| **Parallel file reading** | The compressed chunks of an `.h5ad` file are inflated across all cores while the file loads. |
| **Vectorized kernels** | PCA, the neighbor graph and UMAP share kernels with a fixed summation order, so results repeat from run to run. |

## Algorithms and implementation

| Method | Based on | Implementation |
|---|---|---|
| **Multilevel spectral UMAP initialization** | New | Computes the spectral embedding on a coarse graph of cell groups, lifts it to individual cells and refines it on the full graph. Initializes millions of cells in seconds and captures the large-scale structure of the dataset. |
| **Atlas-tuned defaults** | New | Selects highly variable genes among those detected in more than 1% of cells, renormalizes each cell over the selected genes and standardizes genes before PCA. |
| **Paris hierarchy** | scikit-network; Scarf (balanced cut) | Reproduces the reference bit for bit. Sorted adjacency vectors, extended in order as clusters merge, avoid hash tables, and a single nearest-neighbor-chain pass yields the full dendrogram, from which clusters at any resolution are cut. |
| **Leiden** | Traag, Waltman and van Eck (2019); CWTS networkanalysis | Array-based throughout: dense scratch buffers with touched-entry lists in place of hash maps, and reduced graphs built by counting sort with node weights carried as sums. A single seeded random stream makes results reproducible for a given graph and seed. |
| **Harmony** | Korsunsky et al. (2019); harmony-pytorch; R harmony's documented options | Computes centroid distances in tiles, so the soft assignments are the only working array the size of the data. Tallies and ridge systems use double precision, and results are independent of the thread count. Corrections agree with harmonypy's as closely as its own runs with different seeds. |
| **Scrublet** | Scrublet | Represents simulated doublets implicitly, as rescaled sums of their parents' projections, keeping memory at 30 values per cell. Lanczos PCA and parallel batch scoring process 443,797 cells (141 batches, 8 cores) in 22 s, against 71 s for the reference on 8 processes. |
| **Pseudobulk testing** | The edgeR and limma methods (Robinson and Oshlack 2010; Smyth 2004; Phipson et al. 2016) | Gives edgeR's and limma's results: the same genes and p-values within 1e-11 on 350 generated datasets. Streams raw counts once, from memory or the block store, and aggregates them per replicate, condition and cell type before limma-trend testing with donors as replicates. |
| **Randomized PCA** | Halko, Martinsson and Tropp (2011) | Implicit mean-centering keeps the matrix sparse throughout; an exact solver handles the selected-gene matrix. |
| **NN-Descent** | pynndescent | Random-projection-forest initialization with a bounded copy per subtree, and block-bounded memory; the resulting graph is identical at any thread count. Exact search is used for smaller datasets. |
| **UMAP** | umap-learn | Builds the fuzzy graph once as a symmetric sparse matrix shared by initialization and optimization. Optimization is multithreaded; `deterministic=True` gives bit-reproducible layouts. |

Reference implementations are credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## An atlas on a laptop

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/hero_dark.png">
    <img src="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/hero_light.png" alt="UMAP of the 4.06 million cells of the human fetal atlas, coloured by cell type, with the laptop time and memory" width="100%">
  </picture>
  <br>
  <sub>Polariseq with default settings on the human fetal atlas (Cao et al. 2020). Laptop figures: out of core, 8 GB
  laptop, median of 3 runs (table below).</sub>
</p>

| dataset | cells | machine | time | peak resident memory |
|---|---:|---|---:|---:|
| mouse brain (10x Genomics) | 1.3 million | laptop, 8 GB, 4 threads | 3.6 min | 3.2 GB |
| human fetal atlas (Cao et al. 2020) | 4.06 million | laptop, 8 GB, 4 threads | 5.9 min | 4.2 GB |

Quality control through clustering, out of core, default settings, 6 GB
budget, median of 3 runs. Peak resident memory is the memory the budget
governs. The macOS physical footprint, which also counts memory the system
compressed, peaked at 3.9 GB and 5.7 GB. The benchmark scripts and their results are on the
[`reproduce`](https://github.com/DarkOxygen123/Polariseq/tree/reproduce)
branch.

## Install

Polariseq is installed from this repository with pip, which compiles its Rust core during installation:

```bash
pip install "git+https://github.com/DarkOxygen123/Polariseq.git"
```

Python 3.10 to 3.14 on macOS, Linux or Windows. The build needs the stable Rust toolchain, a C compiler and
CMake 3.16 or newer, which builds the HDF5 library into the package. pip fetches CMake, and Rust when it is
missing, by itself; the C compiler is the one tool to have installed (macOS: `xcode-select --install`;
Windows: Build Tools for Visual Studio; Linux: usually present). The first build takes a few minutes. Check
the install:

```bash
python -c "import polariseq as ps; print(ps.__version__); print(ps.resources())"
```

### What each install gives you

**The core**, `pip install "git+https://github.com/DarkOxygen123/Polariseq.git"`, is the whole analysis,
with NumPy and pandas as its only Python dependencies:

- **Reading:** `.h5ad`, 10x `.h5` and 10x MatrixMarket folders; one file, or several analysed as one dataset.
- **Planning:** `ps.plan` predicts the memory and disk of every step before reading; the analysis runs in
  memory or out of core.
- **Analysis:** quality control, cell and gene filters, normalization, gene selection, PCA, the neighbour
  graph, Leiden and Louvain clustering, UMAP, the Paris hierarchy and PAGA; `ps.process_diskbacked` runs
  it in one call.
- **Biology:** marker genes (`ps.get.markers`), gene scores, cluster annotation from built-in marker lists,
  merging and subclustering, pseudotime.
- **Several samples:** doublet detection (Scrublet), integration (Harmony), and comparisons between
  conditions with donors as replicates.
- **Results:** cell and gene tables, clusters, UMAP coordinates and marker lists as pandas and NumPy
  objects, ready to save.

It suits a server, a pipeline or any run where the numbers are what you need. Figures, export to AnnData and
SciPy matrices come with the extras below; a function that needs one names the extra to add.

**The extras** add one purpose each, named in brackets:

| extra | installs | lets you |
|---|---|---|
| `plot` | Matplotlib | draw every figure with `ps.pl`: UMAPs by cluster or gene, dot plots, violins, heatmaps, volcano plots, QC and overview figures, in a light or dark theme |
| `anndata` | AnnData, SciPy | export with `adata.to_anndata()` or to `.h5ad`, and continue in Scanpy, scVI or any tool built on AnnData |
| `scipy` | SciPy | take results as SciPy sparse matrices (the neighbour graph in `adata.obsp`, `.to_scipy()`, `adata.sample()`) for code that expects them |
| `celltypist` | CellTypist, AnnData, SciPy | annotate cell types with CellTypist's trained models: `ps.tl.auto_annotate(adata, method="celltypist")` |

**Everything**, `pip install "polariseq[all] @ git+https://github.com/DarkOxygen123/Polariseq.git"`, is
the core with every extra: Matplotlib, AnnData, SciPy and CellTypist. It suits notebook work with figures
and the hand-over to Scanpy, and it is the largest install, since CellTypist brings Scanpy and
scikit-learn.

**Most people need part of it:**

```bash
pip install "polariseq[plot] @ git+https://github.com/DarkOxygen123/Polariseq.git"           # the analysis and its figures (the quick start below)
pip install "polariseq[plot,anndata] @ git+https://github.com/DarkOxygen123/Polariseq.git"   # and the hand-over to Scanpy (every example script)
```

An extra can be added later with the same command.

Each operating system step by step, older Linux systems (CentOS 7 and its kind), cluster nodes, offline
machines, and the complete build from source with the tests: [INSTALL.md](INSTALL.md).

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
ps.pl.umap(adata, color="leiden", show=False, save="umap.png")   # figures: the plot extra
```

The same code runs in memory or out of core; `ps.read_h5ad(path,
mode="memory")` or `mode="out-of-core"` fixes the mode. Other inputs:
`ps.read_10x_h5("filtered_feature_bc_matrix.h5")` and
`ps.read_10x_mtx("filtered_feature_bc_matrix/")`. A list of files,
`ps.read_h5ad(["donor1.h5ad", "donor2.h5ad"])`, is analyzed as one dataset,
with each cell's file in `adata.obs["dataset"]`.

<details>
<summary><b>The whole analysis in one call, out of core</b></summary>

<br>

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
ps.pl.umap(adata, color="leiden", show=False, save="umap.png")   # figures: the plot extra
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

</details>

## Figures from the package

Every figure below comes from `ps.pl`, in the default light theme and in the
dark theme (`ps.pl.set_theme("dark")`).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/genes_dark.png">
    <img src="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/genes_light.png" alt="UMAPs colored by the expression of four genes" width="100%">
  </picture>
  <br><sub><code>ps.pl.umap</code>: gene expression on the embedding</sub>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/dotplot_dark.png">
    <img src="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/dotplot_light.png" alt="Dot plot of marker genes by cell type" width="560">
  </picture>
  <br><sub><code>ps.pl.dotplot</code>: marker genes by cell type</sub>
</p>

Dot plots, violins, heatmaps, volcano plots and multi-panel figures are shown
in [`examples/04_plotting.py`](examples/04_plotting.py).

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
pip install ".[plot,anndata]"     # 01 and 03 need the core only; 02, 04 and 06 the plot extra; 05 plot and anndata
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

## Citation

If you use Polariseq, or results produced with it, cite it with the version you used. The citation is in
[CITATION.cff](CITATION.cff), and GitHub's "Cite this repository" button gives it in APA and BibTeX:

> Uthamkumar M, Dutta A, Basak T. Polariseq: atlas-scale single-cell analysis on the computer you have.
> Version 0.2.0 (2026). https://github.com/DarkOxygen123/Polariseq

## License

Polariseq is released under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
with additional terms, all in [LICENSE](LICENSE):

- **Free for noncommercial use**: academic research, teaching, and use by universities, research
  institutes and nonprofit organizations, whatever their funding, as long as the work is not a paid
  service or product for a company.
- **Keep the notices**: copies and changed versions carry [LICENSE](LICENSE) and
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and a changed version says it is based on Polariseq.
- **Cite it**: work that uses Polariseq cites it (see [Citation](#citation)).
- **Commercial use requires an agreement with the authors**, including use inside a company and analyses
  run as a paid service. Contact: trayambak@iitmandi.ac.in.

Some steps follow code published by others under permissive licences; those projects and their licences
are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright (C) 2026 Uthamkumar M, Abhi Dutta and Trayambak Basak.
