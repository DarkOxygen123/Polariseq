# Methods and optimizations

How Polariseq reaches atlas scale on ordinary hardware: the engineering under every step, then each
algorithm, the work it follows and what this implementation does differently. Reference implementations
are credited in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

**Contents:** [Optimizations](#optimizations) ·
[UMAP initialization](#multilevel-spectral-umap-initialization) ·
[Atlas-tuned defaults](#atlas-tuned-defaults) · [Paris](#paris-hierarchy) · [Leiden](#leiden) ·
[Harmony](#harmony) · [Scrublet](#scrublet) · [Pseudobulk](#pseudobulk-testing) ·
[PCA](#randomized-pca) · [NN-Descent](#nn-descent) · [UMAP](#umap)

## Optimizations

**Compact block store.** Out of core, the counts are copied once into a lossless layout of little more
than one byte per stored value (a plain layout takes eight). Results are identical to those from the plain
layout, and every chunk carries a checksum.

**Memory budget.** `ps.set_budget` fixes memory and threads, blocks are sized to it, and `ps.plan`
reports the peak of every step before anything runs.

**Parallel file reading.** The compressed chunks of an `.h5ad` file are inflated across all cores while
the file loads.

**Vectorized kernels.** PCA, the neighbor graph and UMAP share kernels with a fixed summation order, so
results repeat from run to run.

## Algorithms

### Multilevel spectral UMAP initialization

*New in Polariseq.*

Computes the spectral embedding on a coarse graph of cell groups, lifts it to individual cells and refines
it on the full graph. Initializes millions of cells in seconds and captures the large-scale structure of
the dataset.

### Atlas-tuned defaults

*New in Polariseq.*

Selects highly variable genes among those detected in more than 1% of cells, renormalizes each cell over
the selected genes and standardizes genes before PCA. `ps.set_workflow("scanpy")` switches every default
to Scanpy's own steps for a step-by-step comparison.

### Paris hierarchy

*Based on scikit-network; balanced cut from Scarf.*

Reproduces the reference bit for bit. Sorted adjacency vectors, extended in order as clusters merge, avoid
hash tables, and a single nearest-neighbor-chain pass yields the full dendrogram, from which clusters at
any resolution are cut.

### Leiden

*Based on Traag, Waltman and van Eck (2019) and CWTS networkanalysis.*

Array-based throughout: dense scratch buffers with touched-entry lists in place of hash maps, and reduced
graphs built by counting sort with node weights carried as sums. A single seeded random stream makes
results reproducible for a given graph and seed.

### Harmony

*Based on Korsunsky et al. (2019), harmony-pytorch and R harmony's documented options.*

Computes centroid distances in tiles, so the soft assignments are the only working array the size of the
data. Tallies and ridge systems use double precision, and results are independent of the thread count.
Corrections agree with harmonypy's as closely as its own runs with different seeds.

### Scrublet

*Based on Scrublet.*

Represents simulated doublets implicitly, as rescaled sums of their parents' projections, keeping memory
at 30 values per cell. Lanczos PCA and parallel batch scoring process 443,797 cells (141 batches,
8 cores) in 22 s, against 71 s for the reference on 8 processes.

### Pseudobulk testing

*Based on the edgeR and limma methods (Robinson and Oshlack 2010; Smyth 2004; Phipson et al. 2016).*

Gives edgeR's and limma's results: the same genes and p-values within 1e-11 on 350 generated datasets.
Streams raw counts once, from memory or the block store, and aggregates them per replicate, condition and
cell type before limma-trend testing with donors as replicates.

### Randomized PCA

*Based on Halko, Martinsson and Tropp (2011).*

Implicit mean-centering keeps the matrix sparse throughout; an exact solver handles the selected-gene
matrix.

### NN-Descent

*Based on pynndescent.*

Random-projection-forest initialization with a bounded copy per subtree, and block-bounded memory; the
resulting graph is identical at any thread count. Exact search is used for smaller datasets.

### UMAP

*Based on umap-learn.*

Builds the fuzzy graph once as a symmetric sparse matrix shared by initialization and optimization.
Optimization is multithreaded; `deterministic=True` gives bit-reproducible layouts.
