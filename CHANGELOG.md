# Changelog

## 0.2.0 (2026-10-16)

**Licence.** From 0.2.0, Polariseq is released under the PolyForm Noncommercial License 1.0.0 with
additional terms (keep the notices, cite Polariseq; commercial use by agreement with the authors), in
`LICENSE`. Versions 0.1.0 and 0.1.1 were released under GPL-3.0-or-later and are withdrawn. Polariseq is
installed from this repository (`pip install "git+https://github.com/DarkOxygen123/Polariseq.git"`); the
PyPI packages of 0.1.x were removed.

**A light install.** `pip install "git+https://github.com/DarkOxygen123/Polariseq.git"` installs the
analysis with NumPy and pandas only. Figures (`plot`, Matplotlib), SciPy matrices for other tools (`scipy`),
AnnData export (`anndata`) and CellTypist annotation (`celltypist`) are extras, and
`all` adds every one. Without SciPy the neighbour graph in `adata.obsp` holds its CSR arrays and gives the
same clusters and layouts. pip fetches CMake for the build, and maturin installs Rust when it is missing,
so a C compiler is the one build tool to have installed.

Three modules are written anew, so that none of Polariseq follows code under the GNU General Public
License; NN-Descent and the planner change where measurements at 100 million cells showed they had to.
Details and every comparison are in `benchmarks/relicence/README.md` on the reproduce branch.

- **Leiden** follows Traag, Waltman and van Eck (2019) and the authors' CWTS code (MIT). Reduced graphs
  carry node weights as sums, which fixes an error of 0.1.x: every aggregated level counted the internal
  weight of a community twice too often. On nine datasets from 2,700 to 4 million cells, the clusters agree
  with 0.1.x as well as 0.1.x agrees with itself across seeds, at equal or higher modularity, equal or
  better agreement with published labels, and 12 to 25% less time. Louvain (`method="fast"`) gains most
  from the fix (Kang et al.: agreement with the published types 0.42 to 0.51).
- **Pseudobulk testing** is written from the edgeR and limma publications and documentation, its open
  details settled by comparing outputs with edgeR 4.10.5 and limma 3.68.5: on 350 generated datasets the
  same genes pass the filter and p-values agree within 4e-12 (0.1.x: 195 of 200 datasets, 2e-9). The
  special functions follow the NIST DLMF, with libm's `lgamma` and `erfc`. Results on Kang et al. are
  unchanged (6,106 genes at adjusted p < 0.05, as with R).
- **Harmony** follows Korsunsky et al. (2019) and harmony-pytorch (BSD-3), with R harmony's documented
  options. Corrections agree with harmonypy 2.0.2 as closely as harmonypy agrees with itself across
  seeds, at the speed of 0.1.x (four to seven times harmonypy's).
- **NN-Descent** splits subtrees of more than 2^20 points in place, and copies the rows of smaller ones
  as before. Random hyperplanes split unevenly (on the 1.3M-cell brain atlas up to half of all cells
  reached one copy), so one copy per worker thread was the peak of the neighbours step, about 1 KB per
  cell at 16 threads. The graph is identical bit for bit.
- **Planner**: the neighbours step is charged `680 + 1 x threads` bytes per cell above the embedding,
  fitted with 0.2.0 on slices of the Tahoe-100M embedding (20 million cells at 4 to 16 threads, 40 million,
  and all 95.6 million cells). 0.1.1 planned 129 GB for the 95.6-million-cell Tahoe-100M run, which held
  159.5 GB (178.9 GB with mapped files); 0.2.0's step holds 62.6 GB there. The graph steps are refitted
  on every run of the final build (the Mac, Windows and a compute node, 1.3 and 4.0 million cells): inputs
  read into memory count twice in resident memory (the copy, and the pages of the file the step before
  wrote), inputs read in place once; UMAP also counts the embedding, which its multilevel start reads; the
  run's per-cell annotations are charged 64 bytes per cell; UMAP's working memory is 42 bytes per graph
  entry (31 before). The fetal atlas at 6 GB is now planned at 7.3 GB, against 6.3 to 7.3 GB measured
  (5.4 GB planned before); `benchmarks/relicence/planner_v020_check.py` (reproduce branch) sets plan beside measurement.
- **Graph steps read their inputs into memory when they fit.** Neighbours, Leiden and UMAP read the
  embedding and the graphs in random order. 0.1.x read them in place from the project's files, which on a
  cluster's network file system stalled whenever the node dropped the cached pages: on Tahoe-100M the
  neighbours step and Leiden ran at 0.03 to 0.2 cores for hours, and a failed read there kills the process.
  Each step now reads its inputs into memory in one pass when the run, the inputs and the step's working
  memory fit the budget and the memory free at the time (within a Slurm allocation or a cgroup limit);
  read this way, Tahoe-100M's neighbours step takes 1.0 h (3.8 h in the 0.1.1 run) and Leiden 45 min
  (65 min). Otherwise a step reads them in place; when the project is on a network file system it first
  copies them to a local disk, if one has room, and warns if none has. `ps.plan()` applies the same rule,
  shows each graph step's choice and warns about a network project; `adata.uns[step]["inputs"]` and
  example 06's report record the choice made. UMAP builds its layout in memory and writes it once.

## 0.1.1 (2026-10)

Doublet detection (`ps.pp.scrublet`) is about ten times faster; nothing else changes.

- The leading principal components of each batch come from a Lanczos iteration on the sparse counts
  (full reorthogonalization, 1e-10 residual tolerance, seeded) in place of the dense eigendecomposition
  of a genes-by-genes matrix, which took 90% of the time. The dense route remains for small batches
  and as a fallback.
- Batches are scored in parallel, the largest first, and the per-gene sums are freed before the last pass.
- The exact neighbor search works in cache-sized blocks. Neighbors are now exact at every batch size
  (`exact_knn_max` defaults to no limit; set it to use NN-descent above a size). NN-descent was slower at
  every size measured, because Scrublet's neighbor count grows with the square root of the batch.
- Measured on 443,797 cells of the scTab atlas in 141 donor batches (8 cores): 240 s to 22 s, against
  71 s for the reference Scrublet on 8 processes. Calls and thresholds equal those of 0.1.0, scores
  differ at the rounding of 32-bit values, and results do not depend on the number of threads. Agreement
  with the reference Scrublet is as close as the reference is to itself with another seed.

## 0.1.0 (2026-10)

First public release.

- Reading `.h5ad` (dense, CSR, CSC), 10x Genomics `.h5` and MatrixMarket
  folders; several files analyzed as one dataset.
- A planning step (`ps.plan`) that predicts each step's peak memory and the
  disk a run will use from the file's description, before any data are read,
  and chooses between the in-memory and the out-of-core mode.
- An out-of-core mode over a compact, lossless block store with 64-bit row
  pointers, read in blocks sized to the memory budget, running the same
  kernels as the in-memory mode.
- Quality control, filters, normalization, gene selection among genes
  detected in more than 1% of cells, renormalization over the selected genes,
  PCA (exact or randomized), kNN graph, Leiden, Paris hierarchy, UMAP from a
  multilevel spectral start.
- Harmony integration, Scrublet doublet scores, marker genes, cluster
  annotation, merging and sub-clustering, PAGA, pseudotime, and pseudobulk
  comparison of conditions with donors as replicates.
- Plots for every step and overview figures drawn from every cell at any size.
- `ps.set_workflow("scanpy")` for Scanpy's own step definitions.
- Wheels for macOS (Apple Silicon and Intel), Linux x86_64 (manylinux2014:
  glibc 2.17 or newer, so also CentOS and RHEL 7 clusters) and Windows x86_64,
  Python 3.10 to 3.14.
