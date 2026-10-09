# Changelog

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
