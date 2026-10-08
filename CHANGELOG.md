# Changelog

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
