#!/usr/bin/env Rscript
# One run of the fixed pipeline in BPCells' own functions, no Seurat.
#
# BPCells is designed to be used directly: lazy, streamed operations over a
# bit-packed matrix on disk, then its own PCA (svds), kNN (HNSW), graph and
# Leiden. Running it through Seurat (the seurat-bpcells arm) measures Seurat's
# functions reading a BPCells matrix, not BPCells. This arm follows the
# BPCells tutorial's structure while holding the pipeline fixed:
#
#   * highly variable genes by scanpy's "seurat" flavour (log dispersion of
#     the normalised counts, z-scored within 20 equal-width bins of log1p
#     mean, top n), computed from BPCells' streamed row statistics, so the
#     gene set is comparable with every other arm;
#   * PCA of the centred log data (no per-gene scaling), as in every other
#     arm; the BPCells tutorial also scales, which the fixed pipeline omits;
#   * a UMAP-style fuzzy (geodesic) kNN graph, like scanpy's, and Leiden at
#     the shared resolution;
#   * when a UMAP is asked for, uwot::umap on the principal components, as the
#     BPCells tutorial draws its UMAP (BPCells has no UMAP of its own), with
#     the shared number of neighbours and the run's threads.
#
# Same contract as seurat_pipeline.R: STEP markers on stderr, artifacts as
# csv/txt outside the timed region, a JSON summary on the last stdout line.
#
# Usage: Rscript bpcells_native.R <config.json>

suppressPackageStartupMessages({
  library(jsonlite)
})

cfg <- fromJSON(commandArgs(trailingOnly = TRUE)[1])
T0 <- as.numeric(Sys.time())

step <- function(name, expr) {
  b <- as.numeric(Sys.time()) - T0
  force(expr)
  e <- as.numeric(Sys.time()) - T0
  cat(sprintf("STEP %s %.4f %.4f\n", name, b, e), file = stderr())
  invisible(NULL)
}

threads <- max(1L, as.integer(cfg$threads))
Sys.setenv(OMP_NUM_THREADS = threads, OPENBLAS_NUM_THREADS = threads)
try(RhpcBLASctl::blas_set_num_threads(threads), silent = TRUE)

step("import", {
  suppressPackageStartupMessages({
    library(BPCells)
    library(Matrix)
  })
})

# genes x cells, as BPCells stores it.
step("load", {
  mat <- open_matrix_dir(dir = cfg$bpcells_dir)
})

step("qc", {
  cs <- matrix_stats(mat, col_stats = "mean", threads = threads)$col_stats
  genes_per_cell <- cs["nonzero", ]
  counts_per_cell <- cs["mean", ] * nrow(mat)
})

step("filter_cells", {
  keep <- genes_per_cell >= cfg$min_genes
  mat <- mat[, keep]
  counts_per_cell <- counts_per_cell[keep]
})

step("normalize", {
  norm <- multiply_cols(mat, cfg$target_sum / counts_per_cell)
})

step("log1p", {
  logm <- log1p(norm)
})

step("hvg", {
  # scanpy flavor="seurat": mean and ddof=1 variance of the normalised
  # (not logged) counts, one streamed pass.
  rs <- matrix_stats(norm, row_stats = "variance", threads = threads)$row_stats
  mu <- rs["mean", ]
  v <- rs["variance", ]
  mu[mu == 0] <- 1e-12
  disp <- v / mu
  disp[disp == 0] <- NA
  disp <- log(disp)
  lmu <- log1p(mu)
  bins <- cut(lmu, breaks = 20)
  bmean <- tapply(disp, bins, mean, na.rm = TRUE)
  bsd <- tapply(disp, bins, sd, na.rm = TRUE)
  one <- is.na(bsd) & !is.na(bmean)
  bsd[one] <- bmean[one]
  bmean[one] <- 0
  dnorm <- (disp - bmean[as.integer(bins)]) / bsd[as.integer(bins)]
  dnorm[is.na(dnorm)] <- -Inf
  hvg <- order(dnorm, decreasing = TRUE)[seq_len(cfg$n_hvg)]
  hvg <- sort(hvg)
})

step("pca", {
  # The HVG subset is small; materialise it once in memory, as the BPCells
  # tutorial does, then centre lazily and run BPCells' own svds.
  sub <- write_matrix_memory(logm[hvg, ], compress = FALSE)
  gm <- matrix_stats(sub, row_stats = "mean", threads = threads)$row_stats["mean", ]
  centred <- sub - gm
  sv <- svds(centred, k = cfg$n_pcs, threads = threads)
  pca <- sweep(sv$v, 2, sv$d, `*`)
})

step("neighbors", {
  knn <- knn_hnsw(pca, k = cfg$n_neighbors, threads = threads, verbose = FALSE)
  graph <- knn_to_geodesic_graph(knn, threads = threads)
})

step("leiden", {
  # cluster_graph_leiden is a thin wrapper around igraph, but its
  # graph_from_adjacency_matrix(mode = "lower") builds a dense n x n
  # intermediate (5.7 GB at 20K cells; the 50K-cell run was killed at a
  # 43 GB footprint). Building the same undirected weighted graph from the
  # edge list avoids that and leaves the clustering itself unchanged.
  e <- Matrix::summary(Matrix::tril(graph, -1))
  ig <- igraph::make_graph(rbind(e$i, e$j), n = nrow(graph), directed = FALSE)
  set.seed(cfg$seed)
  clusters <- as.factor(igraph::membership(igraph::cluster_leiden(
    ig, resolution = cfg$resolution, objective_function = "modularity",
    weights = e$x)))
})

if (isTRUE(cfg$with_umap)) {
  step("umap", {
    set.seed(cfg$seed)
    umap <- uwot::umap(pca, n_neighbors = cfg$n_neighbors, n_threads = threads,
                       n_sgd_threads = threads)
  })
} else {
  step("umap", { invisible(NULL) })
}

# --- artifacts, outside the timed region -----------------------------------
a <- cfg$artifacts_dir
dir.create(a, showWarnings = FALSE, recursive = TRUE)
write.table(pca, file.path(a, "pca.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write.table(as.integer(clusters) - 1L, file.path(a, "labels.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
if (isTRUE(cfg$with_umap)) {
  write.table(umap, file.path(a, "umap.csv"), sep = ",",
              row.names = FALSE, col.names = FALSE)
}
writeLines(rownames(mat)[hvg], file.path(a, "hvg.txt"))
writeLines(colnames(mat), file.path(a, "obs_names.txt"))

cat(toJSON(list(
  n_obs_after = ncol(mat),
  n_vars_after = length(hvg),
  n_clusters = length(unique(clusters)),
  bpcells_version = as.character(packageVersion("BPCells")),
  blas = tryCatch(sessionInfo()$BLAS, error = function(e) NA_character_),
  r_version = paste(R.version$major, R.version$minor, sep = ".")
), auto_unbox = TRUE))
