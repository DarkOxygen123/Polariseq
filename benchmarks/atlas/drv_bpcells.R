#!/usr/bin/env Rscript
# BPCells arm of the showdown (BPCells 0.3.1 from GitHub, R 4.4).
#
#   Rscript drv_bpcells.R convert --input SHARDS --dir DIR
#   Rscript drv_bpcells.R run --dir DIR --out ART --threads 16 [--doublet-flags PREFIX] [--doublets flag|remove]
#
# convert  opens every .h5ad file with open_matrix_anndata_hdf5 (genes x cells, read lazily), names each
#          cell by its global row, joins the files with cbind and writes one bit-packed matrix directory
#          (counts as uint32, BPCells' packable type).
# run      BPCells' own analysis as its PBMC tutorial gives it: library-size normalization and log1p
#          (lazy), genes ranked by the variance of the log-normalized values, the selected genes written
#          once (to disk: the tutorial names write_matrix_dir for larger data), z-scored as the tutorial
#          does ((x - mean) / variance), svds for 50 components, knn_hnsw (ef 500) and the shared-
#          neighbor graph. Clustering is Leiden at resolution 1 through igraph from the graph's edge list:
#          BPCells' cluster_graph_leiden calls igraph::graph_from_adjacency_matrix, which builds a dense
#          n x n intermediate (5.7 GB at 20,000 cells in our earlier benchmark), impossible at this scale.
#          BPCells has no UMAP: none is computed.
#
# BPCells has no doublet detection. --doublet-flags PREFIX reads the reference Scrublet's flags written by
# doublets_ref.py (PREFIX.global_row.f64, PREFIX.score.f32, PREFIX.predicted.u8) and keeps them as cell
# vectors for review; with --doublets remove the flagged cells are dropped (a lazy column subset).
#
# Contract (common.py): STEP <name> <t_begin> <t_end> on stderr in epoch seconds, JSON summary last on stdout.
# Artifacts in --out: global_rows.f64, labels.i32, hvg.txt.

suppressPackageStartupMessages({ library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
phase <- args[1]
opt <- list(threads = Sys.getenv("SLURM_CPUS_PER_TASK", "16"), doublets = "flag")
i <- 2
while (i < length(args)) {
  opt[[gsub("-", "_", sub("^--", "", args[i]))]] <- args[i + 1]
  i <- i + 2
}
threads <- max(1L, as.integer(opt$threads))
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
P <- fromJSON(file.path(here, "params.json"))
Sys.setenv(OMP_NUM_THREADS = threads, OPENBLAS_NUM_THREADS = threads)

step <- function(name, expr) {
  b <- as.numeric(Sys.time())
  on.exit(cat(sprintf("STEP %s %.3f %.3f\n", name, b, as.numeric(Sys.time())), file = stderr()))
  force(expr)
  invisible(NULL)
}

shard_files <- function(spec) {
  out <- character(0)
  for (p in strsplit(spec, ",")[[1]]) {
    if (dir.exists(p)) {
      f <- list.files(p, pattern = "\\.h5ad$", full.names = TRUE)
      num <- suppressWarnings(as.numeric(sub("\\.h5ad$", "", basename(f))))
      out <- c(out, if (all(!is.na(num))) f[order(num)] else sort(f))
    } else out <- c(out, p)
  }
  out
}

du_gb <- function(path) {
  r <- tryCatch(system2("du", c("-sk", path), stdout = TRUE), error = function(e) NA)
  round(as.numeric(strsplit(r[1], "\t")[[1]][1]) * 1024 / 1e9, 2)
}

step("import", {
  suppressPackageStartupMessages({ library(BPCells); library(Matrix) })
})

if (phase == "convert") {
  files <- shard_files(opt$input)
  step("convert", {
    off <- 0
    mats <- vector("list", length(files))
    for (k in seq_along(files)) {
      m <- open_matrix_anndata_hdf5(files[k])
      m <- convert_matrix_type(m, "uint32_t")
      colnames(m) <- as.character(off + seq_len(ncol(m)) - 1)
      off <- off + ncol(m)
      mats[[k]] <- m
    }
    mat <- if (length(mats) == 1) mats[[1]] else do.call(cbind, mats)
    write_matrix_dir(mat = mat, dir = opt$dir, compress = TRUE)
  })
  cat(toJSON(list(tool = "bpcells", version = as.character(packageVersion("BPCells")), phase = "convert",
                  cells_in = ncol(mat), genes_in = nrow(mat), dir_gb = du_gb(opt$dir)), auto_unbox = TRUE), "\n")
  quit(status = 0)
}

# ---- run --------------------------------------------------------------------------------------------
summary <- list(tool = "bpcells", version = as.character(packageVersion("BPCells")), phase = "run",
                threads = threads, r_version = paste(R.version$major, R.version$minor, sep = "."))
step("load", { mat <- open_matrix_dir(dir = opt$dir) })
summary$cells_in <- ncol(mat)

step("qc", {
  s <- matrix_stats(mat, row_stats = "nonzero", col_stats = "mean", threads = threads)
  genes_per_cell <- s$col_stats["nonzero", ]
  counts_per_cell <- s$col_stats["mean", ] * nrow(mat)
  keep <- genes_per_cell >= P$min_genes
  gkeep <- s$row_stats["nonzero", ] >= P$min_cells
  mat <- mat[gkeep, keep]
  counts_per_cell <- counts_per_cell[keep]
})
summary$cells_after_qc <- ncol(mat)
summary$genes_after_qc <- nrow(mat)

if (!is.null(opt$doublet_flags)) {
  step("doublets_attach", {
    rd <- function(suffix, what, size, signed = TRUE) {
      fn <- paste0(opt$doublet_flags, suffix)
      readBin(fn, what = what, n = file.size(fn) / size, size = size, signed = signed)
    }
    grow <- rd(".global_row.f64", "double", 8)
    dscore <- rd(".score.f32", "numeric", 4)
    dflag <- rd(".predicted.u8", "integer", 1, signed = FALSE)
    pos <- match(grow, as.numeric(colnames(mat)))
    doublet_score <- rep(NA_real_, ncol(mat))
    doublet_flag <- rep(0L, ncol(mat))
    ok <- !is.na(pos)
    doublet_score[pos[ok]] <- dscore[ok]
    doublet_flag[pos[ok]] <- dflag[ok]
  })
  summary$doublets_flagged <- sum(doublet_flag)
  if (opt$doublets == "remove") {
    step("doublet_removal", {
      mat <- mat[, doublet_flag == 0]
      counts_per_cell <- counts_per_cell[doublet_flag == 0]
    })
    summary$cells_after_doublets <- ncol(mat)
  }
}

step("normalize", {
  logm <- log1p(multiply_cols(mat, P$target_sum / counts_per_cell))
})

step("hvg", {
  st <- matrix_stats(logm, row_stats = "variance", threads = threads)
  hvg <- sort(head(order(st$row_stats["variance", ], decreasing = TRUE), P$n_hvg))
})

step("pca", {
  tmp <- file.path(opt$out, "hvg_matrix")
  unlink(tmp, recursive = TRUE)
  sub <- write_matrix_dir(logm[hvg, ], tmp, compress = TRUE)
  gene_means <- st$row_stats["mean", hvg]
  gene_vars <- st$row_stats["variance", hvg]
  z <- (sub - gene_means) / gene_vars
  sv <- svds(z, k = P$n_pcs, threads = threads)
  pca <- multiply_cols(sv$v, sv$d)
})
summary$hvg_matrix_gb <- du_gb(file.path(opt$out, "hvg_matrix"))

step("neighbors", {
  knn <- knn_hnsw(pca, k = P$neighbors, ef = 500, threads = threads, verbose = FALSE)
  snn <- knn_to_snn_graph(knn, return_type = "list")
})

step("leiden", {
  lower <- snn$i > snn$j
  use <- if (any(lower) && !all(lower)) lower else rep(TRUE, length(lower))
  ig <- igraph::make_graph(rbind(snn$i[use] + 1L, snn$j[use] + 1L), n = snn$dim, directed = FALSE)
  set.seed(P$seed)
  clusters <- igraph::membership(igraph::cluster_leiden(
    ig, resolution = P$resolution, objective_function = "modularity", weights = snn$weight[use]))
})

# ---- artifacts, outside the timed steps ---------------------------------------------------------------
dir.create(opt$out, showWarnings = FALSE, recursive = TRUE)
writeBin(as.numeric(colnames(mat)), file.path(opt$out, "global_rows.f64"), size = 8)
writeBin(as.integer(clusters) - 1L, file.path(opt$out, "labels.i32"), size = 4)
writeLines(rownames(mat)[hvg], file.path(opt$out, "hvg.txt"))
unlink(file.path(opt$out, "hvg_matrix"), recursive = TRUE)
summary$cells_analyzed <- ncol(mat)
summary$hvg <- length(hvg)
summary$clusters <- length(unique(clusters))
summary$dir_gb <- du_gb(opt$dir)
cat(toJSON(summary, auto_unbox = TRUE), "\n")
