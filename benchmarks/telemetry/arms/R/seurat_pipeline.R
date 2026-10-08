#!/usr/bin/env Rscript
# One Seurat run of the fixed pipeline, emitting the same step markers and the
# same artifacts as every Python arm.
#
# Contract with the harness (see arms/base.py):
#   * step markers go to stderr as  STEP <name> <begin_s> <end_s>
#     on a clock whose origin is process start, so the parent can align them
#     with its memory samples exactly as it does for a Python arm.
#   * artifacts are written as plain files the Python side already reads:
#       pca.npy is NOT written from R (no numpy writer); instead pca.csv,
#       labels.csv, hvg.txt and obs_names.txt are written and the Python
#       wrapper converts them. Conversion happens after the timed region.
#
# Usage: Rscript seurat_pipeline.R <config.json>

suppressPackageStartupMessages({
  library(jsonlite)
})

cfg <- fromJSON(commandArgs(trailingOnly = TRUE)[1])
T0 <- as.numeric(Sys.time())

# R caps its own vector heap independently of the machine, and the macOS
# default of 16 Gb refused a 200K-cell run while the process held 3.25 GB
# RSS. The ceiling in force is therefore part of the result, not part of
# the setup, and is recorded so that a record measured under one ceiling
# can never be mistaken for a record measured under another.
R_VSIZE_LIMIT_MB <- tryCatch(mem.maxVSize(), error = function(e) NA_real_)

# Emit a marker on the same clock the parent sampler uses. Written to stderr
# so stdout stays clean for the result JSON.
step <- function(name, expr) {
  b <- as.numeric(Sys.time()) - T0
  force(expr)
  e <- as.numeric(Sys.time()) - T0
  cat(sprintf("STEP %s %.4f %.4f\n", name, b, e), file = stderr())
  invisible(NULL)
}

# Thread pinning. Seurat/BLAS threading is set from the same budget every
# other arm gets — see the fairness note in scanpy_arm.py: an arm given more
# workers than the others measures the launcher, not the implementation.
if (!is.null(cfg$threads) && cfg$threads > 0) {
  Sys.setenv(OMP_NUM_THREADS = cfg$threads,
             OPENBLAS_NUM_THREADS = cfg$threads)
  try(RhpcBLASctl::blas_set_num_threads(cfg$threads), silent = TRUE)
}

step("import", {
  suppressPackageStartupMessages({
    library(Seurat)
    library(Matrix)
    if (isTRUE(cfg$use_bpcells)) library(BPCells)
  })
})

step("load", {
  if (isTRUE(cfg$use_bpcells)) {
    # BPCells wants its own on-disk directory. Converting is NOT part of the
    # timed pipeline for the other arms, so it is done once outside by the
    # Python wrapper and only the open is timed here.
    mat <- open_matrix_dir(dir = cfg$bpcells_dir)
    obj <- CreateSeuratObject(counts = mat, min.cells = 0, min.features = 0)
  } else {
    # The wrapper wrote the matrix as raw little-endian binary before timing
    # started, so no arm pays a format-conversion cost inside its timed
    # region. CSC, genes x cells, which is dgCMatrix's own layout.
    stem <- cfg$counts_stem
    d <- fromJSON(paste0(stem, ".dims.json"))
    x  <- readBin(paste0(stem, ".data.bin"),    "double",  n = d$nnz,     size = 8)
    i  <- readBin(paste0(stem, ".indices.bin"), "integer", n = d$nnz,     size = 4)
    pp <- readBin(paste0(stem, ".indptr.bin"),  "integer", n = d$ncol + 1, size = 4)
    counts <- new("dgCMatrix", i = i, p = pp, x = x,
                  Dim = c(d$nrow, d$ncol),
                  Dimnames = list(d$genes, d$cells))
    obj <- CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0)
  }
})

step("qc", {
  # scanpy's calculate_qc_metrics equivalent: per-cell counts and features are
  # computed by CreateSeuratObject, so this step records the mitochondrial
  # fraction to keep the step list aligned across arms.
  obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^(MT|mt)-")
})

step("filter_cells", {
  obj <- subset(obj, subset = nFeature_RNA >= cfg$min_genes)
})

step("normalize", {
  obj <- NormalizeData(obj, normalization.method = "LogNormalize",
                       scale.factor = cfg$target_sum, verbose = FALSE)
})

# Seurat fuses normalization and log transform in NormalizeData. The marker is
# still emitted, near-zero, so the panels line up and the fusion is visible
# rather than hidden in a differently-named bar (arms/base.py, obligation 1).
step("log1p", { invisible(NULL) })

step("hvg", {
  obj <- FindVariableFeatures(obj, selection.method = "vst",
                              nfeatures = cfg$n_hvg, verbose = FALSE)
})

step("pca", {
  obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
  obj <- RunPCA(obj, features = VariableFeatures(obj),
                npcs = cfg$n_pcs, seed.use = cfg$seed, verbose = FALSE)
})

# Seurat's own defaults differ from scanpy's on three axes: k.param 20 vs 15,
# resolution 0.8 vs 1.0, and algorithm 1 (Louvain) vs Leiden. The `default`
# arm keeps Seurat's, because that is what a Seurat user actually runs; the
# `tuned` arm matches scanpy's where Seurat has the same knob.
#
# What is NOT matched, deliberately: Seurat clusters on a *pruned shared*
# nearest-neighbour graph, scanpy on a fuzzy kNN graph. That is a real
# methodological difference between the tools, not a parameter, and forcing
# either to imitate the other would measure something no user runs.
k_param    <- if (isTRUE(cfg$tuned)) cfg$n_neighbors else 20
resolution <- if (isTRUE(cfg$tuned)) cfg$resolution   else 0.8
algo       <- if (isTRUE(cfg$tuned)) 4 else 1          # 4 = Leiden, 1 = Louvain

step("neighbors", {
  obj <- FindNeighbors(obj, dims = seq_len(cfg$n_pcs),
                       k.param = k_param, verbose = FALSE)
})

step("leiden", {
  obj <- if (algo == 4) {
    FindClusters(obj, resolution = resolution, algorithm = 4,
                 method = "igraph", random.seed = cfg$seed, verbose = FALSE)
  } else {
    FindClusters(obj, resolution = resolution, algorithm = 1,
                 random.seed = cfg$seed, verbose = FALSE)
  }
})

if (isTRUE(cfg$with_umap)) {
  step("umap", {
    obj <- RunUMAP(obj, dims = seq_len(cfg$n_pcs), seed.use = cfg$seed,
                   verbose = FALSE)
  })
} else {
  step("umap", { invisible(NULL) })
}

# --- artifacts, outside the timed region -----------------------------------
a <- cfg$artifacts_dir
dir.create(a, showWarnings = FALSE, recursive = TRUE)
write.table(Embeddings(obj, "pca"), file.path(a, "pca.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(as.integer(Idents(obj)) - 1L, file.path(a, "labels.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
writeLines(VariableFeatures(obj), file.path(a, "hvg.txt"))
writeLines(colnames(obj), file.path(a, "obs_names.txt"))
if (isTRUE(cfg$with_umap)) {
  write.table(Embeddings(obj, "umap"), file.path(a, "umap.csv"),
              sep = ",", row.names = FALSE, col.names = FALSE)
}

cat(toJSON(list(
  k_param_used = k_param,
  resolution_used = resolution,
  algorithm_used = if (algo == 4) "leiden" else "louvain",
  graph_used = "SNN (Seurat native)",
  n_obs_after = ncol(obj),
  n_vars_after = length(VariableFeatures(obj)),
  n_clusters = length(unique(Idents(obj))),
  seurat_version = as.character(packageVersion("Seurat")),
  blas = tryCatch(sessionInfo()$BLAS, error = function(e) NA_character_),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  r_max_vsize_env = Sys.getenv("R_MAX_VSIZE", unset = NA_character_),
  r_vsize_limit_mb = R_VSIZE_LIMIT_MB,
  # Peak vector heap R actually accounted, against the ceiling above.
  # Read after the last timed step, so this gc() cannot perturb a
  # measurement.
  r_vcells_peak_mb = tryCatch({
    g <- gc(); as.numeric(g[2, ncol(g)])
  }, error = function(e) NA_real_)
), auto_unbox = TRUE))
