#!/usr/bin/env Rscript
# Convert the wrapper's raw binary matrix into a BPCells on-disk directory.
#
# This runs as its OWN Rscript invocation, before the timed pipeline, for the
# same reason the Python arms read an already-converted `.h5ad`: a format
# conversion that only one arm pays would measure the storage format rather
# than the implementation. BPCells' bit-packed directory is its native input,
# exactly as `.h5ad` is scanpy's, so building it is setup, not analysis.
#
# Usage: Rscript bpcells_convert.R <config.json>

suppressPackageStartupMessages({
  library(jsonlite); library(Matrix); library(BPCells)
})

cfg <- fromJSON(commandArgs(trailingOnly = TRUE)[1])
stem <- cfg$counts_stem
out  <- cfg$bpcells_dir

if (dir.exists(out)) {
  cat("bpcells dir already present, reusing\n"); quit(status = 0)
}

d  <- fromJSON(paste0(stem, ".dims.json"))
x  <- readBin(paste0(stem, ".data.bin"),    "double",  n = d$nnz,      size = 8)
i  <- readBin(paste0(stem, ".indices.bin"), "integer", n = d$nnz,      size = 4)
pp <- readBin(paste0(stem, ".indptr.bin"),  "integer", n = d$ncol + 1, size = 4)

m <- new("dgCMatrix", i = i, p = pp, x = x,
         Dim = c(d$nrow, d$ncol),
         Dimnames = list(d$genes, d$cells))

# BPCells stores integer counts bit-packed. write_matrix_dir takes the
# matrix and lays it out on disk in its own format.
write_matrix_dir(mat = m, dir = out, compress = TRUE)
cat(sprintf("wrote BPCells dir: %s (%d x %d, %d nnz)\n",
            out, d$nrow, d$ncol, d$nnz))
