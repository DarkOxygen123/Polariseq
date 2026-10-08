#!/usr/bin/env Rscript
# Stream an .h5ad matrix into a BPCells directory (untimed setup).
#
# bpcells_convert.R builds a dgCMatrix in memory first, which at 400K brain
# cells is ~10 GB of doubles and so cannot run on the 8 GB laptop at the very
# sizes the disk-backed arms exist for. BPCells reads AnnData itself; reading
# it lazily and writing the bit-packed directory streams the conversion.
# Counts are stored as uint32, BPCells' native (bit-packable) type.
#
# Usage: Rscript bpcells_from_h5ad.R <config.json>

suppressPackageStartupMessages({
  library(jsonlite); library(BPCells)
})

cfg <- fromJSON(commandArgs(trailingOnly = TRUE)[1])
out <- cfg$bpcells_dir
if (dir.exists(out)) {
  cat("bpcells dir already present, reusing\n"); quit(status = 0)
}
m <- open_matrix_anndata_hdf5(cfg$h5ad_path)
# Seurat rewrites '_' to '-' in feature names; the Seurat-on-BPCells arm
# would otherwise see different names from the h5ad (see seurat_arm.py).
rownames(m) <- gsub("_", "-", rownames(m))
m <- convert_matrix_type(m, "uint32_t")
write_matrix_dir(mat = m, dir = out, compress = TRUE)
cat(sprintf("wrote BPCells dir: %s (%d x %d)\n", out, nrow(m), ncol(m)))
