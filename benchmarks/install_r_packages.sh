#!/usr/bin/env bash
# Install the R side of the comparison. COMPILES FROM SOURCE — this pegs all
# cores, so it must not run while a gated campaign is in flight (the gate
# requires cpu_busy <= ~26% and would stall indefinitely).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== R $(Rscript -e 'cat(as.character(getRversion()))') =="
echo "== BLAS: $(Rscript -e 'cat(sessionInfo()$BLAS)') =="

Rscript - <<'RS'
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4)

need <- function(p) !requireNamespace(p, quietly = TRUE)

# jsonlite and Matrix ship with the Homebrew build; everything else is ours.
if (need("remotes"))     install.packages("remotes")
if (need("RhpcBLASctl")) install.packages("RhpcBLASctl")   # thread pinning
if (need("igraph"))      install.packages("igraph")        # Leiden backend
if (need("Seurat"))      install.packages("Seurat")        # pulls SeuratObject

# BPCells is not on CRAN. It is the disk-backed backend behind Seurat v5's
# large-data path and is the nearest peer to our out-of-core mode, so the
# comparison needs it.
if (need("BPCells")) {
  remotes::install_github("bnprks/BPCells/r", upgrade = "never")
}

for (p in c("Seurat","SeuratObject","BPCells","igraph","RhpcBLASctl","Matrix","jsonlite")) {
  v <- tryCatch(as.character(packageVersion(p)), error = function(e) "FAILED")
  cat(sprintf("  %-14s %s\n", p, v))
}
RS
