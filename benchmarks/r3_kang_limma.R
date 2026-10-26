# edgeR and limma on the pseudobulk sums benchmarks/r3_kang.py wrote: the
# reference Polariseq's comparison is checked against.
#
#   Rscript benchmarks/r3_kang_limma.R
#
# The recommended pseudobulk workflow (edgeR 4.10, limma 3.68), per cell type:
# filterByExpr on the conditions, TMM (normLibSizes), log2 CPM with a prior
# count of 3, a design blocking on the donor, eBayes with the variance trend.
suppressMessages({library(edgeR); library(limma)})
d <- file.path("benchmarks", "results", "r3", "kang")
counts <- as.matrix(read.csv(file.path(d, "pseudobulk_counts.csv"), row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(d, "pseudobulk_meta.csv"), row.names = 1, check.names = FALSE)
t0 <- Sys.time()
out <- list()
for (ct in unique(meta$cell_type)) {
  s <- meta$cell_type == ct
  cond <- factor(meta$condition[s], levels = c("control", "IFN-beta"))
  if (length(unique(cond)) < 2) next
  y <- DGEList(t(counts[s, , drop = FALSE]))
  donor <- factor(meta$donor[s])
  keep <- filterByExpr(y, group = cond)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- suppressMessages(normLibSizes(y))
  logcpm <- cpm(y, log = TRUE, prior.count = 3)
  fit <- eBayes(lmFit(logcpm, model.matrix(~ donor + cond)), trend = TRUE)
  tt <- topTable(fit, coef = "condIFN-beta", number = Inf, sort.by = "none")
  out[[ct]] <- data.frame(cell_type = ct, gene = rownames(tt), logFC = tt$logFC, t = tt$t,
                          P.Value = tt$P.Value, adj.P.Val = tt$adj.P.Val)
}
seconds <- as.numeric(Sys.time() - t0, units = "secs")
write.csv(do.call(rbind, out), file.path(d, "limma.csv"), row.names = FALSE)
writeLines(sprintf('{"seconds": %.3f, "limma": "%s", "edgeR": "%s"}', seconds,
                   packageVersion("limma"), packageVersion("edgeR")), file.path(d, "limma.json"))
cat(sprintf("edgeR + limma on the sums: %.2f s\n", seconds))
