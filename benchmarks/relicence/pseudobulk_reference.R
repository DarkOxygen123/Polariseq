# Black-box reference for the pseudobulk test: edgeR and limma's outputs on many generated pseudobulk
# datasets, step by step, so a replacement can be checked against R output by output.
#
#   Rscript pseudobulk_reference.R OUT_DIR [N_CASES]
#
# Each case: samples x genes counts from a negative binomial with a gene-wise mean and dispersion,
# library sizes spread over a 10-fold range, some genes all-zero or nearly so, a few very large counts;
# 2 to 4 conditions; replicates either paired (every donor in every condition) or not; 3 to 12 samples
# per condition. The workflow is the module's: filterByExpr(group = condition), keep.lib.sizes = FALSE,
# normLibSizes (TMM), cpm(log = TRUE, prior.count = 3), lmFit on ~ condition or ~ donor + condition,
# eBayes(trend = TRUE), topTable per condition with BH. Writes case_<i>.json per case.
suppressPackageStartupMessages({ library(edgeR); library(limma); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]; n_cases <- if (length(args) > 1) as.integer(args[2]) else 200L
# "small": few genes, all well expressed, so the prior's trend is fitted to 3 to 60 genes
small <- length(args) > 2 && args[3] == "small"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(if (length(args) > 2 && args[3] == "small") 7L else 20261012L)
for (i in seq_len(n_cases)) {
  n_cond <- sample(2:4, 1, prob = c(0.6, 0.25, 0.15))
  paired <- runif(1) < 0.4
  per <- if (paired) rep(sample(3:8, 1), n_cond) else sample(3:12, n_cond, replace = TRUE)
  condition <- rep(seq_len(n_cond) - 1L, per)
  donor <- if (paired) rep(seq_len(per[1]) - 1L, n_cond) else seq_along(condition) - 1L
  g <- if (small) sample(c(3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 29, 30, 31, 40, 60), 1) else sample(c(200, 500, 1500), 1)
  n <- length(condition)
  mu <- if (small) exp(rnorm(g, 7, 1)) else exp(rnorm(g, 3, 2.2))   # gene means, near zero to tens of thousands
  disp <- 0.02 + 0.6 * runif(g)^2
  lib <- exp(runif(n, log(0.3), log(3)))          # library sizes over a 10-fold range
  effect <- matrix(0, n_cond, g)
  de <- sample(g, round(0.1 * g))
  effect[-1, de] <- rnorm((n_cond - 1) * length(de), 0, 1.2)
  dfx <- if (paired) rnorm(per[1], 0, 0.3) else rep(0, n)
  y <- matrix(0, n, g)
  for (s in seq_len(n)) {
    m <- mu * lib[s] * exp(effect[condition[s] + 1, ] + (if (paired) dfx[donor[s] + 1] else 0))
    y[s, ] <- rnbinom(g, mu = m, size = 1 / disp)
  }
  if (!small) y[, sample(g, 5)] <- 0              # genes never seen
  y[sample(length(y), 3)] <- y[sample(length(y), 3)] * 50 + 1e5   # a few very large counts
  ref <- 0L
  grp <- factor(condition)
  dge <- DGEList(counts = t(y), group = grp)
  keep <- filterByExpr(dge, group = grp)
  case <- list(counts = y, condition = condition, donor = donor, paired = paired, reference = ref,
               keep = which(keep) - 1L)
  if (sum(keep) >= 2) {
    dge <- dge[keep, , keep.lib.sizes = FALSE]
    dge <- normLibSizes(dge)
    logcpm <- cpm(dge, log = TRUE, prior.count = 3)
    cf <- relevel(grp, ref = as.character(ref))
    design <- if (paired) model.matrix(~ factor(donor) + cf) else model.matrix(~ cf)
    fit <- eBayes(lmFit(logcpm, design), trend = TRUE)
    cols <- grep("^cf", colnames(design))
    comps <- lapply(cols, function(k) {
      tt <- topTable(fit, coef = k, number = Inf, sort.by = "none", adjust.method = "BH")
      list(condition = as.integer(sub("^cf", "", colnames(design)[k])), logFC = tt$logFC, t = tt$t,
           p = tt$P.Value, adjp = tt$adj.P.Val, ave = tt$AveExpr)
    })
    case$norm_factors <- dge$samples$norm.factors
    case$lib_size <- dge$samples$lib.size
    case$logcpm <- t(logcpm)
    case$df_residual <- fit$df.residual[1]
    case$df_prior <- fit$df.prior
    case$s2_prior <- fit$s2.prior
    case$comparisons <- comps
  }
  write_json(case, file.path(out, sprintf("case_%03d.json", i)), digits = NA, matrix = "rowmajor")
}
cat("wrote", n_cases, "cases to", out, "\n")
