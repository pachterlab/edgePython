# Writes the edgeR reference values in edger5_parity.json.gz, inputs included.
# Run from tests/data:  Rscript edger5_parity.R
# Values in the file come from R 4.5.2, edgeR 4.99.8 (Bioconductor devel,
# released as edgeR 5.0.0) and limma 3.99.2.
suppressMessages({library(edgeR); library(jsonlite)})
set.seed(20261017)
inputs <- list()
out <- list()

# Counts with all-zero genes, genes with an all-zero group and a wide range of
# abundances, so the fitters meet their edge cases.
ngenes <- 300
nlibs <- 9
group <- factor(rep(c("A", "B", "C"), each = 3))
batch <- factor(rep(1:3, 3))
mu <- outer(rgamma(ngenes, 0.6, 0.04), rep(1, nlibs))
mu[1:30, group == "B"] <- mu[1:30, group == "B"] * 3
counts <- matrix(rnbinom(ngenes * nlibs, mu = mu, size = 10), ngenes, nlibs)
counts[41:50, ] <- 0
counts[51:60, group == "C"] <- 0
storage.mode(counts) <- "double"
lib.size <- colSums(counts) * exp(rnorm(nlibs, 0, 0.1))
offset <- log(lib.size)
dispersion <- 0.05 + rgamma(ngenes, 2, 40)
weights <- matrix(runif(ngenes * nlibs, 0.5, 1.5), ngenes, nlibs)
inputs$counts <- counts
inputs$offset <- offset
inputs$dispersion <- dispersion
inputs$weights <- weights
inputs$oneway <- unname(model.matrix(~group))
inputs$additive <- unname(model.matrix(~batch + group))

fit_values <- function(fit) list(
  coefficients = fit$coefficients,
  unshrunk = fit$unshrunk.coefficients,
  fitted = fit$fitted.values,
  deviance = fit$deviance)

for (d in c("oneway", "additive")) {
  design <- inputs[[d]]
  out[[paste0("glmfit_", d)]] <- fit_values(
    glmFit(counts, design, dispersion = dispersion, offset = offset))
  out[[paste0("glmfit_", d, "_weights")]] <- fit_values(
    glmFit(counts, design, dispersion = dispersion, offset = offset, weights = weights))
  out[[paste0("glmfit_", d, "_poisson")]] <- fit_values(
    glmFit(counts, design, dispersion = 0, offset = offset))
}
# Contrast LRT, including the genes with an all-zero group
fit <- glmFit(counts, inputs$oneway, dispersion = dispersion, offset = offset)
out$glmlrt_contrast <- glmLRT(fit, contrast = c(0, 1, -1))$table$PValue
out$pred_fc <- predFC(counts, inputs$additive, offset = offset, dispersion = dispersion,
                      prior.count = 1)
apc <- addPriorCount(counts, offset = offset, prior.count = 2)
out$add_prior_count <- list(y = apc$y, offset = as.matrix(apc$offset))
out$mglm_one_group <- mglmOneGroup(counts, dispersion = dispersion, offset = offset)
out$mglm_levenberg <- mglmLevenberg(counts, inputs$additive, dispersion = dispersion,
                                    offset = offset)$coefficients
out$nbinom_deviance <- nbinomDeviance(counts, glmFit(counts, inputs$additive,
  dispersion = dispersion, offset = offset)$fitted.values, dispersion = dispersion)
out$nbinom_deviance_small_disp <- nbinomDeviance(counts[1:40, ],
  matrix(rowMeans(counts[1:40, ]) + 0.5, 40, nlibs), dispersion = 1e-6)

# glmQLFit with legacy = FALSE (the default). From a DGEList holding
# estimateDisp's dispersions, which edgeR 5.0 no longer uses; with
# observation weights; with a preset dispersion above 4, which is not capped.
ql_values <- function(fit, test = TRUE) {
  res <- list(dispersion = fit$dispersion, average_ql = fit$average.ql.dispersion,
              df_adj = fit$df.residual.adj, deviance_adj = fit$deviance.adj,
              s2_prior = fit$s2.prior, df_prior = fit$df.prior, s2_post = fit$s2.post)
  if (test) res$pvalue <- glmQLFTest(fit, coef = 5)$table$PValue
  res
}
dge <- estimateDisp(normLibSizes(DGEList(counts)), inputs$additive)
inputs$norm_factors <- dge$samples$norm.factors
out$qlfit_dgelist <- ql_values(glmQLFit(dge, inputs$additive, robust = TRUE))
out$qlfit_nonrobust <- ql_values(glmQLFit(dge, inputs$additive))
out$qlfit_no_trend <- ql_values(glmQLFit(dge, inputs$additive, abundance.trend = FALSE))
out$qlfit_weights <- ql_values(glmQLFit(counts, inputs$additive, offset = offset,
                                        weights = weights, robust = TRUE))
out$qlfit_disp10 <- ql_values(glmQLFit(counts, inputs$additive, offset = offset,
                                       dispersion = 10), test = FALSE)

# limma's squeezeVar with unequal df (fitFDistUnequalDF1)
s2 <- out$qlfit_weights$deviance_adj / pmax(out$qlfit_weights$df_adj, 1e-8)
inputs$sv_var <- s2
inputs$sv_df <- out$qlfit_weights$df_adj
inputs$sv_cov <- aveLogCPM(counts, offset = offset)
for (robust in c(FALSE, TRUE)) {
  s <- squeezeVar(s2, out$qlfit_weights$df_adj, covariate = inputs$sv_cov, robust = robust)
  out[[paste0("squeezevar_unequal", if (robust) "_robust" else "")]] <-
    list(var_prior = s$var.prior, df_prior = s$df.prior, var_post = s$var.post)
}

# glmQLFTest with fold-change thresholds (TREAT, UPSHOT by default), on
# default and legacy fits, for coefficients and contrasts
ftest_values <- function(res) list(logFC = as.matrix(res$table[, grep("^logFC", names(res$table))]),
                                   F = res$table$F, pvalue = res$table$PValue,
                                   df_total = res$df.total)
fit_new <- glmQLFit(dge, inputs$additive, robust = TRUE)
fit_legacy <- glmQLFit(dge, inputs$additive, robust = TRUE, legacy = TRUE)
lfc <- log2(1.5)
inputs$contrast2 <- cbind(c(0, 0, 0, 1, 0), c(0, 0, 0, 1, -1))
out$ftest_new <- ftest_values(glmQLFTest(fit_new, coef = 5))
out$ftest_new_upshot <- ftest_values(glmQLFTest(fit_new, coef = 5, lfc = lfc))
out$ftest_new_treat <- ftest_values(glmQLFTest(fit_new, coef = 5, lfc = lfc, upshot = FALSE))
out$ftest_new_fc <- ftest_values(glmQLFTest(fit_new, coef = 5, fc = 1.5))
out$ftest_new_contrast <- ftest_values(glmQLFTest(fit_new, contrast = c(0, 0, 0, 1, -1), lfc = lfc))
out$ftest_new_twocoef <- ftest_values(glmQLFTest(fit_new, coef = 4:5, lfc = lfc))
out$ftest_new_twolfc <- ftest_values(glmQLFTest(fit_new, coef = 4:5, lfc = c(0.3, 0.8), upshot = FALSE))
out$ftest_new_contrast2 <- ftest_values(glmQLFTest(fit_new, contrast = inputs$contrast2, lfc = lfc))
out$ftest_legacy <- ftest_values(glmQLFTest(fit_legacy, coef = 5))
out$ftest_legacy_upshot <- ftest_values(glmQLFTest(fit_legacy, coef = 5, lfc = lfc))

# Unequal-df squeezeVar on enough genes that the loess span is below 1
set.seed(7)
inputs$sv2_var <- rchisq(2000, 4) / 4 * exp(rnorm(2000, 0, 0.3))
inputs$sv2_df <- runif(2000, 2, 6)
inputs$sv2_cov <- rnorm(2000)
s <- squeezeVar(inputs$sv2_var, inputs$sv2_df, covariate = inputs$sv2_cov)
out$squeezevar_unequal_2000 <- list(var_prior = s$var.prior, df_prior = s$df.prior)
out$choose_lowess_span <- c(chooseLowessSpan(20000, small.n = 500), chooseLowessSpan(1000))
# R's optimize(), including a minimum at the upper bound
out$optimize <- c(optimize(function(x) (x - 0.3)^2 + sin(5 * x), c(0, 2))$minimum,
                  optimize(function(x) -log(x), c(0.5, 0.9998))$minimum)

# Cox-Reid common dispersion and adjusted profile likelihoods
out$disp_cox_reid <- c(oneway = dispCoxReid(counts, inputs$oneway, offset = offset),
                       additive = dispCoxReid(counts, inputs$additive, offset = offset))
out$apl <- adjustedProfileLik(0.1, counts, inputs$additive, offset = matrix(offset, ngenes, nlibs, byrow = TRUE))
out$apl_oneway <- adjustedProfileLik(0.1, counts, inputs$oneway, offset = matrix(offset, ngenes, nlibs, byrow = TRUE))

inputs$version <- c(R = as.character(getRversion()),
                    edgeR = as.character(packageVersion("edgeR")),
                    limma = as.character(packageVersion("limma")))
con <- gzfile("edger5_parity.json.gz", "w", compression = 9)
writeLines(toJSON(list(inputs = inputs, edgeR = out), digits = NA, na = "string"), con)
close(con)
