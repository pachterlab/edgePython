# Writes the edgeR reference values in edger5_parity.json.gz, inputs included.
# Run from tests/data:  Rscript edger5_parity.R
# Values in the file come from R 4.5.2, edgeR 4.99.8 (Bioconductor devel,
# released as edgeR 5.0.0) and limma 3.99.2.
suppressMessages({library(edgeR); library(limma); library(jsonlite)})
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

# glmTreat: the new default (legacy = FALSE) and legacy methods, for LR and
# QL fits. With legacy = TRUE and the interval null, edgeR drops NAs from a logical
# index (j <- j[!is.na(j)]), which misaligns the p-values of later genes
# when some genes are NA (here the all-zero genes). The references use
# edgeR's glmTreat with that index corrected (NAs set to FALSE).
src <- deparse(edgeR::glmTreat)
stopifnot(sum(grepl("j <- j[!is.na(j)]", src, fixed = TRUE)) == 1)
glmTreatFixed <- eval(parse(text = sub("j <- j[!is.na(j)]", "j[is.na(j)] <- FALSE", src, fixed = TRUE)),
                      envir = asNamespace("edgeR"))
treat_values <- function(res) {
  tab <- res$table
  list(logFC = tab$logFC, unshrunk_logFC = tab$unshrunk.logFC, pvalue = tab$PValue)
}
fit_lrt <- glmFit(dge, inputs$additive)
fit_lrt0 <- glmFit(dge, inputs$additive, prior.count = 0)
for (legacy in c(FALSE, TRUE)) {
  sfx <- if (legacy) "_legacy" else ""
  treat <- function(fit, ...) treat_values(glmTreatFixed(fit, lfc = lfc, legacy = legacy, ...))
  out[[paste0("treat_lrt", sfx)]] <- treat(fit_lrt, coef = 5)
  out[[paste0("treat_lrt_worst", sfx)]] <- treat(fit_lrt, coef = 5, null = "worst.case")
  out[[paste0("treat_lrt_unshrunk", sfx)]] <- treat(fit_lrt0, coef = 5)
  out[[paste0("treat_ql", sfx)]] <- treat(fit_new, coef = 5)
  out[[paste0("treat_ql_worst", sfx)]] <- treat(fit_new, coef = 5, null = "worst.case")
  out[[paste0("treat_ql_contrast", sfx)]] <- treat(fit_new, contrast = c(0, 0, 0, 1, -1))
  # edgeR 4.99.8 never sets poisson.bound for legacy QL fits and so fails
  # on them; a global poisson.bound = TRUE (edgeR 4's value, found by R's
  # scoping) gives the intended result
  poisson.bound <- TRUE
  out[[paste0("treat_qllegacy", sfx)]] <- treat(fit_legacy, coef = 5)
  rm(poisson.bound)
}
# estimateDisp: GLM and classic modes, robust, weights and trend methods
disp_values <- function(e) list(common = e$common.dispersion, trended = e$trended.dispersion,
                                tagwise = e$tagwise.dispersion, prior_df = e$prior.df,
                                span = e$span)
dge0 <- normLibSizes(DGEList(counts, group = group))
out$disp_glm <- disp_values(estimateDisp(dge0, inputs$additive))
out$disp_glm_robust <- disp_values(estimateDisp(dge0, inputs$additive, robust = TRUE))
dgew <- dge0
dgew$weights <- weights
out$disp_glm_weights <- disp_values(estimateDisp(dgew, inputs$additive))
out$disp_classic <- disp_values(estimateDisp(dge0))
out$disp_classic_robust <- disp_values(estimateDisp(dge0, robust = TRUE))
out$disp_classic_onegroup <- disp_values(estimateDisp(normLibSizes(DGEList(counts))))
for (tm in c("movingave", "loess", "locfit.mixed"))
  out[[paste0("disp_trend_", tm)]] <- disp_values(estimateDisp(dge0, inputs$additive, trend.method = tm))
# With trend.method = "none", edgeR builds m0 with one row per gene, not per
# gene passing min.row.sum, and fails if any gene is filtered (edgeR 4 too).
# The reference uses estimateDisp.default with nrow(l0) rows.
src <- deparse(edgeR:::estimateDisp.default)
stopifnot(sum(grepl("matrix(colMeans(l0), ntags,", src, fixed = TRUE)) == 1)
estimateDispFixed <- eval(parse(text = sub("matrix(colMeans(l0), ntags,", "matrix(colMeans(l0), nrow(l0),",
                                           src, fixed = TRUE)), envir = asNamespace("edgeR"))
out$disp_trend_none <- disp_values(estimateDispFixed(counts, inputs$additive, group = group,
  lib.size = dge0$samples$lib.size * dge0$samples$norm.factors, trend.method = "none"))
out$common_disp_classic <- estimateCommonDisp(dge0)$common.dispersion
# locfit degree 1, which locfit.mixed uses, with prior weights and ties
set.seed(11)
inputs$loc1_x <- sort(rnorm(600))
inputs$loc1_x[5:8] <- inputs$loc1_x[5]
inputs$loc1_y <- cbind(sin(2 * inputs$loc1_x) + rnorm(600, 0, 0.3), inputs$loc1_x^2)
inputs$loc1_w <- runif(600, 0.5, 2)
out$locfit_degree1 <- locfitByCol(inputs$loc1_y, inputs$loc1_x, weights = inputs$loc1_w,
                                  span = 0.3, degree = 1)
# Genewise offsets without a design: edgeR 5.0 does not use the classic
# model but then fails; the intended result uses the group design
inputs$genewise_offset <- matrix(log(dge0$samples$lib.size * dge0$samples$norm.factors),
                                 ngenes, nlibs, byrow = TRUE) +
                          matrix(seq(-0.05, 0.05, length.out = ngenes * nlibs), ngenes, nlibs)
out$disp_genewise_offset <- disp_values(estimateDisp(counts, design = model.matrix(~group),
                                                     group = group, offset = inputs$genewise_offset))

# The older dispersion estimators, each from an unmodified DGEList
dgen <- normLibSizes(DGEList(counts, group = group))
for (d in c("oneway", "additive")) {
  X <- inputs[[d]]
  for (m in c("CoxReid", "Pearson", "deviance"))
    out[[paste0("glmcommon_", m, "_", d)]] <- estimateGLMCommonDisp(dgen, X, method = m)$common.dispersion
  for (m in c("bin.spline", "bin.loess", "power", "spline"))
    out[[paste0("glmtrended_", m, "_", d)]] <- estimateGLMTrendedDisp(dgen, X, method = m)$trended.dispersion
  chain <- estimateGLMTrendedDisp(estimateGLMCommonDisp(dgen, X), X)
  out[[paste0("glmtagwise_", d)]] <- estimateGLMTagwiseDisp(chain, X)$tagwise.dispersion
  out[[paste0("glmtagwise_notrend_", d)]] <- estimateGLMTagwiseDisp(estimateGLMCommonDisp(dgen, X), X,
                                                                    trend = FALSE)$tagwise.dispersion
  out[[paste0("glmrobust_", d)]] <- estimateGLMRobustDisp(dgen, X)$tagwise.dispersion
}
classic <- estimateCommonDisp(dgen)
out$classic_tagwise <- estimateTagwiseDisp(classic)$tagwise.dispersion
out$classic_tagwise_loess <- estimateTagwiseDisp(classic, trend = "loess", span = 0.8,
                                                 prior.df = 20)$tagwise.dispersion
out$classic_tagwise_none <- estimateTagwiseDisp(classic, trend = "none")$tagwise.dispersion
# Binned trends on 2000 genes: enough for 7 or more bins in dispBinTrend
# and for estimateTrendedDisp's 50 bins
set.seed(5)
mu_big <- outer(rgamma(2000, 0.8, 0.02), rep(1, nlibs))
inputs$big_counts <- matrix(rnbinom(2000 * nlibs, mu = mu_big, size = 8), 2000, nlibs)
storage.mode(inputs$big_counts) <- "double"
dgebig <- normLibSizes(DGEList(inputs$big_counts, group = group))
for (m in c("bin.spline", "bin.loess")) {
  out[[paste0("bigtrended_", m)]] <- estimateGLMTrendedDisp(dgebig, inputs$additive, method = m)$trended.dispersion
  out[[paste0("classic_trended_", m)]] <- estimateTrendedDisp(estimateCommonDisp(dgebig), method = m)$trended.dispersion
}

# filterByExpr: groups, designs (hat() adds an intercept), presets, and the
# max.n.samples cap with two groups of 70
keep_values <- function(k) as.integer(k)
# cpm as edgeR computes it: y * 1e6 / lib.size, and the log with prior counts
out$cpm_raw <- cpm(counts, lib.size = lib.size)
out$cpm_log <- cpm(counts, lib.size = lib.size, log = TRUE, prior.count = 2)
inputs$lib_size <- lib.size
dgef <- DGEList(counts, group = group)
out$filter_group <- keep_values(filterByExpr(dgef))
out$filter_design <- keep_values(filterByExpr(dgef, design = inputs$additive))
inputs$filter_covariate <- seq(-1, 1, length.out = nlibs)^3
out$filter_covariate <- keep_values(filterByExpr(counts, design = cbind(inputs$filter_covariate)))
out$filter_min_sample_size <- keep_values(filterByExpr(dgef, min.sample.size = 2))
for (p in c("stringent", "moderate", "lenient"))
  out[[paste0("filter_", p)]] <- keep_values(filterByExpr(dgef, preset = p))
set.seed(9)
inputs$filter_large <- matrix(rnbinom(400 * 140, mu = rep(rgamma(400, 0.4, 0.03), 140), size = 3), 400, 140)
storage.mode(inputs$filter_large) <- "double"
inputs$filter_large_group <- rep(1:2, each = 70)
out$filter_large <- keep_values(filterByExpr(inputs$filter_large, group = inputs$filter_large_group))
out$filter_large_nocap <- keep_values(filterByExpr(inputs$filter_large, group = inputs$filter_large_group,
                                                   max.n.samples = Inf))

# offset.prior: genewise log adjustments to the library sizes (rows centred,
# as the catch functions make them), used by getOffset, cpm, rpkm and every
# model fitted from the DGEList
set.seed(13)
op <- matrix(rnorm(ngenes * nlibs, 0, 0.2), ngenes, nlibs)
inputs$offset_prior <- op - rowMeans(op)
inputs$ave_length <- 500 + 10 * seq_len(ngenes)
dgeop <- normLibSizes(DGEList(counts, group = group, genes = data.frame(AveLength = inputs$ave_length)))
dgeop$offset.prior <- inputs$offset_prior
out$op_get_offset <- getOffset(dgeop)
out$op_cpm <- cpm(dgeop)
out$op_cpm_log <- cpm(dgeop, log = TRUE)
out$op_rpkm <- rpkm(dgeop)
eop <- estimateDisp(dgeop, inputs$additive)
out$op_disp <- list(common = eop$common.dispersion, trended = eop$trended.dispersion)
out$op_qlftest <- glmQLFTest(glmQLFit(eop, inputs$additive), coef = 5)$table$PValue
out$op_subset_cpm <- cpm(dgeop[11:60, 2:8])

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

# limma 3.99: voom (adaptive span by default), voomLmFit, arrayWeights,
# duplicateCorrelation, camera and fry, on the 2000-gene counts. Gene-by-sample
# matrices keep their first 200 genes.
top <- function(m) m[1:200, , drop = FALSE]
v <- voom(dgebig, inputs$additive)
out$voom <- list(E = top(v$E), weights = top(v$weights), span = v$span)
out$voom_fixed_span <- top(voom(dgebig, inputs$additive, adaptive.span = FALSE)$weights)
out$voom_offset_prior <- voom(dgeop, inputs$additive)$weights
lmfit_values <- function(fit) list(coefficients = top(fit$coefficients), sigma = fit$sigma[1:200],
  df_residual = fit$df.residual, stdev_unscaled = top(fit$stdev.unscaled))
out$voomlmfit <- lmfit_values(voomLmFit(dgebig, inputs$additive))
out$voomlmfit_sw <- lmfit_values(voomLmFit(dgebig, inputs$additive, sample.weights = TRUE))
inputs$block <- rep(1:3, each = 3)
oneway <- inputs$oneway
out$voomlmfit_block <- lmfit_values(voomLmFit(dgebig, oneway, block = inputs$block))
dgez <- dgebig; dgez$counts[1:20, group == "C"] <- 0
out$voomlmfit_zeros <- lmfit_values(voomLmFit(dgez, inputs$additive))
out$array_weights <- arrayWeights(v, inputs$additive)
out$duplicate_correlation <- duplicateCorrelation(voom(dgebig, oneway), oneway, block = rep(1:3, 3))$consensus.correlation
set.seed(1)
inputs$sets <- list(s1 = 1:40, s2 = 101:160, s3 = sample(2000, 80), s4 = 7L)
inputs$set_weights <- seq(-1, 1, length.out = 40)
inputs$array_w <- c(1, 0.5, 2, 1.5, 1, 0.8, 1.2, 0.7, 1.1)
wsets <- list(w1 = data.frame(id = 1:40, w = inputs$set_weights))
gs_values <- function(tab, sets = inputs$sets) {
  tab <- tab[names(sets), ]
  tab$Direction <- if (is.null(tab$Direction)) NULL else as.integer(tab$Direction == "Up")
  as.list(tab)
}
out$camera <- gs_values(camera(v, inputs$sets, inputs$additive))
out$camera_array_weights <- gs_values(camera(v$E, inputs$sets, inputs$additive, weights = inputs$array_w))
out$camera_correlation <- gs_values(camera(v, inputs$sets, inputs$additive, inter.gene.cor = NA))
out$camera_ranks <- gs_values(camera(v, inputs$sets, inputs$additive, use.ranks = TRUE))
out$camera_nondirectional <- gs_values(suppressWarnings(camera(v, inputs$sets, inputs$additive, directional = FALSE)))
out$camera_contrast <- gs_values(camera(v, inputs$sets, inputs$additive, contrast = c(0, 1, -1, 0, 0)))
out$camera_weighted_set <- gs_values(camera(v, wsets, inputs$additive, inter.gene.cor = NA), wsets)
out$camera_dgelist <- gs_values(camera(estimateDisp(dgebig, inputs$additive), inputs$sets, inputs$additive))
out$fry <- gs_values(fry(v, inputs$sets, inputs$additive))
for (s in c("none", "residual.sd", "p2"))
  out[[paste0("fry_", s)]] <- gs_values(fry(v, inputs$sets, inputs$additive, standardize = s))
out$fry_robust <- gs_values(fry(v, inputs$sets, inputs$additive, robust = TRUE))
out$fry_trend <- gs_values(fry(v$E, inputs$sets, inputs$additive, trend.var = TRUE))
out$fry_array_weights <- gs_values(fry(v$E, inputs$sets, inputs$additive, array.weights = inputs$array_w))
out$fry_gene_weights <- gs_values(fry(v$E, inputs$sets, inputs$additive, gene.weights = seq(0.5, 2, length.out = 2000)))
out$fry_contrast <- gs_values(fry(v, inputs$sets, inputs$additive, contrast = c(0, 1, -1, 0, 0)))
out$fry_block <- gs_values(fry(v, inputs$sets, oneway, block = inputs$block, correlation = 0.3))
out$fry_weighted_set <- gs_values(fry(v, wsets, inputs$additive), wsets)
out$fry_dgelist <- gs_values(fry(estimateDisp(dgebig), inputs$sets))

# sampleWeights from glmQLFit fits: from a DGEList, with observation weights
# (which glmQLFit.DGEList takes from y$weights),
# from a matrix with a fixed dispersion, and a refit using the weights
dgeb <- estimateDisp(dgebig, inputs$additive)
qb <- glmQLFit(dgeb, inputs$additive)
out$sample_weights <- sampleWeights(qb)[1, ]
# Weights from a formula, rebuilt in the tests rather than stored
big_weights <- matrix(0.5 + ((seq_len(2000 * nlibs) * 7919) %% 1000) / 1000, 2000, nlibs)
dgebw <- dgeb; dgebw$weights <- big_weights
out$sample_weights_obs <- sampleWeights(glmQLFit(dgebw, inputs$additive))[1, ]
qm <- glmQLFit(inputs$big_counts, inputs$oneway, dispersion = 0.1, offset = log(dgeb$samples$lib.size))
out$sample_weights_matrix <- sampleWeights(qm)[1, ]
dgebs <- dgeb; dgebs$weights <- as.matrix(sampleWeights(qb))
out$sample_weights_refit <- glmQLFTest(glmQLFit(dgebs, inputs$additive), coef = 5)$table$PValue[1:200]
# glmLRT on a fit from a plain matrix: aveLogCPM uses the fit's offset and dispersion
lm_fit <- glmFit(counts, inputs$additive, dispersion = dispersion, offset = offset)
out$lrt_matrix <- as.list(glmLRT(lm_fit, coef = 5)$table[, c("logCPM", "LR", "PValue")])

inputs$version <- c(R = as.character(getRversion()),
                    edgeR = as.character(packageVersion("edgeR")),
                    limma = as.character(packageVersion("limma")))
con <- gzfile("edger5_parity.json.gz", "w", compression = 9)
writeLines(toJSON(list(inputs = inputs, edgeR = out), digits = NA, na = "string"), con)
close(con)
