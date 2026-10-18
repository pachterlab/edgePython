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

inputs$version <- c(R = as.character(getRversion()),
                    edgeR = as.character(packageVersion("edgeR")),
                    limma = as.character(packageVersion("limma")))
con <- gzfile("edger5_parity.json.gz", "w", compression = 9)
writeLines(toJSON(list(inputs = inputs, edgeR = out), digits = NA, na = "string"), con)
close(con)
