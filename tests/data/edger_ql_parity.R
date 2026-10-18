# Writes the edgeR and limma reference values in edger_ql_parity.json from
# its inputs. Run from tests/data:  Rscript edger_ql_parity.R
# Values in the file come from R 4.5.2, edgeR 4.8.2, limma 3.66.0.
suppressMessages({library(edgeR); library(jsonlite)})
fixture <- fromJSON("edger_ql_parity.json", simplifyMatrix = TRUE)
e <- fixture$inputs
out <- list()

y <- e$tmm
out$tmm <- calcNormFactors(DGEList(y, lib.size = colSums(y)), method = "TMM")$samples$norm.factors

out$span105 <- c(chooseLowessSpan(105),
                 chooseLowessSpan(105, small.n = 50, min.span = 0.25, power = 0.5))

out$locfit <- locfitByCol(e$loc_y, e$loc_x, weights = e$loc_w, span = 0.6, degree = 0)

out$nonrobust_df <- squeezeVar(e$var, 6, covariate = e$cov)$df.prior
s <- squeezeVar(e$var, 6, covariate = e$cov, robust = TRUE)
out$robust_var_prior <- s$var.prior
out$robust_df_prior <- s$df.prior
s <- squeezeVar(e$var, e$df_unequal, covariate = e$cov, robust = TRUE, legacy = TRUE)
out$unequal_var_prior <- s$var.prior
out$unequal_df_prior <- s$df.prior
s <- squeezeVar(e$var, e$df_zero, covariate = e$cov, robust = TRUE)
out$zero_var_prior <- s$var.prior
out$zero_df_prior <- s$df.prior

y <- e$zg
design <- model.matrix(~ factor(rep(1:3, each = 2)))
d <- estimateDisp(calcNormFactors(DGEList(y)), design)
out$zg_common <- d$common.dispersion
out$zg_trended <- d$trended.dispersion
out$zg_tagwise <- d$tagwise.dispersion

fixture$edgeR <- out
write_json(fixture, "edger_ql_parity.json", digits = NA, pretty = TRUE)
