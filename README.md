# edgePython

`edgePython` is a Python implementation of the Bioconductor `edgeR` package for differential analysis of genomics count data. It also includes a new single-cell differential expression method that extends the NEBULA-LN negative binomial mixed model with edgeR's TMM normalization and empirical Bayes dispersion shrinkage.

**edgePython 0.5.0 matches edgeR 5.0.** edgePython's version numbers follow edgeR's major version with a leading 0 (edgePython 0.5 corresponds to edgeR 5). Results are checked function by function against edgeR 4.99.8 and limma 3.99.2, the Bioconductor development versions that become edgeR 5.0, and agree to within numerical precision (typically a relative difference of 1e-8 or less). The reference values come from [tests/data/edger5_parity.R](tests/data/edger5_parity.R) and are checked by [tests/test_edger5_parity.py](tests/test_edger5_parity.py). Where edgeR 5.0 has a bug, edgePython does what edgeR intends instead; these cases are listed under [edgeR bugs corrected in edgePython](#edger-bugs-corrected-in-edgepython).

[![PyPI version](https://img.shields.io/pypi/v/edgepython)](https://pypi.org/project/edgepython/)
[![PyPI downloads](https://static.pepy.tech/badge/edgepython)](https://pepy.tech/project/edgepython)
[![License](https://img.shields.io/pypi/l/edgepython)](https://pypi.org/project/edgepython/)

## Installation

From PyPI:

```bash
pip install edgepython
```

With optional extras from PyPI:

```bash
pip install "edgepython[all]"
```

From source:

```bash
pip install .
```

With optional extras:

```bash
pip install .[all]
```

## Quick Start

```python
import numpy as np
import edgepython as ep

# genes x samples count matrix
counts = np.random.poisson(lam=10, size=(1000, 6))
group = np.array(["A", "A", "A", "B", "B", "B"])

y = ep.make_dgelist(counts=counts, group=group)
y = ep.calc_norm_factors(y)
y = ep.estimate_disp(y)

design = np.column_stack([np.ones(6), (group == "B").astype(float)])
fit = ep.glm_ql_fit(y, design)
res = ep.glm_ql_ftest(fit, coef=1)
top = ep.top_tags(res, n=10)
print(top["table"].head())
```

## Features

### Data Structures

`DGEList`-style data structures (`make_dgelist`, `cbind_dgelist`, `rbind_dgelist`, `valid_dgelist`) with accessor functions (`get_counts`, `get_dispersion`, `get_norm_lib_sizes`, `get_offset`).

### Normalization

TMM, TMMwsp, RLE, and upper-quartile normalization via `calc_norm_factors` (also available as `norm_lib_sizes`, edgeR's newer name). Normalized expression values via `cpm`, `rpkm`, `tpm`, `ave_log_cpm`, `cpm_by_group`, and `rpkm_by_group`.

### Filtering

Gene filtering by expression level via `filter_by_expr`, including edgeR 5.0's `preset` and `max_n_samples` options.

### Dispersion Estimation

Common, trended, and tagwise dispersion estimation (`estimate_disp`, `estimate_common_disp`, `estimate_trended_disp`, `estimate_tagwise_disp`) with GLM variants (`estimate_glm_common_disp`, `estimate_glm_trended_disp`, `estimate_glm_tagwise_disp`). Weighted likelihood empirical Bayes shrinkage via `WLEB`.

### Differential Expression Testing

- **Exact test**: `exact_test` for two-group comparisons with exact negative binomial tests, plus helpers (`exact_test_double_tail`, `equalize_lib_sizes`, `q2q_nbinom`, `split_into_groups`).
- **GLM fitting**: `glm_fit`, `glm_ql_fit` for generalized linear model fitting.
- **GLM testing**: likelihood ratio tests (`glm_lrt`), quasi-likelihood F-tests (`glm_ql_ftest`, with edgeR 5.0's TREAT and UPSHOT fold-change thresholds), and fold-change threshold testing (`glm_treat`).
- **Results**: `top_tags` for extracting top DE genes with p-value adjustment, `decide_tests` for classifying genes as up/down/unchanged.

### Linear Models and voom (limma)

Ports of limma 3.99: `voom` (with offsets and an adaptive lowess span), `voom_lmfit`, `array_weights`, `duplicate_correlation` and `squeeze_var`.

### Gene Set Testing

Competitive and self-contained gene set tests: `camera`, `fry`, `roast`, `mroast`, `romer`. `camera` and `fry` follow limma 3.99, including observation weights and weighted gene sets. Gene ontology and KEGG pathway enrichment via `goana` and `kegga`.

### Differential Splicing

Differential exon and transcript usage testing via `diff_splice` (GLM-based with LRT or QL tests), `diff_splice_dge` (exact test for two-group comparisons), and `splice_variants` (chi-squared tests for homogeneity of proportions across exons).

### Paired Counts and Binomial Models

Differential proportion analysis of paired counts, such as methylated and unmethylated reads, following edgeR 5.0: `make_pclist` and `read_bismark2pc` build a PCList; `bin_fit` fits binomial GLMs; `bin_ql_fit` and `bin_ql_ftest` give quasi-binomial F-tests with bias-adjusted deviances, empirical Bayes quasi-dispersions and TREAT/UPSHOT odds-ratio thresholds; `sample_weights` estimates sample quality weights from binomial or negative binomial QL fits.

### Quantification Uncertainty

Reading quantification output with bootstrap or Gibbs sampling uncertainty from Salmon (`catch_salmon`), kallisto (`catch_kallisto`), lr-kallisto (`catch_lr_kallisto`), oarfish (`catch_oarfish`) and RSEM (`catch_rsem`), each with a gene-level version (`catch_salmon_gene`, `catch_kallisto_gene`, `catch_lr_kallisto_gene`, `catch_oarfish_gene`, `catch_rsem_gene`). Overdispersion estimates from quantification uncertainty are used for differential transcript expression following the approach of Baldoni et al. (2024). Gencode transcript names are unpacked into annotation (`split_gencode_tx_names`, `collapse_gencode_tx2gene`). `dgelist_from_tximport` and `dgelist_from_tximeta` build a DGEList from tximport or tximeta output.

### I/O

- **Universal reader**: `read_data` with auto-detection for kallisto (H5/TSV), lr-kallisto, Salmon, oarfish, RSEM, 10X CellRanger, CSV/TSV count tables, AnnData (`.h5ad`), and RDS files. Quantifier output is read with the `catch_*` readers.
- **Specialized readers**: `read_dge` (collates per-sample count files), `read_10x` (10X Genomics output), `feature_counts_to_dgelist` (featureCounts output), `read_bismark2dge` (Bismark methylation coverage).
- **Single-cell aggregation**: `seurat_to_pb` for pseudo-bulk aggregation.
- **Export**: `to_anndata` for converting DGEList and results to AnnData format.

### Visualization

`plot_md` (mean-difference plots), `plot_bcv` (biological coefficient of variation), `plot_mds` (multidimensional scaling), `plot_ql_disp` (quasi-likelihood dispersion), `plot_smear` (smear plots), `ma_plot` (MA plots), and `gof` (goodness of fit).

### Single-Cell Mixed Model

NEBULA-LN-style negative binomial gamma mixed model for multi-subject single-cell data: `glm_sc_fit`, `shrink_sc_disp`, `glm_sc_test`.

### ChIP-Seq

ChIP-seq normalization to matched input controls via `normalize_chip_to_input` and `calc_norm_offsets_for_chip`.

### Methylation/RRBS

Bismark coverage file reader (`read_bismark2dge`) and methylation-specific design matrix construction (`model_matrix_meth`).

### Utilities

Design matrix construction (`model_matrix`), prior count addition (`add_prior_count`), predicted fold changes (`pred_fc`), Good-Turing smoothing (`good_turing`), count thinning/downsampling (`thin_counts`), Gini coefficient (`gini`), sum technical replicates (`sum_tech_reps`), negative binomial z-scores (`zscore_nbinom`), nearest TSS annotation (`nearest_tss`), and variance shrinkage (`squeeze_var`).

## Features not in edgeR

edgePython includes some functionality that edgeR does not have:

- **Single-cell mixed model**: a NEBULA-LN negative binomial gamma mixed model for multi-subject single-cell data, combined with edgeR's TMM normalization and empirical Bayes dispersion shrinkage (`glm_sc_fit`, `shrink_sc_disp`, `glm_sc_test`).
- **lr-kallisto**: `catch_lr_kallisto` and `catch_lr_kallisto_gene` read long-read quantifications from kallisto `quant-tcc`. Run with `-b` and `--matrix-to-directories` (or `--matrix-to-files`), the bootstraps give overdispersion estimates exactly as for kallisto.
- **More gene-level readers**: `catch_oarfish_gene` (oarfish), and `catch_rsem_gene`, which uses the Gibbs summaries in RSEM's own `genes.results` files, so a gene-level reader is available for every supported quantifier.
- **kallisto plaintext output**: `catch_kallisto` also reads `abundance.tsv` with plaintext bootstraps (`bs_abundance_*.tsv`); edgeR requires `abundance.h5`.
- **tximeta without R**: `dgelist_from_tximeta` accepts an AnnData or a dictionary of assays in place of a SummarizedExperiment.
- **Universal reader and AnnData export**: `read_data` detects and reads quantifier output, 10X data, count tables, AnnData (`.h5ad`) and, with R installed, R `.rds` objects; `to_anndata` exports results.
- **MCP server**: [MCP/edgepython_server.py](MCP/edgepython_server.py) lets AI agents run a differential expression analysis through tool calls.

## edgeR bugs corrected in edgePython

These were found in edgeR 4.99.8 (the development version of edgeR 5.0) while checking edgePython against it. Each was confirmed by running edgeR. In each case edgePython follows what edgeR's code and documentation intend:

- **`glmTreat` on legacy QL fits** (`glmQLFit(legacy=TRUE)`) fails with "object 'poisson.bound' not found". edgePython applies the Poisson bound as edgeR 4 did.
- **`glmTreat` with `legacy=TRUE` and the interval null** gives misaligned p-values when any gene has NA statistics (for example all-zero genes): `j <- j[!is.na(j)]` removes entries from a logical index, so later genes receive another gene's adjustment. This was also edgeR 4's only `glmTreat` method. edgePython computes each gene separately.
- **`estimateDisp(trend.method="none")`** fails when `min.row.sum` filters out any gene (also in edgeR 4). edgePython builds the prior for the genes that are kept.
- **`estimateDisp` with gene-specific offsets and no design** fails. edgePython uses the group design that classic mode would use, so the offsets are honored.
- **Weights passed with a DGEList or PCList**: `glmFit`, `glmQLFit`, `estimateDisp`, `estimateGLMCommonDisp`, `estimateGLMTrendedDisp`, `cpmByGroup`, `binFit` and `binQLFit` fail with "formal argument 'weights' matched by multiple actual arguments", and `aveLogCPM` silently ignores the weights. edgePython uses the weights as the object's own.
- **`catchSalmonGene` and `catchKallistoGene` with `verbose=FALSE`** stop with "tx2gene not provided" on Gencode transcript names, because an `else` binds to `if(verbose)`. edgePython's gene readers work regardless of `verbose`.
- **`catchSalmonGene`/`catchKallistoGene` with `gene.length="tximport"`** return the counts in place of the effective gene lengths when no gene has all-zero counts (an empty negative index). edgePython computes the TPM-weighted lengths in all cases.
- **`catchSalmon` without bootstrap or Gibbs samples** (and the other transcript-level readers, which share the code) divides the counts by missing overdispersions and fails with "NA counts not allowed". edgePython leaves the counts undivided.
- **`catchKallistoGene` with `remove.version=TRUE`** keeps version numbers on the gene IDs from `tx2gene`, unlike `catchSalmonGene`, whose documentation says `remove.version` applies to transcript and gene IDs. edgePython removes them.
- **`binFit`/`binQLFit` with a rank-deficient weighted design** (for example a locus with no coverage in a batch) return that locus's coefficients in LINPACK's pivoted column order, so values appear under the wrong coefficients, which also affects `binQLFTest` TREAT/UPSHOT p-values. edgePython keeps the design's column order, with 0 for the inestimable coefficient.

## Examples

The [examples/mammary](examples/mammary) directory contains two notebooks for the GSE60450 mouse mammary dataset ([Fu et al. 2015](https://www.nature.com/articles/ncb3117)):

- [mouse_mammary_tutorial.ipynb](examples/mammary/mouse_mammary_tutorial.ipynb) — edgePython-only tutorial (Colab-ready)
- [mouse_mammary_R_vs_Python.ipynb](examples/mammary/mouse_mammary_R_vs_Python.ipynb) — side-by-side edgeR vs edgePython comparison

The [examples/hoxa1](examples/hoxa1) directory contains two notebooks for the GSE37704 HOXA1 knockdown dataset ([Trapnell et al. 2013](https://doi.org/10.1038/nbt.2594)), with transcript-level quantification by kallisto:

- [hoxa1_tutorial.ipynb](examples/hoxa1/hoxa1_tutorial.ipynb) — edgePython-only tutorial with scaled analysis using bootstrap overdispersion (Colab-ready)
- [hoxa1_R_vs_Python.ipynb](examples/hoxa1/hoxa1_R_vs_Python.ipynb) — side-by-side edgeR vs edgePython comparison reproducing Figure 1 panels

The [examples/clytia](examples/clytia) directory contains a notebook for the *Clytia hemisphaerica* single-cell RNA-seq dataset ([Chari et al. 2021](https://doi.org/10.1016/j.celrep.2021.109751)), demonstrating the NEBULA-LN mixed model with empirical Bayes dispersion shrinkage:

- [clytia_tutorial.ipynb](examples/clytia/clytia_tutorial.ipynb) — single-cell differential expression of fed vs starved gastrodigestive cells across 10 organisms, reproducing Figure 2 panels (Colab-ready)

## Development

Run tests:

```bash
pytest -q
```

## Authorship

The code was written primarily by Claude (Anthropic) and Codex (OpenAI). The project was directed by Lior Pachter.

A detailed description of the project, methods, and benchmarks is available in the associated preprint:

Pachter, L., Differential analysis of genomics count data with edge* (2026). *bioRxiv*. https://doi.org/10.64898/2026.02.16.706223


## edgeR

`edgePython` is based on the [edgeR](https://bioconductor.org/packages/edgeR/) Bioconductor package. The edgeR publications are:

- Robinson MD, Smyth GK (2007). Moderated statistical tests for assessing differences in tag abundance. *Bioinformatics*, 23(21), 2881-2887. [doi:10.1093/bioinformatics/btm453](https://doi.org/10.1093/bioinformatics/btm453)

- Robinson MD, Smyth GK (2007). Small-sample estimation of negative binomial dispersion, with applications to SAGE data. *Biostatistics*, 9(2), 321-332. [doi:10.1093/biostatistics/kxm030](https://doi.org/10.1093/biostatistics/kxm030)

- Robinson MD, McCarthy DJ, Smyth GK (2010). edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics*, 26(1), 139-140. [doi:10.1093/bioinformatics/btp616](https://doi.org/10.1093/bioinformatics/btp616)

- Robinson MD, Oshlack A (2010). A scaling normalization method for differential expression analysis of RNA-seq data. *Genome Biology*, 11(3), R25. [doi:10.1186/gb-2010-11-3-r25](https://doi.org/10.1186/gb-2010-11-3-r25)

- McCarthy DJ, Chen Y, Smyth GK (2012). Differential expression analysis of multifactor RNA-Seq experiments with respect to biological variation. *Nucleic Acids Research*, 40(10), 4288-4297. [doi:10.1093/nar/gks042](https://doi.org/10.1093/nar/gks042)

- Chen Y, Lun ATL, Smyth GK (2014). Differential expression analysis of complex RNA-seq experiments using edgeR. In *Statistical Analysis of Next Generation Sequencing Data*, Springer, 51-74. [doi:10.1007/978-3-319-07212-8_3](https://doi.org/10.1007/978-3-319-07212-8_3)

- Zhou X, Lindsay H, Robinson MD (2014). Robustly detecting differential expression in RNA sequencing data using observation weights. *Nucleic Acids Research*, 42(11), e91. [doi:10.1093/nar/gku310](https://doi.org/10.1093/nar/gku310)

- Dai Z, Sheridan JM, Gearing LJ, Moore DL, Su S, Wormald S, Wilcox S, O'Connor L, Dickins RA, Blewitt ME, Ritchie ME (2014). edgeR: a versatile tool for the analysis of shRNA-seq and CRISPR-Cas9 genetic screens. *F1000Research*, 3, 95. [doi:10.12688/f1000research.3928.2](https://doi.org/10.12688/f1000research.3928.2)

- Lun ATL, Chen Y, Smyth GK (2016). It's DE-licious: A recipe for differential expression analyses of RNA-seq experiments using quasi-likelihood methods in edgeR. In *Statistical Genomics*, Springer, 391-416. [doi:10.1007/978-1-4939-3578-9_19](https://doi.org/10.1007/978-1-4939-3578-9_19)

- Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential expression analysis of RNA-Seq experiments using Rsubread and the edgeR quasi-likelihood pipeline. *F1000Research*, 5, 1438. [doi:10.12688/f1000research.8987.2](https://doi.org/10.12688/f1000research.8987.2)

- Chen Y, Pal B, Visvader JE, Smyth GK (2018). Differential methylation analysis of reduced representation bisulfite sequencing experiments using edgeR. *F1000Research*, 6, 2055. [doi:10.12688/f1000research.13196.2](https://doi.org/10.12688/f1000research.13196.2)

- Baldoni PL, Chen Y, Hediyeh-zadeh S, Liao Y, Dong X, Ritchie ME, Shi W, Smyth GK (2024). Dividing out quantification uncertainty allows efficient assessment of differential transcript expression with edgeR. *Nucleic Acids Research*, 52(3), e13. [doi:10.1093/nar/gkad1167](https://doi.org/10.1093/nar/gkad1167)

- Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful differential analysis of sequencing data with expanded functionality and improved support for small counts and larger datasets. *Nucleic Acids Research*, 53(2), gkaf018. [doi:10.1093/nar/gkaf018](https://doi.org/10.1093/nar/gkaf018)

The single-cell mixed model in `edgePython` is based on NEBULA:

- He L, Davila-Velderrain J, Sumida TS, Hafler DA, Kellis M, Kulminski AM (2021). NEBULA is a fast negative binomial mixed model for differential or co-expression analysis of large-scale multi-subject single-cell data. *Communications Biology*, 4, 629. [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)

## License

This project is licensed under the GNU General Public License v3.0.
