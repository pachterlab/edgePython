# edgePython

`edgePython` is a Python implementation of the Bioconductor `edgeR` package for differential analysis of genomics count data. It also includes a new single-cell differential expression method that extends the NEBULA-LN negative binomial mixed model with edgeR's TMM normalization and empirical Bayes dispersion shrinkage.

[![PyPI version](https://img.shields.io/pypi/v/edgepython)](https://pypi.org/project/edgepython/)
[![PyPI downloads](https://static.pepy.tech/badge/edgepython)](https://pepy.tech/project/edgepython)
[![License](https://img.shields.io/pypi/l/edgepython)](https://pypi.org/project/edgepython/)
[![Python versions](https://img.shields.io/pypi/pyversions/edgepython)](https://pypi.org/project/edgepython/)

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

TMM, TMMwsp, RLE, and upper-quartile normalization via `calc_norm_factors`. Normalized expression values via `cpm`, `rpkm`, `tpm`, `ave_log_cpm`, `cpm_by_group`, and `rpkm_by_group`.

### Filtering

Gene filtering by expression level via `filter_by_expr`.

### Dispersion Estimation

Common, trended, and tagwise dispersion estimation (`estimate_disp`, `estimate_common_disp`, `estimate_trended_disp`, `estimate_tagwise_disp`) with GLM variants (`estimate_glm_common_disp`, `estimate_glm_trended_disp`, `estimate_glm_tagwise_disp`). Weighted likelihood empirical Bayes shrinkage via `WLEB`.

### Differential Expression Testing

- **Exact test**: `exact_test` for two-group comparisons with exact negative binomial tests, plus helpers (`exact_test_double_tail`, `equalize_lib_sizes`, `q2q_nbinom`, `split_into_groups`).
- **GLM fitting**: `glm_fit`, `glm_ql_fit` for generalized linear model fitting.
- **GLM testing**: likelihood ratio tests (`glm_lrt`), quasi-likelihood F-tests (`glm_ql_ftest`), and fold-change threshold testing (`glm_treat`).
- **Results**: `top_tags` for extracting top DE genes with p-value adjustment, `decide_tests` for classifying genes as up/down/unchanged.

### Gene Set Testing

Competitive and self-contained gene set tests: `camera`, `fry`, `roast`, `mroast`, `romer`. Gene ontology and KEGG pathway enrichment via `goana` and `kegga`.

### Differential Splicing

Differential exon and transcript usage testing via `diff_splice` (GLM-based with LRT or QL tests), `diff_splice_dge` (exact test for two-group comparisons), and `splice_variants` (chi-squared tests for homogeneity of proportions across exons).

### Quantification Uncertainty

Reading quantification output with bootstrap or Gibbs sampling uncertainty from Salmon (`catch_salmon`), kallisto (`catch_kallisto`), and RSEM (`catch_rsem`). Overdispersion estimates from quantification uncertainty are used for differential transcript expression following the approach of Baldoni et al. (2024).

### I/O

- **Universal reader**: `read_data` with auto-detection for kallisto (H5/TSV), Salmon, oarfish, RSEM, 10X CellRanger, CSV/TSV count tables, AnnData (`.h5ad`), and RDS files.
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

This code was written by Claude (Anthropic). The project was directed by Lior Pachter.

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
