# edgePython

`edgePython` is a Python implementation of the Bioconductor `edgeR` package for differential analysis of genomics count data. It also includes a new single-cell differential expression method that extends the NEBULA-LN negative binomial mixed model with edgeR's TMM normalization and empirical Bayes dispersion shrinkage.

The package includes:
- `DGEList`-style data structures
- normalization and filtering
- dispersion estimation
- exact test and GLM-based testing (`LRT`, `QLF`, `TREAT`)
- gene set testing (`camera`, `fry`, `roast`, `mroast`, `romer`, `goana`, `kegga`)
- differential splicing utilities
- I/O helpers for common RNA-seq quantification outputs
- single-cell negative binomial mixed models (`glm_sc_fit`, `glm_sc_test`, `shrink_sc_disp`)
- ChIP-seq normalization utilities (`normalize_chip_to_input`, `calc_norm_offsets_for_chip`)
- methylation/RRBS helpers (`read_bismark2dge`, `model_matrix_meth`)

## Installation

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

## Single-Cell Mixed Model

`edgePython` includes a NEBULA-LN-style single-cell workflow:

```python
# y_sc: genes x cells counts
# cell_meta: per-cell metadata DataFrame
# sample: column in cell_meta identifying subject/sample
fit_sc = ep.glm_sc_fit(y_sc, cell_meta=cell_meta, design="~ group", sample="subject_id")
fit_sc = ep.shrink_sc_disp(fit_sc)
res_sc = ep.glm_sc_test(fit_sc, coef=1)
top_sc = ep.top_tags(res_sc, n=20)
```

## ChIP-Seq and Methylation

ChIP-seq normalization to matched input controls:

```python
# input_counts and response are arrays (features x samples)
offsets = ep.calc_norm_offsets_for_chip(input_counts, response)
```

RRBS/methylation workflow helpers:

```python
dge_bs = ep.read_bismark2dge(files)
design_bs = ep.model_matrix_meth(dge_bs, design=design)
```

## Development

Run tests:

```bash
pytest -q
```

## Authorship

This code was written by Claude (Anthropic). The project was directed by Lior Pachter.

## edgeR

`edgePython` is based on the [edgeR](https://bioconductor.org/packages/edgeR/) Bioconductor package. The edgeR publications are:

- Robinson MD, McCarthy DJ, Smyth GK (2010). edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics*, 26(1), 139-140. [doi:10.1093/bioinformatics/btp616](https://doi.org/10.1093/bioinformatics/btp616)

- McCarthy DJ, Chen Y, Smyth GK (2012). Differential expression analysis of multifactor RNA-Seq experiments with respect to biological variation. *Nucleic Acids Research*, 40(10), 4288-4297. [doi:10.1093/nar/gks042](https://doi.org/10.1093/nar/gks042)

- Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential expression analysis of RNA-Seq experiments using Rsubread and the edgeR quasi-likelihood pipeline. *F1000Research*, 5, 1438. [doi:10.12688/f1000research.8987.2](https://doi.org/10.12688/f1000research.8987.2)

- Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful differential analysis of sequencing data with expanded functionality and improved support for small counts and larger datasets. *Nucleic Acids Research*, 53(2), gkaf018. [doi:10.1093/nar/gkaf018](https://doi.org/10.1093/nar/gkaf018)

The single-cell mixed model in `edgePython` is based on NEBULA:

- He L, Davila-Velderrain J, Sumida TS, Hafler DA, Kellis M, Kulminski AM (2021). NEBULA is a fast negative binomial mixed model for differential or co-expression analysis of large-scale multi-subject single-cell data. *Communications Biology*, 4, 629. [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)

## License

This project is licensed under the GNU General Public License v3.0.
