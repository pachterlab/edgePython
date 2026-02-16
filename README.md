# edgePython

`edgePython` is a Python implementation of the Bioconductor `edgeR` package for differential analysis of genomics count data.
 
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

## License

This project is licensed under the GNU General Public License v3.0.
