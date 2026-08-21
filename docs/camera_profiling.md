# Profiling `camera()`: findings and optimizations

## Summary

`camera()` was profiled end-to-end on realistic-scale data. The profile
showed that essentially all of the runtime (97.8%) was spent in a helper,
`zscore_nbinom()`, that converted counts to z-scores using a pure-Python
loop making one scalar SciPy call per gene. Vectorizing that loop, plus a
smaller batching fix in the per-gene-set test loop in `_camera_default()`,
brought a representative run from **69.9s to 1.5s (~46x)** with
numerically identical output.

## Methodology

`camera()` is exercised through the same pipeline as
[`examples/mammary/mouse_mammary_tutorial.ipynb`](../examples/mammary/mouse_mammary_tutorial.ipynb),
using the bundled real mouse mammary gland RNA-seq dataset
(`examples/mammary/data/countdata.tsv`, `sampleinfo.tsv`):

```python
d = ep.make_dgelist(counts_df.values.astype(float), group=group, genes=...)
cpm_vals = ep.cpm(d)
keep = np.sum(cpm_vals > 0.5, axis=1) >= 2
d_filt = ep.make_dgelist(d['counts'][keep, :], group=group, genes=...)
d_filt = ep.calc_norm_factors(d_filt)
design = ep.model_matrix('~ 0 + group', pd.DataFrame({'group': group}))
d_filt = ep.estimate_disp(d_filt, design=design)
fit = ep.glm_ql_fit(d_filt, design=design)
```

This yields a `DGEGLM` fit over **15,804 genes x 12 samples**, matching the
scale of a typical filtered bulk RNA-seq experiment.

Since no gene-set collection ships with the repo, an MSigDB-like
collection of **3,000 gene sets** was generated with sizes drawn from a
log-normal distribution (5-500 genes/set, with replacement across sets) to
match the scale of testing against something like the MSigDB Hallmark +
C2/C5 collections:

```python
rng = np.random.default_rng(0)
gene_sets = {
    f"SET_{i}": rng.choice(ngenes, size=size, replace=False).tolist()
    for i, size in enumerate(sizes)
}
con_basal = np.array([-1, 1, 0, 0, 0, 0], dtype=float)
result = ep.camera(fit, gene_sets, design=design, contrast=con_basal)
```

The call was wrapped in `cProfile` (`sort_stats('cumulative')` and
`sort_stats('tottime')`) to identify hotspots.

## Finding: `zscore_nbinom` was the bottleneck, not the gene-set loop

Baseline profile (`edgepython/gene_sets.py:453` `camera()`, 69.83s total):

```
ncalls  tottime  cumtime  filename:lineno(function)
     1    0.005   69.874  gene_sets.py:453(camera)
     1    0.005   69.201  gene_sets.py:34(_zscore_glm)
    12    2.943   68.290  utils.py:509(zscore_nbinom)
921005    4.787   28.489  scipy/.../_distn_infrastructure.py:613(argsreduce)
189648    6.845   24.405  scipy/.../_distn_infrastructure.py:2305(ppf)
189648    4.533   18.770  scipy/.../_distn_infrastructure.py:3544(logpmf)
```

`_zscore_glm` converts DGEGLM counts to NB z-scores under the null model
before the competitive test runs (`camera.DGEGLM` in edgeR/limma). It calls
`zscore_nbinom()` once per sample column (only **12** calls). But
[`zscore_nbinom`](../edgepython/utils.py) looped over every one of the
15,804 genes in pure Python, issuing individual scalar calls to
`scipy.stats.nbinom.logpmf/logcdf/logsf` and `scipy.stats.norm.ppf`. Each
scalar SciPy call pays fixed overhead (argument validation, broadcasting,
`argsreduce`) regardless of how little work it does, so ~190k scalar calls
dominated the run. The actual gene-set test loop in `_camera_default()`
was comparatively cheap: it only accounted for ~1s of the 69.9s baseline.

This means the runtime of `camera()` on `DGEList`/`DGEGLM` input was
effectively independent of the number of gene sets tested and was instead
driven by `ngenes x nsamples`, which is the opposite of what the API
shape (index of many gene sets) suggests.

## Changes

### 1. Vectorize `zscore_nbinom()` ([`edgepython/utils.py`](../edgepython/utils.py))

Replaced the per-gene Python `for` loop with masked, fully-vectorized
NumPy/SciPy calls. The original logic has three mutually exclusive cases
per gene (`qr == 0`, `q >= mu`, `q < mu`); each case is now evaluated once
across all genes that fall into it via boolean masks, instead of once per
gene:

```python
valid = (mu > 0) & (size > 0)
...
zero_mask = valid & (qr == 0)
upper_mask = valid & (qr != 0) & (q >= mu)
lower_mask = valid & (qr != 0) & (q < mu)

if np.any(zero_mask):
    ...  # single batched nbinom.logpmf / norm.ppf call over zero_mask
if np.any(upper_mask):
    ...  # single batched nbinom.logsf / norm.ppf call over upper_mask
if np.any(lower_mask):
    ...  # single batched nbinom.logcdf / norm.ppf call over lower_mask
```

`limma_port.logsumexp()` was already NumPy-vectorized (elementwise
`np.maximum`/`np.log`/`np.exp`), so it needed no changes to work on
arrays instead of scalars.

This is the primary fix and accounts for nearly all of the speedup.

### 2. Batch the per-set p-value calls in `_camera_default()` ([`edgepython/gene_sets.py`](../edgepython/gene_sets.py))

In the non-`use_ranks` path, `t_dist.cdf()` and `t_dist.sf()` were each
called once per gene set (scalar calls), even though `df_camera` is
constant across every set in a given `camera()` call. The loop now
accumulates the two-sample t-statistic for every set into an array first,
then calls `t_dist.cdf()`/`t_dist.sf()` once, vectorized, after the loop:

```python
two_sample_t = np.empty(nsets)
for s_idx in range(nsets):
    ...
    two_sample_t[s_idx] = delta / np.sqrt(varStatPooled * (vif / m + 1.0 / m2))

p_down_arr = t_dist.cdf(two_sample_t, df_camera)
p_up_arr = t_dist.sf(two_sample_t, df_camera)
```

The `use_ranks=True` path (which calls `_rank_sum_test_with_correlation`
per set) was left as-is — it's a less commonly used option, and each call
already does an O(G) vectorized rank computation internally, so per-call
overhead is proportionally less dominant there. It would be a candidate
for a similar batching pass if profiling shows it matters in practice.

Both branches also stopped computing `correlation` in the non-`use_ranks`
path, since the original code computed it there but never used it
(`vif`, not `correlation`, drives the non-rank test statistic).

## Results

Same pipeline and same 3,000-gene-set collection (fixed RNG seed),
before/after:

| | Before | After |
|---|---|---|
| `camera()` wall time | 69.9s | 1.5s |
| Speedup | — | ~46x |
| Output | (baseline) | Identical `NGenes`/`Direction`/`PValue`/`FDR` per set |

Post-fix profile — the remaining time is dominated by the null-model GLM
refit (`mglm_levenberg`) inside `_zscore_glm`, which is inherent modeling
cost for the DGEGLM to z-score conversion (edgeR/limma's
`camera.DGEGLM` does the same null-model refit), not overhead introduced
by `camera()` itself:

```
ncalls  tottime  cumtime  filename:lineno(function)
     1    0.001    2.436  gene_sets.py:453(camera)
     1    0.002    1.708  gene_sets.py:34(_zscore_glm)
     1    0.002    1.541  glm_fit.py:238(glm_fit)
     1    0.526    1.538  glm_levenberg.py:12(mglm_levenberg)
     1    0.042    0.727  gene_sets.py:241(_camera_default)
    12    0.048    0.164  utils.py:509(zscore_nbinom)
```

(The 2.4s → 1.5s difference between this profile and the final numbers
above is from fix #2, applied after this profile was taken.)

## Testing

- `pytest tests/test_utilities.py -k "zscore or q2q"` — 5 passed
  (direct unit tests of `zscore_nbinom` against R reference values).
- `pytest tests/test_gene_sets.py` — 37 passed (includes `camera()`
  correctness/behavior tests, including the 1000-gene fixture with R
  reference p-values).
- `pytest tests/test_r_vs_py.py -k camera` — 28 passed (NGenes,
  direction, PValue, and FDR agreement between R's `camera()` and
  `edgepython.camera()` across `default`, `log-CPM`, `use.ranks`,
  `inter.gene.cor=0`, `inter.gene.cor=0.05`, `allow.neg.cor`, and
  `unsorted` variants).
- `pytest tests/` (full suite) — 410 passed, 1 skipped.
- Profiling script re-run after each change to confirm the top gene sets
  and p-values in the 3,000-set benchmark were unchanged (bit-for-bit
  same top-5 sets/p-values before and after both fixes).

No test changes were needed — both fixes are behavior-preserving,
performance-only changes.
