"""Figure 2: Single-cell NB mixed model with empirical Bayes shrinkage.

Four panels:
  (a) NEBULA-LN QQ plot of z-statistics
  (b) BCV plot: per-gene MLE dispersion + EB prior trend (full jellyfish)
  (c) Volcano plot: fed vs starved DE (x-axis clipped)
  (d) Shrinkage demo on subsampled data (~30 cells)

Usage: python figure2.py
Dependencies: numpy, pandas, matplotlib (with LaTeX), scipy
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 9,
})

DATA = os.path.join(os.path.dirname(__file__), 'data', 'figure2')

BLUE = '#4878cf'
ORANGE = '#e8823a'
RED = '#c44e52'
GREY = '#aaaaaa'

# ── Load precomputed CSVs ──
qq = pd.read_csv(f'{DATA}/qq_zstats.csv')
bcv = pd.read_csv(f'{DATA}/bcv_full.csv')
de_results = pd.read_csv(f'{DATA}/de_results.csv', index_col=0)
shrink = pd.read_csv(f'{DATA}/shrinkage_sub30.csv')

# ── Make figure ──
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# --- Panel (a): QQ plot of z-statistics ---
ax = axes[0, 0]
theoretical = qq['theoretical'].values
observed = qq['observed'].values
ax.scatter(theoretical, observed, c=BLUE, s=2, alpha=0.3, edgecolors='none',
           rasterized=True)
lims = [min(theoretical.min(), observed.min()) * 1.05,
        max(theoretical.max(), observed.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Observed $z$-statistics')
ax.set_title('NEBULA-LN QQ plot')
ax.tick_params(labelsize=7)

# --- Panel (b): BCV plot with EB trend (full data) ---
ax = axes[0, 1]
abund_ok = bcv['ave_log_abundance'].values
bcv_mle = bcv['bcv_mle'].values
bcv_prior = bcv['bcv_prior'].values

ax.scatter(abund_ok, bcv_mle, c=BLUE, s=2, alpha=0.15, edgecolors='none',
           rasterized=True)
order = np.argsort(abund_ok)
ax.plot(abund_ok[order], bcv_prior[order], color=RED,
        lw=1.8, label='EB prior trend', zorder=5)
ax.set_yscale('log')
ax.set_xlabel(r'Average $\log_2$ abundance')
ax.set_ylabel('BCV')
ax.set_title('Cell-level dispersion (full data)')
ax.tick_params(labelsize=7)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE,
           markersize=4, alpha=0.5, label='Per-gene MLE'),
    Line2D([0], [0], color=RED, lw=1.8, label='EB prior trend'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper right', framealpha=0.9)

# --- Panel (c): Volcano plot (clipped x-axis) ---
ax = axes[1, 0]
logfc = de_results['logFC'].values
logp = -np.log10(de_results['PValue'].values + 1e-300)
fdr = de_results['FDR'].values
sig = fdr < 0.05

# Clip x-axis based on data range (exclude extreme outliers)
fc_lim = max(np.percentile(np.abs(logfc), 99.5), 3.0)
fc_lim = min(fc_lim, 5.0)  # cap at 5

ax.scatter(logfc[~sig], logp[~sig], c=GREY, s=2, alpha=0.3,
           edgecolors='none', rasterized=True)
ax.scatter(logfc[sig], logp[sig], c=RED, s=3, alpha=0.5,
           edgecolors='none', rasterized=True)

# Label top genes
top_idx = de_results.head(5).index
for idx in top_idx:
    row = de_results.loc[idx]
    gene = row.get('gene', str(idx))
    if len(gene) > 15:
        gene = gene[:12] + '...'
    fc_val = row['logFC']
    pval = -np.log10(row['PValue'] + 1e-300)
    # Clamp annotation to visible range
    fc_plot = np.clip(fc_val, -fc_lim * 0.95, fc_lim * 0.95)
    ax.annotate(gene, (fc_plot, pval),
                fontsize=6, alpha=0.8, ha='center', va='bottom')

ax.set_xlim(-fc_lim, fc_lim)
ax.set_xlabel(r'$\log_2$ fold change (fed / starved)')
ax.set_ylabel(r'$-\log_{10}(p)$')
ax.set_title('Fed vs starved DE')
ax.axhline(-np.log10(0.05), color='k', ls='--', lw=0.5, alpha=0.3)
ax.tick_params(labelsize=7)

# --- Panel (d): Shrinkage on subsampled data ---
ax = axes[1, 1]
abund_s = shrink['ave_log_abundance'].values
bcv_mle_s = shrink['bcv_mle'].values
bcv_post_s = shrink['bcv_post'].values
bcv_prior_s = shrink['bcv_prior'].values

BLUE_DARK = '#1a3670'
ORANGE_BRIGHT = '#ff8c00'

ax.scatter(abund_s, bcv_mle_s, c=BLUE_DARK, s=6, alpha=0.3,
           edgecolors='none', rasterized=True, zorder=2)
ax.scatter(abund_s, bcv_post_s, c=ORANGE_BRIGHT, s=6, alpha=0.3,
           edgecolors='none', rasterized=True, zorder=3)

order_s = np.argsort(abund_s)
ax.plot(abund_s[order_s], bcv_prior_s[order_s], color=RED,
        lw=2.0, zorder=5)

ax.set_yscale('log')
ax.set_xlabel(r'Average $\log_2$ abundance')
ax.set_ylabel('BCV')
ax.set_title('Shrinkage effect (30 cells)')
ax.tick_params(labelsize=7)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE_DARK,
           markersize=5, alpha=0.6, label='Raw MLE'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ORANGE_BRIGHT,
           markersize=5, alpha=0.6, label='EB posterior'),
    Line2D([0], [0], color=RED, lw=2.0, label='Prior trend'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper right', framealpha=0.9)

# Panel labels
for ax, letter in zip(axes.flat, 'abcd'):
    ax.text(-0.15, 1.08, r'\textbf{' + letter + '}',
            transform=ax.transAxes, fontsize=12, va='top')

plt.tight_layout(pad=1.0)
out = os.path.dirname(__file__)
plt.savefig(f'{out}/figure2.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{out}/figure2.png', dpi=300, bbox_inches='tight')
print("Saved figure2.pdf and figure2.png")
