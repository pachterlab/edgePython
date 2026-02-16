"""Figure 1: 16-panel edgePython validation against R edgeR (4x4 grid).

Row 1 (a-d): Core pipeline — normalization, dispersion, effect sizes, shrinkage
Row 2 (e-h): Testing frameworks — exact, QL F-test, LRT TREAT, Galaxy QL
Row 3 (i-l): Multi-factor & gene sets — TREAT, GLM coefficients, interaction, camera
Row 4 (m-p): Gene sets & specialized — fry, DTU gene, DTU exon, scaled analysis

Usage: python figure1.py
Dependencies: numpy, pandas, matplotlib (with LaTeX)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 9,
})

DATA = os.path.join(os.path.dirname(__file__), 'data', 'figure1')

BLUE = '#4878cf'
scatter_kw = dict(s=4, alpha=0.4, edgecolors='none', rasterized=True)

fig, axes = plt.subplots(4, 4, figsize=(14, 12.5))

# ═══════════════════════════════════════════════════════════════════
# ROW 1: Core pipeline (HOXA1)
# ═══════════════════════════════════════════════════════════════════

# ── Panel (a): TMM normalization factors ──
ax = axes[0, 0]
r_nf = pd.read_csv(f'{DATA}/R_norm_factors.csv')
p_nf = pd.read_csv(f'{DATA}/Python_norm_factors.csv')
m = r_nf.merge(p_nf, on='sample', suffixes=('_R', '_Py'))
x, y = m['norm.factors_R'].values, m['norm.factors_Py'].values
ax.scatter(x, y, c=BLUE, s=50, alpha=0.8, edgecolors='none', zorder=5)
lims = [min(x.min(), y.min()) * 0.98, max(x.max(), y.max()) * 1.02]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title('TMM norm factors')
ax.tick_params(labelsize=7)

# ── Panel (b): Tagwise BCV (HOXA1) ──
ax = axes[0, 1]
r_d = pd.read_csv(f'{DATA}/R_dispersions.csv', index_col=0)
p_d = pd.read_csv(f'{DATA}/Python_dispersions.csv')
p_d = p_d.set_index('transcript')
common = r_d.index.intersection(p_d.index)
x = np.sqrt(r_d.loc[common, 'tagwise.dispersion'].values)
y = np.sqrt(p_d.loc[common, 'tagwise.dispersion'].values)
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title('Tagwise BCV')
ax.tick_params(labelsize=7)

# ── Panel (c): QL F-test logFC (HOXA1) ──
ax = axes[0, 2]
r_ql = pd.read_csv(f'{DATA}/R_qlf_all_results.csv', index_col=0)
p_ql = pd.read_csv(f'{DATA}/Python_qlf_all_results.csv', index_col=0)
r_ql.index = r_ql.index.astype(str)
p_ql.index = p_ql.index.astype(str)
common = r_ql.index.intersection(p_ql.index)
x = r_ql.loc[common, 'logFC'].values
y = p_ql.loc[common, 'logFC'].values
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [min(x.min(), y.min()) * 1.05, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title('QL F-test logFC')
ax.tick_params(labelsize=7)

# ── Panel (d): QL posterior variance s2.post (HOXA1) ──
ax = axes[0, 3]
r_fit = pd.read_csv(f'{DATA}/R_ql_fit_details.csv', index_col=0)
p_fit = pd.read_csv(f'{DATA}/Python_ql_fit_details.csv', index_col=0)
r_fit.index = r_fit.index.astype(str)
p_fit.index = p_fit.index.astype(str)
common = r_fit.index.intersection(p_fit.index)
x = r_fit.loc[common, 's2.post'].values
y = p_fit.loc[common, 's2.post'].values
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'QL posterior $s^2_{\mathrm{post}}$')
ax.tick_params(labelsize=7)

# ═══════════════════════════════════════════════════════════════════
# ROW 2: Testing frameworks
# ═══════════════════════════════════════════════════════════════════

# ── Panel (e): Exact test p-values (HOXA1) ──
ax = axes[1, 0]
r_et = pd.read_csv(f'{DATA}/R_exact_all_results.csv', index_col=0)
p_et = pd.read_csv(f'{DATA}/Python_exact_all_results.csv', index_col=0)
r_et.index = r_et.index.astype(str)
p_et.index = p_et.index.astype(str)
common = r_et.index.intersection(p_et.index)
rp = r_et.loc[common, 'PValue'].values
pp = p_et.loc[common, 'PValue'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'Exact test $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel (f): QL F-test p-values (HOXA1) ──
ax = axes[1, 1]
r_ql = pd.read_csv(f'{DATA}/R_qlf_all_results.csv', index_col=0)
p_ql = pd.read_csv(f'{DATA}/Python_qlf_all_results.csv', index_col=0)
r_ql.index = r_ql.index.astype(str)
p_ql.index = p_ql.index.astype(str)
common = r_ql.index.intersection(p_ql.index)
rp = r_ql.loc[common, 'PValue'].values
pp = p_ql.loc[common, 'PValue'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'QL F-test $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel (g): LRT TREAT p-values (pooled across 4 thresholds) ──
ax = axes[1, 2]
lrt_R, lrt_Py = [], []
for suffix in ['lfc10', 'lfc12', 'lfc15', 'worst']:
    r = pd.read_csv(f'{DATA}/R_treat_lrt_{suffix}.csv', index_col=0)
    p = pd.read_csv(f'{DATA}/Py_treat_lrt_{suffix}.csv', index_col=0)
    lrt_R.extend(r['PValue'].values)
    lrt_Py.extend(p['PValue'].values)
lrt_R, lrt_Py = np.array(lrt_R), np.array(lrt_Py)
mask = (lrt_R > 0) & (lrt_Py > 0)
x, y = -np.log10(lrt_R[mask]), -np.log10(lrt_Py[mask])
ax.scatter(x, y, c=BLUE, s=10, alpha=0.6, edgecolors='none', rasterized=True)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'LRT TREAT $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel (h): QL F-test p-values (Galaxy) ──
ax = axes[1, 3]
r_gal = pd.read_csv(f'{DATA}/R_basal_preg_vs_lact.csv', index_col=0)
p_gal = pd.read_csv(f'{DATA}/Py_basal_preg_vs_lact.csv')
if 'EntrezGeneID' in p_gal.columns:
    p_gal = p_gal.set_index('EntrezGeneID')
r_gal.index = r_gal.index.astype(str)
p_gal.index = p_gal.index.astype(str)
common = r_gal.index.intersection(p_gal.index)
rp = r_gal.loc[common, 'PValue'].values
pp = p_gal.loc[common, 'PValue'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'QL F-test $-\log_{10}(p)$ (Galaxy)')
ax.tick_params(labelsize=7)

# ═══════════════════════════════════════════════════════════════════
# ROW 3: Multi-factor design & gene sets
# ═══════════════════════════════════════════════════════════════════

# ── Panel (i): TREAT p-values (Galaxy) ──
ax = axes[2, 0]
r_tr = pd.read_csv(f'{DATA}/R_TREAT_basal.csv', index_col=0)
p_tr = pd.read_csv(f'{DATA}/Py_TREAT_basal.csv')
if 'EntrezGeneID' in p_tr.columns:
    p_tr = p_tr.set_index('EntrezGeneID')
r_tr.index = r_tr.index.astype(str)
p_tr.index = p_tr.index.astype(str)
common = r_tr.index.intersection(p_tr.index)
rp = r_tr.loc[common, 'PValue'].values
pp = p_tr.loc[common, 'PValue'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'TREAT $-\log_{10}(p)$ (Galaxy)')
ax.tick_params(labelsize=7)

# ── Panel (j): GLM coefficients (Galaxy, all 6 columns pooled) ──
ax = axes[2, 1]
r_coef = pd.read_csv(f'{DATA}/R_coefficients.csv', index_col=0)
p_coef = pd.read_csv(f'{DATA}/Py_coefficients.csv', header=None)
x = r_coef.values.ravel()
y = p_coef.values.ravel()
ax.scatter(x, y, c=BLUE, s=2, alpha=0.15, edgecolors='none', rasterized=True)
lims = [min(x.min(), y.min()) * 1.05, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title('GLM coefficients')
ax.tick_params(labelsize=7)

# ── Panel (k): Multi-df interaction F-test (Galaxy) ──
ax = axes[2, 2]
r_int = pd.read_csv(f'{DATA}/R_QLFtest_Interaction.csv', index_col=0)
p_int = pd.read_csv(f'{DATA}/Py_QLFtest_Interaction.csv')
if 'EntrezGeneID' in p_int.columns:
    p_int = p_int.set_index('EntrezGeneID')
r_int.index = r_int.index.astype(str)
p_int.index = p_int.index.astype(str)
common = r_int.index.intersection(p_int.index)
rp = r_int.loc[common, 'PValue'].values
pp = p_int.loc[common, 'PValue'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'Interaction $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel (l): camera gene set p-values ──
ax = axes[2, 3]
cam_R, cam_Py = [], []
for suffix in ['default', 'logcpm', 'ranks', 'cor0', 'cor05', 'allowneg']:
    r = pd.read_csv(f'{DATA}/R_camera_{suffix}.csv', index_col=0)
    p = pd.read_csv(f'{DATA}/Py_camera_{suffix}.csv', index_col=0)
    cam_R.extend(r['PValue'].values)
    cam_Py.extend(p['PValue'].values)
cam_R, cam_Py = np.array(cam_R), np.array(cam_Py)
mask = (cam_R > 0) & (cam_Py > 0)
x, y = -np.log10(cam_R[mask]), -np.log10(cam_Py[mask])
ax.scatter(x, y, c=BLUE, s=20, alpha=0.7, edgecolors='none')
lims = [0, max(x.max(), y.max()) * 1.1]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'camera $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ═══════════════════════════════════════════════════════════════════
# ROW 4: Gene sets & specialized analyses
# ═══════════════════════════════════════════════════════════════════

# ── Panel (m): fry gene set p-values ──
ax = axes[3, 0]
r_fry = pd.read_csv(f'{DATA}/R_fry.csv', index_col=0)
p_fry = pd.read_csv(f'{DATA}/Py_fry.csv', index_col=0)
rp_list, pp_list = [], []
for col in ['PValue', 'PValue.Mixed']:
    if col in r_fry.columns and col in p_fry.columns:
        rp_list.extend(r_fry[col].values)
        pp_list.extend(p_fry[col].values)
rp, pp = np.array(rp_list), np.array(pp_list)
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, s=20, alpha=0.7, edgecolors='none')
lims = [0, max(x.max(), y.max()) * 1.1]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'fry $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel (n): DTU gene-level p-values ──
ax = axes[3, 1]
r_dtu = pd.read_csv(f'{DATA}/R_dtu_gene_results.csv', index_col=0)
p_dtu = pd.read_csv(f'{DATA}/Python_dtu_gene_results.csv', index_col=0)
r_dtu.index = r_dtu.index.astype(str)
p_dtu.index = p_dtu.index.astype(str)
common = r_dtu.index.intersection(p_dtu.index)
rp = r_dtu.loc[common, 'gene.Simes.p.value'].values
pp = p_dtu.loc[common, 'gene.Simes.p.value'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'DTU gene $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel (o): DTU exon-level p-values ──
ax = axes[3, 2]
r_dtu_ex = pd.read_csv(f'{DATA}/R_dtu_exon_results.csv')
p_dtu_ex = pd.read_csv(f'{DATA}/Python_dtu_exon_results.csv')
rp = r_dtu_ex['exon.p.value'].values
pp = p_dtu_ex['exon.p.value'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'DTU exon $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel (p): Scaled analysis QL p-values ──
ax = axes[3, 3]
r_sc = pd.read_csv(f'{DATA}/R_scaled_qlf_all_results.csv', index_col=0)
p_sc = pd.read_csv(f'{DATA}/Python_scaled_qlf_all_results.csv', index_col=0)
r_sc.index = r_sc.index.astype(str)
p_sc.index = p_sc.index.astype(str)
common = r_sc.index.intersection(p_sc.index)
rp = r_sc.loc[common, 'PValue'].values
pp = p_sc.loc[common, 'PValue'].values
mask = (rp > 0) & (pp > 0)
x, y = -np.log10(rp[mask]), -np.log10(pp[mask])
ax.scatter(x, y, c=BLUE, **scatter_kw)
lims = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('edgeR (R)'); ax.set_ylabel('edgePython')
ax.set_title(r'Scaled QL $-\log_{10}(p)$')
ax.tick_params(labelsize=7)

# ── Panel labels ──
for ax, letter in zip(axes.flat, 'abcdefghijklmnop'):
    ax.text(-0.18, 1.10, r'\textbf{' + letter + '}',
            transform=ax.transAxes, fontsize=12, va='top')

plt.tight_layout(pad=1.0)
out = os.path.dirname(__file__)
plt.savefig(f'{out}/figure1.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{out}/figure1.png', dpi=300, bbox_inches='tight')
print("Saved figure1.pdf and figure1.png")
