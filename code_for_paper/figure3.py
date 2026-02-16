"""Figure 3: Runtime comparison between R and Python.

Jitter plots with replicate timings for:
  (a) Bulk DE pipeline: R edgeR vs edgePython (HOXA1 dataset)
  (b) Single-cell DE: R NEBULA vs edgePython (jellyfish dataset)

Usage: python figure3.py
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

DATA = os.path.join(os.path.dirname(__file__), 'data', 'figure3')

# ── Load precomputed timings ──
df = pd.read_csv(f'{DATA}/benchmark_timings.csv')

bulk_r_times = df[(df['implementation'] == 'R') & (df['pipeline'] == 'bulk')]['time_seconds'].values
bulk_py_times = df[(df['implementation'] == 'Python') & (df['pipeline'] == 'bulk')]['time_seconds'].values
sc_r_times = df[(df['implementation'] == 'R') & (df['pipeline'] == 'SC')]['time_seconds'].values
sc_py_times = df[(df['implementation'] == 'Python') & (df['pipeline'] == 'SC')]['time_seconds'].values

print(f"Bulk:  R {bulk_r_times.mean():.2f} +/- {bulk_r_times.std():.2f}s  |  "
      f"Py {bulk_py_times.mean():.2f} +/- {bulk_py_times.std():.2f}s")
print(f"SC:    R {sc_r_times.mean():.1f} +/- {sc_r_times.std():.1f}s  |  "
      f"Py {sc_py_times.mean():.1f} +/- {sc_py_times.std():.1f}s")

# ── Make figure ──
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

COLOR_R = '#d65f5f'
COLOR_PY = '#4878cf'

def jitter_strip(ax, positions, data_list, colors, jitter_width=0.12):
    """Draw jitter strip plot with mean bars."""
    rng_j = np.random.RandomState(0)
    for pos, data, color in zip(positions, data_list, colors):
        jitter = rng_j.uniform(-jitter_width, jitter_width, size=len(data))
        ax.scatter(pos + jitter, data, c=color, s=25, alpha=0.7,
                   edgecolors='white', linewidths=0.5, zorder=3)
        mean_val = np.mean(data)
        ax.hlines(mean_val, pos - 0.2, pos + 0.2, colors=color,
                  linewidths=2.0, zorder=4)

# Panel (a): Bulk
ax = axes[0]
jitter_strip(ax, [0, 1],
             [bulk_r_times, bulk_py_times],
             [COLOR_R, COLOR_PY])
ax.set_xticks([0, 1])
ax.set_xticklabels(['R (edgeR)', 'Python\n(edgePython)'], fontsize=8)
ax.set_ylabel('Time (seconds)')
ax.set_title('Bulk RNA-seq (HOXA1)')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=7)

# Panel (b): Single-cell
ax = axes[1]
jitter_strip(ax, [0, 1],
             [sc_r_times, sc_py_times],
             [COLOR_R, COLOR_PY])
ax.set_xticks([0, 1])
ax.set_xticklabels(['R (NEBULA)', 'Python\n(edgePython)'], fontsize=8)
ax.set_ylabel('Time (seconds)')
ax.set_title(r'Single-cell (\textit{Clytia})')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=7)

# Panel labels
for ax, letter in zip(axes.flat, 'ab'):
    ax.text(-0.15, 1.10, r'\textbf{' + letter + '}',
            transform=ax.transAxes, fontsize=12, va='top')

plt.tight_layout(pad=1.0)
out = os.path.dirname(__file__)
plt.savefig(f'{out}/figure3.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{out}/figure3.png', dpi=300, bbox_inches='tight')
print("Saved figure3.pdf and figure3.png")
