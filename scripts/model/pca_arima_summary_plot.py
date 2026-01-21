import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PCA_FILE = os.path.join(ROOT, 'outputs', 'processed_data', 'summerOly_pca_features_long.csv')
DIAG_FILE = os.path.join(ROOT, 'outputs', 'processed_data', 'pca_arima_diagnostics_long.csv')
OUT_FIG = os.path.join(ROOT, 'outputs', 'figures', 'pca_arima_summary.png')

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

df = pd.read_csv(PCA_FILE)
diag = pd.read_csv(DIAG_FILE)

# find PC columns
pc_cols = [c for c in df.columns if c.startswith('Sport_PC_')]
years = sorted(df['Year'].unique())

# compute yearly mean series for each PC
pc_series = {}
for pc in pc_cols:
    s = df.groupby('Year')[pc].mean().reindex(years)
    pc_series[pc] = s.values

# diagnostics mapping
diag_map = diag.set_index('PC') if 'PC' in diag.columns else pd.DataFrame()

fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios':[3,1,1]})

# 1) overlay timeseries (all PCs)
ax = axes[0]
cm = plt.get_cmap('tab20')
for i, pc in enumerate(pc_cols):
    vals = pc_series[pc]
    ax.plot(years, vals, label=pc, color=cm(i % 20), alpha=0.9 if i<8 else 0.45, linewidth=1 if i>=8 else 1.5)
ax.set_title('Yearly mean of Sport PCs (1992-2024)')
ax.set_xlabel('Year')
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize='small', frameon=False)

# 2) Ljung-Box p-values bar
ax = axes[1]
lb_p = []
sh_p = []
pc_labels = []
for pc in pc_cols:
    pc_labels.append(pc)
    if pc in diag_map.index:
        lb_p.append(float(diag_map.loc[pc].get('LjungBox_p', np.nan)))
        sh_p.append(float(diag_map.loc[pc].get('Shapiro_p', np.nan)))
    else:
        lb_p.append(np.nan)
        sh_p.append(np.nan)

xi = np.arange(len(pc_labels))
ax.bar(xi, lb_p, color='C1')
ax.axhline(0.05, color='k', linestyle='--', linewidth=1)
ax.set_xticks(xi)
ax.set_xticklabels(pc_labels, rotation=90, fontsize='small')
ax.set_ylabel('Ljung-Box p')
ax.set_title('Ljung-Box p-values (per PC) — line at p=0.05')

# 3) Shapiro p-values
ax = axes[2]
ax.bar(xi, sh_p, color='C2')
ax.axhline(0.05, color='k', linestyle='--', linewidth=1)
ax.set_xticks(xi)
ax.set_xticklabels(pc_labels, rotation=90, fontsize='small')
ax.set_ylabel('Shapiro p')
ax.set_title('Shapiro p-values (residuals)')

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=150)
print('Saved combined summary figure to', OUT_FIG)
