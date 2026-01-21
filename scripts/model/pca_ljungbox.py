import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PCA_FN = os.path.join(ROOT, 'outputs', 'processed_data', 'summerOly_pca_features.csv')
OUT_DIR = os.path.join(ROOT, 'outputs', 'figures')
OUT_DATA = os.path.join(ROOT, 'outputs', 'processed_data', 'pca_ljungbox_results.csv')

os.makedirs(OUT_DIR, exist_ok=True)

print('Reading', PCA_FN)
df = pd.read_csv(PCA_FN)
# identify PC columns
pc_cols = [c for c in df.columns if c.startswith('Sport_PC_')]
if not pc_cols:
    raise SystemExit('No PC columns found')

results = []
for pc in pc_cols:
    ts = df.groupby('Year')[pc].mean().sort_index()
    ts = ts.dropna()
    n = len(ts)
    if n < 6:
        print(f'Skipping {pc}: too short ({n} points)')
        continue
    max_lag = min(10, n-2)
    lags = list(range(1, max_lag+1))
    try:
        lb = acorr_ljungbox(ts, lags=lags, return_df=True)
    except Exception as e:
        print(f'Failed Ljung-Box for {pc}:', e)
        continue
    lb = lb.reset_index().rename(columns={'index': 'lag', 'lb_stat': 'LjungBox_stat', 'lb_pvalue': 'pvalue'})
    lb['PC'] = pc
    results.append(lb[['PC','lag','LjungBox_stat','pvalue']])

    # plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    # timeseries
    axs[0].plot(ts.index, ts.values, marker='o')
    axs[0].set_title(f'{pc} - Yearly mean time series')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('PC value')
    # ACF
    plot_acf(ts.values, lags=max_lag, ax=axs[1], alpha=0.05)
    axs[1].set_title('ACF')
    # p-values
    axs[2].plot(lb['lag'], lb['pvalue'], marker='o')
    axs[2].axhline(0.05, color='red', linestyle='--', label='alpha=0.05')
    axs[2].set_xlabel('lag')
    axs[2].set_ylabel('p-value')
    axs[2].set_title('Ljung-Box p-values')
    axs[2].legend()
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f'pca_ljungbox_{pc}.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print('Saved', out_png)

if results:
    allres = pd.concat(results, ignore_index=True)
    allres.to_csv(OUT_DATA, index=False)
    print('Saved summary results to', OUT_DATA)
else:
    print('No results produced')
