import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PCA_FN = os.path.join(ROOT, 'outputs', 'processed_data', 'summerOly_pca_features.csv')
OUT_DIR = os.path.join(ROOT, 'outputs', 'figures')
OUT_SUM = os.path.join(ROOT, 'outputs', 'processed_data', 'pca_residuals_summary.csv')
os.makedirs(OUT_DIR, exist_ok=True)

print('Reading', PCA_FN)
df = pd.read_csv(PCA_FN)
pc_cols = [c for c in df.columns if c.startswith('Sport_PC_')]
if not pc_cols:
    raise SystemExit('No PC columns found')

# compute yearly mean series for each PC
pc_ts = {}
years = sorted(df['Year'].unique())
for pc in pc_cols:
    ts = df.groupby('Year')[pc].mean().reindex(years)
    pc_ts[pc] = ts

# overlay plot of all PCs
plt.figure(figsize=(12,6))
for pc,ts in pc_ts.items():
    plt.plot(ts.index, ts.values, label=pc, alpha=0.7)
plt.legend(ncol=3, fontsize='small')
plt.xlabel('Year')
plt.title('Yearly mean of all Sport PCs')
plt.tight_layout()
all_png = os.path.join(OUT_DIR, 'pca_all_pcs_timeseries.png')
plt.savefig(all_png, dpi=150)
plt.close()
print('Saved overlay timeseries:', all_png)

summary_rows = []
for pc, ts in pc_ts.items():
    ts = ts.dropna()
    n = len(ts)
    if n < 8:
        print(f'Skipping {pc}: too short ({n})')
        continue
    # fit ARIMA(1,0,0)
    try:
        model = ARIMA(ts.values, order=(1,0,0)).fit()
        resid = model.resid
    except Exception as e:
        print('ARIMA failed for', pc, e)
        continue
    # ACF/PACF and plots + Ljung-Box and Shapiro
    max_lag = min(10, n-2)
    lb = acorr_ljungbox(resid, lags=[max_lag], return_df=True)
    lb_p = float(lb['lb_pvalue'].iloc[0])
    lb_stat = float(lb['lb_stat'].iloc[0])
    try:
        sh_w_p = float(shapiro(resid)[1])
    except Exception:
        sh_w_p = np.nan
    summary_rows.append({'PC': pc, 'n': n, 'AR1_coef': float(model.params[1]) if len(model.params)>1 else np.nan,
                         'sigma2': float(model.sigma2), 'LjungBox_stat': lb_stat, 'LjungBox_p': lb_p, 'Shapiro_p': sh_w_p})

    # diagnostic figure: timeseries, resid, ACF, PACF, hist+QQ
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0,0].plot(ts.index, ts.values, marker='o')
    axs[0,0].set_title(f'{pc} series')
    axs[0,1].plot(ts.index, resid, marker='o')
    axs[0,1].axhline(0, color='k', linestyle='--')
    axs[0,1].set_title('Residuals (ARIMA(1,0,0))')
    plot_acf(resid, lags=max_lag, ax=axs[1,0])
    axs[1,0].set_title('ACF of residuals')
    plot_pacf(resid, lags=max_lag, ax=axs[1,1], method='ywm')
    axs[1,1].set_title('PACF of residuals')
    # hist + qq
    axs[2,0].hist(resid, bins=15)
    axs[2,0].set_title(f'Histogram (Shapiro p={sh_w_p:.3f})')
    # QQ plot
    import statsmodels.api as sm
    sm.qqplot(resid, line='s', ax=axs[2,1])
    axs[2,1].set_title('QQ plot')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f'pca_resid_diag_{pc}.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print('Saved', out_png)

if summary_rows:
    pd.DataFrame(summary_rows).to_csv(OUT_SUM, index=False)
    print('Saved summary to', OUT_SUM)
else:
    print('No diagnostics produced')
