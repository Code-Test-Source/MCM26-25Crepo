import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import Holt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IN_FILE = os.path.join(ROOT, 'outputs', 'processed_data', 'summerOly_athletes_cleaned_1992_2024.csv')
# use the curated mapping that contains Full_Name -> Current_NOC
MAP_FILE = os.path.join(ROOT, 'outputs', 'processed_data', 'country_name_mapping3.csv')
OUT_DIR = os.path.join(ROOT, 'outputs', 'processed_data')
FIG_DIR = os.path.join(ROOT, 'outputs', 'figures')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print('Loading', IN_FILE)
df = pd.read_csv(IN_FILE)
# build mapping dict early so it's available to all code paths
map_dict = {}
if os.path.exists(MAP_FILE):
    try:
        mapdf = pd.read_csv(MAP_FILE)
        for _, r in mapdf.iterrows():
            full = r.get('Full_Name') or r.get('Full name') or ''
            curr = r.get('Current_NOC') or r.get('Current_NOC')
            if pd.notna(full) and pd.notna(curr):
                map_dict[str(full).strip()] = str(curr).strip()
    except Exception as _:
        map_dict = {}
# Ensure mapping of historical countries -> Mapped_NOC
if 'Mapped_NOC' not in df.columns:
    if map_dict:
        df['country_name'] = df.get('country_name', df.get('NOC',''))
        df['Mapped_NOC'] = df['country_name'].map(lambda x: map_dict.get(str(x).strip(), None))
        df.loc[df['Mapped_NOC'].isna(), 'Mapped_NOC'] = df.loc[df['Mapped_NOC'].isna(), 'NOC']
    else:
        df['Mapped_NOC'] = df['NOC']

# ensure is_host flag using hosts file
hosts_fn = os.path.join(ROOT, '2025_Problem_C_Data', 'summerOly_hosts.csv')
if os.path.exists(hosts_fn):
    hosts_df = pd.read_csv(hosts_fn)
    # normalize host country string (take last token after comma)
    def host_country(host_str):
        try:
            parts = str(host_str).split(',')
            return parts[-1].strip()
        except Exception:
            return str(host_str).strip()
    hosts_df['host_country'] = hosts_df['Host'].astype(str).apply(host_country)
    # build mapping from year -> NOC using mapping file (Full_Name)
    host_year_noc = {}
    if os.path.exists(MAP_FILE):
        for _, r in hosts_df.iterrows():
            try:
                yr = int(r['Year'])
            except Exception:
                continue
            c = r['host_country']
            noc = map_dict.get(c)
            if noc is None:
                # try uppercase variants
                noc = map_dict.get(c.title()) or map_dict.get(c.upper())
            host_year_noc[yr] = noc
    else:
        for _, r in hosts_df.iterrows():
            try:
                yr = int(r['Year'])
            except Exception:
                continue
            host_year_noc[yr] = r['host_country']
    # attach is_host=1 where Year matches and Mapped_NOC matches host noc
    df['is_host'] = 0
    for yr, noc in host_year_noc.items():
        if noc is None:
            continue
        df.loc[(df['Year']==yr) & (df['Mapped_NOC']==noc), 'is_host'] = 1
else:
    df['is_host'] = df.get('is_host', 0)

# filter years 1992-2024
df = df[(df['Year']>=1992) & (df['Year']<=2024)].copy()
print('Filtered years:', df['Year'].min(), df['Year'].max(), 'rows:', len(df))

# create is_host per (Year,Mapped_NOC) if exists
if 'is_host' not in df.columns:
    df['is_host'] = 0
host_map = df.groupby(['Year','Mapped_NOC'])['is_host'].max()

# pivot sports
pivot = df.pivot_table(index=['Year','Mapped_NOC'], columns='Sport', values='Total', aggfunc='sum', fill_value=0)
print('Pivot shape:', pivot.shape)

# restore is_host index
pivot = pivot.reset_index()
pivot['is_host'] = pivot.apply(lambda r: host_map.get((r['Year'], r['Mapped_NOC']), 0), axis=1)

# Standardize sport columns
sport_cols = [c for c in pivot.columns if c not in ['Year','Mapped_NOC','is_host']]
X = pivot[sport_cols].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# PCA retain 85% variance
pca = PCA(n_components=0.85)
Xp = pca.fit_transform(Xs)
pc_cols = [f'Sport_PC_{i+1}' for i in range(Xp.shape[1])]

pca_df = pd.DataFrame(Xp, columns=pc_cols)
out_df = pd.concat([pivot[['Year','Mapped_NOC','is_host']].reset_index(drop=True), pca_df], axis=1)
OUT_PCA = os.path.join(OUT_DIR, 'summerOly_pca_features_long.csv')
out_df.to_csv(OUT_PCA, index=False)
print('Saved PCA features to', OUT_PCA)

# save loadings
loadings = pd.DataFrame(pca.components_.T, index=sport_cols, columns=pc_cols)
OUT_LOAD = os.path.join(OUT_DIR, 'pca_sport_loadings_long.csv')
loadings.to_csv(OUT_LOAD)
print('Saved PCA loadings to', OUT_LOAD)

# ARIMA diagnostics on yearly mean PC series
years = list(range(1992,2025))
# thresholds
MIN_ARIMA_LEN = 15
MIN_HOLT_LEN = 8
summary_rows = []
for pc in pc_cols:
    series = out_df.groupby('Year')[pc].mean().reindex(years)
    series_na = series.dropna()
    n = len(series_na)
    if n < MIN_HOLT_LEN:
        print(f'Skipping {pc}: too short for Holt/ARIMA ({n})')
        continue
    use_holt = False
    if n < MIN_ARIMA_LEN:
        use_holt = True
    try:
        if use_holt:
            print(f'Using Holt fallback for {pc} (n={n})')
            holt = Holt(series_na.values).fit(optimized=True)
            fitted = holt.fittedvalues
            resid = series_na.values - fitted
            model = None
        else:
            model = ARIMA(series_na.values, order=(1,0,0)).fit()
            resid = model.resid
    except Exception as e:
        print('Model fit failed for', pc, e)
        continue
    # Ljung-Box for lags 1..10 (bounded by sample)
    max_lag = min(10, max(1, len(resid)//3))
    lb = acorr_ljungbox(resid, lags=list(range(1,max_lag+1)), return_df=True)
    # take max p-value and min p? We'll record p at lag=max_lag
    lb_p = float(lb['lb_pvalue'].iloc[-1])
    lb_stat = float(lb['lb_stat'].iloc[-1])
    try:
        sh_p = float(shapiro(resid)[1])
    except Exception:
        sh_p = np.nan
    ar1_coef = np.nan
    sigma2 = np.nan
    if model is not None:
        try:
            ar1_coef = float(model.params[1]) if len(model.params)>1 else np.nan
            sigma2 = float(model.sigma2)
        except Exception:
            ar1_coef = np.nan
            sigma2 = np.nan
    summary_rows.append({'PC':pc, 'level':'global', 'n':len(series_na), 'AR1_coef': ar1_coef,
                         'sigma2': sigma2, 'LjungBox_stat': lb_stat, 'LjungBox_p': lb_p, 'Shapiro_p': sh_p})
    # plot diagnostics
    fig, axs = plt.subplots(3,2, figsize=(10,10))
    axs[0,0].plot(series_na.index, series_na.values, marker='o')
    axs[0,0].set_title(f'{pc} yearly mean (1992-2024)')
    axs[0,1].plot(series_na.index, resid, marker='o')
    axs[0,1].axhline(0, color='k', linestyle='--')
    axs[0,1].set_title('Residuals')
    plot_acf(resid, lags=max_lag, ax=axs[1,0])
    axs[1,0].set_title('ACF resid')
    plot_pacf(resid, lags=max_lag, ax=axs[1,1], method='ywm')
    axs[1,1].set_title('PACF resid')
    axs[2,0].hist(resid, bins=15)
    axs[2,0].set_title(f'Histogram (Shapiro p={sh_p:.3f})')
    sm.qqplot(resid, line='s', ax=axs[2,1])
    axs[2,1].set_title('QQ plot')
    plt.tight_layout()
    png = os.path.join(FIG_DIR, f'pca_arima_{pc}_long.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print('Saved', png)

# per-NOC PC ARIMA diagnostics (summary only)
summary_noc = []
nocs = out_df['Mapped_NOC'].unique()
for noc in nocs:
    sub = out_df[out_df['Mapped_NOC']==noc]
    for pc in pc_cols:
        s = sub.set_index('Year')[pc].reindex(years).dropna()
        n = len(s)
        if n < MIN_HOLT_LEN:
            continue
        try:
            if n < MIN_ARIMA_LEN:
                # Holt fallback
                h = Holt(s.values).fit(optimized=True)
                resid = s.values - h.fittedvalues
                m = None
            else:
                m = ARIMA(s.values, order=(1,0,0)).fit()
                resid = m.resid
            max_lag = min(10, max(1, len(resid)//3))
            lb = acorr_ljungbox(resid, lags=[max_lag], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[0])
            sh_p = float(shapiro(resid)[1]) if len(resid)>=3 else np.nan
            ar1c = float(m.params[1]) if (m is not None and len(m.params)>1) else np.nan
            sig2 = float(m.sigma2) if (m is not None) else np.nan
            summary_noc.append({'PC':pc, 'Mapped_NOC':noc, 'n':n, 'AR1_coef': ar1c,
                                'sigma2': sig2, 'LjungBox_p': lb_p, 'Shapiro_p': sh_p})
        except Exception:
            continue

OUT_SUM = os.path.join(OUT_DIR, 'pca_arima_diagnostics_long.csv')
pd.DataFrame(summary_rows).to_csv(OUT_SUM, index=False)
print('Saved global diagnostics to', OUT_SUM)
OUT_SUM_NOC = os.path.join(OUT_DIR, 'pca_arima_diagnostics_long_noc.csv')
pd.DataFrame(summary_noc).to_csv(OUT_SUM_NOC, index=False)
print('Saved per-NOC diagnostics to', OUT_SUM_NOC)
print('Done')
