import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
try:
    from scipy.stats import linregress
    scipy_available = True
except Exception:
    scipy_available = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'outputs', 'processed_data')
FIG_DIR = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

PROCESSED_ATHLETES = os.path.join(OUT_DIR, 'summerOly_athletes_processed.csv')
PER_YEAR_OUT = os.path.join(OUT_DIR, 'model_cv_per_year.csv')
SUMMARY_OUT = os.path.join(OUT_DIR, 'eval_model_cv_summary.csv')
BOOTSTRAP_OUT = os.path.join(OUT_DIR, 'model_cv_bootstrap_distributions.csv')
CV_RF_2024_2028_OUT = os.path.join(OUT_DIR, 'model_cv_2024_2028_check.csv')
FIGS_ZIP = os.path.join(FIG_DIR, 'model_cv_figures_package.zip')

try:
    from xgboost import XGBRegressor
    xgb_available = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    xgb_available = False

from sklearn.ensemble import RandomForestRegressor

# Load processed athletes aggregated to year x country
if os.path.exists(PROCESSED_ATHLETES):
    p = pd.read_csv(PROCESSED_ATHLETES)
    df = p.groupby(['Year','Mapped_NOC'], as_index=False)['Total'].sum()
    # bring is_host flag if present in processed file
    if 'is_host' in p.columns:
        host_flags = p.groupby(['Year','Mapped_NOC'], as_index=False)['is_host'].max()
        df = df.merge(host_flags, on=['Year','Mapped_NOC'], how='left')
    else:
        df['is_host'] = 0
else:
    raise FileNotFoundError('Processed athletes file not found: ' + PROCESSED_ATHLETES)

# Create Prev_Total feature (previous edition) grouped by Mapped_NOC
df = df.sort_values(['Mapped_NOC','Year']).copy()
df['Prev_Total'] = df.groupby('Mapped_NOC')['Total'].shift(1)

# Rolling-origin CV: evaluate for target years 2004..2020 (inclusive)
target_years = [y for y in sorted(df['Year'].unique()) if 2004 <= y <= 2020]
models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, random_state=42) if xgb_available else XGBRegressor(n_estimators=200)
}

per_year_rows = []
agg = {m: {'mae':[], 'mse':[], 'mape':[], 'rmse':[]} for m in models}
# collect aggregated prediction/true pairs across all target years for bootstrap
agg_preds_trues = {m: {'preds':[], 'trues':[]} for m in models}

for ty in target_years:
    train = df[(df['Year'] < ty) & (df['Prev_Total'].notna())]
    test = df[(df['Year'] == ty) & (df['Prev_Total'].notna())]
    if train.empty or test.empty:
        continue
    X_train = train[['Prev_Total']].values
    y_train = train['Total'].values
    X_test = test[['Prev_Total']].values
    y_test = test['Total'].values
    for name, model in models.items():
        m = model
        m.set_params(**({'verbosity':0} if xgb_available and name=='XGBoost' else {}))
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        mae_t = float(np.mean(np.abs(preds - y_test)))
        mse_t = float(np.mean((preds - y_test)**2))
        rmse_t = float(np.sqrt(mse_t))
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = y_test != 0
            mape_t = float(np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask]))*100.0) if mask.sum()>0 else np.nan
        per_year_rows.append({'Year': ty, 'Model': name, 'MAE': mae_t, 'MSE': mse_t, 'RMSE': rmse_t, 'MAPE_pct': mape_t, 'n_test': len(y_test)})
        agg[name]['mae'].append(mae_t)
        agg[name]['mse'].append(mse_t)
        agg[name]['rmse'].append(rmse_t)
        agg[name]['mape'].append(mape_t)
        # store per-sample preds/trues for bootstrap (flatten)
        agg_preds_trues[name]['preds'].append(preds)
        agg_preds_trues[name]['trues'].append(y_test)

# Save per-year CV results
per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(PER_YEAR_OUT, index=False)

# Aggregate summary across target years
summary_rows = []
for name in models:
    maes = np.array(agg[name]['mae'])
    mses = np.array(agg[name]['mse'])
    rmse_arr = np.array(agg[name]['rmse'])
    mapes = np.array([v for v in agg[name]['mape'] if not np.isnan(v)])
    summary_rows.append({
        'Model': name,
        'MAE_mean': float(np.nanmean(maes)) if maes.size>0 else np.nan,
        'MAE_std': float(np.nanstd(maes)) if maes.size>0 else np.nan,
        'MSE_mean': float(np.nanmean(mses)) if mses.size>0 else np.nan,
        'RMSE_mean': float(np.sqrt(float(np.nanmean(mses)))) if mses.size>0 else np.nan,
        'MAPE_mean_pct': float(np.nanmean(mapes)) if mapes.size>0 else np.nan,
        'n_years': int(len(maes))
    })

pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)

# --- Four-panel per-year plots (MSE, RMSE, MAE, MAPE) with RF/XGB together
if not per_year_df.empty:
    metrics = [('MSE','MSE'), ('RMSE','RMSE'), ('MAE','MAE'), ('MAPE_pct','MAPE')]
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    axes = axes.ravel()
    years = sorted(per_year_df['Year'].unique())
    for i,(col,label) in enumerate(metrics):
        ax = axes[i]
        for j,model_name in enumerate(models):
            sub = per_year_df[per_year_df['Model']==model_name]
            sub = sub.set_index('Year').reindex(years)
            vals = sub[col].values if col in sub.columns else np.array([np.nan]*len(years))
            ax.plot(years, vals, marker='o', label=model_name)
            for x,y in zip(years, vals):
                if np.isnan(y):
                    continue
                ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_title(label)
        ax.set_xlabel('Year')
        ax.set_xticks(years[::2])
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,'model_cv_4panel_metrics_compare.png'), dpi=150)
    plt.close()

# --- Bootstrap distributions across aggregated test samples (per model)
bootstrap_rows = []
for name in models:
    preds_list = agg_preds_trues[name]['preds']
    trues_list = agg_preds_trues[name]['trues']
    if len(preds_list)==0:
        continue
    preds_all = np.concatenate(preds_list)
    trues_all = np.concatenate(trues_list)
    n = len(preds_all)
    n_boot = 1000
    boot_metrics = {'mae':[], 'mse':[], 'rmse':[], 'mape':[]}
    rng = np.random.default_rng(42)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        p = preds_all[idx]
        t = trues_all[idx]
        mae_b = float(np.mean(np.abs(p-t)))
        mse_b = float(np.mean((p-t)**2))
        rmse_b = float(np.sqrt(mse_b))
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = t!=0
            mape_b = float(np.mean(np.abs((t[mask]-p[mask])/t[mask]))*100.0) if mask.sum()>0 else np.nan
        boot_metrics['mae'].append(mae_b)
        boot_metrics['mse'].append(mse_b)
        boot_metrics['rmse'].append(rmse_b)
        boot_metrics['mape'].append(mape_b)
    # save summary rows
    for k,v in boot_metrics.items():
        bootstrap_rows.append({'Model':name,'metric':k,'mean':float(np.nanmean(v)),'std':float(np.nanstd(v))})
    # plot histogram figure for this model (4 panels)
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    axes = axes.ravel()
    keys = ['mse','rmse','mae','mape']
    titles = ['MSE','RMSE','MAE','MAPE_pct']
    for ii,k in enumerate(keys):
        ax = axes[ii]
        vals = np.array(boot_metrics[k])
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=40, color='C0', alpha=0.7)
        ax.set_title(titles[ii])
        ax.axvline(np.nanmean(vals), color='k', linestyle='--')
    plt.suptitle(f'Bootstrap metric distributions - {name}')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(FIG_DIR, f'bootstrap_metrics_{name}.png'), dpi=150)
    plt.close()

# save bootstrap summary
if bootstrap_rows:
    pd.DataFrame(bootstrap_rows).to_csv(BOOTSTRAP_OUT, index=False)

# --- Cross-check RF (provided with-host files) vs locally-trained XGB for 2024 and 2028
def load_with_host_preds(year):
    f = os.path.join(OUT_DIR, f'medal_predictions_{year}_with_host.csv')
    if os.path.exists(f):
        return pd.read_csv(f)
    return None

check_rows = []
for target in [2024, 2028]:
    train = df[(df['Year'] < target) & (df['Prev_Total'].notna())]
    test = df[(df['Year'] == target) & (df['Prev_Total'].notna())]
    if train.empty or test.empty:
        continue
    X_train = train[['Prev_Total']].values
    y_train = train['Total'].values
    X_test = test[['Prev_Total']].values
    y_test = test['Total'].values
    # train XGB
    xgb = models['XGBoost']
    xgb.set_params(**({'verbosity':0} if xgb_available else {}))
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    test_df = test[['Mapped_NOC']].copy()
    test_df['XGB_Predicted_Total'] = xgb_preds
    # load RF provided file
    rf_df = load_with_host_preds(target)
    if rf_df is None:
        continue
    rf_df = rf_df[['Mapped_NOC','Predicted_Total']].rename(columns={'Predicted_Total':'RF_Predicted_Total'})
    merged = test_df.merge(rf_df, on='Mapped_NOC', how='inner')
    if merged.empty:
        continue
    merged['abs_diff'] = np.abs(merged['XGB_Predicted_Total'] - merged['RF_Predicted_Total'])
    mae_between = float(np.mean(merged['abs_diff']))
    # regression / correlation stats
    try:
        r2 = float(r2_score(merged['RF_Predicted_Total'], merged['XGB_Predicted_Total']))
    except Exception:
        r2 = np.nan
    slope = intercept = r_value = p_value = np.nan
    if scipy_available and len(merged)>=2:
        try:
            lr = linregress(merged['RF_Predicted_Total'], merged['XGB_Predicted_Total'])
            slope = float(lr.slope)
            intercept = float(lr.intercept)
            r_value = float(lr.rvalue)
            p_value = float(lr.pvalue)
        except Exception:
            pass
    check_rows.append({'Year':target,'n':len(merged),'MAE_between_models': mae_between,'slope':slope,'intercept':intercept,'r_squared': (r_value**2 if not np.isnan(r_value) else r2),'p_value':p_value})
    # scatter plot RF vs XGB with absolute diff annotated and stats
    plt.figure(figsize=(7,6))
    plt.scatter(merged['RF_Predicted_Total'], merged['XGB_Predicted_Total'], alpha=0.7)
    mn = min(merged['RF_Predicted_Total'].min(), merged['XGB_Predicted_Total'].min())
    mx = max(merged['RF_Predicted_Total'].max(), merged['XGB_Predicted_Total'].max())
    plt.plot([mn, mx], [mn, mx], color='k', linestyle='--')
    for idx,row in merged.iterrows():
        plt.text(row['RF_Predicted_Total'], row['XGB_Predicted_Total'], f"{row['abs_diff']:.1f}", fontsize=7, va='bottom', ha='center')
    plt.xlabel('RF Predicted Total (with host)')
    plt.ylabel('XGB Predicted Total')
    stats_txt = f"n={len(merged)}\nR²={(r_value**2 if not np.isnan(r_value) else r2):.3f}\np={p_value:.3g}"
    plt.gca().text(0.02, 0.98, stats_txt, transform=plt.gca().transAxes, va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.title(f'RF vs XGB predictions comparison {target} (annotated abs diff)')
    plt.tight_layout()
    figpath = os.path.join(FIG_DIR, f'rf_xgb_compare_{target}.png')
    plt.savefig(figpath, dpi=150)
    plt.close()
    # collect figure path for packaging
    try:
        figs_for_zip.append(figpath)
    except NameError:
        figs_for_zip = [figpath]

if check_rows:
    pd.DataFrame(check_rows).to_csv(CV_RF_2024_2028_OUT, index=False)

# --- Package key figure files into a zip
import zipfile
if 'figs_for_zip' in globals():
    # also include the 4-panel and bootstrap figures if exist
    extra = [os.path.join(FIG_DIR,'model_cv_4panel_metrics_compare.png')]
    for name in models:
        extra.append(os.path.join(FIG_DIR, f'bootstrap_metrics_{name}.png'))
    candidates = figs_for_zip + extra
    existing = [p for p in candidates if os.path.exists(p)]
    if existing:
        with zipfile.ZipFile(FIGS_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for p in existing:
                zf.write(p, arcname=os.path.basename(p))

        print('Packaged figures to', FIGS_ZIP)

print('Saved per-year CV to', PER_YEAR_OUT)
print('Saved summary to', SUMMARY_OUT)
print('Saved bootstrap summary to', BOOTSTRAP_OUT)
print('Saved RF/XGB cross-check to', CV_RF_2024_2028_OUT)
print('Figures saved to', FIG_DIR)
print('Done')
