import os
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Paths
BASE_DIR = os.path.abspath("./")
DATA_DIR = os.path.join(BASE_DIR, "2025_Problem_C_Data")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "processed_data")
os.makedirs(OUT_DIR, exist_ok=True)
MEDAL_COUNTS = os.path.join(DATA_DIR, "summerOly_medal_counts.csv")
ATHLETES = os.path.join(DATA_DIR, "summerOly_athletes.csv")
HOSTS = os.path.join(DATA_DIR, "summerOly_hosts.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('reliable_pipeline')

# Helpers
import re

HISTORICAL_MAPPING = {
    'urs': 'RUS', 'soviet union': 'RUS', 'eun': 'RUS', 'unified team': 'RUS', 'roc': 'RUS',
    'frg': 'GER', 'gdr': 'GER', 'west germany': 'GER', 'east germany': 'GER',
    'tch': 'CZE', 'czechoslovakia': 'CZE', 'yug': 'SRB', 'yugoslavia': 'SRB',
    'great britain': 'GBR', 'united kingdom': 'GBR', 'united states': 'USA', 'china': 'CHN'
}


def normalize_noc(x):
    if pd.isna(x):
        return x
    s = str(x).replace('\u00A0', ' ').strip()
    s = re.sub(r"\s+", ' ', s)
    return s


def map_noc_to_modern(val):
    if pd.isna(val):
        return val
    s = normalize_noc(val)
    k = s.lower()
    if k in HISTORICAL_MAPPING:
        return HISTORICAL_MAPPING[k]
    if len(s) == 3 and s.isalpha():
        return s.upper()
    up = s.upper()
    if len(up) == 3 and up.isalpha():
        return up
    return s


def safe_history_forecast(series):
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    if len(s) == 1:
        return float(s.iloc[-1])
    vals = s.sort_index().iloc[-3:]
    return float(vals.mean())


def load_data():
    medals = pd.read_csv(MEDAL_COUNTS) if os.path.exists(MEDAL_COUNTS) else pd.DataFrame()
    athletes = pd.read_csv(ATHLETES) if os.path.exists(ATHLETES) else pd.DataFrame()
    hosts = pd.read_csv(HOSTS) if os.path.exists(HOSTS) else pd.DataFrame()
    return medals, athletes, hosts


def data_overview(medals):
    # basic diagnostics
    if medals.empty:
        log.warning('Medals dataframe is empty')
        return {}
    medals['Year'] = pd.to_numeric(medals['Year'], errors='coerce')
    summary = {
        'min_year': int(medals['Year'].min()),
        'max_year': int(medals['Year'].max()),
        'count_rows': len(medals),
        'count_2024': int((medals['Year']==2024).sum()),
        'n_unique_noc_2020': int(medals[medals['Year']==2020]['NOC'].nunique()),
        'n_unique_noc_2024': int(medals[medals['Year']==2024]['NOC'].nunique())
    }
    log.info('Data overview: %s', summary)
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR,'data_summary.csv'), index=False)
    return summary


def build_panel(medals):
    df = medals.copy()
    # normalize and map NOC
    if 'NOC' in df.columns:
        df['NOC_raw'] = df['NOC'].astype(str)
        df['NOC'] = df['NOC'].apply(normalize_noc).apply(map_noc_to_modern)
    else:
        df['NOC'] = ''
    # coerce year and keep >0
    df['Year'] = pd.to_numeric(df.get('Year', np.nan), errors='coerce')
    df = df[df['Year'].notna() & (df['Year']>0)].copy()
    df['Year'] = df['Year'].astype(int)
    # ensure medal columns
    for c in ['Gold','Silver','Bronze','Total']:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    # aggregate to Year,NOC (keep Sport if present)
    if 'Sport' in df.columns:
        agg_cols = ['Year','NOC','Sport']
    else:
        agg_cols = ['Year','NOC']
    num_cols = ['Gold','Silver','Bronze','Total']
    panel = df.groupby(agg_cols, dropna=False)[num_cols].sum().reset_index()
    panel.to_csv(os.path.join(OUT_DIR,'panel.csv'), index=False)
    log.info('Built panel rows=%d', len(panel))
    return panel


def apply_host_flags(panel, hosts_df):
    # hosts_df: Year, Host (string like 'Los Angeles, United States')
    if hosts_df is None or hosts_df.empty:
        panel['is_host'] = 0
        return panel
    hosts_map = {}
    for _, r in hosts_df.iterrows():
        try:
            yr = int(r['Year'])
        except Exception:
            continue
        host_str = str(r['Host'])
        # extract trailing country after comma
        parts = host_str.split(',')
        country = parts[-1].strip() if len(parts)>0 else host_str.strip()
        hosts_map[yr] = country
    panel = panel.copy()
    panel['is_host'] = 0
    for yr, country in hosts_map.items():
        mapped = map_noc_to_modern(country)
        panel.loc[(panel['Year']==yr) & (panel['NOC']==mapped), 'is_host'] = 1
    return panel


def compute_holt_trends(panel):
    # collapse historical names by mapping NOC already done
    cols = ['NOC','Year','Total','Gold']
    df = panel.copy()
    results = []
    groups = df.groupby('NOC')
    for noc, g in groups:
        series_total = g.sort_values('Year').groupby('Year')['Total'].sum()
        series_gold = g.sort_values('Year').groupby('Year')['Gold'].sum()
        t = safe_history_forecast(series_total)
        gld = safe_history_forecast(series_gold)
        results.append({'NOC': noc, 'HOLT_Total_2028': t, 'HOLT_Gold_2028': gld})
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR,'holt_trends.csv'), index=False)
    log.info('Holt trends rows=%d', len(res))
    return res


def compute_pca(panel, n_components=5):
    # sport-weighted value per Year,NOC,Sport
    if 'Sport' not in panel.columns:
        # create trivial sport column
        panel = panel.copy()
        panel['Sport'] = 'All'
    panel['Weighted'] = panel['Gold']*0.5 + panel['Silver']*0.3 + panel['Bronze']*0.2
    wide = panel.pivot_table(index=['Year','NOC'], columns='Sport', values='Weighted', aggfunc='sum', fill_value=0)
    if wide.shape[0] == 0:
        log.warning('PCA input empty')
        return pd.DataFrame()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(wide.values)
    n_comp = max(1, min(n_components, Xs.shape[1], Xs.shape[0]))
    pca = PCA(n_components=n_comp, random_state=42)
    pcs = pca.fit_transform(Xs)
    cols = [f'PC{i+1}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, columns=cols, index=wide.index).reset_index()
    pca_df.to_csv(os.path.join(OUT_DIR,'pca.csv'), index=False)
    log.info('PCA rows=%d cols=%d', pca_df.shape[0], pca_df.shape[1])
    return pca_df


def build_features(panel, pca_df, holt_df):
    df = panel.copy()
    # ensure is_host exists
    if 'is_host' not in df.columns:
        df['is_host'] = 0
    # merge holt (NOC level)
    df = df.merge(holt_df, on='NOC', how='left')
    # merge pca on Year,NOC
    if not pca_df.empty:
        df = df.merge(pca_df, on=['Year','NOC'], how='left')
    # lags and stats
    df = df.sort_values(['NOC','Year'])
    df['Gold_Lag1'] = df.groupby('NOC')['Gold'].shift(1).fillna(0)
    df['Total_Lag1'] = df.groupby('NOC')['Total'].shift(1).fillna(0)
    df['Gold_MA3'] = df.groupby('NOC')['Gold'].rolling(3, min_periods=1).mean().reset_index(0,drop=True).fillna(0)
    df['Total_MA3'] = df.groupby('NOC')['Total'].rolling(3, min_periods=1).mean().reset_index(0,drop=True).fillna(0)
    df['Gold_Best'] = df.groupby('NOC')['Gold'].cummax().fillna(0)
    df['Total_Best'] = df.groupby('NOC')['Total'].cummax().fillna(0)
    # fill remaining na
    df.fillna(0, inplace=True)
    # expected_russia_impact_2028: scale by correlation with RUS and RUS Holt forecast
    try:
        rus_holt = float(holt_df[holt_df['NOC']=='RUS']['HOLT_Total_2028'].iloc[0]) if not holt_df[holt_df['NOC']=='RUS'].empty else 0.0
        pivot = panel.pivot_table(index='Year', columns='NOC', values='Total', aggfunc='sum', fill_value=0)
        if 'RUS' in pivot.columns and rus_holt>0:
            rseries = pivot['RUS']
            cors = pivot.corrwith(rseries).to_dict()
            df['expected_russia_impact_2028'] = df['NOC'].map(lambda x: cors.get(x,0.0) * rus_holt)
        else:
            df['expected_russia_impact_2028'] = 0.0
    except Exception:
        df['expected_russia_impact_2028'] = 0.0
    # host effect interaction: host multiplier times HOLT_Total_2028
    df['host_holt_interaction'] = df['is_host'] * df.get('HOLT_Total_2028', 0)
    df.to_csv(os.path.join(OUT_DIR,'features.csv'), index=False)
    log.info('Features rows=%d cols=%d', df.shape[0], df.shape[1])
    return df


def train_xgb(features_df, target='Total', B=50):
    # train using data Year<=2024
    train = features_df[features_df['Year']<=2024].copy()
    if train.empty:
        raise RuntimeError('No training rows found')
    drop_cols = ['Year','NOC','Sport','Gold','Total']
    feat_cols = [c for c in train.columns if c not in drop_cols and train[c].dtype in [np.float64, np.int64]]
    X = train[feat_cols].copy()
    y = train[target].copy()
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {'n_estimators':200, 'max_depth':5, 'learning_rate':0.05, 'subsample':0.8, 'colsample_bytree':0.8, 'random_state':42}
    base = XGBRegressor(**params)
    base.fit(X_tr, y_tr)
    val_pred = base.predict(X_val)
    log.info('Validation RMSE: %.3f', np.sqrt(mean_squared_error(y_val, val_pred)))
    # bootstrap
    rng = np.random.default_rng(42)
    n = len(X)
    def train_one(i):
        idx = rng.integers(0, n, n)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        m = XGBRegressor(**params)
        m.fit(Xb, yb)
        return m
    n_jobs = min(8, cpu_count())
    models = Parallel(n_jobs=n_jobs)(delayed(train_one)(i) for i in range(B))
    return base, models, feat_cols


def predict_ensemble(models, X_pred):
    preds = np.vstack([m.predict(X_pred) for m in models])
    median = np.median(preds, axis=0)
    lo = np.percentile(preds, 2.5, axis=0)
    hi = np.percentile(preds, 97.5, axis=0)
    return median, lo, hi, preds


def run(B=20):
    medals, athletes, hosts = load_data()
    data_overview(medals)
    panel = build_panel(medals)
    # apply host flags from hosts file
    panel = apply_host_flags(panel, hosts)
    # ensure 2024 rows exist
    c2024 = int((panel['Year']==2024).sum())
    log.info('Panel count for 2024: %d', c2024)
    holt = compute_holt_trends(panel)
    pca_df = compute_pca(panel)
    features = build_features(panel, pca_df, holt)
    base, models, feat_cols = train_xgb(features, B=B)
    # prepare predictions
    # allowed NOCs present in 2020 or 2024
    allowed = set(panel[panel['Year'].isin([2020,2024])]['NOC'].unique())
    log.info('Allowed NOCs count=%d', len(allowed))
    pred_rows = features[features['Year']==2028].copy()
    if pred_rows.empty:
        log.warning('No Year==2028 rows; generating from 2024 base')
        base_rows = features[features['Year']==2024].copy()
        base_rows['Year'] = 2028
        pred_rows = base_rows
    if len(allowed)>0:
        pred_rows = pred_rows[pred_rows['NOC'].isin(allowed)].copy()
    if pred_rows.empty:
        log.warning('No prediction rows after filtering; aborting save')
        return
    # show feature importances
    try:
        importances = pd.Series(base.feature_importances_, index=feat_cols).sort_values(ascending=False)
        importances.to_csv(os.path.join(OUT_DIR,'feature_importances.csv'))
        log.info('Top features:\n%s', importances.head(10).to_string())
    except Exception:
        log.exception('Failed to compute feature importances')
    X_pred = pred_rows[feat_cols].copy()
    X_pred = X_pred.select_dtypes(include=[np.number])
    median, lo, hi, allp = predict_ensemble(models, X_pred)
    out = pd.DataFrame({'NOC': pred_rows['NOC'].values, 'Pred_Total_median': median, 'Pred_Total_lo': lo, 'Pred_Total_hi': hi})
    out.to_csv(os.path.join(OUT_DIR,'predictions_reliable.csv'), index=False)
    log.info('Saved predictions_reliable.csv rows=%d', len(out))


if __name__ == '__main__':
    run(B=20)
