#!/usr/bin/env python3
"""Predict 2028 medals using multiple approaches:
- PCA for dimensionality reduction (used for ML models)
- Time-series: ARIMA per-country
- LSTM across country sequences
- ML regressors: LightGBM, XGBoost, SVM, RandomForest

Outputs: outputs/predictions_2028.csv (per-country predictions per model)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
PROCESSED_DIR = OUT / 'processed_data'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROC = PROCESSED_DIR / 'processed_data.csv'
RESULTS_DIR = OUT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_panel():
    if not PROC.exists():
        raise FileNotFoundError('processed_data.csv not found; run data prep')
    return pd.read_csv(PROC)


def make_predict_features(panel, target_year=2028):
    # For each country, take last available row and construct features for next year
    last_year = int(panel['Year'].max())
    countries = sorted(panel['Country'].unique())
    rows = []
    for c in countries:
        sub = panel[panel['Country'] == c].sort_values('Year')
        if sub.empty:
            continue
        last = sub.iloc[-1].to_dict()
        # build new row
        new = last.copy()
        new['Year'] = target_year
        # shift lags: new Total_lag1 = last['Total']
        for k in [1,2,3]:
            if f'Total_lag{k}' in last:
                new[f'Total_lag{k}'] = last.get('Total') if k==1 else last.get(f'Total_lag{k-1}', 0)
            if f'Gold_lag{k}' in last:
                new[f'Gold_lag{k}'] = last.get('Gold') if k==1 else last.get(f'Gold_lag{k-1}', 0)
        # roll stats: approximate by using last values
        if 'Total_avg_3' in last:
            new['Total_avg_3'] = last.get('Total_avg_3', 0)
        if 'Total_std_3' in last:
            new['Total_std_3'] = last.get('Total_std_3', 0)
        if 'DominantRatio' in last:
            new['DominantRatio'] = last.get('DominantRatio', 0)
        rows.append(new)
    return pd.DataFrame(rows)


def pca_features(X, n_components=5):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    Z = pca.fit_transform(Xs)
    cols = [f'PCA{i+1}' for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols, index=X.index)


def arima_forecasts(panel, target_year=2028):
    res = {}
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        print('statsmodels not available — skipping ARIMA')
        return res
    for c, g in panel.groupby('Country'):
        ser = g.sort_values('Year')['Gold'].astype(float).fillna(0).values
        if len(ser) < 2:
            continue
        try:
            model = ARIMA(ser, order=(1,0,0))
            fit = model.fit()
            # forecast 1 step ahead (next Olympics)
            f = fit.forecast(steps=1)
            res[c] = max(0, float(f[0]))
        except Exception:
            continue
    return res


def lstm_forecast(panel, target_year=2028):
    res = {}
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
    except Exception:
        print('tensorflow not available — skipping LSTM')
        return res

    # prepare sequences across countries using Gold series
    seqs = []
    countries = []
    maxlen = 6
    for c, g in panel.groupby('Country'):
        vals = g.sort_values('Year')['Gold'].astype(float).fillna(0).values
        if len(vals) < 3:
            continue
        countries.append(c)
        # pad or truncate to maxlen
        if len(vals) < maxlen:
            pad = np.zeros(maxlen - len(vals))
            arr = np.concatenate([pad, vals])
        else:
            arr = vals[-maxlen:]
        seqs.append(arr)
    if not seqs:
        return res
    X = np.array(seqs)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # use first maxlen-1 as input, last as target
    X_in = Xs[:, :-1]
    y = Xs[:, -1]
    X_in = X_in.reshape((X_in.shape[0], X_in.shape[1], 1))

    model = Sequential([LSTM(32, input_shape=(X_in.shape[1], 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_in, y, epochs=30, batch_size=8, verbose=0)

    preds = model.predict(X_in)
    preds_inv = scaler.inverse_transform(np.hstack([np.zeros((preds.shape[0], X.shape[1]-1)), preds]))[:, -1]
    for i, c in enumerate(countries):
        res[c] = max(0, float(preds_inv[i]))
    return res


def ml_models(panel, predict_df, target='Gold'):
    out = {}
    num_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    # remove target and Year index-like
    features = [c for c in num_cols if c not in ['Year', 'Gold', 'Total']]
    if not features:
        return out

    # train/test split using historical rows
    X = panel[features].fillna(0)
    y = panel[target].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # reduce dims with PCA
    pca = PCA(n_components=min(10, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.2, random_state=42)

    models = {}
    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models['RF'] = rf

    # SVM (SVR)
    svr = SVR()
    try:
        svr.fit(X_train, y_train)
        models['SVM'] = svr
    except Exception:
        pass

    # XGBoost
    try:
        import xgboost as xgb
        xg = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        xg.fit(X_train, y_train)
        models['XGB'] = xg
    except Exception:
        print('xgboost not available — skipping')

    # LightGBM
    try:
        import lightgbm as lgb
        lg = lgb.LGBMRegressor(n_estimators=200, random_state=42)
        lg.fit(X_train, y_train)
        models['LGBM'] = lg
    except Exception:
        print('lightgbm not available — skipping')

    # prepare predict features
    Xp_pred = pca.transform(scaler.transform(predict_df[features].fillna(0)))

    for name, m in models.items():
        preds = m.predict(Xp_pred)
        out[name] = dict(zip(predict_df['Country'], [float(max(0, p)) for p in preds]))

    return out


def ensemble_results(model_preds, countries):
    # model_preds: dict model -> {country: pred}
    rows = []
    for c in countries:
        row = {'Country': c}
        vals = []
        for m, mp in model_preds.items():
            v = mp.get(c, np.nan)
            row[f'pred_{m}'] = v
            vals.append(v)
        # simple ensemble: median of available
        valid = [v for v in vals if not (pd.isna(v))]
        row['pred_ensemble'] = float(np.median(valid)) if valid else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    target_year = 2028
    predict_df = make_predict_features(panel, target_year=target_year)
    countries = predict_df['Country'].tolist()

    # ARIMA (per-country) for Gold
    arima_preds = arima_forecasts(panel, target_year=target_year)

    # LSTM
    lstm_preds = lstm_forecast(panel, target_year=target_year)

    # ML models for Gold
    ml_preds_gold = ml_models(panel, predict_df, target='Gold')

    # ML models for Total
    ml_preds_total = ml_models(panel, predict_df, target='Total')

    # collate predictions into dataframe
    models_all = {}
    # add ARIMA and LSTM under Gold
    if arima_preds:
        models_all['ARIMA'] = {c: arima_preds.get(c, np.nan) for c in countries}
    if lstm_preds:
        models_all['LSTM'] = {c: lstm_preds.get(c, np.nan) for c in countries}
    # add ML gold models
    for m, mp in ml_preds_gold.items():
        models_all[f'Gold_{m}'] = mp
    for m, mp in ml_preds_total.items():
        models_all[f'Total_{m}'] = mp

    df_out = ensemble_results(models_all, countries)
    outpath = RESULTS_DIR / 'predictions_2028.csv'
    df_out.to_csv(outpath, index=False)
    print('Wrote', outpath)


if __name__ == '__main__':
    main()
