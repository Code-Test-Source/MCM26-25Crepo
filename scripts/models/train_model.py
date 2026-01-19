#!/usr/bin/env python3
"""Train a RandomForest on `outputs/model_ready.csv` and save model, preds, importances."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
DATA = OUT / 'model_ready.csv'
RESULTS = OUT / 'results'


def prepare_features(df):
    meta = [c for c in ['Country','Year'] if c in df.columns]
    X = df.drop(columns=[c for c in df.columns if c in meta + ['Total']])
    X = X.select_dtypes(include=[np.number]).copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c].fillna(X[c].median(), inplace=True)
    return X


def main():
    if not DATA.exists():
        print('model_ready.csv not found; run data prep first')
        return
    df = pd.read_csv(DATA)
    if 'Total' not in df.columns:
        print('No Total column in model_ready.csv')
        return
    y = df['Total'].values
    X = prepare_features(df)
    if X.shape[1] == 0:
        print('No numeric features to train on')
        return

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    print(f'RandomForest RMSE: {rmse:.4f}')

    # save artifacts
    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RESULTS / 'rf_model.joblib')
    pd.DataFrame({'y_true': y_test, 'y_pred': pred}).to_csv(RESULTS / 'model_predictions.csv', index=False)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi.to_csv(RESULTS / 'feature_importances.csv')


if __name__ == '__main__':
    main()
