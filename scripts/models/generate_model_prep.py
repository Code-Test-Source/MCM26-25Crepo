#!/usr/bin/env python3
"""Generate model-ready CSV and simple EDA summaries from processed_data.csv

Outputs written to outputs/:
- model_ready.csv
- eda_country_top10.csv
- feature_correlation.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs'
PROCESSED = OUT_DIR / 'processed_data'
PROCESSED.mkdir(parents=True, exist_ok=True)
PROC = PROCESSED / 'processed_data.csv'


def main():
    if not PROC.exists():
        print('processed_data.csv not found at', PROC)
        return
    df = pd.read_csv(PROC)

    # basic numeric columns selection
    num = df.select_dtypes(include=['number']).copy()

    # correlation matrix of numeric features
    corr = num.corr()
    corr.to_csv(PROCESSED / 'feature_correlation.csv')

    # top 10 countries by total medals (sum over years)
    if 'Total' in df.columns:
        top = df.groupby('Country', as_index=False)['Total'].sum().sort_values('Total', ascending=False).head(10)
        top.to_csv(PROCESSED / 'eda_country_top10.csv', index=False)

    # prepare model-ready dataset: predict 'Total' using lag features if available
    features = []
    for c in ['Total_lag1','Gold_lag1','Total_avg_3','DominantRatio','IsHost']:
        if c in df.columns:
            features.append(c)
    if 'Total' in df.columns and features:
        model_df = df[['Country','Year','Total'] + features].copy()
        model_df.to_csv(PROCESSED / 'model_ready.csv', index=False)
        print('Wrote model_ready.csv with', len(model_df), 'rows')
    else:
        print('Required columns not found to build model_ready.csv; wrote summaries only')

    print('Wrote eda and correlation summaries to', PROCESSED)


if __name__ == '__main__':
    main()
