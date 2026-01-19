#!/usr/bin/env python3
"""Generate preprocessing visualizations from outputs/processed_data.csv."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG_DIR = OUT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROC = OUT / 'processed_data.csv'


def main():
    if not PROC.exists():
        print('processed_data.csv not found; run data prep first')
        return
    df = pd.read_csv(PROC)
    # Total distribution
    if 'Total' in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df['Total'], bins=30, kde=False)
        plt.title('Total medals distribution')
        plt.savefig(FIG_DIR / 'total_distribution.png', bbox_inches='tight')
        plt.close()

    # missingness
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    sns.barplot(x=miss.index, y=miss.values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('missing_fraction')
    plt.title('Fraction missing per column')
    plt.savefig(FIG_DIR / 'missing_fraction.png', bbox_inches='tight')
    plt.close()

    # top countries
    if 'Country' in df.columns and 'Total' in df.columns:
        top = df.groupby('Country', as_index=False)['Total'].sum().sort_values('Total', ascending=False).head(10)
        plt.figure(figsize=(8,4))
        sns.barplot(data=top, x='Country', y='Total')
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 countries by total medals')
        plt.savefig(FIG_DIR / 'top10_countries.png', bbox_inches='tight')
        plt.close()

    # correlation heatmap
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 1:
        corr = num.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, cmap='RdBu', center=0)
        plt.title('Numeric feature correlations')
        plt.savefig(FIG_DIR / 'correlation_heatmap.png', bbox_inches='tight')
        plt.close()

    print('Wrote figures to', FIG_DIR)


if __name__ == '__main__':
    main()
