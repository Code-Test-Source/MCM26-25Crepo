#!/usr/bin/env python3
"""Plot top-10 countries by 2028 ensemble prediction and save CSV/PNG."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
FIG_DIR = OUT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_CANDIDATES = [OUT / 'predictions_2028.csv', OUT / 'results' / 'predictions_2028.csv']


def find_pred_file():
    for p in PRED_CANDIDATES:
        if p.exists():
            return p
    return None


def main():
    pred_path = find_pred_file()
    if pred_path is None:
        print('predictions_2028.csv not found in outputs/ or outputs/results/')
        return
    df = pd.read_csv(pred_path)
    # prefer ensemble column
    if 'pred_ensemble' in df.columns:
        score_col = 'pred_ensemble'
    else:
        # fallback: pick first numeric prediction column after Country
        cols = [c for c in df.columns if c.startswith('pred_')]
        if not cols:
            print('No prediction columns found in', PRED)
            return
        score_col = cols[0]

    df_top = df.sort_values(score_col, ascending=False).head(10).reset_index(drop=True)
    results_dir = OUT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(results_dir / 'predictions_2028_top10.csv', index=False)

    plt.figure(figsize=(10,6))
    plt.bar(df_top['Country'], df_top[score_col], color='tab:blue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Predicted Gold (ensemble)' if score_col=='pred_ensemble' else score_col)
    plt.title('Top 10 predicted countries for 2028 (by {})'.format(score_col))
    plt.tight_layout()
    outpng = FIG_DIR / 'pred_2028_top10.png'
    plt.savefig(outpng)
    plt.close()
    print('Wrote', outpng)


if __name__ == '__main__':
    main()
