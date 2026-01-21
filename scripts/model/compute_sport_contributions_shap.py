import os
import warnings
from typing import List

import pandas as pd
import numpy as np


def load_data(base_dir: str):
    """
    Load per-year metrics and join on (Year, Sport):
    - china_sport_stability_cv.csv -> CV2YR as Stability_CV2YR
    - china_sport_participation_share.csv -> SharePct as Participation_SharePct
    - china_sport_weighted_share.csv -> SharePct as Weighted_SharePct
    Modeling target comes from china_medals_by_year_sport.csv (Weighted321).
    Country totals from china_medal_weighted.csv.
    """
    stability_path = os.path.join(base_dir, "china_sport_stability_cv.csv")
    participation_path = os.path.join(base_dir, "china_sport_participation_share.csv")
    weighted_share_path = os.path.join(base_dir, "china_sport_weighted_share.csv")
    sport_year_path = os.path.join(base_dir, "china_medals_by_year_sport.csv")
    total_year_path = os.path.join(base_dir, "china_medal_weighted.csv")

    stability = pd.read_csv(stability_path)
    participation = pd.read_csv(participation_path)
    weighted_share = pd.read_csv(weighted_share_path)
    sport_year = pd.read_csv(sport_year_path)
    total_year = pd.read_csv(total_year_path)

    # Select and rename columns
    stab_cols = [c for c in ["Year", "Sport", "CV2YR"] if c in stability.columns]
    stability = stability[stab_cols].rename(columns={"CV2YR": "Stability_CV2YR"})

    part_cols = [c for c in ["Year", "Sport", "SharePct"] if c in participation.columns]
    participation = participation[part_cols].rename(columns={"SharePct": "Participation_SharePct"})

    wshare_cols = [c for c in ["Year", "Sport", "SharePct"] if c in weighted_share.columns]
    weighted_share = weighted_share[wshare_cols].rename(columns={"SharePct": "Weighted_SharePct"})

    # Ensure required columns
    if not {"Year", "Sport", "Weighted321"}.issubset(sport_year.columns):
        raise ValueError("Expected columns Year,Sport,Weighted321 in china_medals_by_year_sport.csv")
    if not {"Year", "Weighted321"}.issubset(total_year.columns):
        raise ValueError("Expected columns Year,Weighted321 in china_medal_weighted.csv")

    # Merge per-year features into sport-year medals
    merged = (
        sport_year.merge(stability, on=["Year", "Sport"], how="left")
                  .merge(participation, on=["Year", "Sport"], how="left")
                  .merge(weighted_share, on=["Year", "Sport"], how="left")
    )

    # Fill missing Stability_CV2YR using global median (first year often missing)
    if "Stability_CV2YR" in merged.columns:
        med = merged["Stability_CV2YR"].median(skipna=True)
        merged["Stability_CV2YR"] = merged["Stability_CV2YR"].fillna(med if pd.notna(med) else 0.0)

    # Drop rows missing key shares (if any); otherwise fill 0
    for c in ["Participation_SharePct", "Weighted_SharePct"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    return (stability, participation, weighted_share, sport_year, total_year, merged)


def train_model_shap(X: pd.DataFrame, y: pd.Series):
    """
    Train a tree-based model and compute SHAP values.
    Prefers RandomForestRegressor (no heavy deps). Uses shap.TreeExplainer.
    """
    from sklearn.ensemble import RandomForestRegressor
    import shap

    # Suppress noisy warnings from shap
    warnings.filterwarnings("ignore", category=UserWarning)

    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    # Global feature weights via mean absolute SHAP values
    mean_abs = np.abs(shap_values).mean(axis=0)
    if mean_abs.sum() == 0:
        weights = np.ones_like(mean_abs) / len(mean_abs)
    else:
        weights = mean_abs / mean_abs.sum()

    return model, explainer, shap_values, expected_value, weights, mean_abs


def allocate_total_medals_by_year(
    base_dir: str,
    feature_cols: List[str] = ["Stability_CV2YR", "Participation_SharePct", "Weighted_SharePct"],
    output_prefix: str = "sport_contributions"
):
    stability, participation, weighted_share, sport_year, total_year, merged = load_data(base_dir)

    # Modeling features and target
    X = merged[feature_cols].copy()
    y = merged["Weighted321"].astype(float).values

    # Train and compute SHAP
    model, explainer, shap_values, expected_value, weights, mean_abs = train_model_shap(X, y)

    # Predictions per row (sport-year)
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, None)  # ensure non-negative

    merged_pred = merged.copy()
    merged_pred["PredWeighted321"] = y_pred

    # Per-year normalization and allocation of national total
    total_year_small = total_year[["Year", "Weighted321"]].rename(columns={"Weighted321": "CountryTotalWeighted321"})

    # Compute per year shares from predictions
    year_sums = merged_pred.groupby("Year")["PredWeighted321"].sum().rename("YearPredSum").reset_index()
    merged_pred = merged_pred.merge(year_sums, on="Year", how="left")

    # Avoid divide-by-zero: if a year's sum is 0, assign uniform share among sports seen that year
    def safe_share(row):
        if row["YearPredSum"] > 0:
            return row["PredWeighted321"] / row["YearPredSum"]
        return 0.0

    merged_pred["ShareWithinYear"] = merged_pred.apply(safe_share, axis=1)

    # Bring in actual national totals and allocate
    merged_pred = merged_pred.merge(total_year_small, on="Year", how="left")
    merged_pred["AllocatedContributionToTotal"] = merged_pred["ShareWithinYear"] * merged_pred["CountryTotalWeighted321"]

    # Outputs
    by_year_cols = [
        "Year", "Sport", "Weighted321", "PredWeighted321", "ShareWithinYear", "CountryTotalWeighted321", "AllocatedContributionToTotal"
    ] + feature_cols
    by_year = merged_pred[by_year_cols].sort_values(["Year", "AllocatedContributionToTotal"], ascending=[True, False])

    overall = (
        by_year.groupby("Sport")["AllocatedContributionToTotal"].sum().rename("OverallContributionToTotal").reset_index()
        .sort_values("OverallContributionToTotal", ascending=False)
    )

    # SHAP global weights table
    shap_weights_df = pd.DataFrame({
        "Feature": feature_cols,
        "MeanAbsSHAP": mean_abs,
        "NormalizedWeight": weights,
    }).sort_values("MeanAbsSHAP", ascending=False)

    # Save outputs
    by_year_path = os.path.join(base_dir, f"{output_prefix}_by_year.csv")
    overall_path = os.path.join(base_dir, f"{output_prefix}_overall.csv")
    weights_path = os.path.join(base_dir, f"{output_prefix}_shap_weights.csv")

    by_year.to_csv(by_year_path, index=False)
    overall.to_csv(overall_path, index=False)
    shap_weights_df.to_csv(weights_path, index=False)

    # Optional: SHAP summary plot (skip in headless environments if it errors)
    try:
        import matplotlib.pyplot as plt
        import shap
        shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{output_prefix}_shap_summary.png"), dpi=180)
        plt.close()
    except Exception as e:
        # Silently continue if plotting fails (e.g., no display backend)
        pass

    return {
        "by_year_csv": by_year_path,
        "overall_csv": overall_path,
        "shap_weights_csv": weights_path,
    }


if __name__ == "__main__":
    # Use repository outputs processed_data directory as default data source
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    BASE_DIR = os.path.join(ROOT, 'outputs', 'processed_data')
    results = allocate_total_medals_by_year(BASE_DIR)
    for k, v in results.items():
        print(f"{k}: {v}")
