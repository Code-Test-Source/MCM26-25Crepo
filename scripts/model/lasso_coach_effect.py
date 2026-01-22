"""Per-event Lasso modeling of Olympic medal scores with coach-effect detection.

Steps
- Load cleaned athlete-level data.
- Aggregate to year x Sport_Event level.
- Create lag features (previous medal scores, rolling means, historical max).
- Train Lasso models per Sport_Event with 5-fold CV (or fewer folds if limited rows).
- Evaluate and flag potential "great coach" effects based on large positive residuals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Configuration
# ---------------------------
DATA_PATH = Path(__file__).with_name("athletes_cleaned_chow_ready.csv")
# Residual thresholds for detecting a potential great-coach effect.
COACH_MARGIN = 3  # positive residual threshold for two consecutive Games
COACH_EXTREME = 4  # single-Games extreme positive residual threshold
MIN_EVENT_ROWS = 5  # minimum athlete rows per Sport_Event to be included
OUT_METRICS = Path(__file__).with_name("metrics_per_event.csv")
OUT_PREDS = Path(__file__).with_name("preds_per_event.csv")
OUT_COACH = Path(__file__).with_name("coach_effect_candidates.csv")
OUT_COEFFS = Path(__file__).with_name("coeffs_per_event.csv")


# ---------------------------
# Data preparation helpers
# ---------------------------
def load_and_filter(path: Path) -> pd.DataFrame:
    """Load the CSV and drop Sport_Event groups with < MIN_EVENT_ROWS rows."""
    df = pd.read_csv(path)
    filtered = df.groupby("Sport_Event").filter(lambda x: len(x) >= MIN_EVENT_ROWS)
    return filtered


def aggregate_year_event(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate athlete rows to year x Sport_Event level with medal stats."""
    grouped = (
        df.groupby(["Sport_Event", "Year"], as_index=False)
        .agg(
            total_medal_score=("Medal_Score", "sum"),
            avg_medal_score=("Medal_Score", "mean"),
            athletes=("Name", "count"),
            is_host=("is_host", "max"),
        )
        .sort_values(["Sport_Event", "Year"])
    )
    return grouped


def add_athlete_features(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Add athlete-history features aggregated to Sport_Event x Year."""
    df_sorted = df.sort_values(["Sport_Event", "Name", "Year"]).copy()

    grouped = df_sorted.groupby(["Sport_Event", "Name"])
    df_sorted["prior_participations"] = grouped.cumcount()
    df_sorted["returning_flag"] = df_sorted["prior_participations"] > 0
    # Cumulative Medal_Score up to previous appearances per athlete-event
    df_sorted["hist_medal_score"] = grouped["Medal_Score"].cumsum().shift(1).fillna(0)

    athlete_feats = (
        df_sorted.groupby(["Sport_Event", "Year"], as_index=False)
        .agg(
            returning_athlete_pct=("returning_flag", "mean"),
            avg_athlete_hist_score=("hist_medal_score", "mean"),
        )
        .sort_values(["Sport_Event", "Year"])
    )

    merged = agg.merge(athlete_feats, on=["Sport_Event", "Year"], how="left")
    merged[["returning_athlete_pct", "avg_athlete_hist_score"]] = merged[[
        "returning_athlete_pct",
        "avg_athlete_hist_score",
    ]].fillna(0)
    return merged


def add_time_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Add lag-based time-series features per Sport_Event."""
    def _add_features(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Year").copy()
        g["lag1_total"] = g["total_medal_score"].shift(1)
        # Average of medal totals over the previous two Games
        g["lag2_avg_total"] = g["total_medal_score"].shift(1).rolling(window=2).mean()
        # Previous Games average medal score (athlete-level mean in the prior edition)
        g["lag1_avg_medal"] = g["avg_medal_score"].shift(1)
        # Historical max medal total up to the previous Games
        g["hist_max_total"] = g["total_medal_score"].shift(1).expanding().max()
        g["lag1_returning_pct"] = g["returning_athlete_pct"].shift(1)
        g["lag1_avg_athlete_hist"] = g["avg_athlete_hist_score"].shift(1)
        return g

    featured = agg.groupby("Sport_Event", group_keys=False).apply(_add_features)
    return featured


# ---------------------------
# Modeling and evaluation
# ---------------------------
def train_lasso_per_event(featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train per-event Lasso models and return metrics, predictions, and coefficients."""
    records: List[Dict[str, object]] = []
    preds: List[Dict[str, object]] = []
    coef_records: List[Dict[str, object]] = []

    feature_cols = [
        "lag1_total",
        "lag2_avg_total",
        "hist_max_total",
        "is_host",
        "lag1_avg_medal",
        "lag1_returning_pct",
        "lag1_avg_athlete_hist",
    ]

    for event, g in featured.groupby("Sport_Event"):
        # Drop rows lacking sufficient history for lag features
        g_model = g.dropna(subset=["lag1_total", "lag2_avg_total", "hist_max_total"])
        if len(g_model) < 2:
            continue  # not enough data to fit any model

        X = g_model[feature_cols].fillna(0)
        y = g_model["total_medal_score"]

        cv_folds = max(2, min(5, len(g_model)))
        model = LassoCV(cv=cv_folds, random_state=42)
        model.fit(X, y)

        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        coef_entry = {"Sport_Event": event, "intercept": float(model.intercept_)}
        coef_entry.update({col: float(coef) for col, coef in zip(feature_cols, model.coef_)})
        coef_records.append(coef_entry)

        records.append(
            {
                "Sport_Event": event,
                "n_samples": len(g_model),
                "alpha": model.alpha_,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

        # Store per-row predictions for later coach-effect detection
        for _, row in g_model.assign(pred=y_pred, residual=y - y_pred).iterrows():
            preds.append(
                {
                    "Sport_Event": event,
                    "Year": int(row["Year"]),
                    "actual": float(row["total_medal_score"]),
                    "predicted": float(row["pred"]),
                    "residual": float(row["residual"]),
                }
            )

    metrics_df = pd.DataFrame(records).sort_values("rmse")
    preds_df = pd.DataFrame(preds).sort_values(["Sport_Event", "Year"])
    coefs_df = pd.DataFrame(coef_records).sort_values("Sport_Event")
    return metrics_df, preds_df, coefs_df


# ---------------------------
# Coach-effect detection
# ---------------------------
def detect_great_coach(preds_df: pd.DataFrame, margin: float, extreme: float) -> pd.DataFrame:
    """Identify events/years showing sustained or extreme positive residuals."""
    findings: List[Dict[str, object]] = []

    for event, g in preds_df.groupby("Sport_Event"):
        g = g.sort_values("Year")
        positive = g["residual"] > margin

        # Consecutive two-Games positive residuals
        consec_idx = positive & positive.shift(1, fill_value=False)
        for idx in g[consec_idx].index:
            findings.append(
                {
                    "Sport_Event": event,
                    "Year": int(g.loc[idx, "Year"]),
                    "residual": float(g.loc[idx, "residual"]),
                    "pattern": f">{margin} in 2 consecutive Games",
                }
            )

        # Single extreme positive residuals
        extreme_idx = g["residual"] > extreme
        for idx in g[extreme_idx].index:
            findings.append(
                {
                    "Sport_Event": event,
                    "Year": int(g.loc[idx, "Year"]),
                    "residual": float(g.loc[idx, "residual"]),
                    "pattern": f">{extreme} in one Games",
                }
            )

    result = pd.DataFrame(findings).sort_values(["Sport_Event", "Year"])
    return result


# ---------------------------
# Main pipeline
# ---------------------------
def main() -> None:
    df = load_and_filter(DATA_PATH)
    agg = aggregate_year_event(df)
    agg = add_athlete_features(df, agg)
    featured = add_time_features(agg)

    metrics_df, preds_df, coefs_df = train_lasso_per_event(featured)
    coach_df = detect_great_coach(preds_df, margin=COACH_MARGIN, extreme=COACH_EXTREME)

    # Persist results for easy inspection
    metrics_df.to_csv(OUT_METRICS, index=False)
    preds_df.to_csv(OUT_PREDS, index=False)
    coefs_df.to_csv(OUT_COEFFS, index=False)
    coach_df.to_csv(OUT_COACH, index=False)

    print("=== Metrics per Sport_Event ===")
    if metrics_df.empty:
        print("No events had enough data to train.")
    else:
        print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Lasso coefficients per Sport_Event ===")
    if coefs_df.empty:
        print("No coefficient results produced (insufficient data).")
    else:
        print(coefs_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Great-coach effect candidates ===")
    if coach_df.empty:
        print("No great-coach patterns detected with current thresholds.")
    else:
        print(coach_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
