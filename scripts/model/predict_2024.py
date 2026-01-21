import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

warnings.filterwarnings("ignore")

# Paths
processed_file = os.path.join("summerOly_athletes_processed.csv")
pca_file = os.path.join("summerOly_pca_features.csv")
output_file_no_host = os.path.join("medal_predictions_2024_no_host.csv")
output_file_with_host = os.path.join("medal_predictions_2024_with_host.csv")

# 1. Load data
medals_df = pd.read_csv(processed_file)
pca_df = pd.read_csv(pca_file)

# 2. Preprocess
# Map ROC (2020) back to RUS to keep continuity in time series
medals_df['Mapped_NOC'] = medals_df['Mapped_NOC'].replace({'ROC': 'RUS'})
pca_df['Mapped_NOC'] = pca_df['Mapped_NOC'].replace({'ROC': 'RUS'})

# Aggregate per-country per-year metrics: Gold, Total, is_host
# Note: is_host should be unique for each country-year
agg_df = medals_df.groupby(['Year', 'Mapped_NOC']).agg({
    'Gold': 'sum',
    'Total': 'sum',
    'is_host': 'max'
}).reset_index()

# Merge PCA features (built per Year-Country in pca_sport_reduction.py)
data = pd.merge(agg_df, pca_df, on=['Year', 'Mapped_NOC', 'is_host'], how='left')

# Fill PCA NaNs (rare; would come from pivot gaps)
data = data.fillna(0)

# 3. Feature engineering: lag features
# Use previous edition (T-4) to predict current edition (T)
# Sort by country and year
data = data.sort_values(['Mapped_NOC', 'Year'])

# Collect PCA column names
pca_cols = [c for c in data.columns if 'Sport_PC_' in c]

# Build lag columns
for col in ['Gold', 'Total'] + pca_cols:
    data[f'Prev_{col}'] = data.groupby('Mapped_NOC')[col].shift(1)

# Drop rows without previous edition (first appearance per NOC not usable for training)
# For 2024 prediction we need 2020 data, which exists in the table
# Only dropna on training set
train_data = data.dropna(subset=[f'Prev_Total'])

# 4. Prepare train and prediction sets
# Train: 2004-2020 (predict current edition from previous)
# Target: 2024 (build an input row per country)

# Countries eligible for 2024 (participated in 2020 plus RUS)
countries_2020 = data[data['Year'] == 2020]['Mapped_NOC'].unique()
target_countries = list(countries_2020)
if 'RUS' not in target_countries:
    target_countries.append('RUS')

############################################
# Logic:
# - Non-host baseline: Random Forest on previous edition and PCA lags
# - Host effect: apply historical host average uplift additively
############################################

# 1) Compute historical host average absolute uplift (host edition minus prior non-host mean)
host_rows = data[data['is_host'] == 1]
uplift_gold_deltas, uplift_total_deltas = [], []
for _, row in host_rows.iterrows():
    noc = row['Mapped_NOC']
    year = row['Year']
    history_non_host = data[(data['Mapped_NOC'] == noc) & (data['Year'] < year) & (data['is_host'] == 0)]
    if history_non_host.empty:
        continue
    avg_gold = history_non_host['Gold'].mean()
    avg_total = history_non_host['Total'].mean()
    uplift_gold_deltas.append(row['Gold'] - avg_gold)
    uplift_total_deltas.append(row['Total'] - avg_total)

mean_uplift_gold = float(np.mean(uplift_gold_deltas)) if len(uplift_gold_deltas) > 0 else 0.0
mean_uplift_total = float(np.mean(uplift_total_deltas)) if len(uplift_total_deltas) > 0 else 0.0

print(f"Historical host average uplift Gold: {mean_uplift_gold:.3f}, Total: {mean_uplift_total:.3f}")

# 2) Train non-host baseline models
train_non_host = train_data[train_data['is_host'] == 0].copy()
base_features = ['Prev_Gold', 'Prev_Total'] + [f'Prev_{c}' for c in pca_cols]

print("Training baseline Gold model (non-host)...")
rf_gold = RandomForestRegressor(n_estimators=200, random_state=42)
rf_gold.fit(train_non_host[base_features], train_non_host['Gold'])

print("Training baseline Total model (non-host)...")
rf_total = RandomForestRegressor(n_estimators=200, random_state=42)
rf_total.fit(train_non_host[base_features], train_non_host['Total'])

# 3) Generate 2024 baseline predictions (non-host), then add host uplift for FRA
pred_rows_base, pred_rows_with_host = [], []

def rf_pred_with_ci(model, feat_vec, lower_q=0.025, upper_q=0.975):
    tree_preds = np.array([est.predict([feat_vec])[0] for est in model.estimators_])
    lower = float(np.quantile(tree_preds, lower_q))
    upper = float(np.quantile(tree_preds, upper_q))
    pred = float(model.predict([feat_vec])[0])
    return pred, lower, upper
for noc in target_countries:
    last_row = data[data['Mapped_NOC'] == noc].iloc[-1]
    feat_vec = [last_row['Prev_Gold'], last_row['Prev_Total']] + [last_row[f'Prev_{c}'] for c in pca_cols]

    base_gold, base_gold_l, base_gold_u = rf_pred_with_ci(rf_gold, feat_vec)
    base_total, base_total_l, base_total_u = rf_pred_with_ci(rf_total, feat_vec)

    pred_rows_base.append({
        'Mapped_NOC': noc,
        'Predicted_Gold': base_gold,
        'Predicted_Gold_Lower95': base_gold_l,
        'Predicted_Gold_Upper95': base_gold_u,
        'Predicted_Total': base_total,
        'Predicted_Total_Lower95': base_total_l,
        'Predicted_Total_Upper95': base_total_u
    })

    # Apply average uplift additively for FRA as 2024 host
    if noc == 'FRA':
        adj_gold = base_gold + mean_uplift_gold
        adj_gold_l = base_gold_l + mean_uplift_gold
        adj_gold_u = base_gold_u + mean_uplift_gold
        adj_total = base_total + mean_uplift_total
        adj_total_l = base_total_l + mean_uplift_total
        adj_total_u = base_total_u + mean_uplift_total
    else:
        adj_gold, adj_gold_l, adj_gold_u = base_gold, base_gold_l, base_gold_u
        adj_total, adj_total_l, adj_total_u = base_total, base_total_l, base_total_u

    pred_rows_with_host.append({
        'Mapped_NOC': noc,
        'Predicted_Gold': adj_gold,
        'Predicted_Gold_Lower95': adj_gold_l,
        'Predicted_Gold_Upper95': adj_gold_u,
        'Predicted_Total': adj_total,
        'Predicted_Total_Lower95': adj_total_l,
        'Predicted_Total_Upper95': adj_total_u
    })

pred_df_2024_base = pd.DataFrame(pred_rows_base)
pred_df_2024_with_host = pd.DataFrame(pred_rows_with_host)

# 4) Postprocess and save (two outputs)
for df in [pred_df_2024_base, pred_df_2024_with_host]:
    for col in [
        'Predicted_Gold', 'Predicted_Gold_Lower95', 'Predicted_Gold_Upper95',
        'Predicted_Total', 'Predicted_Total_Lower95', 'Predicted_Total_Upper95'
    ]:
        df[col] = df[col].round()
        df[col] = df[col].clip(lower=0).astype(int)

result_no_host = pred_df_2024_base[
    ['Mapped_NOC', 'Predicted_Gold', 'Predicted_Gold_Lower95', 'Predicted_Gold_Upper95',
     'Predicted_Total', 'Predicted_Total_Lower95', 'Predicted_Total_Upper95']
].sort_values('Predicted_Total', ascending=False)

result_with_host = pred_df_2024_with_host[
    ['Mapped_NOC', 'Predicted_Gold', 'Predicted_Gold_Lower95', 'Predicted_Gold_Upper95',
     'Predicted_Total', 'Predicted_Total_Lower95', 'Predicted_Total_Upper95']
].sort_values('Predicted_Total', ascending=False)

result_no_host.to_csv(output_file_no_host, index=False)
result_with_host.to_csv(output_file_with_host, index=False)

print("\n=== 2024 Medal Table Top 20 (with host effect) ===")
print(
    result_with_host.head(20)
    .reset_index(drop=True)
    .assign(Rank=lambda df: df.index+1)
    [['Rank','Mapped_NOC','Predicted_Gold','Predicted_Gold_Lower95','Predicted_Gold_Upper95','Predicted_Total','Predicted_Total_Lower95','Predicted_Total_Upper95']]
    .to_string(index=False)
)

print("\n=== 2024 Medal Table Top 20 (without host effect) ===")
print(
    result_no_host.head(20)
    .reset_index(drop=True)
    .assign(Rank=lambda df: df.index+1)
    [['Rank','Mapped_NOC','Predicted_Gold','Predicted_Gold_Lower95','Predicted_Gold_Upper95','Predicted_Total','Predicted_Total_Lower95','Predicted_Total_Upper95']]
    .to_string(index=False)
)

