# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from statsmodels.tsa.arima.model import ARIMA
# import warnings
# import os

# warnings.filterwarnings("ignore")

# # 路径
# base_dir = r"d:\美赛复现\PCA_prediction\data_processing"
# processed_file = os.path.join(base_dir, "summerOly_athletes_processed.csv")
# pca_file = os.path.join(base_dir, "summerOly_pca_features.csv")
# output_file = os.path.join(base_dir, "medal_predictions_2024.csv")

# # 1. 加载数据
# medals_df = pd.read_csv(processed_file)
# pca_df = pd.read_csv(pca_file)

# # 2. 数据预处理
# # 将 ROC (2020) 映射回 RUS，以便保持时间序列连续性
# medals_df['Mapped_NOC'] = medals_df['Mapped_NOC'].replace({'ROC': 'RUS'})
# pca_df['Mapped_NOC'] = pca_df['Mapped_NOC'].replace({'ROC': 'RUS'})

# # 聚合每个国家每年的这三样数据: Gold, Total, is_host
# # 注意: is_host 在同一年同一国家应该是唯一的
# agg_df = medals_df.groupby(['Year', 'Mapped_NOC']).agg({
#     'Gold': 'sum',
#     'Total': 'sum',
#     'is_host': 'max'
# }).reset_index()

# # 合并 PCA 特征
# # PCA 特征已经在 pca_sport_reduction.py 中按 Year-Country 生成
# data = pd.merge(agg_df, pca_df, on=['Year', 'Mapped_NOC', 'is_host'], how='left')

# # 填充 PCA 特征的 NaN (如果有国家某年没参加某些项目导致缺失，不过通常 PCA 不会产生 NaN 除非 pivot 导致)
# data = data.fillna(0)

# # 3. 特征工程: 构建滞后特征 (Lag features)
# # 我们需要用 T-4 的特征来预测 T
# # 需要对每个国家单独 sorted by Year
# data = data.sort_values(['Mapped_NOC', 'Year'])

# # 获取所有PCA列名
# pca_cols = [c for c in data.columns if 'Sport_PC_' in c]

# # 创建滞后列
# for col in ['Gold', 'Total'] + pca_cols:
#     data[f'Prev_{col}'] = data.groupby('Mapped_NOC')[col].shift(1)

# # 删除没有前一届数据的行 (即每个国家的第一次参赛记录无法用于训练)
# # 但对于 2024 预测，我们需要 2020 的数据，这在表中是存在的
# # 只需要 dropna 训练集
# train_data = data.dropna(subset=[f'Prev_Total'])

# # 4. ARIMA 预测函数
# def get_arima_forecast(series, next_step_idx):
#     # series: 历史数据列表
#     # 如果数据太少，用均值或最后一次
#     if len(series) < 3:
#         return series[-1]
    
#     try:
#         # 简单 ARIMA(1,1,0) 或者 (1,0,0)
#         # 很多时候数据就 5-6 个点，(0,1,0) 是随机游走
#         model = ARIMA(series, order=(1, 1, 0))
#         model_fit = model.fit()
#         forecast = model_fit.forecast(steps=1)
#         return max(0, forecast[0]) # 奖牌不能为负
#     except:
#         return series[-1]

# # 5. 准备训练集和测试集 (或预测集)
# # 训练集: 2004 - 2020 (基于前一届预测当前届)
# # 预测目标: 2024
# # 我们需要构建一个 2024 的输入行

# # 筛选出有资格预测 2024 的国家 (即 2020 参加了的国家，以及俄罗斯)
# countries_2020 = data[data['Year'] == 2020]['Mapped_NOC'].unique()
# target_countries = list(countries_2020)
# if 'RUS' not in target_countries:
#     target_countries.append('RUS')

# # 构建 2024 的输入特征
# pred_rows = []
# for noc in target_countries:
#     # 获取该国最新的一行数据 (应该是 2020)
#     last_row = data[data['Mapped_NOC'] == noc].iloc[-1]
    
#     # 构造 2024 行
#     row_2024 = {
#         'Year': 2024,
#         'Mapped_NOC': noc,
#         'is_host': 1 if noc == 'FRA' else 0, # 法国是 2024 东道主
#     }
    
#     # 填充 Lag 特征 (即 2020 的真实值)
#     row_2024['Prev_Gold'] = last_row['Gold']
#     row_2024['Prev_Total'] = last_row['Total']
#     for col in pca_cols:
#         row_2024[f'Prev_{col}'] = last_row[col]
        
#     # ARIMA 特征: 基于历史所有年份预测 2024
#     history_gold = data[data['Mapped_NOC'] == noc]['Gold'].tolist()
#     history_total = data[data['Mapped_NOC'] == noc]['Total'].tolist()
    
#     row_2024['ARIMA_Gold'] = get_arima_forecast(history_gold, 2024)
#     row_2024['ARIMA_Total'] = get_arima_forecast(history_total, 2024)
    
#     pred_rows.append(row_2024)

# pred_df_2024 = pd.DataFrame(pred_rows)

# # 同时也需要给训练集加上 ARIMA 特征 (Validating on past)
# # 这很慢，所以只对训练数据做
# print("正在生成训练集 ARIMA 特征 (这可能需要一点时间)...")
# arima_gold_train = []
# arima_total_train = []

# for idx, row in train_data.iterrows():
#     noc = row['Mapped_NOC']
#     year = row['Year']
#     # 获取该年之前的所有历史数据 (不包含该年，防止泄露)
#     history = data[(data['Mapped_NOC'] == noc) & (data['Year'] < year)]
    
#     if len(history) > 2:
#         g_pred = get_arima_forecast(history['Gold'].tolist(), year)
#         t_pred = get_arima_forecast(history['Total'].tolist(), year)
#     else:
#         # 数据不足时用前一届的值 (即 Prev_Gold) 代替 ARIMA 预测
#         g_pred = row['Prev_Gold']
#         t_pred = row['Prev_Total']
        
#     arima_gold_train.append(g_pred)
#     arima_total_train.append(t_pred)

# train_data['ARIMA_Gold'] = arima_gold_train
# train_data['ARIMA_Total'] = arima_total_train

# # 6. 训练模型 & 预测
# # 特征列
# features = ['is_host', 'Prev_Gold', 'Prev_Total', 'ARIMA_Gold', 'ARIMA_Total'] + [f'Prev_{c}' for c in pca_cols]

# # 预测 Gold
# print("训练 Gold 模型...")
# rf_gold = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_gold.fit(train_data[features], train_data['Gold'])
# pred_df_2024['Predicted_Gold'] = rf_gold.predict(pred_df_2024[features])

# # 预测 Total
# print("训练 Total 模型...")
# rf_total = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_total.fit(train_data[features], train_data['Total'])
# pred_df_2024['Predicted_Total'] = rf_total.predict(pred_df_2024[features])

# # 7. 后处理和保存
# # 取整
# pred_df_2024['Predicted_Gold'] = pred_df_2024['Predicted_Gold'].round().astype(int)
# pred_df_2024['Predicted_Total'] = pred_df_2024['Predicted_Total'].round().astype(int)

# # 排序
# result = pred_df_2024[['Mapped_NOC', 'Predicted_Gold', 'Predicted_Total']].sort_values('Predicted_Total', ascending=False)

# # 保存
# result.to_csv(output_file, index=False)

# print("\n=== 2024 奖牌榜预测前 20 名 ===")


# print(result.head(20).reset_index(drop=True).assign(Rank=lambda df: df.index+1)[['Rank','Mapped_NOC','Predicted_Gold','Predicted_Total']].to_string(index=False))

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

warnings.filterwarnings("ignore")

# Path configuration
base_dir = r"./outputs/processed_data/"  # Keep system path unchanged
processed_file = os.path.join(base_dir, "summerOly_athletes_processed.csv")
pca_file = os.path.join(base_dir, "summerOly_pca_features.csv")
output_file_2024 = os.path.join(base_dir, "medal_predictions_2024.csv")
output_file_2028 = os.path.join(base_dir, "medal_predictions_2028.csv")

# 1. Load data
medals_df = pd.read_csv(processed_file)
pca_df = pd.read_csv(pca_file)

# 2. Data preprocessing
# Map ROC (2020) back to RUS to maintain time series continuity
medals_df['Mapped_NOC'] = medals_df['Mapped_NOC'].replace({'ROC': 'RUS'})
pca_df['Mapped_NOC'] = pca_df['Mapped_NOC'].replace({'ROC': 'RUS'})

# Aggregate three metrics (Gold, Total, is_host) for each country per year
# Note: is_host should be unique for the same country in the same year
agg_df = medals_df.groupby(['Year', 'Mapped_NOC']).agg({
    'Gold': 'sum',
    'Total': 'sum',
    'is_host': 'max'
}).reset_index()

# Merge PCA features
# PCA features have been generated by Year-Country in pca_sport_reduction.py
data = pd.merge(agg_df, pca_df, on=['Year', 'Mapped_NOC', 'is_host'], how='left')

# Fill NaN values in PCA features (might occur if a country didn't participate in some sports in a year, but PCA usually doesn't generate NaN unless caused by pivot)
data = data.fillna(0)

# 3. Feature engineering: Build lag features
# We use T-4 features to predict T (Olympic Games are held every 4 years)
# Need to sort by Year for each country individually
data = data.sort_values(['Mapped_NOC', 'Year'])

# Get all PCA column names
pca_cols = [c for c in data.columns if 'Sport_PC_' in c]

# Create lag columns
for col in ['Gold', 'Total'] + pca_cols:
    data[f'Prev_{col}'] = data.groupby('Mapped_NOC')[col].shift(1)

# Drop rows without previous session data (i.e., first participation record of each country can't be used for training)
# But for 2024 prediction, we need 2020 data which exists in the table
# Only dropna for training set
train_data = data.dropna(subset=[f'Prev_Total'])

# 4. ARIMA prediction function
def get_arima_forecast(series, next_step_idx):
    # series: list of historical data
    # Use mean or last value if data is insufficient
    if len(series) < 3:
        return series[-1]
    
    try:
        # Simple ARIMA(1,1,0) or (1,0,0)
        # Often only 5-6 data points available, (0,1,0) is random walk
        model = ARIMA(series, order=(1, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return max(0, forecast[0]) # Medals cannot be negative
    except:
        return series[-1]

# 5. Prepare training set and test set (or prediction set)
# Training set: 2004 - 2020 (predict current session based on previous session)
# Prediction target: 2024
# We need to build input rows for 2024

# Filter countries eligible for 2020 prediction (countries that participated in 2020, plus Russia)
countries_2020 = data[data['Year'] == 2020]['Mapped_NOC'].unique()
target_countries = list(countries_2020)
if 'RUS' not in target_countries:
    target_countries.append('RUS')

# Build input features for 2024
pred_rows = []
for noc in target_countries:
    # Get the latest row of data for this country (should be 2020)
    last_row = data[data['Mapped_NOC'] == noc].iloc[-1]
    
    # Construct 2024 row
    row_2024 = {
        'Year': 2024,
        'Mapped_NOC': noc,
        'is_host': 1 if noc == 'FRA' else 0, # France is the host of 2024 Olympics
    }
    
    # Fill Lag features (i.e., actual values from 2020)
    row_2024['Prev_Gold'] = last_row['Gold']
    row_2024['Prev_Total'] = last_row['Total']
    for col in pca_cols:
        row_2024[f'Prev_{col}'] = last_row[col]
        
    # ARIMA features: predict 2024 based on all historical years
    history_gold = data[data['Mapped_NOC'] == noc]['Gold'].tolist()
    history_total = data[data['Mapped_NOC'] == noc]['Total'].tolist()
    
    row_2024['ARIMA_Gold'] = get_arima_forecast(history_gold, 2024)
    row_2024['ARIMA_Total'] = get_arima_forecast(history_total, 2024)
    
    pred_rows.append(row_2024)

pred_df_2024 = pd.DataFrame(pred_rows)

# Also add ARIMA features to training set (Validating on past)
# This is slow, so only do it for training data
print("Generating ARIMA features for training set (this may take a moment)...")
arima_gold_train = []
arima_total_train = []

for idx, row in train_data.iterrows():
    noc = row['Mapped_NOC']
    year = row['Year']
    # Get all historical data before this year (exclude current year to prevent data leakage)
    history = data[(data['Mapped_NOC'] == noc) & (data['Year'] < year)]
    
    if len(history) > 2:
        g_pred = get_arima_forecast(history['Gold'].tolist(), year)
        t_pred = get_arima_forecast(history['Total'].tolist(), year)
    else:
        # Use previous session value (i.e., Prev_Gold) instead of ARIMA prediction when data is insufficient
        g_pred = row['Prev_Gold']
        t_pred = row['Prev_Total']
        
    arima_gold_train.append(g_pred)
    arima_total_train.append(t_pred)

train_data['ARIMA_Gold'] = arima_gold_train
train_data['ARIMA_Total'] = arima_total_train

# 6. Train models & make predictions
# Feature columns
features = ['is_host', 'Prev_Gold', 'Prev_Total', 'ARIMA_Gold', 'ARIMA_Total'] + [f'Prev_{c}' for c in pca_cols]

# Predict Gold medals
print("Training Gold medal prediction model...")
xgb_gold = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_gold.fit(train_data[features], train_data['Gold'])
pred_df_2024['Predicted_Gold'] = xgb_gold.predict(pred_df_2024[features])

# Predict Total medals
print("Training Total medal prediction model...")
xgb_total = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_total.fit(train_data[features], train_data['Total'])
pred_df_2024['Predicted_Total'] = xgb_total.predict(pred_df_2024[features])

# 7. Post-processing and saving results
# Round to integer
pred_df_2024['Predicted_Gold'] = pred_df_2024['Predicted_Gold'].round().astype(int)
pred_df_2024['Predicted_Total'] = pred_df_2024['Predicted_Total'].round().astype(int)

# Sort results
result = pred_df_2024[['Mapped_NOC', 'Predicted_Gold', 'Predicted_Total']].sort_values('Predicted_Total', ascending=False)

# Save results
result.to_csv(output_file_2024, index=False)

print("\n=== Top 20 Predicted Medal Rankings for 2024 ===")

print(result.head(20).reset_index(drop=True).assign(Rank=lambda df: df.index+1)[['Rank','Mapped_NOC','Predicted_Gold','Predicted_Total']].to_string(index=False))

# --- 2028 prediction ---
print("\nIntegrating provided 2024 results and retraining for 2028 prediction...")

# Load user-provided 2024 results
olympics_file = os.path.join(base_dir, "olympics_medals_with_host_tag.csv")
if os.path.exists(olympics_file):
    real_df = pd.read_csv(olympics_file)
    # normalize column names to match our data
    if 'NOC' in real_df.columns:
        real_df = real_df.rename(columns={'NOC': 'Mapped_NOC'})
    real_2024 = real_df[real_df['Year'] == 2024].copy()
    # map ROC -> RUS if present
    if 'Mapped_NOC' in real_2024.columns:
        real_2024['Mapped_NOC'] = real_2024['Mapped_NOC'].replace({'ROC': 'RUS'})
else:
    real_2024 = pd.DataFrame(columns=['Mapped_NOC', 'Year', 'Gold', 'Total', 'is_host'])

# Ensure we have model-predicted Russia for 2024 if real data lacks it
pred_map = pred_df_2024.set_index('Mapped_NOC')[['Predicted_Gold', 'Predicted_Total']]
if 'RUS' not in real_2024.get('Mapped_NOC', []).values and 'RUS' in pred_map.index:
    rus_row = {
        'Mapped_NOC': 'RUS',
        'Year': 2024,
        'Gold': int(pred_map.loc['RUS', 'Predicted_Gold']),
        'Total': int(pred_map.loc['RUS', 'Predicted_Total']),
        'is_host': 0
    }
    real_2024 = pd.concat([real_2024, pd.DataFrame([rus_row])], ignore_index=True)

# Drop any existing 2024 rows from main `data` and replace with provided real 2024 rows
data = data[data['Year'] != 2024]
new_rows = []
for _, r in real_2024.iterrows():
    noc = r['Mapped_NOC']
    # find latest available PCA row for this country (likely 2020)
    prior = data[data['Mapped_NOC'] == noc]
    if not prior.empty:
        last_row = prior.iloc[-1]
        pc_vals = {c: last_row[c] for c in pca_cols}
    else:
        pc_vals = {c: 0 for c in pca_cols}

    new_row = {
        'Year': 2024,
        'Mapped_NOC': noc,
        'Gold': int(r['Gold']),
        'Total': int(r['Total']),
        'is_host': int(r.get('is_host', 0)),
    }
    for c, v in pc_vals.items():
        new_row[c] = v
    new_rows.append(new_row)

if new_rows:
    data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True, sort=False)

# Re-sort and rebuild lag features now that 2024 is included
data = data.sort_values(['Mapped_NOC', 'Year'])
for col in ['Gold', 'Total'] + pca_cols:
    data[f'Prev_{col}'] = data.groupby('Mapped_NOC')[col].shift(1)

# Build training data including 2024 to train models for predicting 2028
train_data_2028 = data.dropna(subset=['Prev_Total']).copy()

print("Generating ARIMA features for 2028 training set (may take a moment)...")
arima_gold_train = []
arima_total_train = []
for idx, row in train_data_2028.iterrows():
    noc = row['Mapped_NOC']
    year = row['Year']
    history = data[(data['Mapped_NOC'] == noc) & (data['Year'] < year)]
    if len(history) > 2:
        g_pred = get_arima_forecast(history['Gold'].tolist(), year)
        t_pred = get_arima_forecast(history['Total'].tolist(), year)
    else:
        g_pred = row['Prev_Gold']
        t_pred = row['Prev_Total']
    arima_gold_train.append(g_pred)
    arima_total_train.append(t_pred)

train_data_2028['ARIMA_Gold'] = arima_gold_train
train_data_2028['ARIMA_Total'] = arima_total_train

# Retrain XGBoost models using data up to 2024
print("Retraining XGBoost models using data through 2024...")
xgb_gold_2028 = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_total_2028 = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_gold_2028.fit(train_data_2028[features], train_data_2028['Gold'])
xgb_total_2028.fit(train_data_2028[features], train_data_2028['Total'])

# Build 2028 prediction inputs using real 2024 as Prev_*
pred_rows_2028 = []
for noc in target_countries:
    # take the country's 2024 row if present, else latest available
    rows = data[(data['Mapped_NOC'] == noc)]
    if rows.empty:
        continue
    last_row = rows[rows['Year'] == 2024]
    if last_row.empty:
        last_row = rows.iloc[-1]
    else:
        last_row = last_row.iloc[0]

    row_2028 = {
        'Year': 2028,
        'Mapped_NOC': noc,
        'is_host': 1 if noc == 'USA' else 0,
    }
    row_2028['Prev_Gold'] = last_row['Gold']
    row_2028['Prev_Total'] = last_row['Total']
    for col in pca_cols:
        row_2028[f'Prev_{col}'] = last_row[col]

    # ARIMA feature using history including 2024
    history_gold = data[data['Mapped_NOC'] == noc]['Gold'].tolist()
    history_total = data[data['Mapped_NOC'] == noc]['Total'].tolist()
    row_2028['ARIMA_Gold'] = get_arima_forecast(history_gold, 2028)
    row_2028['ARIMA_Total'] = get_arima_forecast(history_total, 2028)

    pred_rows_2028.append(row_2028)

pred_df_2028 = pd.DataFrame(pred_rows_2028)

print("Predicting 2028 medals with retrained XGBoost models...")
pred_df_2028['Predicted_Gold'] = xgb_gold_2028.predict(pred_df_2028[features])
pred_df_2028['Predicted_Total'] = xgb_total_2028.predict(pred_df_2028[features])

# Post-process and save 2028 results
pred_df_2028['Predicted_Gold'] = pred_df_2028['Predicted_Gold'].clip(lower=0).round().astype(int)
pred_df_2028['Predicted_Total'] = pred_df_2028['Predicted_Total'].clip(lower=0).round().astype(int)

result_2028 = pred_df_2028[['Mapped_NOC', 'Predicted_Gold', 'Predicted_Total']].sort_values('Predicted_Total', ascending=False)
result_2028.to_csv(output_file_2028, index=False)

print("\n=== Top 20 Predicted Medal Rankings for 2028 (using real 2024 data) ===")
print(result_2028.head(20).reset_index(drop=True).assign(Rank=lambda df: df.index+1)[['Rank','Mapped_NOC','Predicted_Gold','Predicted_Total']].to_string(index=False))