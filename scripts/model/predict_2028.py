import os
import warnings
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import re

warnings.filterwarnings("ignore")

# Paths
base_dir = r"./outputs/processed_data/"
athletes_file = os.path.join(base_dir, "summerOly_athletes_processed.csv")
pca_file = os.path.join(base_dir, "summerOly_pca_features.csv")
real_2024_file = os.path.join(base_dir, "olympics_medals_with_host_tag.csv")
output_2024_pred = os.path.join(base_dir, "medal_predictions_2024_modelpred.csv")
output_2028 = os.path.join(base_dir, "medal_predictions_2028.csv")

# Helpers
def get_arima_forecast(series):
    if len(series) < 3:
        return series[-1] if len(series) > 0 else 0
    try:
        model = ARIMA(series, order=(1,1,0))
        fit = model.fit()
        return max(0, fit.forecast(steps=1)[0])
    except Exception:
        return series[-1]

# Load PCA
pca_df = pd.read_csv(pca_file)
# Normalize PCA ROC -> RUS
if 'Mapped_NOC' in pca_df.columns:
    pca_df['Mapped_NOC'] = pca_df['Mapped_NOC'].replace({'ROC': 'RUS'})

def normalize_country_name(s):
    """Normalize country names for mapping: lowercase, strip, remove team suffixes like ' 1', '(Team)', trailing digits."""
    if not isinstance(s, str):
        return s
    s2 = s.strip().lower()
    # remove trailing parenthetical groups e.g. ' (Team)'
    s2 = re.sub(r"\s*\(.*\)$", "", s2)
    # remove trailing numeric team suffixes like ' 1' or ' 2'
    s2 = re.sub(r"\s+\d+$", "", s2)
    # remove trailing ' team' token
    s2 = re.sub(r"\s+team$", "", s2)
    # collapse multiple spaces
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

# (Mapping helpers defined below; mapping will be derived when needed)

def map_value_to_mapped_noc(noc_in=None, country_name=None, year=None, gold=None, total=None):
    # prefer explicit 3-letter
    if isinstance(noc_in, str):
        s = noc_in.strip()
        if len(s)==3 and s.isupper():
            return s
        if s.upper()=='ROC':
            return 'RUS'
    if isinstance(country_name, str):
        key = normalize_country_name(country_name)
        if key in name_map:
            return name_map[key]
    try:
        if year is not None and gold is not None and total is not None:
            candidates = agg_lookup[(agg_lookup['Year']==int(year))&(agg_lookup['Gold']==int(gold))&(agg_lookup['Total']==int(total))]
            if len(candidates)==1:
                return candidates.iloc[0]['Mapped_NOC']
    except Exception:
        pass
    if isinstance(noc_in, str) and pd.notna(noc_in):
        return noc_in
    if isinstance(country_name, str) and pd.notna(country_name):
        return country_name
    return None

# We'll derive a mapping from the provided datasets (no external country-codes file)
def derive_name_to_mapped_noc(athletes_path, pca_df, raw_counts_path=None):
    # Build aggregated medal table from athletes (has Mapped_NOC)
    agg = pd.read_csv(athletes_path).groupby(['Year','Mapped_NOC']).agg({'Gold':'sum','Total':'sum'}).reset_index()
    name_map = {}
    # If a provided raw counts file exists, use it to match by (Year,Gold,Total) -> Mapped_NOC
    if raw_counts_path and os.path.exists(raw_counts_path):
        raw = pd.read_csv(raw_counts_path)
        # try common column names
        for col in ['country_name','NOC','country']:
            if col in raw.columns:
                name_col = col
                break
        else:
            name_col = None
        for _, r in raw.iterrows():
            try:
                y = int(r.get('Year',0))
            except Exception:
                continue
            g = r.get('Gold', None)
            t = r.get('Total', None)
            if pd.isna(g) or pd.isna(t):
                continue
            candidates = agg[(agg['Year']==y) & (agg['Gold']==int(g)) & (agg['Total']==int(t))]
            if len(candidates)==1 and name_col:
                mapped = candidates.iloc[0]['Mapped_NOC']
                raw_name = str(r.get(name_col)).strip()
                key1 = raw_name.lower()
                key2 = normalize_country_name(raw_name)
                name_map[key1] = mapped
                name_map[key2] = mapped

    codes = set(pca_df['Mapped_NOC'].unique())
    return name_map, codes

# Load medals data: prefer user-provided aggregated file if present (use as full historical dataset), otherwise fall back to athletes file and aggregate
if os.path.exists(real_2024_file):
    raw_medals = pd.read_csv(real_2024_file)
    # normalize column names
    if 'NOC' in raw_medals.columns:
        raw_medals = raw_medals.rename(columns={'NOC': 'NOC_IN'})
    if 'country_name' in raw_medals.columns and 'NOC_IN' not in raw_medals.columns:
        raw_medals = raw_medals.rename(columns={'country_name': 'country_name_in'})

    # Derive mapping from provided datasets and map rows to `Mapped_NOC` (3-letter codes)
    # Try to infer mapping by matching Year/Gold/Total against aggregated athletes data
    name_map, available_codes = derive_name_to_mapped_noc(athletes_file, pca_df,
                                                         raw_counts_path=os.path.join('2025_Problem_C_Data', 'summerOly_medal_counts.csv'))
    agg_lookup = pd.read_csv(athletes_file).groupby(['Year','Mapped_NOC']).agg({'Gold':'sum','Total':'sum'}).reset_index()

    def _map_row_to_mapped_noc(r):
        # prefer explicit 3-letter if present
        if 'NOC_IN' in r and isinstance(r['NOC_IN'], str):
            s = r['NOC_IN'].strip()
            if len(s)==3 and s.isupper():
                return s
            if s.upper()=='ROC':
                return 'RUS'
        # try name_map from earlier derivation
        if 'country_name_in' in r and isinstance(r['country_name_in'], str):
            key = r['country_name_in'].strip().lower()
            if key in name_map:
                return name_map[key]
        # fallback: match by Year/Gold/Total in aggregated athletes data
        try:
            y = int(r.get('Year',0))
            g = int(r.get('Gold',-1))
            t = int(r.get('Total',-1))
            candidates = agg_lookup[(agg_lookup['Year']==y)&(agg_lookup['Gold']==g)&(agg_lookup['Total']==t)]
            if len(candidates)==1:
                return candidates.iloc[0]['Mapped_NOC']
        except Exception:
            pass
        # final fallback: return provided NOC_IN or country_name (unchanged)
        if 'NOC_IN' in r and pd.notna(r['NOC_IN']):
            return r['NOC_IN']
        if 'country_name_in' in r and pd.notna(r['country_name_in']):
            return r['country_name_in']
        return None

    raw_medals['Mapped_NOC'] = raw_medals.apply(_map_row_to_mapped_noc, axis=1)
    # Normalize ROC -> RUS in mapped codes
    raw_medals['Mapped_NOC'] = raw_medals['Mapped_NOC'].replace({'ROC': 'RUS'})

    # Ensure required columns exist
    required = ['Year','Mapped_NOC','Gold','Total','is_host']
    missing = [c for c in required if c not in raw_medals.columns]
    if missing:
        # fall back to aggregating athletes file if provided file doesn't have needed columns
        medals_df = pd.read_csv(athletes_file)
        if 'Mapped_NOC' in medals_df.columns:
            medals_df['Mapped_NOC'] = medals_df['Mapped_NOC'].replace({'ROC':'RUS'})
        agg = medals_df.groupby(['Year','Mapped_NOC']).agg({'Gold':'sum','Total':'sum','is_host':'max'}).reset_index()
        data = pd.merge(agg, pca_df, on=['Year','Mapped_NOC','is_host'], how='left').fillna(0)
    else:
        medals_df = raw_medals[required].copy()
        data = pd.merge(medals_df, pca_df, on=['Year','Mapped_NOC','is_host'], how='left').fillna(0)
else:
    medals_df = pd.read_csv(athletes_file)
    # Normalize ROC -> RUS
    if 'Mapped_NOC' in medals_df.columns:
        medals_df['Mapped_NOC'] = medals_df['Mapped_NOC'].replace({'ROC': 'RUS'})
    # Aggregate medals by Year & NOC
    agg = medals_df.groupby(['Year','Mapped_NOC']).agg({'Gold':'sum','Total':'sum','is_host':'max'}).reset_index()
    # Merge PCA
    data = pd.merge(agg, pca_df, on=['Year','Mapped_NOC','is_host'], how='left').fillna(0)

# Build PCA column list
pca_cols = [c for c in data.columns if c.startswith('Sport_PC_')]

# Create lag features (Prev_*)
data = data.sort_values(['Mapped_NOC','Year'])
for col in ['Gold','Total'] + pca_cols:
    data[f'Prev_{col}'] = data.groupby('Mapped_NOC')[col].shift(1)



# FUNCTION: train on provided years and return trained XGB models
def train_models(train_df, feature_cols):
    X = train_df[feature_cols]
    y_gold = train_df['Gold']
    y_total = train_df['Total']
    model_g = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model_t = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model_g.fit(X, y_gold)
    model_t.fit(X, y_total)
    return model_g, model_t

# 1) Predict 2024 Russia using model trained on 2000-2020
hist_data = data[data['Year'] <= 2020].copy()
# Build ARIMA features for hist_data training
train_rows = hist_data.dropna(subset=['Prev_Total']).copy()
print('Generating ARIMA features for 2000-2020 training...')
arima_g = []
arima_t = []
for idx, row in train_rows.iterrows():
    noc = row['Mapped_NOC']
    year = row['Year']
    hist = hist_data[(hist_data['Mapped_NOC']==noc) & (hist_data['Year'] < year)]
    if len(hist) > 2:
        arima_g.append(get_arima_forecast(hist['Gold'].tolist()))
        arima_t.append(get_arima_forecast(hist['Total'].tolist()))
    else:
        arima_g.append(row['Prev_Gold'])
        arima_t.append(row['Prev_Total'])
train_rows['ARIMA_Gold'] = arima_g
train_rows['ARIMA_Total'] = arima_t

features = ['is_host','Prev_Gold','Prev_Total','ARIMA_Gold','ARIMA_Total'] + [f'Prev_{c}' for c in pca_cols]
# Some Prev_PC columns may be missing in train_rows if NaN - fill with 0
train_rows[features] = train_rows[features].fillna(0)

model_g_2000_2020, model_t_2000_2020 = train_models(train_rows, features)
print('Trained models on 2000-2020.')

# Build 2024 input for RUS
noc = 'RUS'
last_row = data[(data['Mapped_NOC']==noc) & (data['Year']<=2020)].sort_values('Year').iloc[-1]
input_2024 = {}
input_2024['Year'] = 2024
input_2024['Mapped_NOC'] = noc
input_2024['is_host'] = 1 if noc == 'FRA' else 0
input_2024['Prev_Gold'] = last_row['Gold']
input_2024['Prev_Total'] = last_row['Total']
for c in pca_cols:
    input_2024[f'Prev_{c}'] = last_row[c]
# ARIMA using history up to 2020
history = data[(data['Mapped_NOC']==noc) & (data['Year']<=2020)]
input_2024['ARIMA_Gold'] = get_arima_forecast(history['Gold'].tolist())
input_2024['ARIMA_Total'] = get_arima_forecast(history['Total'].tolist())

input_df = pd.DataFrame([input_2024])[features].fillna(0)
pred_rus_2024_gold = model_g_2000_2020.predict(input_df)[0]
pred_rus_2024_total = model_t_2000_2020.predict(input_df)[0]

print(f"Predicted Russia 2024 (model trained 2000-2020): Gold={pred_rus_2024_gold:.2f}, Total={pred_rus_2024_total:.2f}")

# Compute model host-effect: for a selection of countries compute diff when is_host toggled
countries = hist_data['Mapped_NOC'].unique()
host_effects = []
for c in countries:
    rows = data[data['Mapped_NOC']==c]
    if rows.empty:
        continue
    last = rows[rows['Year']<=2020].sort_values('Year').iloc[-1]
    base = {}
    base['Prev_Gold']=last['Gold']; base['Prev_Total']=last['Total']
    for cc_col in pca_cols:
        base[f'Prev_{cc_col}']=last[cc_col]
    base['ARIMA_Gold'] = get_arima_forecast(data[(data['Mapped_NOC']==c)&(data['Year']<=2020)]['Gold'].tolist())
    base['ARIMA_Total'] = get_arima_forecast(data[(data['Mapped_NOC']==c)&(data['Year']<=2020)]['Total'].tolist())
    row0 = pd.DataFrame([{**{'is_host':0}, **base}])[features].fillna(0)
    row1 = pd.DataFrame([{**{'is_host':1}, **base}])[features].fillna(0)
    g0 = model_g_2000_2020.predict(row0)[0]; g1 = model_g_2000_2020.predict(row1)[0]
    t0 = model_t_2000_2020.predict(row0)[0]; t1 = model_t_2000_2020.predict(row1)[0]
    eps = 1e-6
    mult_g = (g1) / (g0 + eps)
    mult_t = (t1) / (t0 + eps)
    host_effects.append({'Mapped_NOC':c, 'Gold_multiplier': mult_g, 'Total_multiplier': mult_t})
host_df = pd.DataFrame(host_effects)
mean_host_gold_mult = host_df['Gold_multiplier'].replace([np.inf, -np.inf], np.nan).dropna().mean()
mean_host_total_mult = host_df['Total_multiplier'].replace([np.inf, -np.inf], np.nan).dropna().mean()
print(f"Estimated average host multiplier (model 2000-2020): Gold x{mean_host_gold_mult:.3f}, Total x{mean_host_total_mult:.3f}")

# Save 2024 model predictions for reference
pred_2024_df = pd.DataFrame([{'Mapped_NOC':noc,'Pred_Gold':pred_rus_2024_gold,'Pred_Total':pred_rus_2024_total}])
pred_2024_df.to_csv(output_2024_pred, index=False)

# 2) Prepare combined 2024 data (0.3 * model_pred + 0.7 * actual)
# First generate model predictions for all countries for 2024 using model trained on 2000-2020
print('Generating 2024 model predictions for all countries...')
all_countries = np.unique(data['Mapped_NOC'])
pred_rows = []
for c in all_countries:
    rows = data[(data['Mapped_NOC']==c) & (data['Year']<=2020)].sort_values('Year')
    if rows.empty:
        continue
    last = rows.iloc[-1]
    row = {}
    row['Mapped_NOC']=c
    row['Prev_Gold']=last['Gold']; row['Prev_Total']=last['Total']
    for col in pca_cols:
        row[f'Prev_{col}']=last[col]
    row['ARIMA_Gold']=get_arima_forecast(rows['Gold'].tolist())
    row['ARIMA_Total']=get_arima_forecast(rows['Total'].tolist())
    row['is_host']=1 if c=='FRA' else 0
    pred_rows.append(row)

pred_all_2024 = pd.DataFrame(pred_rows)
if not pred_all_2024.empty:
    pred_all_2024[features] = pred_all_2024[features].fillna(0)
    pred_all_2024['Pred_Gold'] = model_g_2000_2020.predict(pred_all_2024[features])
    pred_all_2024['Pred_Total'] = model_t_2000_2020.predict(pred_all_2024[features])
else:
    pred_all_2024 = pd.DataFrame(columns=['Mapped_NOC','Pred_Gold','Pred_Total'])

# Load provided real 2024
if os.path.exists(real_2024_file):
    real_df = pd.read_csv(real_2024_file)
    # normalize name column
    if 'NOC' in real_df.columns:
        real_df = real_df.rename(columns={'NOC':'NOC_IN'})
    if 'country_name' in real_df.columns:
        real_df = real_df.rename(columns={'country_name':'country_name_in'})
    # Map to Mapped_NOC (3-letter) using dataset-derived mapping
    real_df['Mapped_NOC'] = real_df.apply(lambda r: map_value_to_mapped_noc(r.get('NOC_IN'), r.get('country_name_in'), r.get('Year'), r.get('Gold'), r.get('Total')), axis=1)
    real_2024 = real_df[real_df['Year']==2024].copy()
    # ensure columns Gold, Total, is_host exist
    if 'Gold' not in real_2024.columns or 'Total' not in real_2024.columns:
        raise RuntimeError('Provided 2024 file missing Gold/Total columns')
else:
    real_2024 = pd.DataFrame(columns=['Mapped_NOC','Year','Gold','Total','is_host'])

# Merge model preds with real
pred_all_2024 = pred_all_2024.set_index('Mapped_NOC')
combined_rows = []
for c in all_countries:
    pred_row = pred_all_2024.loc[c] if c in pred_all_2024.index else None
    real_row = real_2024[real_2024['Mapped_NOC']==c]
    real_exists = not real_row.empty
    pred_g = float(pred_row['Pred_Gold']) if pred_row is not None else 0.0
    pred_t = float(pred_row['Pred_Total']) if pred_row is not None else 0.0
    if real_exists:
        act_g = float(real_row.iloc[0]['Gold'])
        act_t = float(real_row.iloc[0]['Total'])
        comb_g = 0.3*pred_g + 0.7*act_g
        comb_t = 0.3*pred_t + 0.7*act_t
        is_host = int(real_row.iloc[0].get('is_host',0))
    else:
        # use predicted if no actual
        comb_g = pred_g
        comb_t = pred_t
        is_host = 1 if c=='FRA' else 0
    combined_rows.append({'Mapped_NOC':c,'Year':2024,'Gold':comb_g,'Total':comb_t,'is_host':is_host})

combined_2024 = pd.DataFrame(combined_rows)

# If Russia missing real 2024, ensure we included model pred for RUS (user requirement)
if 'RUS' not in combined_2024['Mapped_NOC'].values and 'RUS' in pred_all_2024.index:
    rpred = pred_all_2024.loc['RUS']
    combined_2024 = pd.concat([combined_2024, pd.DataFrame([{'Mapped_NOC':'RUS','Year':2024,'Gold':float(rpred['Pred_Gold']),'Total':float(rpred['Pred_Total']),'is_host':0}])], ignore_index=True)

# Replace any existing 2024 rows in data with combined_2024
data = data[data['Year']!=2024]
# attach PCA vals for each country (from latest available prior row)
new_rows = []
for _,r in combined_2024.iterrows():
    c = r['Mapped_NOC']
    prior = data[data['Mapped_NOC']==c]
    if not prior.empty:
        last = prior.sort_values('Year').iloc[-1]
        pc_vals = {c0: last[c0] for c0 in pca_cols}
    else:
        pc_vals = {c0:0 for c0 in pca_cols}
    row = {'Year':2024,'Mapped_NOC':c,'Gold':r['Gold'],'Total':r['Total'],'is_host':r['is_host']}
    row.update(pc_vals)
    new_rows.append(row)
if new_rows:
    data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True, sort=False)

# Rebuild Prev features and ARIMA for training up to 2024
data = data.sort_values(['Mapped_NOC','Year'])
for col in ['Gold','Total'] + pca_cols:
    data[f'Prev_{col}'] = data.groupby('Mapped_NOC')[col].shift(1)

train_all = data[data['Year']<=2024].dropna(subset=['Prev_Total']).copy()
print('Generating ARIMA features for training through 2024...')
ag=[]; at=[]
for idx,row in train_all.iterrows():
    noc=row['Mapped_NOC']; year=row['Year']
    history=data[(data['Mapped_NOC']==noc)&(data['Year']<year)]
    if len(history)>2:
        ag.append(get_arima_forecast(history['Gold'].tolist()))
        at.append(get_arima_forecast(history['Total'].tolist()))
    else:
        ag.append(row['Prev_Gold']); at.append(row['Prev_Total'])
train_all['ARIMA_Gold']=ag; train_all['ARIMA_Total']=at
train_all[features]=train_all[features].fillna(0)

# Retrain models using data through 2024
print('Retraining models on data through 2024 (including combined 2024)...')
model_g_all, model_t_all = train_models(train_all, features)

# Build 2028 prediction inputs using 2024 combined as Prev_*
pred_rows_2028 = []
for c in np.unique(data['Mapped_NOC']):
    rows = data[data['Mapped_NOC']==c].sort_values('Year')
    if rows.empty:
        continue
    last_2024 = rows[rows['Year']==2024]
    if not last_2024.empty:
        last = last_2024.iloc[0]
    else:
        last = rows.iloc[-1]
    row = {'Mapped_NOC':c,'Year':2028,'is_host':1 if c=='USA' else 0}
    row['Prev_Gold']=last['Gold']; row['Prev_Total']=last['Total']
    for col in pca_cols:
        row[f'Prev_{col}']=last[col]
    history = data[(data['Mapped_NOC']==c)]['Gold'].tolist()
    row['ARIMA_Gold']=get_arima_forecast(history)
    row['ARIMA_Total']=get_arima_forecast(data[(data['Mapped_NOC']==c)]['Total'].tolist())
    pred_rows_2028.append(row)

pred_df_2028 = pd.DataFrame(pred_rows_2028)
pred_df_2028[features]=pred_df_2028[features].fillna(0)

print('Predicting 2028...')
# We'll compute baseline (no-host) and host predictions via the retrained models,
# derive per-country multiplicative host multipliers, and apply them multiplicatively.
pred_df_2028_nohost = pred_df_2028.copy()
pred_df_2028_nohost['is_host'] = 0
pred_df_2028_nohost[features] = pred_df_2028_nohost[features].fillna(0)
g_no = model_g_all.predict(pred_df_2028_nohost[features])
t_no = model_t_all.predict(pred_df_2028_nohost[features])

pred_df_2028_host = pred_df_2028.copy()
pred_df_2028_host['is_host'] = 1
pred_df_2028_host[features] = pred_df_2028_host[features].fillna(0)
g_host = model_g_all.predict(pred_df_2028_host[features])
t_host = model_t_all.predict(pred_df_2028_host[features])

eps = 1e-6
g_mult = (g_host) / (g_no + eps)
t_mult = (t_host) / (t_no + eps)

pred_df_2028['Pred_Gold_base_nohost'] = g_no
pred_df_2028['Pred_Total_base_nohost'] = t_no
pred_df_2028['Gold_mult'] = g_mult
pred_df_2028['Total_mult'] = t_mult

# Use per-country multiplier when is_host==1, otherwise keep baseline no-host prediction.
final_g = []
final_t = []
mean_g_mult = np.nanmean(np.where(np.isfinite(g_mult), g_mult, np.nan))
mean_t_mult = np.nanmean(np.where(np.isfinite(t_mult), t_mult, np.nan))
for i, r in pred_df_2028.iterrows():
    if int(r['is_host'])==1:
        multg = r['Gold_mult'] if np.isfinite(r['Gold_mult']) and r['Gold_mult']>0 else mean_g_mult
        multt = r['Total_mult'] if np.isfinite(r['Total_mult']) and r['Total_mult']>0 else mean_t_mult
        final_g.append(max(0.0, r['Pred_Gold_base_nohost'] * (multg if multg is not None else 1.0)))
        final_t.append(max(0.0, r['Pred_Total_base_nohost'] * (multt if multt is not None else 1.0)))
    else:
        final_g.append(max(0.0, r['Pred_Gold_base_nohost']))
        final_t.append(max(0.0, r['Pred_Total_base_nohost']))

pred_df_2028['Pred_Gold'] = np.round(final_g).astype(int)
pred_df_2028['Pred_Total'] = np.round(final_t).astype(int)

result_2028 = pred_df_2028[['Mapped_NOC','Pred_Gold','Pred_Total']].sort_values('Pred_Total', ascending=False)
result_2028.to_csv(output_2028, index=False)

print('\n=== Summary ===')
print(f"Russia 2024 prediction (trained on 2000-2020): Gold={pred_rus_2024_gold:.1f}, Total={pred_rus_2024_total:.1f}")
print(f"Estimated average host multiplier (2000-2020 model): Gold x{mean_host_gold_mult:.3f}, Total x{mean_host_total_mult:.3f}")
print(f"Russia 2028 prediction (using combined 2024 data):")
rus_2028 = result_2028[result_2028['Mapped_NOC']=='RUS']
if not rus_2028.empty:
    r2028 = rus_2028.iloc[0]
    print(f"  Gold={r2028['Pred_Gold']}, Total={r2028['Pred_Total']}")
else:
    print('  Russia not in predicted 2028 results')

usa_2028 = result_2028[result_2028['Mapped_NOC']=='USA']
if not usa_2028.empty:
    u = usa_2028.iloc[0]
    # extract multipliers/computed host effect from pred_df_2028
    usa_row_pred = pred_df_2028[pred_df_2028['Mapped_NOC']=='USA'].iloc[0]
    usa_mult_g = usa_row_pred.get('Gold_mult', np.nan)
    usa_mult_t = usa_row_pred.get('Total_mult', np.nan)
    print(f"USA 2028 predicted: Gold={u['Pred_Gold']}, Total={u['Pred_Total']}")
    print(f"USA host multiplier (model): Gold x{usa_mult_g:.3f}, Total x{usa_mult_t:.3f}")
else:
    print('USA not in predicted 2028 results')

print('\nSaved 2028 predictions to:', output_2028)
print('Saved 2024 model-predictions to:', output_2024_pred)
