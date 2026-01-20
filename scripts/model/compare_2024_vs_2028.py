import os
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'outputs', 'processed_data')
PRED_FN = os.path.join(OUT, 'predictions_reliable.csv')
PROC_MEDALS = os.path.join(OUT, 'processed_medal_counts.csv')
MAP_FN = os.path.join(OUT, 'country_mapping_auto.csv')
OUT_COMP = os.path.join(OUT, 'pred_vs_actual_2024.csv')

pred = pd.read_csv(PRED_FN)
med = pd.read_csv(PROC_MEDALS)
mapdf = pd.read_csv(MAP_FN)

# build mapping dict from original_name and standardized_name -> olympic_noc
map_dict = {}
for _,r in mapdf.iterrows():
    orig = str(r.get('original_name','')).strip()
    std = str(r.get('standardized_name','')).strip()
    noc = str(r.get('olympic_noc','')).strip()
    if orig:
        map_dict[orig] = noc
    if std and std!=orig:
        map_dict[std] = noc

# map med country_name to noc
med['country_name'] = med['country_name'].astype(str)
med['mapped_noc'] = med['country_name'].map(lambda x: map_dict.get(x, None))

# fallback: try direct NOC-like strings in NOC column (if already 3-letter)
med.loc[med['mapped_noc'].isna(), 'mapped_noc'] = med.loc[med['mapped_noc'].isna(), 'NOC']

# compute 2024 totals by mapped_noc
med2024 = med[med['Year']==2024].copy()
actual = med2024.groupby('mapped_noc', dropna=False)['Total'].sum().reset_index()
actual.columns = ['NOC', 'Actual_Total_2024']

# merge with predictions
cmp = pred.merge(actual, on='NOC', how='left')
# if Actual missing, set 0
cmp['Actual_Total_2024'] = cmp['Actual_Total_2024'].fillna(0.0)
cmp['Pred'] = cmp['Pred_Total_median']
cmp['AbsDiff'] = (cmp['Pred'] - cmp['Actual_Total_2024']).abs()
cmp['RelChangePct'] = np.where(cmp['Actual_Total_2024']>0, (cmp['Pred']/cmp['Actual_Total_2024'] - 1.0)*100.0, np.nan)

# summary stats
from math import sqrt
valid = cmp[~cmp['Actual_Total_2024'].isna()]
mae = valid['AbsDiff'].mean()
medae = valid['AbsDiff'].median()
rmse = sqrt(((valid['Pred']-valid['Actual_Total_2024'])**2).mean())
corr = valid[['Pred','Actual_Total_2024']].corr().iloc[0,1]

# save
cmp.to_csv(OUT_COMP, index=False)

print('Saved comparison to', OUT_COMP)
print('N rows compared:', len(cmp))
print('MAE:', round(mae,3), 'MedianAE:', round(medae,3), 'RMSE:', round(rmse,3), 'Corr:', round(corr,3))

# show top 8 biggest relative increases (excluding actual==0)
inc = cmp[cmp['Actual_Total_2024']>0].copy()
inc['RelAbs'] = inc['RelChangePct'].abs()
print('\nTop 8 by relative change (abs %):')
print(inc.sort_values('RelAbs', ascending=False).head(8)[['NOC','Actual_Total_2024','Pred','RelChangePct']].to_string(index=False))

# print USA and RUS rows if present
for x in ['USA','RUS']:
    row = cmp[cmp['NOC']==x]
    if not row.empty:
        r = row.iloc[0]
        print(f"\n{x}: actual_2024={r['Actual_Total_2024']}, pred_2028={r['Pred']}, rel%={r['RelChangePct']}")
    else:
        print(f"\n{x}: not in predictions file")
