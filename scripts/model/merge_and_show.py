import os
import pandas as pd
OUT_DIR = os.path.join(os.path.abspath('./'), 'outputs', 'processed_data')
SRC = os.path.join(OUT_DIR, 'predictions_reliable.csv')
DST = os.path.join(OUT_DIR, 'predictions_2028_final.csv')

if not os.path.exists(SRC):
    print('Source predictions_reliable.csv not found; run reliable pipeline first')
    raise SystemExit(1)

df = pd.read_csv(SRC)
# ensure expected columns
required = ['NOC','Pred_Total_median','Pred_Total_lo','Pred_Total_hi']
for c in required:
    if c not in df.columns:
        print('Missing column', c)
        raise SystemExit(1)
# save to final path
df.to_csv(DST, index=False)
print('Saved merged predictions to', DST)
# show top 10 and interval widths
print('\nSample predictions (top 10):')
print(df.head(10).to_string(index=False))
print('\nInterval width stats:')
print('median interval width:', (df['Pred_Total_hi']-df['Pred_Total_lo']).median())
print('max interval width:', (df['Pred_Total_hi']-df['Pred_Total_lo']).max())
print('min interval width:', (df['Pred_Total_hi']-df['Pred_Total_lo']).min())
