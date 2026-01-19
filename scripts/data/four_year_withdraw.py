import pandas as pd

# 1. Read the processed athletes CSV file (from outputs)
df = pd.read_csv('./outputs/processed_data/processed_summer_athletes.csv')

# 2. Define the most recent four Olympic years (2012, 2016, 2020, 2024)
recent_olympic_years = [2012, 2016, 2020, 2024]

# 3. Filter data for the most recent four editions
df_recent_four = df[df['Year'].isin(recent_olympic_years)].copy()
df_recent_four = df_recent_four[df_recent_four['Medal'] != 'No medal']

df_recent_four.drop(columns=["Event", "City", "NOC"], axis=1, inplace=True)

# 4. Save to a new file (without index column)
output_file_path = './outputs/processed_data/recent_four_summer_athletes.csv'
df_recent_four.to_csv(output_file_path, index=False, encoding='utf-8')

# 5. Print processing summary
print("=== Processing Results ===")
print(f"Original total rows: {len(df):,}")
print(f"Filtered total rows: {len(df_recent_four):,}")
print(f"Retention rate: {len(df_recent_four)/len(df)*100:.2f}%")
print("\nDistribution by year:")
for year in sorted(recent_olympic_years):
    count = len(df_recent_four[df_recent_four['Year'] == year])
    print(f"  {year}: {count:,} records")
print(f"\nNew file saved to: {output_file_path}")