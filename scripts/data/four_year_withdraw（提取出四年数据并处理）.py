import pandas as pd

# 1. 读取原始CSV文件
df = pd.read_csv('processed_summer_athletes1.csv')

# 2. 定义最近四届奥运会年份（2012、2016、2020、2024） 
recent_olympic_years = [2012, 2016, 2020, 2024]

# 3. 筛选最近四届数据
df_recent_four = df[df['Year'].isin(recent_olympic_years)].copy()
df_recent_four = df_recent_four[df['Medal'] != 'No medal']

df_recent_four.drop(columns=["Event","City", "NOC"], axis=1, inplace=True)

# 4. 保存为新文件（不包含索引列）
output_file_path = 'recent_four_summer_athletes2.csv'
df_recent_four.to_csv(output_file_path, index=False, encoding='utf-8')

# 5. 输出处理结果统计
print("=== 数据处理结果 ===")
print(f"原始数据总行数：{len(df):,}")
print(f"筛选后数据总行数：{len(df_recent_four):,}")
print(f"保留数据比例：{len(df_recent_four)/len(df)*100:.2f}%")
print(f"\n各年份数据分布：")
for year in sorted(recent_olympic_years):
    count = len(df_recent_four[df_recent_four['Year'] == year])
    print(f"  {year}年：{count:,} 条记录")
print(f"\n新文件已保存至：{output_file_path}")