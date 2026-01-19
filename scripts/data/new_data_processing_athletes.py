
# Import necessary libraries
import pandas as pd

# --------------------------
# Step 1: Read the original data
# --------------------------
# Read data (keep the file path consistent with the original path)
df = pd.read_csv('summerOly_athletes.csv')

# View basic information of the original data (optional, for verification)
print("=== Basic Information of Original Data ===")
print(f"Data shape (rows × columns): {df.shape}")
print("\nFirst 5 rows of original data:")
print(df.head())

# --------------------------
# Step 2: Read external mapping table and perform country name unification
# --------------------------
# --------------------------
# Sub-step 2.1: Read external country name mapping table (CSV format)
# --------------------------
# Path to the external mapping table (replace with your actual file path)
mapping_table_path = 'country_name_mapping2.csv'  
try:
    # Read mapping table (assume columns: 'original_name' = original NOC name, 'unified_name' = target name)
    mapping_df = pd.read_csv(mapping_table_path, encoding='utf-8')
    
    # Convert mapping table to dictionary (key: original name, value: unified name)
    country_mapping = dict(zip(mapping_df['Old_NOC'], mapping_df['Full_Name']))
    print(f"\n=== Mapping Table Loaded Successfully ===")
    print(f"Number of mapping rules: {len(country_mapping)}")
    #print(f"First 5 mapping rules: {dict(list(country_mapping.items())[:5])}")

except FileNotFoundError:
    print(f"Error: Mapping table file '{mapping_table_path}' not found!")
    raise  # Terminate program if mapping table is missing
except KeyError as e:
    print(f"Error: Mapping table missing required column {e}! Please ensure columns are 'original_name' and 'unified_name'")
    raise

# --------------------------
# Sub-step 2.2: Unify country names (same logic as before)
# --------------------------
# Add a new 'country_name' column (only for name mapping, does not modify the original NOC column)
# Names that do not match the mapping rules retain their original values
df['country_name'] = df['NOC'].map(lambda x: country_mapping.get(x, x))

# --------------------------
# Step 3: Save the new file with only unified country names
# --------------------------
# Save path (add "_name_unified" identifier to distinguish files with only unified names)
name_unified_file_path = 'processed_summer_athletes2.csv'
df.to_csv(name_unified_file_path, index=False, encoding='utf-8')

# Output verification information
