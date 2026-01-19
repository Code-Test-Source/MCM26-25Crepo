
# Import necessary libraries
import pandas as pd

# --------------------------
# Step 1: Read the original data
# --------------------------
# Read data (keep the file path consistent with the repository)
df = pd.read_csv('./2025_Problem_C_Data/summerOly_medal_counts.csv')

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
# Path to the external mapping table (use repository data folder)
mapping_table_path = './outputs/processed_data/country_name_mapping.csv'
try:
    # Read mapping table (assume columns: 'original_name' = original NOC name, 'unified_name' = target name)
    mapping_df = pd.read_csv(mapping_table_path, encoding='utf-8')
    
    # Convert mapping table to dictionary (key: original code/name, value: unified name)
    # mapping CSV uses columns: 'Old_NOC' and 'Full_Name'
    country_mapping = dict(zip(mapping_df['Old_NOC'], mapping_df['Full_Name']))
    print(f"\n=== Mapping Table Loaded Successfully ===")
    print(f"Number of mapping rules: {len(country_mapping)}")
    print(f"First 5 mapping rules: {dict(list(country_mapping.items())[:5])}")

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
name_unified_file_path = './outputs/processed_data/processed_medal_counts.csv'
df.to_csv(name_unified_file_path, index=False, encoding='utf-8')

# Output verification information
print("\n=== Country Name Unification Only Completed ===")
print(f"Number of rows after processing: {df.shape[0]} (consistent with original data)")
print(f"New column added: country_name (Unified Country Name)")
print(f"\nFirst 5 rows of processed data (showing comparison between NOC and unified country name):")
print(df[['Year', 'NOC', 'country_name', 'Gold', 'Silver', 'Bronze', 'Total']].head())
print(f"\nFile with only unified names has been saved to: {name_unified_file_path}")