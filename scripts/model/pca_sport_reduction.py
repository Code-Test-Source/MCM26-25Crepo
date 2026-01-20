 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

# Define file paths
base_dir = r"./outputs/processed_data/"  # Keep the original path as it's a system path
input_file = os.path.join(base_dir, "summerOly_athletes_processed.csv")
output_file = os.path.join(base_dir, "summerOly_pca_features.csv")

# 1. Read data
df = pd.read_csv(input_file)

# 2. Data Pivoting
# We want each row to be a (Year, Country) pair, with columns representing medal performance for various Sports
# Here we use the 'Total' number of medals as the evaluation metric
# Include 'is_host' in the index to retain this label after PCA
pivot_df = df.pivot_table(index=['Year', 'Mapped_NOC', 'is_host'], 
                          columns='Sport', 
                          values='Total', 
                          aggfunc='sum', 
                          fill_value=0)

print(f"Pivot table shape: {pivot_df.shape}")

# 3. Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_df)

# 4. Apply PCA
# Retain 85% of the variance
pca = PCA(n_components=0.85) 
X_pca = pca.fit_transform(X_scaled)

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Original number of features (sports count): {pivot_df.shape[1]}")
print(f"Number of principal components after retaining 85% information: {pca.n_components_}")
print(f"Total explained variance: {sum(explained_variance):.4f}")

# 5. Construct result DataFrame
pc_columns = [f'Sport_PC_{i+1}' for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pc_columns, index=pivot_df.index)

# Reset index
pca_df = pca_df.reset_index()

# 6. Save results
pca_df.to_csv(output_file, index=False)
print(f"PCA processing completed, results saved to: {output_file}")

# 7. Save PCA loadings matrix - i.e., which sports compose each principal component
loadings_file = os.path.join(base_dir, "pca_sport_loadings.csv")
loadings_df = pd.DataFrame(pca.components_.T, columns=pc_columns, index=pivot_df.columns)
loadings_df.to_csv(loadings_file)
print(f"PCA loadings matrix saved to: {loadings_file}")

# 8. Analyze and display the meaning of principal components
print("\n=== Detailed Composition of Principal Components (Top 5 Weights) ===")
# Iterate over the first 5 principal components (or all if fewer)
n_show = min(5, pca.n_components_)
for i in range(n_show):
    pc_name = f'Sport_PC_{i+1}'
    print(f"\n--- {pc_name} (Explained Variance: {explained_variance[i]:.2%}) ---")
    
    # Get all sport weights for this principal component
    weights = loadings_df[pc_name]
    
    # Sort weights
    sorted_weights = weights.sort_values(ascending=False)
    
    # Get top 5 positively correlated sports
    top_pos = sorted_weights.head(5)
    # Get top 5 negatively correlated sports
    top_neg = sorted_weights.tail(5)
    
    print("Positive Correlation (representing high-score features for this PC):")
    for sport, weight in top_pos.items():
        print(f"  {sport:<25} : {weight:.4f}")
        
    print("Negative Correlation (representing low-score features for this PC):")
    for sport, weight in top_neg.items():
        print(f"  {sport:<25} : {weight:.4f}")