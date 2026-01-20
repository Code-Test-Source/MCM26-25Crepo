import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
data = pd.read_csv('.outputs/processed_data/athletes_after_sum.csv')
# Create a pivot table: rows are country + year, columns are sports, values are total medals
pivot_data = data.pivot_table(index=['country_name', 'Year'], columns='Sport', values='Total', fill_value=0)

# Standardize data (PCA usually requires standardization)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_data)

# Perform PCA, retaining components that explain up to 75% cumulative variance
pca = PCA(n_components=0.75)
pca.fit(scaled_data)

# Output results
print("PCA Results:")
print(f"Original data shape: {pivot_data.shape}")
print(f"Standardized data shape: {scaled_data.shape}")

# Explained variance
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("\nExplained Variance:")
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i+1}: {var:.4f}")

print("\nExplained Variance Ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Principal Component {i+1}: {ratio:.4f}")

print("\nCumulative Explained Variance Ratio:")
for i, cum_ratio in enumerate(cumulative_variance_ratio):
    print(f"First {i+1} Principal Components: {cum_ratio:.4f}")

# Component loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, index=pivot_data.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
print("\nComponent Loadings (First 5 Principal Components):")
print(loadings_df.iloc[:, :5])

# Transformed data
transformed_data = pca.transform(scaled_data)
transformed_df = pd.DataFrame(transformed_data, index=pivot_data.index, columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])])
print("\nTransformed Data (First 5 Rows, First 5 Principal Components):")
print(transformed_df.iloc[:5, :5])

# Save results to CSV
loadings_df.to_csv('.outputs/processed_data/pca_loadings.csv')
transformed_df.to_csv('.outputs/processed_data/pca_transformed_data.csv')

print("\nResults saved to .outputs/processed_data/pca_loadings.csv and .outputs/processed_data/pca_transformed_data.csv")

# Use principal components combined with medal numbers to predict 2028 medal situations for countries using machine learning models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare training data: collect principal components and medal numbers for all countries over the years
X_data = []
y_data = []

for country in transformed_df.index.get_level_values('country_name').unique():
    country_data = transformed_df.loc[country]
    years = sorted(country_data.index)
    pc_seq = country_data.loc[years].values  # Principal component sequence
    
    # Get corresponding total medals
    country_pivot = pivot_data.loc[country]
    medal_totals = country_pivot.loc[years].sum(axis=1).values  # Total medals per year
    
    for i in range(len(pc_seq) - 1):
        X_data.append(pc_seq[i])  # Principal components of the current year
        y_data.append(medal_totals[i + 1])  # Total medals of the next year

X_data = np.array(X_data)
y_data = np.array(y_data)

if len(X_data) > 0:
    # Train Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_data, y_data)
    
    # Predict 2028 medal situations for countries
    predictions_2028 = {}
    for country in transformed_df.index.get_level_values('country_name').unique():
        country_data = transformed_df.loc[country]
        years = sorted(country_data.index)
        if len(years) > 0:
            last_pc = country_data.loc[years[-1]].values  # Principal components of the latest year
            pred_medal = rf_model.predict(last_pc.reshape(1, -1))[0]
            predictions_2028[country] = max(0, pred_medal)
    
    # Sort and display top 10
    sorted_predictions = sorted(predictions_2028.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Predicted Olympic Medal Table for 2028 (Based on Principal Components and Medal Numbers Machine Learning Model):")
    for rank, (country, total) in enumerate(sorted_predictions, 1):
        print(f"{rank}. {country}: {total:.0f} medals")
else:
    print("\nNot enough data for training.")


