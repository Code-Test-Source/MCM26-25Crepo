import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('.output/processed_data/recent_4_olympics_medals.csv')

# Known total medals for 2028
total_medals_2028 = 1025

# Calculate total medals per year
total_medals_per_year = df.groupby('Year')['Total'].sum()

# Prepare predictions list
predictions = []

# Group by country and predict proportion for each
for country in df['country_name'].unique():
    country_data = df[df['country_name'] == country].sort_values('Year')
    # Calculate proportion for each year the country participated
    proportions = []
    years = []
    for year in country_data['Year']:
        total_for_year = total_medals_per_year[year]
        medals_for_year = country_data[country_data['Year'] == year]['Total'].values[0]
        proportion = medals_for_year / total_for_year
        proportions.append(proportion)
        years.append(year)

    if len(proportions) >= 3:  # At least 3 data points for regression
        # Use 'Year' as X and 'Proportion' as y
        X = np.array(years).reshape(-1, 1)
        y = np.array(proportions)
        try:
            # Fit Random Forest Regressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            # Predict proportion for 2028
            predicted_proportion = max(0, model.predict([[2028]])[0])
            # Calculate predicted medals
            predicted_medals = int(predicted_proportion * total_medals_2028)
            predictions.append({'Country': country, 'Predicted_Medals_2028': predicted_medals})
        except Exception as e:
            # If fitting fails, use the mean proportion
            mean_proportion = np.mean(proportions)
            predicted_medals = int(mean_proportion * total_medals_2028)
            predictions.append({'Country': country, 'Predicted_Medals_2028': predicted_medals})
    else:
        # Use mean proportion if not enough data
        mean_proportion = np.mean(proportions) if proportions else 0
        predicted_medals = int(mean_proportion * total_medals_2028)
        predictions.append({'Country': country, 'Predicted_Medals_2028': predicted_medals})

# Create DataFrame and sort by predicted medals
predictions_df = pd.DataFrame(predictions).sort_values('Predicted_Medals_2028', ascending=False)

# Print top 10 predictions
print("Top 10 Predicted Olympic Medal Table for 2028 (Direct Proportion Prediction with Random Forest):")
for rank, (idx, row) in enumerate(predictions_df.head(10).iterrows(), start=1):
    print(f"{rank}. {row['Country']}: {row['Predicted_Medals_2028']} medals")

# Verify total
print(f"\nTotal Predicted Medals: {predictions_df['Predicted_Medals_2028'].sum()}")

# Save full predictions to CSV
predictions_df.to_csv('.output/processed_data/predicted_2028_medals_proportion_direct.csv', index=False)