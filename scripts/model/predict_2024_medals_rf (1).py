import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def predict_2024_medals():
    """
    Predict 2024 Olympic medal counts based on 2012, 2016, 2020 data
    Considering is_host factor, assuming Russia participates
    """

    # Read data
    df = pd.read_csv('.output/processed_data/olympics_medals_cleaned.csv')

    # Filter historical data (2012-2020)
    historical_data = df[df['Year'].isin([2012, 2016, 2020])].copy()

    # Handle Russia data: merge ROC as Russia
    historical_data['country_name'] = historical_data['country_name'].replace({'ROC': 'Russia'})

    # Aggregate medal data for same year and country (to avoid duplicates)
    historical_data = historical_data.groupby(['country_name', 'Year', 'is_host']).agg({
        'Gold': 'sum',
        'Total': 'sum'
    }).reset_index()

    # Calculate historical statistics for each country
    country_stats = []

    for country in historical_data['country_name'].unique():
        country_data = historical_data[historical_data['country_name'] == country]

        # Calculate average medal counts
        avg_gold = country_data['Gold'].mean()
        avg_total = country_data['Total'].mean()

        # Calculate trend (latest vs average)
        latest_gold = country_data[country_data['Year'] == country_data['Year'].max()]['Gold'].values[0]
        latest_total = country_data[country_data['Year'] == country_data['Year'].max()]['Total'].values[0]

        # Whether ever been host
        ever_host = country_data['is_host'].max()

        # Participation count
        participation_count = len(country_data)

        country_stats.append({
            'country_name': country,
            'avg_gold': avg_gold,
            'avg_total': avg_total,
            'latest_gold': latest_gold,
            'latest_total': latest_total,
            'ever_host': ever_host,
            'participation_count': participation_count
        })

    # Convert to DataFrame
    features_df = pd.DataFrame(country_stats)

    # Prepare training data
    # Use historical data to train model, target is medal count for next Olympics
    train_data = []

    for country in features_df['country_name']:
        country_hist = historical_data[historical_data['country_name'] == country]

        for i, row in country_hist.iterrows():
            # Create training sample for each Olympics
            year = row['Year']
            is_host = row['is_host']

            # Find statistics for the country before this year
            prev_data = historical_data[
                (historical_data['country_name'] == country) &
                (historical_data['Year'] < year)
            ]

            if len(prev_data) > 0:
                prev_avg_gold = prev_data['Gold'].mean()
                prev_avg_total = prev_data['Total'].mean()
                prev_participation = len(prev_data)
                prev_ever_host = prev_data['is_host'].max()
            else:
                # If first participation, use 0
                prev_avg_gold = 0
                prev_avg_total = 0
                prev_participation = 0
                prev_ever_host = 0

            train_data.append({
                'country_name': country,
                'year': year,
                'is_host': is_host,
                'prev_avg_gold': prev_avg_gold,
                'prev_avg_total': prev_avg_total,
                'prev_participation': prev_participation,
                'prev_ever_host': prev_ever_host,
                'target_gold': row['Gold'],
                'target_total': row['Total']
            })

    train_df = pd.DataFrame(train_data)

    # Train model
    feature_cols = ['is_host', 'prev_avg_gold', 'prev_avg_total', 'prev_participation', 'prev_ever_host']

    # Train gold medal prediction model
    X_gold = train_df[feature_cols]
    y_gold = train_df['target_gold']

    # Train total medal prediction model
    X_total = train_df[feature_cols]
    y_total = train_df['target_total']

    # Standardize features
    scaler_gold = StandardScaler()
    scaler_total = StandardScaler()

    X_gold_scaled = scaler_gold.fit_transform(X_gold)
    X_total_scaled = scaler_total.fit_transform(X_total)

    # Use Random Forest regression
    rf_gold = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_total = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_gold.fit(X_gold_scaled, y_gold)
    rf_total.fit(X_total_scaled, y_total)

    # Predict 2024 data
    predictions_2024 = []

    # Get all participating countries (including Russia)
    countries_2024 = df[df['Year'] == 2024]['country_name'].unique()
    # Ensure Russia is included
    if 'Russia' not in countries_2024:
        countries_2024 = np.append(countries_2024, 'Russia')

    for country in countries_2024:
        # Get historical data for the country
        country_hist = historical_data[historical_data['country_name'] == country]

        if len(country_hist) > 0:
            # Calculate historical statistics
            prev_avg_gold = country_hist['Gold'].mean()
            prev_avg_total = country_hist['Total'].mean()
            prev_participation = len(country_hist)
            prev_ever_host = country_hist['is_host'].max()
        else:
            # New participating country
            prev_avg_gold = 0
            prev_avg_total = 0
            prev_participation = 0
            prev_ever_host = 0

        # 2024 is_host: France is the host
        is_host_2024 = 1 if country == 'France' else 0

        # Prepare prediction features
        features = np.array([[is_host_2024, prev_avg_gold, prev_avg_total, prev_participation, prev_ever_host]])

        # Standardize
        features_gold_scaled = scaler_gold.transform(features)
        features_total_scaled = scaler_total.transform(features)

        # Predict
        pred_gold = max(0, rf_gold.predict(features_gold_scaled)[0])  # Ensure non-negative
        pred_total = max(pred_gold, rf_total.predict(features_total_scaled)[0])  # Ensure total >= gold

        predictions_2024.append({
            'country_name': country,
            'NOC': country,  # Simplified processing
            'Year': 2024,
            'Gold': round(pred_gold),
            'Total': round(pred_total),
            'is_host': is_host_2024
        })

    # Convert to DataFrame and sort
    pred_df = pd.DataFrame(predictions_2024)
    pred_df = pred_df.sort_values(['Gold', 'Total'], ascending=False).reset_index(drop=True)

    # Save prediction results
    pred_df.to_csv('.output/processed_data/predicted_2024_medals.csv', index=False)

    print("✅ 2024 Olympic medal prediction completed!")
    print(f"Predicted medal data for {len(pred_df)} countries")
    print("Prediction results saved to: predicted_2024_medals.csv")

    # Display top 10 predicted results
    print("\n🏆 Top 10 predicted 2024 medal rankings:")
    print(pred_df.head(10)[['country_name', 'Gold', 'Total', 'is_host']].to_string(index=False))

    # Evaluate model performance (using historical data)
    print("\n📊 Model evaluation:")
    # Use last year data as test
    test_data = train_df[train_df['year'] == 2020]
    if len(test_data) > 0:
        X_test_gold = scaler_gold.transform(test_data[feature_cols])
        X_test_total = scaler_total.transform(test_data[feature_cols])

        pred_test_gold = rf_gold.predict(X_test_gold)
        pred_test_total = rf_total.predict(X_test_total)

        mae_gold = mean_absolute_error(test_data['target_gold'], pred_test_gold)
        mae_total = mean_absolute_error(test_data['target_total'], pred_test_total)

        print(f"Gold medal prediction MAE: {mae_gold:.2f}")
        print(f"Total medal prediction MAE: {mae_total:.2f}")
    return pred_df

def predict_2028_medals():
    """
    Predict 2028 Olympic medal counts based on 2012, 2016, 2020, 2024 data
    2024 data uses predicted values, considering host effect, assuming Russia participates
    """

    # First, get 2024 predictions
    pred_2024_df = predict_2024_medals()

    # Read original data
    df = pd.read_csv('olympics_medals_cleaned.csv')

    # Filter historical data (2012-2020)
    historical_data = df[df['Year'].isin([2012, 2016, 2020])].copy()

    # Handle Russia data: merge ROC as Russia
    historical_data['country_name'] = historical_data['country_name'].replace({'ROC': 'Russia'})

    # Aggregate medal data
    historical_data = historical_data.groupby(['country_name', 'Year', 'is_host']).agg({
        'Gold': 'sum',
        'Total': 'sum'
    }).reset_index()

    # Add predicted 2024 data to historical data
    pred_2024_for_training = pred_2024_df[['country_name', 'Gold', 'Total', 'is_host']].copy()
    pred_2024_for_training['Year'] = 2024
    pred_2024_for_training = pred_2024_for_training[['country_name', 'Year', 'Gold', 'Total', 'is_host']]

    # Combine historical and predicted 2024 data
    extended_historical = pd.concat([historical_data, pred_2024_for_training], ignore_index=True)

    # Prepare training data using 2012-2024 data
    train_data = []

    for country in extended_historical['country_name'].unique():
        country_hist = extended_historical[extended_historical['country_name'] == country].sort_values('Year')

        for i, row in country_hist.iterrows():
            year = row['Year']
            is_host = row['is_host']

            # Find data before this year
            prev_data = extended_historical[
                (extended_historical['country_name'] == country) &
                (extended_historical['Year'] < year)
            ]

            if len(prev_data) > 0:
                prev_avg_gold = prev_data['Gold'].mean()
                prev_avg_total = prev_data['Total'].mean()
                prev_participation = len(prev_data)
                prev_ever_host = prev_data['is_host'].max()
            else:
                prev_avg_gold = 0
                prev_avg_total = 0
                prev_participation = 0
                prev_ever_host = 0

            train_data.append({
                'country_name': country,
                'year': year,
                'is_host': is_host,
                'prev_avg_gold': prev_avg_gold,
                'prev_avg_total': prev_avg_total,
                'prev_participation': prev_participation,
                'prev_ever_host': prev_ever_host,
                'target_gold': row['Gold'],
                'target_total': row['Total']
            })

    train_df = pd.DataFrame(train_data)

    # Train model
    feature_cols = ['is_host', 'prev_avg_gold', 'prev_avg_total', 'prev_participation', 'prev_ever_host']

    X_gold = train_df[feature_cols]
    y_gold = train_df['target_gold']

    X_total = train_df[feature_cols]
    y_total = train_df['target_total']

    # Standardize features
    scaler_gold = StandardScaler()
    scaler_total = StandardScaler()

    X_gold_scaled = scaler_gold.fit_transform(X_gold)
    X_total_scaled = scaler_total.fit_transform(X_total)

    # Use Random Forest regression
    rf_gold = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_total = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_gold.fit(X_gold_scaled, y_gold)
    rf_total.fit(X_total_scaled, y_total)

    # Predict 2028 data
    predictions_2028 = []

    # Get all countries from 2024 predictions (including Russia)
    countries_2028 = pred_2024_df['country_name'].unique()

    for country in countries_2028:
        # Get historical data including predicted 2024
        country_hist = extended_historical[extended_historical['country_name'] == country]

        if len(country_hist) > 0:
            prev_avg_gold = country_hist['Gold'].mean()
            prev_avg_total = country_hist['Total'].mean()
            prev_participation = len(country_hist)
            prev_ever_host = country_hist['is_host'].max()
        else:
            prev_avg_gold = 0
            prev_avg_total = 0
            prev_participation = 0
            prev_ever_host = 0

        # 2028 is_host: USA is the host (Los Angeles)
        is_host_2028 = 1 if country == 'United States' else 0

        features = np.array([[is_host_2028, prev_avg_gold, prev_avg_total, prev_participation, prev_ever_host]])

        features_gold_scaled = scaler_gold.transform(features)
        features_total_scaled = scaler_total.transform(features)

        pred_gold = max(0, rf_gold.predict(features_gold_scaled)[0])
        pred_total = max(pred_gold, rf_total.predict(features_total_scaled)[0])

        predictions_2028.append({
            'country_name': country,
            'NOC': country,
            'Year': 2028,
            'Gold': round(pred_gold),
            'Total': round(pred_total),
            'is_host': is_host_2028
        })

    pred_2028_df = pd.DataFrame(predictions_2028)
    pred_2028_df = pred_2028_df.sort_values(['Gold', 'Total'], ascending=False).reset_index(drop=True)

    # Save prediction results
    pred_2028_df.to_csv('predicted_2028_medals.csv', index=False)

    print("✅ 2028 Olympic medal prediction completed!")
    print(f"Predicted medal data for {len(pred_2028_df)} countries")
    print("Prediction results saved to: predicted_2028_medals.csv")

    # Display top 10 predicted results
    print("\n🏆 Top 10 predicted 2028 medal rankings:")
    print(pred_2028_df.head(10)[['country_name', 'Gold', 'Total', 'is_host']].to_string(index=False))

    return pred_2028_df

if __name__ == "__main__":
    predict_2024_medals()
    predict_2028_medals()