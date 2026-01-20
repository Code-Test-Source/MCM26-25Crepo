import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('recent_4_olympics_medals.csv')

# Calculate total medals per year
total_medals = df.groupby('Year')['Total'].sum()

print("Historical Total Medals per Olympic Year:")
print(total_medals)

# Fit ARIMA model to the total medals time series
model = ARIMA(total_medals, order=(1, 1, 1))
model_fit = model.fit()

# Forecast for 2028
forecast = model_fit.forecast(steps=1)
predicted_total = int(forecast.iloc[0])

print(f"\nPredicted Total Medals for 2028: {predicted_total}")

# Save the prediction to CSV
pd.DataFrame({'Year': [2028], 'Predicted_Total_Medals': [predicted_total]}).to_csv('predicted_total_medals_2028.csv', index=False)