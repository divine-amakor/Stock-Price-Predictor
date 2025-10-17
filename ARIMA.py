import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import normaltest
from statsmodels.tsa.stattools import acf, pacf
# from pmdarima.arima import auto_arima
import scipy.interpolate as sci
import scipy.optimize as sco
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import scipy.optimize as sco

START = "2021-01-01"
END = "2025-01-01"


def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data


data = load_data('AAPL')
df = data
df.head()
print(df)

#Data Cleaning
df = df.dropna()
df.index = pd.to_datetime(df.Date)

df = df.iloc[:, 1]
print(df.head())
print(df.describe())
print(type(df))

# Create table visualization of Close prices and dates
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Create table data from the first 10 rows of df
table_data = [[date.strftime('%Y-%m-%d'), f'${price:.2f}']
              for date, price in list(zip(df.index, df.values))[:20]]
column_headers = ['Date', 'Close Price']

plt.suptitle('AAPL Stock Close Prices', y=0.99)
table = ax.table(cellText=table_data,
                 colLabels=column_headers,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.show()

# Data Exploration
plt.figure(figsize=(16, 7))
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time Frame')
ax1.set_ylabel('Stock Price for APPLE')
plt.title("Close Price Visualization")
ax1.plot(df)
plt.show()

# Checking stationarity
# Determining rolling statistics
rolLmean = df.rolling(12).mean()
rolLstd = df.rolling(12).std()

plt.figure(figsize=(16, 7))
fig = plt.figure(1)

# Plot rolling statistics:
orig = plt.plot(df, color='blue', label='Original')
mean = plt.plot(rolLmean, color='red', label='Rolling Mean')
std = plt.plot(rolLstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

# Making data stationary

# Transformation
plt.figure(figsize=(16, 7))
fig = plt.figure(1)

ts_log = np.log(df)
plt.plot(ts_log)
plt.show()

# Decomposition
decomposition = seasonal_decompose(ts_log, period=21, model='multiplicative')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16, 7))
fig = plt.figure(1)

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.show()

# Differencing
ts_log_diff = ts_log.diff().dropna() # First-order difference and drop the initial NaN
plt.figure(figsize=(16,7))
plt.plot(ts_log_diff, label='Differenced Log Transformed Series (Log Returns)')
plt.title('Differenced Log Transformed Series (Log Returns)')
plt.legend()
plt.show()

# Determining rolling statistics
rolLmean = ts_log_diff.rolling(12).mean()
rolLstd = ts_log_diff.rolling(12).std()

# Plot rolling statistics:
orig = plt.plot(ts_log_diff, color='blue', label='Original')
mean = plt.plot(rolLmean, color='red', label='Rolling Mean')
std = plt.plot(rolLstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

# Splitting data into training and testing sets
# We will split ts_log_diff, which is stationary.
# The split will be 70% for training and 30% for testing.
split_ratio = 0.7
split_point = int(len(ts_log_diff) * split_ratio)

train_data = ts_log_diff[:split_point]
test_data = ts_log_diff[split_point:]

# print(f"Training data length: {len(train_data)}")
# print(f"Test data length: {len(test_data)}")
# print(f"Train data head:\n{train_data.head()}")
# print(f"Test data head:\n{test_data.head()}")

# For reconstructing predictions, we need the ts_log value that corresponds to
# the day *before* the first day of test_data (i.e., the last day of train_data on the log scale).
# ts_log_diff starts one index later than ts_log due to .diff().dropna().
# The index of train_data.index[-1] is the last day of the training period for the differenced series.
# We need the ts_log value for this specific date.
last_day_of_train_diff_index = train_data.index[-1]
last_log_train_value_for_reconstruction = ts_log.loc[last_day_of_train_diff_index]

# Align actual test data (ts_log_test_actual, series_data_test_actual) with the test_data index
# This ensures we have the correct actual values for comparison against predictions.
ts_log_test_actual = ts_log.loc[test_data.index]
series_data_test_actual = df.loc[test_data.index]

# print(f"Length of ts_log_test_actual: {len(ts_log_test_actual)}")
# print(f"Length of series_data_test_actual: {len(series_data_test_actual)}")
# print(f"Last log train value for reconstruction (ts_log at {last_day_of_train_diff_index}): {last_log_train_value_for_reconstruc

# Plot ACF and PACF
# Plot ACF and PACF on training data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
sm.graphics.tsa.plot_acf(train_data, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF) on Training Data')
sm.graphics.tsa.plot_pacf(train_data, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF) on Training Data')
plt.tight_layout()
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# ARIMA model is fit on train_data (which is ts_log_diff).
# The order (p, d, q) is chosen based on ACF/PACF plots of train_data.
# Since train_data is already the first difference of log data (ts_log_diff),
# the 'd' in ARIMA for this series should typically be 0 if train_data is stationary.
# If train_data still showed trends/non-stationarity, d=1 (or higher) might be considered.
# For this example, we'll use (2,0,2) assuming train_data is stationary.
# This means an ARMA(2,2) model is applied to the already differenced ts_log_diff.
arima_order = (0, 0, 0) # p=2, d=0, q=2

model = ARIMA(train_data, order=arima_order)
results_ARIMA = model.fit()

# print(results_ARIMA.summary())

# Plotting the fitted values on the training data
plt.figure(figsize=(16, 8))
plt.plot(train_data, label='Training Data (Differenced Log)')
plt.plot(results_ARIMA.fittedvalues, color='red', label=f'Fitted Values on Training Data (ARIMA{arima_order})')
plt.title(f'ARIMA{arima_order} Model Fit on Training Data')
plt.legend()
plt.show()

# Make predictions for the test set period.
# The number of steps to forecast is the length of the test_data.
n_periods_forecast = len(test_data)
forecast_results = results_ARIMA.get_forecast(steps=n_periods_forecast)
predicted_diff_test = forecast_results.predicted_mean # These are predictions for ts_log_diff values

# Reversing transformations for test set predictions

# Step 1: Reverse differencing for ts_log_diff predictions.
# We use the last actual log value corresponding to the last day of train_data's index,
# stored in last_log_train_value_for_reconstruction, as the anchor point.

# Ensure predicted_diff_test is a pandas Series with an appropriate index for cumsum.
# Its index should align with test_data for proper cumsum and addition.
if not isinstance(predicted_diff_test, pd.Series):
    # If predicted_diff_test is a NumPy array, convert it to a Series
    # and assign the index from test_data (or a portion if lengths differ)
    predicted_diff_test = pd.Series(predicted_diff_test, index=test_data.index[:len(predicted_diff_test)])
else:
    # If it's already a Series, ensure its index matches test_data for alignment
    predicted_diff_test.index = test_data.index[:len(predicted_diff_test)]


# Cumulative sum of predicted differences and add to the last known log value from the training phase.
predicted_log_test = predicted_diff_test.cumsum() + last_log_train_value_for_reconstruction

# Crucial: Align the index of predicted_log_test with the full test_data.index if necessary.
# This step ensures that predicted_log_test has the exact same DateTimeIndex as test_data.
# This is important if the forecast produced slightly fewer/more points or had a misaligned index.
if len(predicted_log_test) == len(test_data.index):
    predicted_log_test.index = test_data.index
elif len(predicted_log_test) < len(test_data.index):
    print(f"Warning: Predictions length ({len(predicted_log_test)}) is shorter than test_data index length ({len(test_data.index)}). Plotting and RMSE will use available predictions.")
    # Adjust actual test data to match the length of predictions for comparison
    ts_log_test_actual = ts_log_test_actual.iloc[:len(predicted_log_test)]
    series_data_test_actual = series_data_test_actual.iloc[:len(predicted_log_test)]
    predicted_log_test.index = test_data.index[:len(predicted_log_test)] # Ensure index is still from test_data
else: # Predictions are longer
    print(f"Warning: Predictions length ({len(predicted_log_test)}) is longer than test_data index length ({len(test_data.index)}). Trimming predictions.")
    predicted_log_test = predicted_log_test.iloc[:len(test_data.index)]
    predicted_log_test.index = test_data.index


# Step 2: Reverse log transformation (exponentiate) to get predicted prices.
predictions_ARIMA_test = np.exp(predicted_log_test)

# print("Actual log values for test set (ts_log_test_actual head):")
# print(ts_log_test_actual.head())
# print("Predicted log values for test set (predicted_log_test head):")
# print(predicted_log_test.head())

# print("Actual prices for test set (series_data_test_actual head):")
# print(series_data_test_actual.head())
# print("Predicted prices for test set (predictions_ARIMA_test head):")
# print(predictions_ARIMA_test.head())


# Align actual test data with the (potentially adjusted) index of predictions_ARIMA_test.
# This ensures we are comparing the correct corresponding values, especially if lengths were adjusted.
aligned_series_data_test_actual = series_data_test_actual.loc[predictions_ARIMA_test.index]
aligned_ts_log_test_actual = ts_log_test_actual.loc[predicted_log_test.index]

# Ensure lengths match for RMSE calculation after alignment.
if len(predictions_ARIMA_test) == len(aligned_series_data_test_actual):
    rmse_test = np.sqrt(mean_squared_error(aligned_series_data_test_actual, predictions_ARIMA_test))
    mae_test = mean_absolute_error(aligned_series_data_test_actual, predictions_ARIMA_test)
    print(f'Test RMSE: {rmse_test:.4f}')
    print(f'Test MAE: {mae_test:.4f}')

    # Plotting actual vs predicted prices for the test set
    plt.figure(figsize=(16, 8))
    # Plot full historical data for context, slightly transparent
    plt.plot(df.index, df, label='Historical Actual Price (Full Range)', alpha=0.3)
    # Plot actual prices for the test period
    plt.plot(aligned_series_data_test_actual.index, aligned_series_data_test_actual, color='blue', label='Actual Price (Test Set)')
    # Plot predicted prices for the test period
    plt.plot(predictions_ARIMA_test.index, predictions_ARIMA_test, color='red', linestyle='--', label='Predicted Price (Test Set)')
    plt.title(f'Stock Price Prediction vs Actual (Test Set) - ARIMA{arima_order} - RMSE: {rmse_test:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (Close)')
    plt.legend()
    plt.show()

    # Plotting log scale comparison for test set
    plt.figure(figsize=(16,8))
    plt.plot(aligned_ts_log_test_actual.index, aligned_ts_log_test_actual, color='blue', label='Actual Log Price (Test Set)')
    plt.plot(predicted_log_test.index, predicted_log_test, color='red', linestyle='--', label='Predicted Log Price (Test Set)')
    plt.title(f'Log Price Prediction vs Actual (Test Set) - ARIMA{arima_order}')
    plt.xlabel('Date')
    plt.ylabel('Log Stock Price (Close)')
    plt.legend()
    plt.show()

else:
    print("Length mismatch between final predictions and aligned actual test data. Cannot calculate RMSE or plot accurately.")
    print(f"Length of predictions_ARIMA_test: {len(predictions_ARIMA_test)}")
    print(f"Length of aligned_series_data_test_actual: {len(aligned_series_data_test_actual)}")

# Create comparison table
comparison_df = pd.DataFrame({
    'Date': aligned_series_data_test_actual.index,
    'Actual': aligned_series_data_test_actual.values,
    'Predicted': predictions_ARIMA_test.values,
    'Error': abs(aligned_series_data_test_actual.values - predictions_ARIMA_test.values)
})

# Display table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data from the first 20 rows
table_data = [[date.strftime('%Y-%m-%d'), f'${actual:.2f}', f'${pred:.2f}', f'{err:.2f}']
                for date, actual, pred, err in comparison_df.values[:20]]
column_headers = ['Date', 'Actual Price', 'Predicted Price', 'Absolute Error']

plt.suptitle('Comparison of Actual vs Predicted Prices', y=0.99)
table = ax.table(cellText=table_data,
                    colLabels=column_headers,
                    loc='center',
                    cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.show()
