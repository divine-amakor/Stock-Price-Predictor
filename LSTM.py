import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from pandas import to_datetime



START = "2021-01-01"
END = "2025-01-01"


# Define a function to load the dataset

def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data


data = load_data('AAPL')
df = data
df.index = pd.to_datetime(df.Date)
df.head()
print(df)

plt.title("Close Price Visualization")
plt.plot(df.Date,df.Close)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

ma100 = df.Close.rolling(100).mean()
print(ma100)

plt.figure(figsize=(12, 6))
plt.plot(df.Date, df.Close)
plt.plot(df.Date, ma100, 'r')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Graph Of Moving Averages Of 100 Days')
plt.legend(['Close Price', '100 Days Moving Average'])
plt.show()

ma200 = df.Close.rolling(200).mean()
print(ma200)

plt.figure(figsize=(12, 6))
plt.plot(df.Date, df.Close)
plt.plot(df.Date, ma200, 'g')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Graph Of Moving Averages Of 200 Days')
plt.legend(['Close Price', '200 Days Moving Average'])
plt.show()

combined_data = pd.DataFrame({
    'Date': df.Date,
    'Close Price': df.Close.values.flatten(),
    'MA100': ma100.values.flatten(),
    'MA200': ma200.values.flatten()
})

# Create Table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = [['Date', 'Close Price', '100-Day MA', '200-Day MA']]
for i in range(199,220):
    date = combined_data['Date'].iloc[i].strftime('%Y-%m-%d')
    close = f"{combined_data['Close Price'].iloc[i]:.2f}"
    ma_100 = f"{combined_data['MA100'].iloc[i]:.2f}" if not np.isnan(combined_data['MA100'].iloc[i]) else 'N/A'
    ma_200 = f"{combined_data['MA200'].iloc[i]:.2f}" if not np.isnan(combined_data['MA200'].iloc[i]) else 'N/A'
    table_data.append([date, close, ma_100, ma_200])

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2,1.5)
plt.suptitle('Close Price, 100-Day and 200-Day Moving Average (Days 200-220)', y = 0.99)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df.Date, df.Close)
plt.plot(df.Date, ma100, 'r')
plt.plot(df.Date, ma200, 'g')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Comparison Of 100 Days And 200 Days Moving Averages')
plt.legend(['Close Price', '100 Days Moving Average', '200 Days Moving Average'])
plt.show()

print(df.shape)

# Splitting data into training and testing

train = pd.DataFrame(data[0:int(len(data) * 0.70)])
test = pd.DataFrame(data[int(len(data) * 0.70): int(len(data))])

print(train.shape)
print(test.shape)

print(train.head())
print(test.head())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values

data_training_array = scaler.fit_transform(train_close)
print(data_training_array)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)

from tensorflow.keras.layers import Dense, Dropout, LSTM

model = tf.keras.Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True
               , input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.summary()

model.save('keras_model.h5')

print(test_close.shape)
print(test_close)

past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)

final_df = pd.concat([past_100_days, test_df], ignore_index=True)
print(final_df.head())

input_data = scaler.fit_transform(final_df)
print(input_data)

print(input_data.shape)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

# Making predictions

y_pred = model.predict(x_test)
print(y_pred.shape)

print(y_test)
print(y_pred)

print(scaler.scale_)

scale_factor = 1 / 0.00749936
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

test_dates = to_datetime(test['Date'].values)
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, 'b', label="Original Price")
plt.plot(test_dates, y_pred, 'r', label="Predicted Price")
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Comparison Of Original And Predicted Prices using LSTM Model')
plt.gcf().autofmt_xdate()  # Rotation and alignment of tick labels
plt.legend()
plt.show()



from sklearn.metrics import mean_absolute_error, root_mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("Mean absolute error on test set: ", mae)


# Create comparison table with dates
fig, ax = plt.subplots(figsize=(15, 8))
ax.axis('tight')
ax.axis('off')

comparison_data = [['Date', 'Original Price', 'Predicted Price', 'Error']]
for i in range(20):
    date = test_dates.strftime('%Y-%m-%d')[i]
    original = y_test[i]
    predicted = y_pred[i][0]
    error = abs(original - predicted)
    comparison_data.append([date, f'{original:.2f}', f'{predicted:.2f}', f'{error:.2f}'])

table = ax.table(cellText=comparison_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title('Comparison of Original and Predicted Prices (First 20 Entries)')
plt.show()

# Create a table for metrics
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('tight')
ax.axis('off')
table_data = [['Metric', 'Value'],
              ['RMSE', f'{rmse:.2f}'],
              ['MAE', f'{mae:.2f}']]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title('Model Performance Metrics')
plt.show()



# Data Cleaning
