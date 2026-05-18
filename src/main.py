import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv(
    "data/TimeSeries_TotalSolarGen_and_Load_IT_2016.csv"
)

# View first rows
print(df.head())

# View columns
print(df.columns)

# View dataset information
print(df.info())

# Convert timestamp column to datetime
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Set timestamp as index
df.set_index('utc_timestamp', inplace=True)

# Remove missing values
df = df.dropna()

# Plot original energy demand graph
plt.figure(figsize=(14,6))

plt.plot(df['IT_load_new'])

plt.title("Italy Energy Demand Over Time")

plt.xlabel("Time")

plt.ylabel("Energy Load")

plt.grid(True)

plt.show()

# Create lag features
df['lag_1'] = df['IT_load_new'].shift(1)

df['lag_24'] = df['IT_load_new'].shift(24)

# Create time-based features
df['hour'] = df.index.hour

df['day'] = df.index.day

df['month'] = df.index.month

# Remove NaN values created by lag features
df = df.dropna()

# Select features
features = [
    'lag_1',
    'lag_24',
    'hour',
    'day',
    'month'
]

X = df[features]

y = df['IT_load_new']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Train XGBoost model
model = XGBRegressor()

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)

# Plot actual vs predicted graph with proper dates
plt.figure(figsize=(14,6))

plt.plot(
    y_test.index,
    y_test.values,
    label='Actual'
)

plt.plot(
    y_test.index,
    predictions,
    label='Predicted'
)

plt.title("Actual vs Predicted Energy Demand")

plt.xlabel("Time")

plt.ylabel("Energy Load")

plt.legend()

plt.grid(True)

plt.show()