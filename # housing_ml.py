# housing_ml.py
"""
Intro to Machine Learning Workshop
Dataset: Housing Prices (realtor-data.csv)
Topics:
- Data Preprocessing
- Train-Test Split
- Linear Regression
- Model Evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("realtor-data.csv")
print("First 5 rows of the dataset:")
print(df.head())

# 2. Select Features (X) and Target (y)
# Adjust column names based on your dataset
# Example: Predict 'price' using 'bed', 'bath', and 'house_size'
target_column = 'price'
feature_columns = ['bed', 'bath', 'house_size']

X = df[feature_columns]
y = df[target_column]

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Optional for linear regression, but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. Predictions
y_pred = model.predict(X_test_scaled)

# 7. Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Visualization: Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
