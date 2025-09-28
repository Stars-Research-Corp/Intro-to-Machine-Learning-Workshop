# housing_ml.py
"""
Intro to Machine Learning Workshop
Dataset: Housing Prices (realtor-data.csv)

Topics Covered:
1. Data Preprocessing (cleaning and preparing the data)
2. Train-Test Split (separating data for training and testing)
3. Linear Regression (building a prediction model)
4. Model Evaluation (checking how good the model is)
5. Visualization (comparing actual vs predicted results)
"""

# Import libraries
import pandas as pd              # for handling data in tables (dataframes)
import numpy as np               # for numerical operations
from sklearn.model_selection import train_test_split   # to split data into training & testing
from sklearn.linear_model import LinearRegression      # the model we’ll use
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # for evaluation
from sklearn.preprocessing import StandardScaler       # for scaling/normalizing data
import matplotlib.pyplot as plt   # for creating graphs

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Read the dataset from a CSV file into a DataFrame
df = pd.read_csv("realtor-data.csv")

# Display the first 5 rows to understand what the data looks like
print("First 5 rows of the dataset:")
print(df.head())

# -------------------------------
# 2. Select Features (X) and Target (y)
# -------------------------------
# Features are the inputs (like number of bedrooms, bathrooms, house size).
# Target is what we want to predict (in this case, the house price).

# Define which column is the target (what we want to predict)
target_column = 'price'

# Define which columns are the features (inputs for prediction)
feature_columns = ['bed', 'bath', 'house_size']

# Create X (features) and y (target)
X = df[feature_columns]
y = df[target_column]

# Handle missing values by replacing them with the average of each column
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# -------------------------------
# 3. Train-Test Split
# -------------------------------
# Split the data into:
# - Training set (80%): used to train the model
# - Testing set (20%): used to evaluate the model on new, unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Feature Scaling
# -------------------------------
# Scaling puts all features on the same scale (important if they have very different ranges).
# Example: 'house_size' might be in thousands, while 'bedrooms' is in single digits.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit scaler on training data
X_test_scaled = scaler.transform(X_test)        # apply same transformation to test data

# -------------------------------
# 5. Train Model
# -------------------------------
# Create a Linear Regression model and train it with the scaled training data
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------
# 6. Make Predictions
# -------------------------------
# Use the trained model to predict house prices from the test data
y_pred = model.predict(X_test_scaled)

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
# Calculate how well the model did:
mae = mean_absolute_error(y_test, y_pred)   # average absolute difference
mse = mean_squared_error(y_test, y_pred)    # average squared difference
rmse = np.sqrt(mse)                         # square root of MSE, easier to interpret
r2 = r2_score(y_test, y_pred)               # R² score, how much variance is explained

# Print the results
print("\nModel Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------------
# 8. Visualization
# -------------------------------
# Compare actual prices vs predicted prices on a scatter plot
plt.scatter(y_test, y_pred, alpha=0.5)   # points closer to the diagonal line are better
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
