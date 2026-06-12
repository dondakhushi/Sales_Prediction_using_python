
# Sales Prediction using Machine Learning
# Author: Your Name

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Dataset

df = pd.read_csv("Advertising.csv")

print("\n========== First 5 Rows ==========")
print(df.head())

# Basic Information

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Statistical Summary ==========")
print(df.describe())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

# Data Cleaning

# Remove unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Exploratory Data Analysis

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Feature Selection

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Model Training

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Model Coefficients

print("\n========== Model Coefficients ==========")

coefficients = pd.DataFrame(
    model.coef_,
    X.columns,
    columns=['Coefficient']
)

print(coefficients)

print("\nIntercept:", model.intercept_)

# Prediction

y_pred = model.predict(X_test)

print("\n========== Sample Predictions ==========")

results = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': y_pred
})

print(results.head(10))

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 2))

# Actual vs Predicted Plot

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()

# Prediction Line Plot

plt.figure(figsize=(10, 6))

plt.plot(
    y_test.values,
    label='Actual Sales'
)

plt.plot(
    y_pred,
    label='Predicted Sales'
)

plt.legend()
plt.title("Actual vs Predicted Sales Comparison")
plt.show()

# Predict Sales for New Advertisement Budget

print("\n========== New Sales Prediction ==========")

new_data = pd.DataFrame({
    'TV': [150],
    'Radio': [25],
    'Newspaper': [20]
})

predicted_sales = model.predict(new_data)

print(
    f"Predicted Sales for TV=150, Radio=25, Newspaper=20 : "
    f"{predicted_sales[0]:.2f}"
)

# Save Predictions

results.to_csv("sales_predictions.csv", index=False)

print("\nPredictions saved to 'sales_predictions.csv'")

print("\n========== Project Completed Successfully ==========")