# Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

#  Load Dataset 
df = pd.read_csv("Housing.csv")
print("Dataset Shape:", df.shape)
df.head()

# Basic Info 
print(df.info())
print(df.describe())

# Handle Missing Values 
df = df.dropna()
print("After dropping NA:", df.shape)

# Exploratory Data Analysis (EDA) 
# Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price'], kde=True)
plt.title("Distribution of House Prices")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Feature Selection
# Adjust these based on your dataset columns:
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_above']
target = 'price'

X = df[features]
y = df[target]

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Model Training 
# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Random Forest (optional stronger model)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions 
lin_preds = lin_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Evaluation 

def evaluate_model(true, preds, name):
    mse = mean_squared_error(true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, preds)
    
    print(f"\n{name} Performance:")
    print("---------------------------")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R² Score:", r2)

evaluate_model(y_test, lin_preds, "Linear Regression")
evaluate_model(y_test, rf_preds, "Random Forest")

# Visualization: Actual vs Predicted (Random Forest) 
plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_preds, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()

# Conclusion 
print("\nConclusion:")
print("The Random Forest model typically performs better for non-linear housing data.")
print("Higher R² and lower RMSE indicate better predictions.")
print("Feature importance and dataset quality can further improve performance.")