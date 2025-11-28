import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (Update path if needed)
dataset_path = r"C:\Users\abhij\OneDrive\Desktop\hack_pro\updated8.csv"  # Update with actual file location


if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"File not found: {dataset_path}")

df = pd.read_csv(dataset_path)

# Display basic info
print("Dataset Info:")
print(df.info())

# Handle missing values if any
df = df.dropna()

# Encode 'Traffic Density' as numbers
label_encoder = LabelEncoder()
df['Traffic Density'] = label_encoder.fit_transform(df['Traffic Density'])

# Splitting features and target variable
X = df.drop(columns=['Traffic Density'])  # Features
y = df['Traffic Density']  # Target variable (now numeric)

# Convert categorical columns to numeric if needed
X = pd.get_dummies(X)

# Splitting data (80:20 or 90:10)
split_ratio = 0.8  # Change to 0.9 for 90:10 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge Regression Model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# KPI Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("\nKPI Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save model if needed
import joblib
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(model, "traffic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nModel saved as 'traffic_model.pkl'")


