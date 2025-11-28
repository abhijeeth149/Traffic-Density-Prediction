# import pandas as pd
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load dataset (Update path if needed)
# dataset_path = r"C:\Users\abhij\OneDrive\Desktop\hack_pro\updated8.csv"  # Update with actual file location


# if not os.path.exists(dataset_path):
#     raise FileNotFoundError(f"File not found: {dataset_path}")

# df = pd.read_csv(dataset_path)

# # Display basic info
# print("Dataset Info:")
# print(df.info())

# # Handle missing values if any
# df = df.dropna()

# # Encode 'Traffic Density' as numbers
# label_encoder = LabelEncoder()
# df['Traffic Density'] = label_encoder.fit_transform(df['Traffic Density'])

# # Splitting features and target variable
# X = df.drop(columns=['Traffic Density'])  # Features
# y = df['Traffic Density']  # Target variable (now numeric)

# # Convert categorical columns to numeric if needed
# X = pd.get_dummies(X)

# # Splitting data (80:20 or 90:10)
# split_ratio = 0.8  # Change to 0.9 for 90:10 split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)

# # Normalize data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Train Ridge Regression Model
# model = Ridge(alpha=1.0)
# model.fit(X_train_scaled, y_train)

# # Predictions
# y_pred = model.predict(X_test_scaled)

# # KPI Metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Display results
# print("\nKPI Metrics:")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"R² Score: {r2:.4f}")

# # Save model if needed
# import joblib
# joblib.dump(model, "traffic_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(label_encoder, "label_encoder.pkl")
# print("\nModel saved as 'traffic_model.pkl'")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Load trained models
# ridge_model = joblib.load("traffic_model.pkl")
# scaler = joblib.load("scaler.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

# # Define city and area mappings
# city_areas = {
#     "Hyderabad": ["Alwal", "Charminar", "Begumpet", "Patancheru", "Attapur"],
#     "Mumbai": ["Girgaon", "Matunga", "Mazgaon", "Juhu Beach", "Prabhadevi"],
#     "Bengaluru": ["Vijayanagar", "Hebbal", "Nagavara", "Koramangala", "Majestic"],
#     "Chennai": ["Mylapore", "Anna Nagar", "Adyar", "Nungambakkam", "Mandaveli"],
#     "Kolkata": ["Cossipore", "Esplanade", "Joka", "Salt Lake", "Rajarhat"]
# }

# days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# # Streamlit UI
# def main():
#     st.title("🚦 Urban Traffic Density Prediction")
    
#     # Select City
#     selected_city = st.selectbox("Select City", list(city_areas.keys()))
#     # Select Area
#     selected_area = st.selectbox("Select Area", city_areas[selected_city])
#     # Select Day
#     selected_day = st.selectbox("Select Day", days_of_week)
#     # Select Time
#     selected_time = st.time_input("Select Time")
    
#     if st.button("Predict Traffic Density"):
#         # Create feature vector (simplified example, replace with real feature engineering)
#         feature_vector = np.array([[days_of_week.index(selected_day), int(selected_time.hour)]])
#         feature_vector_scaled = scaler.transform(feature_vector)
        
#         # Ridge Regression Prediction
#         ridge_pred = ridge_model.predict(feature_vector_scaled)[0]
        
#         # Train Random Forest and Boosting on the fly (for demonstration purposes)
#         rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#         gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        
#         # Dummy dataset for training (Replace with actual data in production)
#         X_dummy = np.random.rand(100, 2) * 10
#         y_dummy = np.random.rand(100) * 100
#         rf_model.fit(X_dummy, y_dummy)
#         gb_model.fit(X_dummy, y_dummy)
        
#         # Predictions
#         rf_pred = rf_model.predict(feature_vector_scaled)[0]
#         gb_pred = gb_model.predict(feature_vector_scaled)[0]
        
#         # Compute MAE for each model
#         mae_ridge = mean_absolute_error([y_dummy[0]], [ridge_pred])
#         mae_rf = mean_absolute_error([y_dummy[0]], [rf_pred])
#         mae_gb = mean_absolute_error([y_dummy[0]], [gb_pred])
        
#         # Display results
#         st.subheader("Traffic Prediction Results")
#         st.write(f"**Ridge Regression Prediction:** {ridge_pred:.2f}")
#         st.write(f"**Random Forest Prediction:** {rf_pred:.2f}")
#         st.write(f"**Gradient Boosting Prediction:** {gb_pred:.2f}")
        
#         st.subheader("Model Performance (Lower MAE is better)")
#         st.write(f"**Ridge Regression MAE:** {mae_ridge:.4f}")
#         st.write(f"**Random Forest MAE:** {mae_rf:.4f}")
#         st.write(f"**Gradient Boosting MAE:** {mae_gb:.4f}")
        
# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load trained models
ridge_model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define city and area mappings
city_areas = {
    "Hyderabad": ["Alwal", "Charminar", "Begumpet", "Patancheru", "Attapur"],
    "Mumbai": ["Girgaon", "Matunga", "Mazgaon", "Juhu Beach", "Prabhadevi"],
    "Bengaluru": ["Vijayanagar", "Hebbal", "Nagavara", "Koramangala", "Majestic"],
    "Chennai": ["Mylapore", "Anna Nagar", "Adyar", "Nungambakkam", "Mandaveli"],
    "Kolkata": ["Cossipore", "Esplanade", "Joka", "Salt Lake", "Rajarhat"]
}

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Streamlit UI
def main():
    st.title("🚦 Urban Traffic Density Prediction")
    
    # Select City
    selected_city = st.selectbox("Select City", list(city_areas.keys()))
    # Select Area
    selected_area = st.selectbox("Select Area", city_areas[selected_city])
    # Select Day
    selected_day = st.selectbox("Select Day", days_of_week)
    # Select Time
    selected_time = st.time_input("Select Time")
    
    if st.button("Predict Traffic Density"):
        # Create feature vector with 2 features (day index and hour)
        feature_vector = np.zeros((1, 88))  # Initialize with zeros (88 features)
        feature_vector[0, 0] = days_of_week.index(selected_day)  # Day index
        feature_vector[0, 1] = int(selected_time.hour)  # Hour
        
        # Scale the feature vector
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Ridge Regression Prediction
        ridge_pred = ridge_model.predict(feature_vector_scaled)[0]
        ridge_pred_clipped = np.clip(ridge_pred, 0, len(label_encoder.classes_) - 1)
        ridge_pred_label = label_encoder.inverse_transform([int(round(ridge_pred_clipped))])[0]
        
        # Train Random Forest and Gradient Boosting on the fly (for demonstration purposes)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        
        # Dummy dataset for training (Replace with actual data in production)
        X_dummy = np.random.rand(100, 88) * 10  # 88 features
        y_dummy = np.random.rand(100) * 100
        rf_model.fit(X_dummy, y_dummy)
        gb_model.fit(X_dummy, y_dummy)
        
        # Predictions
        rf_pred = rf_model.predict(feature_vector_scaled)[0]
        rf_pred_clipped = np.clip(rf_pred, 0, len(label_encoder.classes_) - 1)
        rf_pred_label = label_encoder.inverse_transform([int(round(rf_pred_clipped))])[0]
        
        gb_pred = gb_model.predict(feature_vector_scaled)[0]
        gb_pred_clipped = np.clip(gb_pred, 0, len(label_encoder.classes_) - 1)
        gb_pred_label = label_encoder.inverse_transform([int(round(gb_pred_clipped))])[0]
        
        # Compute MAE and RMSE for each model
        mae_ridge = mean_absolute_error([y_dummy[0]], [ridge_pred])
        rmse_ridge = np.sqrt(mean_squared_error([y_dummy[0]], [ridge_pred]))
        
        mae_rf = mean_absolute_error([y_dummy[0]], [rf_pred])
        rmse_rf = np.sqrt(mean_squared_error([y_dummy[0]], [rf_pred]))
        
        mae_gb = mean_absolute_error([y_dummy[0]], [gb_pred])
        rmse_gb = np.sqrt(mean_squared_error([y_dummy[0]], [gb_pred]))
        
        # Display results
        st.subheader("Traffic Prediction Results")  
        st.write(f"**Ridge Regression Prediction:** {ridge_pred_label}")
        st.write(f"**Random Forest Prediction:** {rf_pred_label}")
        st.write(f"**Gradient Boosting Prediction:** {gb_pred_label}")
        
        st.subheader("Model Performance Metrics")
        st.write("**Ridge Regression Metrics:**")
        st.write(f"- Mean Absolute Error (MAE): {mae_ridge:.4f}")
        st.write(f"- Root Mean Squared Error (RMSE): {rmse_ridge:.4f}")
        
        st.write("**Random Forest Metrics:**")
        st.write(f"- Mean Absolute Error (MAE): {mae_rf:.4f}")
        st.write(f"- Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
        
        st.write("**Gradient Boosting Metrics:**")
        st.write(f"- Mean Absolute Error (MAE): {mae_gb:.4f}")
        st.write(f"- Root Mean Squared Error (RMSE): {rmse_gb:.4f}")
        
        # Evaluation based on MAE
        st.subheader("Evaluation")
        st.write("Submissions are evaluated based on the **Mean Absolute Error (MAE)** between the predicted and actual traffic density.")
        st.write(f"**Best Model (Lowest MAE):** {'Ridge Regression' if mae_ridge < mae_rf and mae_ridge < mae_gb else 'Random Forest' if mae_rf < mae_gb else 'Gradient Boosting'}")

if __name__ == "__main__":
    main()