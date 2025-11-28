# 🚦 Traffic Density Prediction System

## 📘 Overview
The Traffic Density Prediction System is an AI-powered application designed to analyze traffic-related data and predict the traffic density of a specific area. The system uses machine learning models trained on parameters such as time, day, vehicle count, working days, population, area name, and weather conditions. It helps city planners, traffic police, and navigation systems make informed decisions about traffic control.

## 🎯 Objective
To develop a prediction model that accurately forecasts traffic density based on real-time or historical traffic parameters.

## ✨ Features
- Predicts traffic density for any given input parameters  
- Visual representation of predicted traffic levels  
- Real-time and historical traffic analysis  
- User-friendly interface  
- Works for multiple areas and cities

## 🏗️ System Architecture
1. **Data Collection** – Dataset `updated8.csv`
2. **Data Preprocessing** – Handling missing values and encoding categorical data
3. **Model Training** – ML algorithms such as Random Forest, Decision Tree, Linear Regression, etc.
4. **Prediction Engine** – Selecting the best performing model
5. **Frontend** – Streamlit-based UI
6. **Backend** – Python ML model hosted locally

## 📊 Dataset Information
The dataset `updated8.csv` contains fields like:
- Time Slot
- Rush Hour
- Day
- Working Day
- City
- Area Name
- Population
- Two-Wheeler Count
- Four-Wheeler Count
- Weather
- Traffic Density

## ⚙️ Technologies Used
- Python
- Machine Learning (Random Forest / Decision Tree)
- Pandas, NumPy, Scikit-Learn
- Streamlit for UI
- Matplotlib / Seaborn for visualizations

## 🚀 How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

### Step 2: Run the Streamlit App
```bash
streamlit run app.py

