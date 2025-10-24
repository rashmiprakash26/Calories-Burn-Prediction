import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load datasets
exercise_data = pd.read_csv("calories.csv")
calories_data = pd.read_csv("exercise.csv")

data = pd.concat([calories_data, exercise_data['Calories']], axis=1)
data.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)
data.drop("User_ID", axis=1, inplace=True)

# Split features and target
X = data.drop("Calories", axis=1)
Y = data["Calories"]

# Standardizing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = XGBRegressor()
model.fit(X_scaled, Y)

# Streamlit UI
st.title("Calories Burnt Predictor")
st.write("Enter your details to predict calories burnt")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
duration = st.number_input("Duration (min)", min_value=1, max_value=300, value=30)
heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=100)
body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)

if st.button("Predict Calories Burnt"):
    input_data = np.array([
        0 if gender == "Male" else 1,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    ]).reshape(1, -1)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    st.success(f"Predicted Calories Burnt: {prediction[0]:.2f} kcal")
