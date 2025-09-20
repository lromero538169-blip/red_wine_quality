import streamlit as st
import joblib
import numpy as np

# Load trained model and preprocessors
model = joblib.load("wine_quality_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🍷 Wine Quality Prediction App")

st.markdown("Enter the wine’s chemical properties below to predict its quality:")

# Input fields for wine features
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")
ph = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict Quality"):
    # Arrange raw features
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, ph, sulphates, alcohol]])
    
    # Apply same preprocessing as training
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)

    # Predict with model
    prediction = model.predict(features_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("✅ Good Quality Wine")
    else:
        st.error("❌ Not Good Quality Wine")
