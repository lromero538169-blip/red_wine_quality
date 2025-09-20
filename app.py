# --- replacement for parts of your app.py ---
import streamlit as st
import joblib
import numpy as np

# Load trained model and preprocessors (wrap in try/except)
try:
    model = joblib.load("wine_quality_model.pkl")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model/preprocessors: {e}")
    st.stop()

st.title("🍷 Wine Quality Prediction App")
st.markdown("Enter the wine’s chemical properties below (leave blank for missing values):")

def float_or_nan(s):
    s = s.strip()
    return np.nan if s == "" else float(s)

# Use text_input so user can leave blank for missing -> imputer can fill
fixed_acidity = st.text_input("Fixed Acidity", "")
volatile_acidity = st.text_input("Volatile Acidity", "")
citric_acid = st.text_input("Citric Acid", "")
residual_sugar = st.text_input("Residual Sugar", "")
chlorides = st.text_input("Chlorides", "")
free_sulfur_dioxide = st.text_input("Free Sulfur Dioxide", "")
total_sulfur_dioxide = st.text_input("Total Sulfur Dioxide", "")
density = st.text_input("Density", "")
ph = st.text_input("pH", "")
sulphates = st.text_input("Sulphates", "")
alcohol = st.text_input("Alcohol", "")

if st.button("Predict Quality"):
    try:
        raw = np.array([[float_or_nan(fixed_acidity), float_or_nan(volatile_acidity), float_or_nan(citric_acid),
                         float_or_nan(residual_sugar), float_or_nan(chlorides), float_or_nan(free_sulfur_dioxide),
                         float_or_nan(total_sulfur_dioxide), float_or_nan(density), float_or_nan(ph),
                         float_or_nan(sulphates), float_or_nan(alcohol)]], dtype=float)
    except ValueError as e:
        st.error(f"Invalid numeric input: {e}")
        st.stop()

    # Apply preprocessing
    try:
        features_imputed = imputer.transform(raw)  # ensure imputer was fitted on same shape/order
        features_scaled = scaler.transform(features_imputed)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Predict class and probability (safe handling)
    try:
        prediction = model.predict(features_scaled)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Attempt to get probability; fallback gracefully
    prob_good = None
    try:
        proba = model.predict_proba(features_scaled)[0]
        # Map probability to class label reliably using model.classes_
        # find index where class == 1 (assuming 1 is Good), else infer
        classes = list(model.classes_)
        if 1 in classes:
            idx_good = classes.index(1)
            idx_bad = 1 - idx_good if len(classes) == 2 else None
            prob_good = proba[idx_good] * 100
            prob_bad = proba[idx_bad] * 100 if idx_bad is not None else None
        else:
            # If classes are e.g. [False, True] or [not good label, good label], assume 'good' is the label mapped during training
            # As fallback, show probabilities mapped to classes
            prob_good = None
            prob_bad = None
    except Exception:
        proba = None

    st.subheader("Prediction Result:")
    # Interpret prediction assuming 1==Good
    if prediction[0] == 1:
        if prob_good is not None:
            st.success(f"✅ Good Quality Wine (Confidence: {prob_good:.2f}%)")
            if prob_bad is not None:
                st.info(f"❌ Not Good Confidence: {prob_bad:.2f}%")
        else:
            st.success("✅ Good Quality Wine")
            st.info("Confidence score not available for this model.")
    else:
        if prob_bad is not None:
            st.error(f"❌ Not Good Quality Wine (Confidence: {prob_bad:.2f}%)")
            if prob_good is not None:
                st.info(f"✅ Good Quality Confidence: {prob_good:.2f}%")
        else:
            st.error("❌ Not Good Quality Wine")
            st.info("Confidence score not available for this model.")

    # Debug info (optional, hidden behind a checkbox)
    if st.checkbox("Show model info"):
        st.write("Model classes:", model.classes_)
        if proba is not None:
            st.write("Raw class probabilities:", proba)
