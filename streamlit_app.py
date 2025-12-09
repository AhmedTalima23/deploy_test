import streamlit as st
import joblib
import numpy as np

# ----------------------------
# Load trained model & preprocessors
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_model()

# ----------------------------
# App layout
# ----------------------------
st.set_page_config(page_title="Machine Failure Prediction", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4B79A1;'>Machine Failure Prediction ⚙️</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("### Enter Machine Features:")

# ----------------------------
# User inputs
# ----------------------------
age = st.number_input("Age", min_value=0, step=1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
pressure = st.number_input("Pressure (bar)", min_value=0.0, step=0.1)
machine_type = st.selectbox("Machine Type", ["TypeA", "TypeB", "TypeC"])
maintenance_level = st.selectbox("Maintenance Level", ["Low", "Medium", "High"])

st.markdown("---")

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Failure"):
    try:
        # Numeric preprocessing
        num_features = np.array([[age, temperature, pressure]])
        num_features_scaled = scaler.transform(num_features)

        # Categorical preprocessing
        cat_features = np.array([[machine_type, maintenance_level]])
        cat_features_encoded = encoder.transform(cat_features)

        # Combine
        X_new = np.hstack([num_features_scaled, cat_features_encoded])

        # Predict
        prediction = model.predict(X_new)[0]

        # Display result
        if prediction == 1:
            st.markdown(
                "<div style='padding:20px; background-color:#ff4b4b; border-radius:10px; color:white; text-align:center;'>"
                "<h2>Predicted Failure Type: Failed ⚠️</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='padding:20px; background-color:#4CAF50; border-radius:10px; color:white; text-align:center;'>"
                "<h2>Predicted Failure Type: Not Failed ✅</h2></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("<hr><p style='text-align:center; color:gray;'>Powered by Streamlit</p>", unsafe_allow_html=True)