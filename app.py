import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("Flight_Delay_Model.pkl")
# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("✈️ Flight Delay Prediction")

st.sidebar.markdown("---")

st.sidebar.success("Model: Optimized Random Forest")

st.sidebar.info("Accuracy: 65.20%")

st.sidebar.markdown("---")

st.sidebar.subheader("👩‍💻 Developer")

st.sidebar.write("Sadia Khatun")

st.sidebar.markdown("---")

st.sidebar.caption("Machine Learning Project")

# -----------------------------
# Title
# -----------------------------
st.title("✈️ Flight Delay Prediction System")

st.markdown("""
This application predicts whether a flight will be delayed using a Machine Learning model.
""")
st.info(
    """
    📌 **Project Information**

    This machine learning application predicts whether a flight
    will be delayed based on flight information.

    **Model:** Optimized Random Forest

    **Dataset:** US Airlines Flight Delay Dataset
    """
)

st.divider()
# -----------------------------
# User Input
# -----------------------------
st.header("✈️ Enter Flight Information")

col1, col2 = st.columns(2)

with col1:

    flight = st.number_input(
        "Flight Number",
        min_value=1,
        value=100
    )

    time = st.number_input(
        "Departure Time",
        min_value=0,
        max_value=2359,
        value=900
    )

    length = st.number_input(
        "Flight Length (Minutes)",
        min_value=10,
        value=120
    )

    day = st.selectbox(
        "Day of Week",
        [1,2,3,4,5,6,7]
    )

    weekend = st.selectbox(
        "Weekend",
        [0,1]
    )

with col2:

    airline = st.number_input(
        "Airline (Encoded)",
        min_value=0,
        value=0
    )

    airport_from = st.number_input(
        "Airport From (Encoded)",
        min_value=0,
        value=0
    )

    airport_to = st.number_input(
        "Airport To (Encoded)",
        min_value=0,
        value=0
    )

    length_category = st.selectbox(
        "Length Category",
        [0,1,2]
    )

    

time_category = st.selectbox(
    "Time Category",
    [0,1,2]
)
# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Flight Delay", use_container_width=True):

    input_data = pd.DataFrame({
        "Flight": [flight],
        "Time": [time],
        "Length": [length],
        "Airline": [airline],
        "AirportFrom": [airport_from],
        "AirportTo": [airport_to],
        "DayOfWeek": [day],
        "Weekend": [weekend],
        "Length_Category": [length_category],
        "Time_Category": [time_category]
    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    confidence = probability.max() * 100

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 Flight Status: DELAYED")
    else:
        st.success("🟢 Flight Status: ON TIME")

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )
    st.divider()

st.caption("Developed by Sadia Khatun | Flight Delay Prediction System")