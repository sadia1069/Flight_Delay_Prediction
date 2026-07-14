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
st.markdown("""
<style>

/* ---------- Main Background ---------- */
.stApp{
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* ---------- Main Title ---------- */
h1{
    color:#38bdf8 !important;
    text-align:center;
    font-size:46px !important;
    font-weight:700;
}

/* ---------- Subtitle ---------- */
h2,h3{
    color:white !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
    background:#111827;
}

/* ---------- Input Box ---------- */
.stNumberInput input{
    border-radius:10px;
}

div[data-baseweb="select"] > div{
    border-radius:10px;
}

/* ---------- Button ---------- */
.stButton>button{
    width:100%;
    height:55px;
    background:#2563eb;
    color:white;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1d4ed8;
    color:white;
}

/* ---------- Metric ---------- */
div[data-testid="metric-container"]{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("Flight_Delay_Model.pkl")

airport_from_mapping = joblib.load("airport_from_mapping.pkl")
airport_to_mapping = joblib.load("airport_to_mapping.pkl")
# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("✈️ Flight Delay Prediction")

st.sidebar.markdown("---")

st.sidebar.success("Model: Optimized Random Forest")

st.sidebar.info("Accuracy: 90.20%")

st.sidebar.markdown("---")

st.sidebar.subheader("👩‍💻 Developer")

st.sidebar.write("Sadia Khatun")
st.sidebar.markdown("📧 **Email:** your_email@example.com")
st.sidebar.markdown("💻 **Built with:** Python, Streamlit")
st.sidebar.markdown("🤖 **Algorithm:** Random Forest")

st.sidebar.markdown("---")

st.sidebar.caption("Machine Learning Project")

# -----------------------------
# Title
# -----------------------------
st.title("✈️ Flight Delay Prediction System")

st.markdown("""
This application predicts whether a flight will be delayed using a Machine Learning model.
""")
st.markdown("""
<div style="
background:#1f3b5b;
padding:20px;
border-radius:15px;
border-left:6px solid #38bdf8;
margin-bottom:25px;">

<h3>📌 Project Information</h3>

<p>
This machine learning application predicts whether a flight
will be delayed based on historical flight information.
</p>

<ul>
<li>🤖 <b>Model:</b> Optimized Random Forest</li>
<li>📊 <b>Accuracy:</b> 90.20%</li>
<li>🗂️ <b>Dataset:</b> US Airlines Flight Delay Dataset</li>
</ul>

</div>
""", unsafe_allow_html=True)

st.divider()
# -----------------------------
# User Input
# -----------------------------
st.header("✈️ Enter Flight Information")
st.caption("Fill in the flight details below and click the prediction button.")

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

    airline = st.selectbox(
        "✈️ Airline",
        [
            "DL","OO","B6","US","FL","WN","CO","AA",
            "YV","EV","XE","9E","OH","UA","MQ",
            "AS","F9","HA"
        ]
    )

    airport_from = st.selectbox(
    "🛫 Airport From",
    list(airport_from_mapping.keys())
)
    airport_to = st.selectbox(
    "🛬 Airport To",
    list(airport_to_mapping.keys())
)

    length_category = st.selectbox(
        "Length Category",
        [0,1,2]
    )
    

time_category = st.selectbox(
    "Time Category",
    [0,1,2]
)
st.markdown("---")
# -----------------------------
# Prediction
# -----------------------------
st.write("")
   
if st.button("🔍 Predict Flight Delay", use_container_width=True):

    airline_mapping = {
        "9E": 0,
        "AA": 1,
        "AS": 2,
        "B6": 3,
        "CO": 4,
        "DL": 5,
        "EV": 6,
        "F9": 7,
        "FL": 8,
        "HA": 9,
        "MQ": 10,
        "OH": 11,
        "OO": 12,
        "UA": 13,
        "US": 14,
        "WN": 15,
        "XE": 16,
        "YV": 17
    }

    airline = airline_mapping[airline]
    airport_from = airport_from_mapping[airport_from]
    airport_to = airport_to_mapping[airport_to]

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

    st.markdown("## 📊 Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 Flight Status: DELAYED")
    else:
        st.success("🟢 Flight Status: ON TIME")
        st.balloons()

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)

    if confidence >= 80:
        st.success("✅ High Confidence Prediction")
    elif confidence >= 60:
        st.warning("🟡 Medium Confidence Prediction")
    else:
        st.error("🔴 Low Confidence Prediction")

    st.divider()

st.caption("Developed by Sadia Khatun | Flight Delay Prediction System")