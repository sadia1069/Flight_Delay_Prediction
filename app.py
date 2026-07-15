import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------
# Custom CSS
# ---------------------------------

st.markdown("""
<style>

/* =========================
   MAIN
========================= */

.stApp{
    background:#111827;
    color:white;
}

/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"]{
    background:#111827;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* =========================
   TITLES
========================= */

h1{
    color:#4FC3F7 !important;
    font-weight:700;
}

h2,h3{
    color:white !important;
}

/* =========================
   LABELS
========================= */

label{
    color:white !important;
    font-size:16px !important;
    font-weight:600 !important;
}

/* =========================
   NUMBER INPUT
========================= */

.stNumberInput input{

    background:#1F2937 !important;
    color:white !important;

    border:2px solid #334155 !important;

    border-radius:12px !important;

    transition:.3s;
}

.stNumberInput input:hover{

    border:2px solid #38BDF8 !important;

}

.stNumberInput input:focus{

    border:2px solid #38BDF8 !important;

    box-shadow:0 0 12px rgba(56,189,248,.45) !important;

}

/* =========================
   SELECT BOX
========================= */

div[data-baseweb="select"]>div{

    background:#1F2937 !important;

    color:white !important;

    border:2px solid #334155 !important;

    border-radius:12px !important;

    transition:.3s;

}

div[data-baseweb="select"]>div:hover{

    border:2px solid #38BDF8 !important;

}

div[data-baseweb="select"]>div:focus-within{

    border:2px solid #38BDF8 !important;

    box-shadow:0 0 12px rgba(56,189,248,.45);

}

/* =========================
   BUTTON
========================= */

.stButton>button{

    width:100%;

    height:58px;

    border:none;

    border-radius:12px;

    background:linear-gradient(90deg,#2563EB,#06B6D4);

    color:white;

    font-size:18px;

    font-weight:bold;

    transition:.3s;

}

.stButton>button:hover{

    background:linear-gradient(90deg,#1D4ED8,#0891B2);

    transform:scale(1.02);

}

/* =========================
   METRIC CARD
========================= */

div[data-testid="metric-container"]{

    background:#1E293B;

    border:1px solid #334155;

    border-radius:15px;

    padding:18px;

}

/* =========================
   PROGRESS BAR
========================= */

.stProgress>div>div>div>div{

    background:#38BDF8;

}

/* =========================
   EXPANDER
========================= */

details{

    background:#111827;

    border:1px solid #334155;

    border-radius:12px;

    padding:10px;

}

/* =========================
   SUCCESS / ERROR
========================= */

div[data-baseweb="notification"]{

    border-radius:12px;

}

/* =========================
   HORIZONTAL LINE
========================= */

hr{

    border:1px solid #334155;

}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar{

    width:8px;

}

::-webkit-scrollbar-track{

    background:#1F2937;

}

::-webkit-scrollbar-thumb{

    background:#38BDF8;

    border-radius:20px;

}

::-webkit-scrollbar-thumb:hover{

    background:#0EA5E9;

}
/* ===========================
   DROPDOWN MENU
=========================== */

/* Closed box */
div[data-baseweb="select"] > div{
    background:#1F2937 !important;
    color:white !important;
    border:2px solid #38BDF8 !important;
    border-radius:12px !important;
}

/* Dropdown list */
div[role="listbox"]{
    background:#1F2937 !important;
    border:2px solid #38BDF8 !important;
    border-radius:12px !important;
}

/* Every option */
div[role="option"]{
    background:#1F2937 !important;
    color:white !important;
}

/* Hover option */
div[role="option"]:hover{
    background:#2563EB !important;
    color:white !important;
}

/* Selected option */
div[aria-selected="true"]{
    background:#38BDF8 !important;
    color:black !important;
}

/* Arrow */
svg{
    color:#38BDF8 !important;
}
/* ===========================
   STREAMLIT 1.59 SELECTBOX
=========================== */

/* Closed Select Box */
[data-baseweb="select"]{
    background:#1F2937 !important;
    border-radius:12px !important;
}

[data-baseweb="select"] > div{
    background:#1E3A5F !important;
    border:2px solid #38BDF8 !important;
    border-radius:12px !important;
}

/* Value */
[data-baseweb="select"] span{
    color:white !important;
    font-weight:500;
}

/* Arrow */
[data-baseweb="select"] svg{
    color:#38BDF8 !important;
}

/* Popup Menu */
div[role="listbox"]{
    background:#1E3A5F !important;
    border:2px solid #38BDF8 !important;
}

/* Option */
div[role="option"]{
    background:#1E3A5F !important;
    color:white !important;
}

/* Hover */
div[role="option"]:hover{
    background:#2563EB !important;
}

/* Selected */
div[aria-selected="true"]{
    background:#38BDF8 !important;
    color:black !important;
}
header[data-testid="stHeader"]{
    background:#1F3B5B !important;
}

div[data-testid="stToolbar"]{
    background:#1F3B5B !important;
}

.stApp{
    background:#1F3B5B;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Model
# ---------------------------------

model = joblib.load("Flight_Delay_Model.pkl")

airport_from_mapping = joblib.load("airport_from_mapping.pkl")
airport_to_mapping = joblib.load("airport_to_mapping.pkl")

# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title("✈️ Flight Delay Prediction")

st.sidebar.markdown("---")

st.sidebar.success("Model : Optimized Random Forest")

st.sidebar.info("Accuracy : 65.20%")

st.sidebar.markdown("---")

st.sidebar.subheader("👩‍💻 Developer")

st.sidebar.markdown("### **Sadia Khatun**")

st.sidebar.markdown("📧 **sadia2305341069@diu.edu.bd**")

st.sidebar.markdown("💻 **Built with:** Python, Streamlit")

st.sidebar.markdown("🤖 **Algorithm:** Random Forest")
st.sidebar.markdown("📊 Dataset: US Airlines")





st.sidebar.markdown("---")

st.sidebar.caption("Machine Learning Project")

# ---------------------------------
# Main Title
# ---------------------------------

st.title("✈️ Flight Delay Prediction System")

st.write(
    "Predict whether a flight will be delayed using Machine Learning."
)

st.markdown("""
<div style="
background:#1E293B;
padding:20px;
border-radius:15px;
border-left:6px solid #38BDF8;
">

### 📌 Project Information

This application predicts flight delay using an Optimized Random Forest Model.

- 🤖 Model : Random Forest
- 📊 Accuracy : 65.20%
- 📁 Dataset : US Airlines Flight Delay Dataset

</div>
""", unsafe_allow_html=True)

st.divider()
# ---------------------------------
# User Input
# ---------------------------------

st.header("✈️ Enter Flight Information")

st.caption("Fill in the flight information below.")

col1, col2 = st.columns(2)

# ---------------- Left Column ----------------

with col1:

    flight = st.number_input(
        "✈️ Flight Number",
        min_value=1,
        value=100
    )

    time = st.number_input(
        "🕒 Departure Time (HHMM)",
        min_value=0,
        max_value=2359,
        value=900
    )

    length = st.number_input(
        "📏 Flight Length (Minutes)",
        min_value=10,
        value=120
    )

    day = st.selectbox(
        "📅 Day of Week",
        [1,2,3,4,5,6,7]
    )

    weekend = st.selectbox(
        "🌴 Weekend",
        [0,1],
        format_func=lambda x: "Yes" if x==1 else "No"
    )

# ---------------- Right Column ----------------

with col2:

    airline = st.selectbox(
        "✈️ Airline",
        [
            "DL","OO","B6","US","FL","WN",
            "CO","AA","YV","EV","XE","9E",
            "OH","UA","MQ","AS","F9","HA"
        ]
    )

    airport_from = st.selectbox(
        "🛫 Airport From",
        sorted(list(airport_from_mapping.keys()))
    )

    airport_to = st.selectbox(
        "🛬 Airport To",
        sorted(list(airport_to_mapping.keys()))
    )

    length_category = st.selectbox(
        "📏 Length Category",
        [0,1,2]
    )

    time_category = st.selectbox(
        "🕒 Time Category",
        [0,1,2]
    )

st.markdown("---")

st.write("")

# ---------------------------------
# Prediction Button
# ---------------------------------

predict_btn = st.button(
    "🔍 Predict Flight Delay",
    use_container_width=True
)
# ---------------------------------
# Prediction
# ---------------------------------

if predict_btn:

    # Airline Encoding
    airline_mapping = {
        "9E":0,
        "AA":1,
        "AS":2,
        "B6":3,
        "CO":4,
        "DL":5,
        "EV":6,
        "F9":7,
        "FL":8,
        "HA":9,
        "MQ":10,
        "OH":11,
        "OO":12,
        "UA":13,
        "US":14,
        "WN":15,
        "XE":16,
        "YV":17
    }

    airline = airline_mapping[airline]

    airport_from = airport_from_mapping[airport_from]
    airport_to = airport_to_mapping[airport_to]

    # Create DataFrame
    input_data = pd.DataFrame({

        "Flight":[flight],
        "Time":[time],
        "Length":[length],
        "Airline":[airline],
        "AirportFrom":[airport_from],
        "AirportTo":[airport_to],
        "DayOfWeek":[day],
        "Weekend":[weekend],
        "Length_Category":[length_category],
        "Time_Category":[time_category]

    })

    # Prediction

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    confidence = probability.max()*100

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        if prediction[0] == 1:
            st.error("🔴 Flight Status : DELAYED")
        else:
            st.success("🟢 Flight Status : ON TIME")
            st.balloons()

    with col2:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        if confidence >= 90:
            st.success("✅ Excellent Confidence")

        elif confidence >= 75:
            st.info("🟢 High Confidence")

        elif confidence >= 60:
            st.warning("🟡 Medium Confidence")

        else:
            st.error("🔴 Low Confidence")

    st.markdown("---")

    with st.expander("📄 View Input Data"):
        st.dataframe(input_data, use_container_width=True)
# Footer
# ---------------------------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:gray">

Made with ❤️ using <b>Python</b>, <b>Streamlit</b> & <b>Machine Learning</b>

<br><br>

© 2026 Sadia Khatun

</div>
""",
unsafe_allow_html=True
)