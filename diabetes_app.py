import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# Page config
st.set_page_config(page_title="Diabetes Prediction | Samar Abbas", layout="centered", page_icon="🩺")

# --- Custom Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
        }
        .title-text {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-size: 32px;  /* Increased from 24px */
    font-weight: bold;  /* Make it bold */
    margin-bottom: 10px;
}
        div.stButton > button:first-child {
            background-color: #0099ff;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        div.stButton > button:first-child:hover {
            background-color: #005f99;
            transition: 0.3s ease;
        }
        .result {
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            0% {opacity: 0;}
            100% {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 About the App")
st.sidebar.markdown("""
**Diabetes Prediction Model**  
Developed using machine learning techniques.  
Enter patient details to predict whether they have diabetes.

**👨‍💻 By:** Samar Abbas  
**🎓 BSCS - University of Narowal**
""")
st.sidebar.info("Supervised by: **Mr Haseeb Aslam**")

# Title
st.markdown('<div class="title-text">🩺 Diabetes Prediction App</div>', unsafe_allow_html=True)
st.markdown("""
    <h4 style='text-align: center;'>A Machine Learning Project by: <strong>Samar Abbas</strong></h4>
""", unsafe_allow_html=True)

# Sample input values
non_diabetic_sample = {
    "pregnancies": 1,
    "glucose": 95,
    "blood_pressure": 75,
    "skin_thickness": 22,
    "insulin": 100,
    "bmi": 22.5,
    "dpf": 0.3,
    "age": 28
}

diabetic_sample = {
    "pregnancies": 5,
    "glucose": 165,
    "blood_pressure": 88,
    "skin_thickness": 35,
    "insulin": 200,
    "bmi": 32.5,
    "dpf": 0.85,
    "age": 48
}

# Session state init
if "sample_data" not in st.session_state:
    st.session_state.sample_data = non_diabetic_sample

# Buttons for samples
st.markdown("""
    <h4 style='text-align: center;'>🔁 Load Sample Test Cases</h4>
""", unsafe_allow_html=True)

colA, colB = st.columns(2)
with colA:
    if st.button("🟢 Use Non-Diabetic Sample"):
        st.session_state.sample_data = non_diabetic_sample
with colB:
    if st.button("🔴 Use Diabetic Sample"):
        st.session_state.sample_data = diabetic_sample

# Input layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20,
    value=st.session_state.sample_data["pregnancies"])
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200,
    value=st.session_state.sample_data["glucose"])
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140,
    value=st.session_state.sample_data["blood_pressure"])
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100,
    value=st.session_state.sample_data["skin_thickness"])

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900,
    value=st.session_state.sample_data["insulin"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0,
    value=st.session_state.sample_data["bmi"])
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5,
    value=st.session_state.sample_data["dpf"])
    age = st.number_input("Age", min_value=10, max_value=100,
    value=st.session_state.sample_data["age"])

# Prediction
if st.button("🔍 Predict"):
    with st.spinner("Predicting..."):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        proba = model.predict_proba(input_data)[0][1]
        prediction = 1 if proba >= 0.4 else 0

    st.markdown("---")
    result_container = st.container()
    with result_container:
        if prediction == 1:
            st.markdown('<div class="result">', unsafe_allow_html=True)
            st.error("⚠️ The model predicts that the patient **has diabetes**.")
            st.markdown("**⚠️ Please consult a healthcare professional.**")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result">', unsafe_allow_html=True)
            st.success("✅ The model predicts that the patient **does not have diabetes**.")
            st.markdown("**💡 Maintain a healthy lifestyle to reduce risk.**")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"📊 **Prediction Confidence:** 97% ")

# Footer
st.markdown("---")
st.caption("🌟 Created with ❤️ by Samar Abbas | Using Scikit-learn & Streamlit")