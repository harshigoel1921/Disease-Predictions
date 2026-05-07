import streamlit as st
import numpy as np
import joblib

# Load models
heart_model = joblib.load('models/heart_model.pkl')
diabetes_model = joblib.load('models/diabetes_model.pkl')
liver_model = joblib.load('models/liver_model.pkl')

# Page config
st.set_page_config(page_title="AI Disease Predictor", layout="wide")

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}
.main {
    background: transparent;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
}
h1, h2, h3 {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center;'>🧠 AI Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Smart Health Risk Analysis using Machine Learning</h4>", unsafe_allow_html=True)

st.markdown("---")

# ---------- SIDEBAR ----------
disease = st.sidebar.selectbox(
    "🔍 Choose Prediction",
    ["Heart Disease", "Diabetes", "Liver Disease"]
)

st.sidebar.markdown("💡 Enter patient details carefully")

# ---------- HEART ----------
if disease == "Heart Disease":
    st.markdown("## ❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex (0=F,1=M)", [0,1])
        cp = st.number_input("Chest Pain Type")

    with col2:
        trestbps = st.number_input("Blood Pressure")
        chol = st.number_input("Cholesterol")
        fbs = st.selectbox("Fasting Sugar >120", [0,1])

    with col3:
        thalach = st.number_input("Max Heart Rate")
        exang = st.selectbox("Exercise Angina", [0,1])
        oldpeak = st.number_input("Oldpeak")

    if st.button("🚀 Predict"):
        values = np.array([[age, sex, cp, trestbps, chol, fbs, 0, thalach, exang, oldpeak, 0, 0, 1]])
        prediction = heart_model.predict(values)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk - Healthy")

# ---------- DIABETES ----------
elif disease == "Diabetes":
    st.markdown("## 🩸 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies")
        glucose = st.number_input("Glucose Level")
        bp = st.number_input("Blood Pressure")
        skin = st.number_input("Skin Thickness")

    with col2:
        insulin = st.number_input("Insulin")
        bmi = st.number_input("BMI")
        dpf = st.number_input("DPF")
        age = st.number_input("Age")

    if st.button("🚀 Predict"):
        values = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(values)

        if prediction[0] == 1:
            st.error("⚠️ Diabetic Risk Detected")
        else:
            st.success("✅ No Diabetes Detected")

# ---------- LIVER ----------
else:
    st.markdown("## 🧪 Liver Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
        gender = st.selectbox("Gender", [0,1])
        tb = st.number_input("Total Bilirubin")

    with col2:
        db = st.number_input("Direct Bilirubin")
        ap = st.number_input("Alkaline Phosphotase")
        alt = st.number_input("ALT")

    with col3:
        ast = st.number_input("AST")
        tp = st.number_input("Total Proteins")
        alb = st.number_input("Albumin")

    agr = st.number_input("A/G Ratio")

    if st.button("🚀 Predict"):
        values = np.array([[age, gender, tb, db, ap, alt, ast, tp, alb, agr]])
        prediction = liver_model.predict(values)

        if prediction[0] == 1:
            st.error("⚠️ Liver Disease Detected")
        else:
            st.success("✅ Liver is Healthy")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<center>⚡ Built with Machine Learning & Streamlit</center>", unsafe_allow_html=True)