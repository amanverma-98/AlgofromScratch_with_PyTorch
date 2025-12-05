import streamlit as st
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile

# -------------------------------
# Load saved model parameters
# -------------------------------
checkpoint = torch.load("models.pth")
w = checkpoint["log_w"]
b = checkpoint["log_b"]
w_svm = checkpoint["svm_w"]
b_svm = checkpoint["svm_b"]

scaler = joblib.load("scaler.pkl")

# -------------------------------
# Prediction functions
# -------------------------------
def logistic_predict(X):
    z = X @ w + b
    return 1 / (1 + torch.exp(-z))

def svm_predict(X):
    z = X @ w_svm + b_svm
    return torch.sign(z)

def map_risk(prob):
    if prob < 0.95:
        return "Low"
    elif prob < 0.97:
        return "Moderate"
    else:
        return "High"

def doctor_advice(risk):
    advice_map = {
        "Low": "Maintain healthy lifestyle, diet, and regular exercise. Routine checkups recommended.",
        "Moderate": "Monitor blood pressure and cholesterol. Consider lifestyle modifications and annual checkups.",
        "High": "Consult a cardiologist. Follow prescribed tests and dietary guidance. Immediate medical attention may be required."
    }
    return advice_map[risk]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.markdown("<h1 style='text-align:center;color:#C2185B;'>❤️ Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Interactive tool with risk gauge, predictions, health report, and recommendations.</p>", unsafe_allow_html=True)

# -------------------------------
# Input cards
# -------------------------------
st.markdown("### 🩺 Enter Your Health Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div style='padding:20px;border-radius:15px;background:#ffe6e6;box-shadow:0 4px 12px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    age = st.number_input("🧑 Age", 20, 90, 45)
    sex = st.selectbox("⚥ Gender", ["Female", "Male"])
    cp = st.selectbox("❤️ Chest Pain Type", ["Typical Chest Pain","Atypical Chest Pain","Non-Cardiac Chest Pain"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div style='padding:20px;border-radius:15px;background:#e6f0ff;box-shadow:0 4px 12px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    rest_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol = st.number_input("🧪 Cholesterol Level (mg/dL)", 100, 600, 240)
    thalach = st.number_input("🏃 Max Heart Rate Achieved", 50, 220, 150)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div style='padding:20px;border-radius:15px;background:#e6ffe6;box-shadow:0 4px 12px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    oldpeak = st.number_input("📉 Heart Oxygen Stress Score", 0.0, 10.0, 1.0)
    ex_angina = st.selectbox("🔥 Chest Pain During Exercise", ["No","Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Feature engineering
# -------------------------------
sex_Male = 1 if sex == "Male" else 0
exercise_induced_angina = 1 if ex_angina == "Yes" else 0

# One-hot encoding for chest pain type
cp_typical = 1 if cp == "Typical Chest Pain" else 0
cp_atypical = 1 if cp == "Atypical Chest Pain" else 0
cp_noncardiac = 1 if cp == "Non-Cardiac Chest Pain" else 0

# Default/median values for remaining features
fbs = 0
restecg_normal = 1
restecg_st = 0
slope_flat = 0
slope_upsloping = 1
vessel_zero = 1
vessel_one = vessel_two = vessel_three = 0
thal_no = 0
thal_normal = 1
thal_reversable = 0
# Build input array in correct order
input_data = np.array([[
    age, rest_bp, chol, thalach, oldpeak,
    sex_Male,
    cp_atypical, cp_noncardiac, cp_typical,
    fbs,
    restecg_normal, restecg_st,
    exercise_induced_angina,
    slope_flat, slope_upsloping,
    vessel_one, vessel_three, vessel_two, vessel_zero,
    thal_no, thal_normal, thal_reversable
]])

# Scale input
scaled_input = scaler.transform(input_data)
X_tensor = torch.tensor(scaled_input, dtype=torch.float32)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🔎 Predict Now"):
    log_pred = logistic_predict(X_tensor).item()
    svm_pred = svm_predict(X_tensor).item()

    log_label = "Low Risk" if log_pred < 0.95 else "Moderate Risk" if log_pred < 0.97 else "High Risk"
    svm_label = "Heart Disease" if svm_pred == 1 else "No Heart Disease"

    risk = map_risk(log_pred)
    advice = doctor_advice(risk)

    # -------------------------------
    # Display prediction cards
    # -------------------------------
    st.markdown("### 📊 Model Predictions")
    col1, col2 = st.columns(2)
    card_style = "padding:20px;border-radius:15px;box-shadow:0 4px 12px rgba(0,0,0,0.1);text-align:center;color:white;font-weight:bold;"

    with col1:
        color = "#FF4C4C" if log_pred >= 0.6 else "#FFD633" if log_pred >= 0.3 else "#4CAF50"
        st.markdown(f"<div style='{card_style}background:{color}'><h4>🔐 Logistic Regression</h4><p>{log_label}</p></div>", unsafe_allow_html=True)

    with col2:
        color = "#FF4C4C" if svm_pred == 1 else "#4CAF50"
        st.markdown(f"<div style='{card_style}background:{color}'><h4>⚖️ SVM</h4><p>{svm_label}</p></div>", unsafe_allow_html=True)

    # -------------------------------
    # Doctor recommendation
    # -------------------------------
    st.subheader("👨‍⚕️ Doctor Recommendations")
    if risk == "Low":
        st.success(advice)
    elif risk == "Moderate":
        st.info(advice)
    else:
        st.error(advice)

    # -------------------------------
    # PDF report
    # -------------------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Heart Disease Health Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Sex: {sex}", ln=True)
    pdf.cell(0, 10, f"Chest Pain Type: {cp}", ln=True)
    pdf.cell(0, 10, f"Max Heart Rate: {thalach}", ln=True)
    pdf.cell(0, 10, f"Heart Oxygen Stress Score: {oldpeak}", ln=True)
    pdf.cell(0, 10, f"Resting BP: {rest_bp}", ln=True)
    pdf.cell(0, 10, f"Cholesterol: {chol}", ln=True)
    pdf.cell(0, 10, f"Exercise Induced Angina: {ex_angina}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, f"Logistic Regression: {log_label}", ln=True)
    pdf.cell(0, 10, f"SVM: {svm_label}", ln=True)
    pdf.cell(0, 10, f"Doctor Recommendation: {advice}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 5, "Disclaimer: This report is for informational purposes only and is not a medical diagnosis. "
                        "Please consult a healthcare professional for personalized medical advice.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)

    st.download_button(
        label="📄 Download Health Report (PDF)",
        data=open(temp_file.name, "rb").read(),
        file_name="Heart_Health_Report.pdf",
        mime="application/pdf"
    )

# -------------------------------
# Disclaimer
# -------------------------------
st.markdown(
    "<p style='color:gray;font-size:12px;text-align:center;'>⚠️ Disclaimer: This tool is for informational purposes only and is not a medical diagnosis. "
    "Consult a qualified healthcare professional for personalized medical advice.</p>", 
    unsafe_allow_html=True
)
