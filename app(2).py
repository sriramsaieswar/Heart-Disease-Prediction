import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# ----------------------- Load Dataset -----------------------
@st.cache_data
def load_data():
    data = pd.read_csv("heart.csv")      # Keep heart.csv in the same folder
    X = data.drop(columns="target", axis=1)
    y = data["target"]
    return X, y

X, y = load_data()

# ----------------------- Train Model ------------------------
@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

model = train_model()

# ----------------------- UI -----------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("Enter the patient details below and click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic",
        ][x],
    )
    trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# Convert Sex
sex = 1 if sex == "Male" else 0

input_data = np.array(
    [
        [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ]
    ]
)

st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: The person is likely to have Heart Disease.")
        st.write(f"**Confidence:** {probability[1]*100:.2f}%")
    else:
        st.success("✅ Low Risk: The person is unlikely to have Heart Disease.")
        st.write(f"**Confidence:** {probability[0]*100:.2f}%")

st.markdown("---")
st.subheader("Dataset Preview")
st.dataframe(pd.read_csv("heart.csv").head())

st.sidebar.header("About")
st.sidebar.info(
    """
    **Heart Disease Prediction System**

    Machine Learning Model:
    - Logistic Regression

    Input Features:
    - Age
    - Sex
    - Chest Pain Type
    - Blood Pressure
    - Cholesterol
    - Fasting Blood Sugar
    - ECG Results
    - Maximum Heart Rate
    - Exercise Induced Angina
    - Oldpeak
    - Slope
    - Major Vessels
    - Thal
    """
)
