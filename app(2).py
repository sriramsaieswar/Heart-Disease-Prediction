# ==========================================
# Heart Disease Prediction Using Logistic Regression
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("framingham.csv")

# ------------------------------------------
# Data Cleaning
# ------------------------------------------

df.drop(columns=["education"], inplace=True)
df.rename(columns={"male": "Sex_male"}, inplace=True)
df.dropna(inplace=True)

# ------------------------------------------
# Select Features
# ------------------------------------------

X = df[['age',
        'Sex_male',
        'cigsPerDay',
        'totChol',
        'sysBP',
        'glucose']]

y = df['TenYearCHD']

# ------------------------------------------
# Feature Scaling
# ------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=4
)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# Model Accuracy
# ------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ------------------------------------------
# Streamlit UI
# ------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Prediction System")

st.write("### Enter Patient Details")

age = st.number_input("Age", 20, 100, 52)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Sex_male = 1 if gender == "Male" else 0

cigsPerDay = st.number_input(
    "Cigarettes Per Day",
    0,
    50,
    10
)

totChol = st.number_input(
    "Total Cholesterol",
    100,
    600,
    220
)

sysBP = st.number_input(
    "Systolic Blood Pressure",
    80,
    250,
    140
)

glucose = st.number_input(
    "Glucose Level",
    40,
    300,
    95
)

if st.button("Predict"):

    patient = np.array([[age,
                         Sex_male,
                         cigsPerDay,
                         totChol,
                         sysBP,
                         glucose]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)

    probability = model.predict_proba(patient)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"Prediction Probability: {probability:.2%}")

st.markdown("---")
st.write(f"**Model Accuracy:** {accuracy:.2%}")
