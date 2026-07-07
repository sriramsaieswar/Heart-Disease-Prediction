# ==========================================
# Heart Disease Prediction Using Logistic Regression + Streamlit
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv("framingham.csv")

# Data Cleaning
df.drop(columns=["education"], inplace=True)
df.rename(columns={"male": "Sex_male"}, inplace=True)
df.dropna(inplace=True)

# Feature Selection
X = df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose']]
y = df['TenYearCHD']

# Feature Scaling
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=4
)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Disease","Disease"],
    yticklabels=["No Disease","Disease"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Streamlit UI
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's health details below.")

age = st.number_input("Age", 20, 100, 52)
gender = st.selectbox("Gender", ["Female", "Male"])
Sex_male = 1 if gender == "Male" else 0
cigsPerDay = st.number_input("Cigarettes Per Day", 0, 50, 10)
totChol = st.number_input("Total Cholesterol", 100, 600, 220)
sysBP = st.number_input("Systolic Blood Pressure", 80, 250, 140)
glucose = st.number_input("Glucose Level", 40, 300, 95)

if st.button("Predict Heart Disease"):
    new_patient = np.array([[age, Sex_male, cigsPerDay, totChol, sysBP, glucose]])
    new_patient = scaler.transform(new_patient)

    prediction = model.predict(new_patient)
    probability = model.predict_proba(new_patient)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"**Probability of Heart Disease:** {probability:.2%}")
