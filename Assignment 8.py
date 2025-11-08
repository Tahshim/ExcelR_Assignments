import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Prediction — Logistic Regression", page_icon="🩺")

st.title("🩺 Diabetes Prediction (Logistic Regression)")
st.write("This app loads a trained Logistic Regression pipeline (imputer + scaler + model) and predicts the probability of diabetes.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("diabetes_logreg_pipeline.pkl")

pipe = load_model()

# Sidebar inputs
st.sidebar.header("Input Features")
def user_inputs():
    # Read feature names from model training (fixed list to align with training columns)
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    vals = []
    for c in columns:
        vals.append(st.sidebar.number_input(c, value=0.0, step=0.1))
    return np.array(vals).reshape(1, -1), list(columns)

X_input, cols = user_inputs()

if st.button("Predict"):
    proba = pipe.predict_proba(X_input)[0,1]
    pred = int(proba >= 0.5)
    st.subheader("Results")
    st.write(f"Predicted probability of diabetes: **{proba:.3f}**")
    st.write(f"Predicted class (0/1): **{pred}**")
    st.info("Threshold is set at 0.5 by default. You can change this in the code if needed.")

st.caption("Model: LogisticRegression (imputer=median, scaler=StandardScaler).")
