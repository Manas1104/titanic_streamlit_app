import streamlit as st
import pandas as pd
import model_utils as mu

st.set_page_config(page_title="Model Accuracy Dashboard", layout="wide")
st.title("Compare Model Accuracy: Raw vs Preprocessed Data")

dataset_name = st.selectbox("Select Dataset", [
    "Heart Disease",
    "Loan Prediction",
    "Student Performance"
])

if dataset_name == "Heart Disease":
    raw_df = mu.load_heart_disease()
    pre_df = mu.preprocess_heart_data(raw_df.copy())
    target = 'target'
elif dataset_name == "Loan Prediction":
    raw_df = mu.load_loan_data()
    pre_df = mu.preprocess_loan_data(raw_df.copy())
    target = 'Loan_Status'
elif dataset_name == "Student Performance":
    raw_df = mu.load_student_data()
    pre_df = mu.preprocess_student_data(raw_df.copy())
    target = 'math score'

st.subheader("Raw Dataset Preview")
st.dataframe(raw_df.head())

st.subheader("Preprocessed Dataset Preview")
st.dataframe(pre_df.head())

raw_acc, raw_report = mu.train_and_evaluate(raw_df.copy(), target)
pre_acc, pre_report = mu.train_and_evaluate(pre_df.copy(), target)

st.subheader("Model Accuracy Comparison")
st.write(f"🔴 Raw Data Accuracy: **{raw_acc:.2f}**")
st.write(f"🟢 Preprocessed Data Accuracy: **{pre_acc:.2f}**")

with st.expander("Classification Report (Raw Data)"):
    st.text(raw_report)

with st.expander("Classification Report (Preprocessed Data)"):
    st.text(pre_report)
