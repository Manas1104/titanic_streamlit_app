import streamlit as st
import pandas as pd
import model_utils as mu

st.set_page_config(page_title="ML Accuracy Comparison", layout="wide")
st.title("🔍 Compare Accuracy: Raw vs Preprocessed Data")

dataset_name = st.selectbox("Choose Dataset", ["Titanic", "Heart Disease", "Loan Prediction", "Student Performance"])

# Load and preprocess
if dataset_name == "Titanic":
    raw_df = mu.load_titanic_data()
    pre_df = mu.preprocess_titanic_data(raw_df)
    target = 'Survived'

elif dataset_name == "Heart Disease":
    raw_df = mu.load_heart_disease()
    pre_df = mu.preprocess_heart_data(raw_df)
    target = 'target'

elif dataset_name == "Loan Prediction":
    raw_df = mu.load_loan_data()
    pre_df = mu.preprocess_loan_data(raw_df)
    target = 'Loan_Status_Y' if 'Loan_Status_Y' in pre_df.columns else 'Loan_Status'

elif dataset_name == "Student Performance":
    raw_df = mu.load_student_data()
    pre_df = mu.preprocess_student_data(raw_df)
    target = 'G3'

# Show data preview
st.subheader("📄 Raw Data")
st.dataframe(raw_df.head())

st.subheader("✅ Preprocessed Data")
st.dataframe(pre_df.head())

# Compare accuracy
raw_acc = mu.evaluate_model(raw_df.dropna(), target)
pre_acc = mu.evaluate_model(pre_df.dropna(), target)

st.markdown("### 📊 Accuracy Comparison")
st.write(f"🔹 Raw Accuracy: **{raw_acc:.4f}**")
st.write(f"✅ Preprocessed Accuracy: **{pre_acc:.4f}**")
