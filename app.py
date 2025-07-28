import streamlit as st
import model_utils as mu

st.set_page_config(page_title="ML Model Comparator", layout="centered")
st.title("📊 ML Model Accuracy Comparator")

dataset_name = st.sidebar.selectbox("Select Dataset", ["Heart Disease", "Loan Prediction", "Student Performance"])
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"])

# Load and preprocess data
if dataset_name == "Heart Disease":
    raw_df = mu.load_heart_disease()
    df = mu.preprocess_heart_data(raw_df)
    target = 'target'

elif dataset_name == "Loan Prediction":
    raw_df = mu.load_loan_data()
    df = mu.preprocess_loan_data(raw_df)
    target = 'Loan_Status'

elif dataset_name == "Student Performance":
    raw_df = mu.load_student_data()
    df = mu.preprocess_student_data(raw_df)
    target = 'G3'

# Show data preview
st.subheader("Dataset Preview")
st.dataframe(raw_df.head())

# Model training and evaluation
st.subheader("Training Result")
mu.run_model_analysis(df, target, model_choice)
