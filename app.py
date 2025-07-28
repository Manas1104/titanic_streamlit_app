# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # ✅ Needed for pd.get_dummies and pd.concat
import model_utils

st.set_page_config(page_title="Data Preprocessing Comparison", layout="wide")
st.title("🧪 Data Preprocessing Mini-Project")
st.markdown("Compare model accuracy on raw vs. preprocessed data")

# Sidebar dataset selector
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ("Heart Disease", "Loan Prediction", "Student Performance")
)

# Load and preprocess
if dataset_choice == "Heart Disease":
    df_raw = model_utils.load_heart_disease()
    df_clean = model_utils.preprocess_heart_data(df_raw)
    target_col = "target"

elif dataset_choice == "Loan Prediction":
    df_raw = model_utils.load_loan_data()
    df_raw = df_raw.dropna(subset=["Loan_Status"])  # remove rows where target is missing
    df_clean = model_utils.preprocess_loan_data(df_raw)
    target_col = "Loan_Status"

elif dataset_choice == "Student Performance":
    df_raw = model_utils.load_student_data()
    df_clean = model_utils.preprocess_student_data(df_raw)
    target_col = "math score"  # You can change to 'reading score' or 'writing score'

# Prepare raw data for baseline model
df_raw_model = df_raw.dropna()
df_raw_encoded = pd.get_dummies(df_raw_model.drop(target_col, axis=1), drop_first=True)
y_raw = df_raw_model[target_col]
X_raw = df_raw_encoded.fillna(0)

acc_raw = model_utils.train_and_score(pd.concat([X_raw, y_raw], axis=1), target_col)
acc_clean, report_clean = model_utils.train_and_report(df_clean, target_col)

# Visuals
st.subheader("📈 Accuracy Comparison")
st.write(f"**Raw Accuracy:** {acc_raw:.2%}")
st.write(f"**Preprocessed Accuracy:** {acc_clean:.2%}")

st.bar_chart({
    "Accuracy": {
        "Raw": acc_raw,
        "Preprocessed": acc_clean
    }
})

st.subheader("📋 Classification Report (Preprocessed)")
st.json(report_clean)

st.subheader("🧼 Preprocessed Data Correlation")
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
st.pyplot(fig_corr)

# Data display
with st.expander("🔍 Raw Data Sample"):
    st.dataframe(df_raw.head())

with st.expander("🧹 Preprocessed Data Sample"):
    st.dataframe(df_clean.head())
