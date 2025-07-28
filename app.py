import streamlit as st
import pandas as pd
import model_utils

st.set_page_config(page_title="Titanic ML - Data Preprocessing", layout="wide")

st.title("🚢 Titanic Survival Prediction")
st.markdown("### Compare model accuracy on raw vs. preprocessed data")

# Load data
df_raw = model_utils.load_data()
df_clean = model_utils.preprocess_data(df_raw)

# Prepare raw data
df_raw_model = df_raw.dropna()
X_raw = pd.get_dummies(df_raw_model.drop("survived", axis=1), drop_first=True)
y_raw = df_raw_model["survived"]
X_raw = X_raw.fillna(0)

# Accuracy comparison
acc_raw = model_utils.train_and_score(pd.concat([X_raw, y_raw], axis=1))
acc_clean = model_utils.train_and_score(df_clean)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize correlation heatmap of preprocessed data
st.subheader("🔍 Feature Correlation (Preprocessed Data)")

fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
st.pyplot(fig_corr)

# Fare distribution before and after log transform
st.subheader("💰 Fare Distribution (Raw vs Log Transformed)")

fig_fare, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df_raw['fare'], kde=True, ax=ax[0])
ax[0].set_title("Original Fare")

if 'fare_log' in df_clean.columns:
    sns.histplot(df_clean['fare_log'], kde=True, ax=ax[1], color='green')
    ax[1].set_title("Log-Transformed Fare")

st.pyplot(fig_fare)


# Display
st.subheader("📈 Model Accuracy")
st.write(f"**Raw Data Accuracy:** {acc_raw:.2%}")
st.write(f"**Preprocessed Data Accuracy:** {acc_clean:.2%}")

st.bar_chart({
    "Accuracy": {
        "Raw": acc_raw,
        "Preprocessed": acc_clean
    }
})
acc_clean, report_clean = model_utils.train_and_report(df_clean)

st.subheader("📋 Classification Report (Preprocessed Data)")
st.json(report_clean)


with st.expander("📄 Raw Dataset"):
    st.dataframe(df_raw.head())

with st.expander("🧹 Preprocessed Dataset"):
    st.dataframe(df_clean.head())
