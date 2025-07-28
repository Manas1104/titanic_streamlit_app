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

with st.expander("📄 Raw Dataset"):
    st.dataframe(df_raw.head())

with st.expander("🧹 Preprocessed Dataset"):
    st.dataframe(df_clean.head())
