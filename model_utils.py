import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Data Loaders
def load_heart_disease():
    return pd.read_csv("Data/heart_data.csv")

def load_loan_data():
    return pd.read_csv("Data/loan_data.csv")

def load_student_data():
    return pd.read_csv("Data/student_data.csv")

# Preprocessing Functions
def preprocess_heart_data(df):
    return df.copy()

def preprocess_loan_data(df):
    df = df.dropna()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])
    return df

def preprocess_student_data(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])
    return df

# Training and Evaluation
def run_model_analysis(df, target_col, model_name):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Accuracy: **{acc:.2f}**")
    
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
