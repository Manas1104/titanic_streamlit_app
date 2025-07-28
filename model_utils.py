# model_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Heart Disease Dataset
# -----------------------

def load_heart_disease():
    return pd.read_csv("heart.csv")

def preprocess_heart_data(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    features = df.columns.drop('target')
    df_scaled[features] = scaler.fit_transform(df_scaled[features])
    return df_scaled


# -----------------------
# Loan Prediction Dataset
# -----------------------

def load_loan_data():
    return pd.read_csv("train.csv")

def preprocess_loan_data(df):
    df = df.drop(columns=["Loan_ID"], errors="ignore")
    df = df.dropna()
    label_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])
    return df


# -----------------------
# Student Performance Dataset
# -----------------------

def load_student_data():
    return pd.read_csv("student.csv")

def preprocess_student_data(df):
    df = df.dropna()
    df = df.copy()
    # Drop free text columns like "name" if they exist
    drop_cols = ['name', 'student id', 'id']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Choose a classification target, here: classify pass/fail in math
    if 'math score' in df.columns:
        df['math score'] = (df['math score'] >= 50).astype(int)

    label_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])
    return df


# -----------------------
# Training & Evaluation
# -----------------------

def train_and_evaluate(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    return acc, report
