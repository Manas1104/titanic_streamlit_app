import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Shared preprocessing function
def scale_numeric_features(df, target_column):
    df = df.dropna()
    X = df.drop(target_column, axis=1).select_dtypes(include=["number"])
    y = df[target_column] if target_column != "Loan_Status" else (df[target_column] == "Y").astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# ---------------- LOAN ------------------
def load_loan_data():
    return pd.read_csv("loan_data.csv")

def preprocess_loan_data():
    df = load_loan_data()
    return scale_numeric_features(df, "Loan_Status")

def train_loan_model(preprocessed=False):
    if preprocessed:
        X, y = preprocess_loan_data()
    else:
        df = load_loan_data().dropna()
        X = df.drop("Loan_Status", axis=1).select_dtypes(include=["number"])
        y = (df["Loan_Status"] == "Y").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# ---------------- HEART ------------------
def load_heart_data():
    return pd.read_csv("heart_data.csv")

def preprocess_heart_data():
    df = load_heart_data()
    return scale_numeric_features(df, "target")

def train_heart_model(preprocessed=False):
    if preprocessed:
        X, y = preprocess_heart_data()
    else:
        df = load_heart_data()
        X = df.drop("target", axis=1)
        y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# ---------------- STUDENT ------------------
def load_student_data():
    return pd.read_csv("student_data.csv")

def preprocess_student_data():
    df = load_student_data()
    return scale_numeric_features(df, "pass")

def train_student_model(preprocessed=False):
    if preprocessed:
        X, y = preprocess_student_data()
    else:
        df = load_student_data().dropna()
        X = df.drop("pass", axis=1).select_dtypes(include=["number"])
        y = df["pass"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy
