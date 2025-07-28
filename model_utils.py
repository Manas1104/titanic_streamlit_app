import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Loan Prediction
def load_loan_data():
    return pd.read_csv("loan_data.csv")

def train_loan_model():
    df = load_loan_data()
    df = df.dropna()

    X = df.drop("Loan_Status", axis=1).select_dtypes(include=["number"])
    y = (df["Loan_Status"] == "Y").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# 2. Heart Disease
def load_heart_data():
    return pd.read_csv("heart_data.csv")

def train_heart_model():
    df = load_heart_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# 3. Student Performance
def load_student_data():
    return pd.read_csv("student_data.csv")

def train_student_model():
    df = load_student_data()

    df = df.dropna()
    X = df.drop("pass", axis=1).select_dtypes(include=["number"])
    y = df["pass"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy
