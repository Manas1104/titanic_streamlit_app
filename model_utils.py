import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_heart_disease():
    return pd.read_csv("heart.csv")

def preprocess_heart_data(df):
    df = df.dropna()
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

def load_loan_data():
    return pd.read_csv("loan.csv")

def preprocess_loan_data(df):
    df = df.dropna()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def load_student_data():
    return pd.read_csv("StudentsPerformance.csv")

def preprocess_student_data(df):
    df = df.dropna()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def train_and_evaluate(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)
