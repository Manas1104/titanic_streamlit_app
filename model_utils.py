# model_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_heart_disease():
    url = "https://raw.githubusercontent.com/ansh941/Machine-Learning/master/Heart%20Disease%20UCI/heart.csv"
    return pd.read_csv(url)

def preprocess_heart_data(df):
    df = df.copy()

    # No missing values
    if df.isnull().sum().sum() > 0:
        df.fillna(df.mean(), inplace=True)

    # Normalize numeric features (exclude 'target')
    features = df.drop('target', axis=1)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['target'] = df['target'].values
    return df_scaled

def load_loan_data():
    url = "https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/loan_prediction.csv"
    df = pd.read_csv(url)
    return df

def preprocess_loan_data(df):
    df = df.copy()
    
    df.drop(columns=['Loan_ID'], inplace=True, errors='ignore')
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    scaler = MinMaxScaler()
    features = df.drop('Loan_Status', axis=1)
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['Loan_Status'] = df['Loan_Status'].values
    return df_scaled

def train_and_report(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report

def train_and_score(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
