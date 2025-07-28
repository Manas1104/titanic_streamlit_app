import pandas as pd
import seaborn as sns
import numpy as np  # ✅ Add this line
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    return sns.load_dataset("titanic")

def preprocess_data(df):
    df = df.copy()

    # Drop irrelevant or high-missing columns
    drop_cols = ['deck', 'embark_town', 'alive']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Fill missing values
    df['age'] = df['age'].fillna(df['age'].mean())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df.dropna(inplace=True)

    # Feature Engineering
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['fare_per_person'] = df['fare'] / df['family_size']

    # Log scale fare
    df['fare_log'] = np.log1p(df['fare'])

    # Encode categorical variables
    for col in df.select_dtypes(include=['object', 'category']):
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Normalize numeric features (except target)
        # Normalize numeric values (exclude target)
    if 'survived' in df.columns:
        numeric_cols = df.select_dtypes(include='number').columns.drop('survived')
    else:
        numeric_cols = df.select_dtypes(include='number').columns

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ✅ Rename columns for better clarity
    df.rename(columns={
        'pclass': 'ticket_class',
        'sex': 'gender',
        'sibsp': 'siblings_spouses_aboard',
        'parch': 'parents_children_aboard',
        'fare': 'ticket_fare',
        'embarked': 'port_of_embarkation',
        'class': 'ticket_class_label',
        'who': 'person_type',
        'adult_male': 'is_adult_male',
        'alone': 'traveling_alone',
        'family_size': 'total_family_members_aboard',
        'fare_per_person': 'fare_divided_by_family_size',
        'fare_log': 'log_fare'
    }, inplace=True)

    return df

    

    return df

def train_and_report(df):
    X = df.drop("survived", axis=1)
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report

def train_and_score(df):
    X = df.drop("survived", axis=1)
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)
