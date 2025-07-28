import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    return sns.load_dataset("titanic")

def preprocess_data(df):
    df = df.copy()
    
    # Drop problematic or redundant columns
    drop_cols = ['deck', 'embark_town', 'alive']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Fill missing values
    if 'age' in df.columns:
        df['age'].fillna(df['age'].mean(), inplace=True)

    if 'embarked' in df.columns:
        df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)  # Ensure all values are string
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Normalize numerical columns (exclude target)
    if 'survived' in df.columns:
        numeric_cols = df.select_dtypes(include='number').columns.drop('survived')
    else:
        numeric_cols = df.select_dtypes(include='number').columns

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def train_and_score(df):
    X = df.drop("survived", axis=1)
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)
