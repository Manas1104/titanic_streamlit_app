import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    return sns.load_dataset("titanic")

def preprocess_data(df):
    df = df.drop(columns=['deck', 'embark_town', 'alive'])
    df = df.copy()
    df['age'].fillna(df['age'].mean(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include='number').columns.drop('survived')
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
