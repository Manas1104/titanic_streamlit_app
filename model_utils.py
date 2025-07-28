import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ---------------- Dataset Loaders ----------------

def load_titanic_data():
    return pd.read_csv("data/titanic.csv")

def load_heart_disease():
    return pd.read_csv("data/heart.csv")

def load_loan_data():
    return pd.read_csv("data/loan.csv")

def load_student_data():
    return pd.read_csv("data/student.csv")


# ---------------- Preprocessing ----------------

def preprocess_titanic_data(df):
    df = df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = pd.get_dummies(df, drop_first=True)
    return df

def preprocess_heart_data(df):
    return pd.get_dummies(df, drop_first=True)

def preprocess_loan_data(df):
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df

def preprocess_student_data(df):
    return pd.get_dummies(df, drop_first=True)


# ---------------- Modeling ----------------

def evaluate_model(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)
