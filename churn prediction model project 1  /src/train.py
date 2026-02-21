import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


# -----------------------------
# 1. Load Data
# -----------------------------
def load_data():
    df = pd.read_csv("/data/Telco-Customer-Churn.csv")
    df = df.drop(columns=["customerID"])
    return df


# -----------------------------
# 2. Preprocess Data
# -----------------------------
def preprocess_data(df):
    # Fix TotalCharges column
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0"})
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Encode target column
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    # Encode categorical features
    encoders = {}
    object_columns = df.select_dtypes(include="object").columns

    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# -----------------------------
# 3. Split Data
# -----------------------------
def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# -----------------------------
# 4. Apply SMOTE
# -----------------------------
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# -----------------------------
# 5. Train Model
# -----------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


# -----------------------------
# 6. Evaluate Model
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nModel Evaluation Results")
    print("-" * 40)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -----------------------------
# 7. Save Model
# -----------------------------
def save_model(model, encoders, feature_names):
    model_data = {
        "model": model,
        "encoders": encoders,
        "feature_names": feature_names
    }

    with open("models/customer_churn_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("\nModel saved successfully inside models/ folder.")


# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    df, encoders = preprocess_data(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Applying SMOTE...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    print("Training model...")
    model = train_model(X_train_smote, y_train_smote)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    save_model(model, encoders, X_train.columns)


if __name__ == "__main__":
    main()