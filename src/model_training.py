import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Load the preprocessed dataset
try:
    df = pd.read_csv('../data/processed/telco_churn_preprocessed.csv')
    print("Preprocessed dataset loaded successfully.")

    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 1. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 2. Feature Scaling
    # Apply StandardScaler to numerical features
    # Identify numerical columns (excluding boolean and already-encoded categorical)
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    print("Numerical features scaled using StandardScaler.")

    # 3. Algorithm Selection & Implementation (Logistic Regression as a baseline)
    model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets and binary classification
    model.fit(X_train, y_train)
    print("Logistic Regression model trained.")

    # 4. Model Evaluation
    y_pred = model.predict(X_test)

    print("--- Model Evaluation Results (Logistic Regression) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))

except FileNotFoundError:
    print("Error: The preprocessed dataset file was not found. Please ensure 'telco_churn_preprocessed.csv' is in the 'data_mining_project/' directory.")
except Exception as e:
    print(f"An error occurred during model training and evaluation: {e}")
