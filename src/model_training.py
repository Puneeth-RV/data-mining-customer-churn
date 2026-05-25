import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path

# Load the raw dataset and fit preprocessing on the training split only.
try:
    current_dir = Path(__file__).parent
    raw_data_path = current_dir.parent / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(raw_data_path)
    print("Raw dataset loaded successfully.")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Separate features (X) and target (y)
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})

    # 1. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 2. Preprocessing
    # Median imputation and scaling are fit on X_train only to avoid data leakage.
    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    passthrough_cols = ['SeniorCitizen']
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), continuous_cols),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('binary_numeric', 'passthrough', passthrough_cols)
        ]
    )
    print("Preprocessing configured: train-only median imputation, one-hot encoding, and scaling for continuous numeric features.")

    # 3. Algorithm Selection & Implementation (Logistic Regression as a baseline)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        model.fit(X_train, y_train)
    print("Logistic Regression model trained.")
    print("Training parameters: test_size=0.2, random_state=42, stratify=y, solver='lbfgs', max_iter=1000")

    # 4. Model Evaluation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
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
    print(f"Error: The dataset file was not found. Please ensure '{raw_data_path.name}' is in the '{raw_data_path.parent}' directory.")
except Exception as e:
    print(f"An error occurred during model training and evaluation: {e}")
