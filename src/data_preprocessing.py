import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
try:
    df = pd.read_csv('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully.")

    # --- Preprocessing Steps ---

    # 1. Handle 'TotalCharges' column
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Check for new missing values after coercion
    # print("Missing values after converting TotalCharges:")
    # print(df.isnull().sum())

    # Fill missing 'TotalCharges' with the median (a common strategy for numerical data)
    # This assumes a small number of missing values.
    # If there are many, a more sophisticated imputation might be needed.
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True)
    print(f"Filled missing TotalCharges with median: {median_total_charges}")


    # 2. Drop 'customerID' column
    df.drop('customerID', axis=1, inplace=True)
    print("Dropped 'customerID' column.")

    # 3. Encode Categorical Variables
    # Identify categorical columns (excluding the target 'Churn')
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols.remove('Churn') # Exclude target variable

    # Apply Label Encoding for binary categorical features
    # (assuming binary features will have only two unique values besides None)
    binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    print(f"Applied Label Encoding to binary columns: {binary_cols}")

    # Apply One-Hot Encoding for multi-category features
    multi_category_cols = [col for col in categorical_cols if df[col].nunique() > 2]
    df = pd.get_dummies(df, columns=multi_category_cols, drop_first=True) # drop_first to avoid dummy variable trap
    print(f"Applied One-Hot Encoding to multi-category columns: {multi_category_cols}")
    
    # Encode the target variable 'Churn'
    le_churn = LabelEncoder()
    df['Churn'] = le_churn.fit_transform(df['Churn'])
    print("Encoded 'Churn' target variable.")

    # Display the processed dataframe info and head
    print("--- Processed Dataset Info ---")
    df.info()
    print("--- Processed Dataset Head ---")
    print(df.head())

    # Save the preprocessed data (optional, but good for next steps)
    df.to_csv('../data/processed/telco_churn_preprocessed.csv', index=False)
    print("Preprocessed data saved to '../data/processed/telco_churn_preprocessed.csv'")

except FileNotFoundError:
    print("Error: The dataset file was not found. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the 'data_mining_project/' directory.")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")
