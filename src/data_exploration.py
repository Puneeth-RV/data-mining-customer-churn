import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully.")
    
    # Initial Data Exploration
    print("--- Dataset Info ---")
    df.info()
    
    print("--- First 5 Rows ---")
    print(df.head())
    
    print("--- Descriptive Statistics ---")
    print(df.describe())
    
    print("--- Missing Values ---")
    print(df.isnull().sum())

except FileNotFoundError:
    print("Error: The dataset file was not found. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the 'data_mining_project/' directory.")
except Exception as e:
    print(f"An error occurred: {e}")
