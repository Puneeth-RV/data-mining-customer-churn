import pandas as pd
from pathlib import Path # Import Pathlib

# Load the dataset
try:
    # Construct an absolute path relative to the current script's directory
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(data_path)
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
    print(f"Error: The dataset file was not found. Please ensure '{data_path.name}' is in the '{data_path.parent}' directory.")
except Exception as e:
    print(f"An error occurred: {e}")
