import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path

st.set_page_config(layout="wide")

# --- Helper Functions (to encapsulate previous steps) ---

def show_centered_plot(fig, width="medium"):
    """Render matplotlib figures at a controlled width."""
    widths = {
        "small": [1.5, 2, 1.5],
        "medium": [1, 3, 1],
        "large": [0.5, 4, 0.5],
    }
    left, center, right = st.columns(widths[width])
    with center:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)

@st.cache_data
def load_raw_data():
    """Loads the raw Telco Customer Churn dataset."""
    # Construct an absolute path relative to the current script's directory
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    try:
        df_raw = pd.read_csv(data_path)
        return df_raw
    except FileNotFoundError:
        st.error(f"Raw dataset not found. Please ensure '{data_path.name}' is in the '{data_path.parent}' directory.")
        return None

@st.cache_data
def preprocess_data(df):
    """Applies the same preprocessing steps as in data_preprocessing.py."""
    df_processed = df.copy()

    # Handle 'TotalCharges' column
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    median_total_charges = df_processed['TotalCharges'].median()
    df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(median_total_charges)

    # Drop 'customerID' column
    df_processed.drop('customerID', axis=1, inplace=True)

    # Encode Categorical Variables
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
    
    # Check if 'Churn' is in categorical_cols before removing
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn') # Exclude target variable

    binary_cols = [col for col in categorical_cols if df_processed[col].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])

    multi_category_cols = [col for col in categorical_cols if df_processed[col].nunique() > 2]
    df_processed = pd.get_dummies(df_processed, columns=multi_category_cols, drop_first=True)
    
    # Encode the target variable 'Churn'
    if 'Churn' in df_processed.columns:
        le_churn = LabelEncoder()
        df_processed['Churn'] = le_churn.fit_transform(df_processed['Churn'])
    
    return df_processed

@st.cache_resource
def train_model(X_train, y_train):
    """Trains a leakage-safe Logistic Regression pipeline."""
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

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        model.fit(X_train, y_train)
    return model

# --- Streamlit App Layout ---
st.title("Customer Churn Prediction Dashboard")
st.markdown("---")

# Load Data
df_raw = load_raw_data()
if df_raw is None:
    st.stop()

st.header("1. Raw Data Overview")
st.write("Full raw dataset:")
st.dataframe(df_raw)
st.write(f"Raw Data Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Preprocess Data
df_processed = preprocess_data(df_raw.copy())

st.header("2. Preprocessed Data Overview")
st.write("Full preprocessed dataset:")
st.dataframe(df_processed)
st.write(f"Processed Data Shape: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")

st.markdown("---")

# Churn Distribution
st.header("3. Churn Distribution")
churn_counts = df_raw['Churn'].value_counts()
fig_churn, ax_churn = plt.subplots(figsize=(5.5, 3.5))
sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax_churn, palette='viridis')
ax_churn.set_title('Distribution of Customer Churn')
ax_churn.set_xlabel('Churn')
ax_churn.set_ylabel('Number of Customers')
fig_churn.tight_layout()
show_centered_plot(fig_churn, width="small")
count_col1, count_col2 = st.columns(2)
count_col1.metric("No-churn customers", churn_counts.get('No', 0))
count_col2.metric("Churn customers", churn_counts.get('Yes', 0))

st.markdown("---")

# Feature Distributions (Interactive)
st.header("4. Feature Distributions (Raw Data)")
selected_feature = st.selectbox(
    "Select a feature to visualize its distribution:",
    df_raw.columns.drop(['customerID', 'Churn', 'TotalCharges', 'MonthlyCharges', 'tenure'])
)

if selected_feature:
    fig_feature, ax_feature = plt.subplots(figsize=(7, 4.2))
    if df_raw[selected_feature].dtype == 'object': # Categorical
        sns.countplot(data=df_raw, x=selected_feature, hue='Churn', ax=ax_feature, palette='coolwarm')
        ax_feature.set_title(f'Distribution of {selected_feature} by Churn')
        ax_feature.set_xticklabels(ax_feature.get_xticklabels(), rotation=45, ha='right')
    else: # Numerical (SeniorCitizen is int, but effectively categorical)
        sns.histplot(data=df_raw, x=selected_feature, hue='Churn', kde=True, ax=ax_feature, palette='coolwarm')
        ax_feature.set_title(f'Distribution of {selected_feature} by Churn')
    fig_feature.tight_layout()
    show_centered_plot(fig_feature, width="medium")

st.markdown("---")

# --- Model Training and Evaluation ---
st.header("5. Model Training and Evaluation")

# Separate raw features and target. Preprocessing is fit after the split to avoid leakage.
df_model = df_raw.copy()
df_model['TotalCharges'] = pd.to_numeric(df_model['TotalCharges'], errors='coerce')
X = df_model.drop(['customerID', 'Churn'], axis=1)
y = df_model['Churn'].map({'No': 0, 'Yes': 1})

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
model = train_model(X_train, y_train)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
    y_pred = model.predict(X_test)

# Model Performance Metrics
st.subheader("Model Performance Metrics (Logistic Regression)")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1-Score", f"{f1:.4f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax_cm.set_ylabel('Actual Label')
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_title('Confusion Matrix')
fig_cm.tight_layout()
show_centered_plot(fig_cm, width="small")

st.markdown("---")

st.header("Conclusion & Future Work")
st.write("This dashboard presents the initial findings of a Logistic Regression model for customer churn prediction. The model achieved an accuracy of approximately 80%, with detailed metrics and visualizations provided. Future work could involve exploring more complex models, handling data imbalance, and performing hyperparameter tuning to further improve performance.")
