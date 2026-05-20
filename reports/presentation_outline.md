# Data Mining Mini-Project: Customer Churn Prediction
## PowerPoint Presentation Outline

### Slide 1: Title Slide
*   **Title:** Customer Churn Prediction using Classification
*   **Student Name:** [Your Name]
*   **Registration Number:** [Your Registration Number]
*   **Course:** Data Mining (CSC302A)
*   **Semester/Batch:** 6th / 2023
*   **University:** Ramaiah University of Applied Sciences

### Slide 2: Problem Statement & Objective
*   **Problem:** Customer churn is a significant challenge for telecommunications companies, leading to revenue loss. Retaining existing customers is more cost-effective than acquiring new ones.
*   **Objective:** To build a predictive model to identify customers at high risk of churning, enabling proactive retention strategies.
*   **Real-world Relevance:** Helps companies reduce churn rate, optimize marketing efforts, and improve customer satisfaction.

### Slide 3: Dataset Overview
*   **Dataset Name:** Telco Customer Churn Dataset
*   **Source:** Publicly available (e.g., Kaggle, IBM)
*   **Size:** 7043 customer records, 20 features + 1 target variable
*   **Key Features:**
    *   Demographics (Gender, SeniorCitizen, Partner, Dependents)
    *   Account Info (Tenure, Contract, MonthlyCharges, TotalCharges, PaymentMethod)
    *   Services (PhoneService, InternetService, OnlineSecurity, TechSupport, etc.)
*   **Target Variable:** `Churn` (Yes/No)

### Slide 4: Methodology - Data Preprocessing
*   **Initial Data Exploration:** Identified `TotalCharges` as `object` type with hidden non-numeric values.
*   **Handling `TotalCharges`:** Converted to numeric, missing values imputed with median (1397.475).
*   **Feature Removal:** Dropped `customerID` (no predictive power).
*   **Categorical Encoding:**
    *   Label Encoding for binary features (e.g., `gender`, `Partner`).
    *   One-Hot Encoding for multi-category features (e.g., `InternetService`, `PaymentMethod`).
*   **Target Encoding:** `Churn` encoded to 0/1.
*   **Feature Scaling:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) scaled using `StandardScaler`.

### Slide 5: Methodology - Model Selection & Implementation
*   **Chosen Technique:** Logistic Regression
    *   **Why?** Simplicity, interpretability, efficiency, and suitable for binary classification.
*   **Tools Used:** Python (Pandas, scikit-learn, NumPy)
*   **Implementation Steps:**
    1.  Loaded preprocessed data.
    2.  Split data into 80% training, 20% testing (stratified).
    3.  Scaled numerical features.
    4.  Trained Logistic Regression model (`solver='liblinear'`, `random_state=42`).

### Slide 6: Results and Analysis
*   **Model:** Logistic Regression
*   **Key Metrics (on Test Set):**
    *   **Accuracy:** 0.8048 (80.48%)
    *   **Precision (Churn=1):** 0.6552
    *   **Recall (Churn=1):** 0.5588
    *   **F1-Score (Churn=1):** 0.6032
*   **Confusion Matrix:**
    *   TN: 925 (Correctly predicted non-churn)
    *   FP: 110 (Incorrectly predicted churn)
    *   FN: 165 (Incorrectly predicted non-churn)
    *   TP: 209 (Correctly predicted churn)
*   **Analysis:** Good overall accuracy, but recall for churners indicates room for improvement. Model is better at identifying non-churners.

### Slide 7: Conclusion & Future Scope
*   **Conclusion:** Successfully developed and evaluated a Logistic Regression model for Telco customer churn prediction. The model achieved reasonable performance, providing a valuable baseline.
*   **Future Scope:**
    *   Explore advanced models (Random Forest, XGBoost).
    *   Address data imbalance (SMOTE, ADASYN).
    *   Feature engineering and importance analysis.
    *   Hyperparameter tuning for optimal performance.
    *   Consider deep learning for complex patterns.

### Slide 8: Q&A / Thank You
*   **Questions?**
*   **Thank You!**

This outline provides a solid foundation for the presentation, covering all necessary aspects within the specified slide count.
