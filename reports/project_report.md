# Data Mining Mini-Project Report: Customer Churn Prediction

## 1. Problem Definition
Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company or service. Predicting customer churn is a critical business problem for many industries, including telecommunications, streaming services, and banking. Retaining existing customers is often more cost-effective than acquiring new ones. By identifying customers at risk of churning, companies can proactively implement retention strategies, such as targeted offers or improved customer service, to prevent their departure.

**Objective:** The primary objective of this mini-project is to build a classification model that accurately predicts whether a telecommunications customer will churn or not, based on their demographic information, service usage, and contract details.

## 2. Dataset Description
The dataset used for this project is the **Telco Customer Churn** dataset, a publicly available dataset widely used for churn prediction tasks. It contains information about a telecommunications company's customers, including their services, account information, and demographic data.

*   **Source:** Publicly available (e.g., Kaggle, IBM).
*   **Number of Instances:** 7043 customers.
*   **Number of Attributes:** 20 features + 1 target variable (`Churn`).

**Key Attributes:**
*   **Demographic Information:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
*   **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
*   **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
*   **Target Variable:** `Churn` (Yes/No, indicating whether the customer churned).

## 3. Data Preprocessing
The raw dataset required several preprocessing steps to prepare it for machine learning model training:

1.  **Handling `TotalCharges`:**
    *   Initially, the `TotalCharges` column was of `object` type due to the presence of some non-numeric values (likely empty strings or spaces).
    *   These non-numeric values were coerced to `NaN` (Not a Number).
    *   The missing `NaN` values were then imputed using the **median** value of the `TotalCharges` column (calculated as 1397.475). This approach was chosen to minimize the impact of potential outliers on the imputed values.

2.  **Dropping `customerID`:**
    *   The `customerID` column, being a unique identifier, has no predictive power for churn and was therefore dropped from the dataset.

3.  **Encoding Categorical Variables:**
    *   **Label Encoding:** Binary categorical features (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) were converted into numerical format (0s and 1s) using `LabelEncoder`.
    *   **One-Hot Encoding:** Multi-category nominal features (`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`) were transformed using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity (the dummy variable trap).
    *   **Target Variable Encoding:** The `Churn` target variable (Yes/No) was also Label Encoded to 1/0 respectively.

4.  **Feature Scaling:**
    *   Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were scaled using `StandardScaler`. This is crucial for algorithms that are sensitive to the magnitude of feature values, such as Logistic Regression, ensuring that no single feature dominates the model due to its larger scale.

## 4. Technique Used: Logistic Regression
**Logistic Regression** is a statistical model that, in its basic form, uses a logistic function to model a binary dependent variable. Despite its name, it is a classification algorithm rather than a regression algorithm. It is widely used for binary classification problems due to its simplicity, interpretability, and efficiency.

**Why Logistic Regression?**
*   **Interpretability:** The coefficients of a Logistic Regression model can be interpreted as the change in the log-odds of the outcome for a one-unit increase in the predictor variable, making it easier to understand the impact of each feature on churn.
*   **Efficiency:** It is computationally less expensive than many other complex classification algorithms.
*   **Baseline:** It provides a strong baseline for comparison with more complex models.
*   **Suitability:** It is well-suited for binary classification problems like customer churn.

## 5. Model Implementation
*   **Tools:** Python programming language, with libraries such as `pandas` for data manipulation, `scikit-learn` for machine learning algorithms and preprocessing, and `numpy` for numerical operations.
*   **Environment:** Standard Python environment with the aforementioned libraries installed.
*   **Steps:**
    1.  The preprocessed dataset (`telco_churn_preprocessed.csv`) was loaded.
    2.  The dataset was split into features (`X`) and the target variable (`y`, which is `Churn`).
    3.  The data was further split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42` for reproducibility and `stratify=y` to maintain the proportion of churners in both sets.
    4.  Numerical features in `X_train` and `X_test` were scaled using `StandardScaler`.
    5.  A `LogisticRegression` model was initialized with `random_state=42` and `solver='liblinear'` (suitable for smaller datasets and binary classification).
    6.  The model was trained using `X_train` and `y_train`.

## 6. Results and Analysis

The Logistic Regression model was evaluated on the unseen test set, yielding the following performance metrics:

*   **Accuracy:** 0.8048 (80.48%)
*   **Precision (for Churn=1):** 0.6552
*   **Recall (for Churn=1):** 0.5588
*   **F1-Score (for Churn=1):0.6032**

**Confusion Matrix:**
```
[[925 110]
 [165 209]]
```
*   **True Negatives (TN):** 925 customers were correctly predicted as non-churn.
*   **False Positives (FP):** 110 customers were incorrectly predicted as churn (Type I error).
*   **False Negatives (FN):** 165 customers were incorrectly predicted as non-churn (Type II error).
*   **True Positives (TP):** 209 customers were correctly predicted as churn.

**Analysis:**
The model achieves an overall accuracy of about 80.5%, which indicates a reasonably good ability to distinguish between churners and non-churners. However, accuracy alone can be misleading, especially in imbalanced datasets (where the number of non-churners is usually much higher than churners).

*   **Precision** for churners (0.6552) means that when the model predicts a customer will churn, it is correct approximately 65.5% of the time. This is important for minimizing wasted resources on customers who were not actually at risk.
*   **Recall** for churners (0.5588) indicates that the model successfully identified about 55.9% of all actual churners. Improving recall is often crucial in churn prediction, as failing to identify a churner (False Negative) can be costly.
*   The **F1-Score** (0.6032) provides a balance between precision and recall.

The confusion matrix highlights that the model is better at predicting non-churners (925 TNs) than churners (209 TPs). The relatively high number of False Negatives (165) suggests there's room for improvement in identifying actual churners. Strategies to address this could include using different classification algorithms, applying techniques for handling imbalanced datasets (e.g., oversampling, undersampling), or further feature engineering.

## 7. Conclusion and Future Scope
The mini-project successfully implemented a Logistic Regression model for customer churn prediction using Python. The model achieved an accuracy of 80.48% and provided insights into its performance through precision, recall, and F1-score.

**Future Scope:**
*   **Explore other algorithms:** Investigate more advanced classification models such as Random Forest, Gradient Boosting (XGBoost, LightGBM), or Support Vector Machines to potentially improve performance.
*   **Handle Imbalanced Data:** Given that churn datasets are often imbalanced, techniques like SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN could be applied to improve the model's ability to predict the minority class (churners).
*   **Feature Importance:** Analyze feature importance to understand which factors contribute most significantly to customer churn.
*   **Hyperparameter Tuning:** Systematically optimize the hyperparameters of the chosen model using techniques like GridSearchCV or RandomizedSearchCV.
*   **Deep Learning Models:** For larger datasets, consider deep learning approaches, which can automatically learn complex feature interactions.

## 8. References
*   [Telco Customer Churn Dataset (Kaggle/IBM)](https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv) (This specific link was used for direct download)
*   Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
*   Pandas documentation: [https://pandas.pydata.org/](https://pandas.pydata.org/)

## 9. Roles and Responsibilities
This was an individual mini-project.
