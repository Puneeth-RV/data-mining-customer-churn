# 🎓 Complete Walkthrough: Customer Churn Prediction Project

> **This document explains the entire project from scratch.**
> Read it top-to-bottom and you'll understand every concept, every file, every result.

---

## Table of Contents

1. [The Big Picture — What is this project?](#1-the-big-picture--what-is-this-project)
2. [Key Concepts You Must Know](#2-key-concepts-you-must-know)
3. [The Dataset](#3-the-dataset)
4. [Project Architecture — File Map](#4-project-architecture--file-map)
5. [Step-by-Step Code Walkthrough](#5-step-by-step-code-walkthrough)
6. [The Model — Logistic Regression Explained](#6-the-model--logistic-regression-explained)
7. [Results & What They Mean](#7-results--what-they-mean)
8. [Charts Explained (with images)](#8-charts-explained-with-images)
9. [The Streamlit Dashboard](#9-the-streamlit-dashboard)
10. [Presentation Q&A Cheat Sheet](#10-presentation-qa-cheat-sheet)

---

## 1. The Big Picture — What is this project?

### What is Data Mining?

Data mining = **extracting useful patterns from large datasets** using statistics, machine learning, and database techniques. Think of it as "digging through data to find gold nuggets of insight."

### What is this project about?

We built a system that **predicts whether a telecom customer will leave (churn) or stay**, using their account information. This is a **classification** problem — we classify each customer into one of two categories: **"will churn"** or **"won't churn."**

### Why does this matter?

- Acquiring a **new customer costs 5–7× more** than retaining an existing one.
- If you can predict who's about to leave, you can offer them a discount or better plan **before** they leave.
- Telecom companies lose billions to churn every year.

### The workflow in one sentence

> **Raw data → Clean it up → Feed it to a Logistic Regression model → Model predicts churn → Evaluate how good the predictions are → Show results on a dashboard.**

```
┌──────────────┐    ┌───────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────────────┐
│ 📄 Raw CSV   │───▶│ 🧹 Preprocess │───▶│ ✂️ Train/Test  │───▶│ 🤖 Logistic     │───▶│ 📊 Evaluate  │───▶│ 🖥️ Streamlit        │
│ 7043 customers│   │ Clean & Encode│    │ Split 80/20   │    │ Regression      │    │ Metrics      │    │ Dashboard           │
└──────────────┘    └───────────────┘    └────────────────┘    └─────────────────┘    └──────────────┘    └──────────────────────┘
```

---

## 2. Key Concepts You Must Know

### Classification

A type of machine learning where the output is a **category** (not a number).
Here: `Churn = Yes` or `Churn = No`.

### Features vs Target

| Term | Meaning | In this project |
|------|---------|-----------------|
| **Features (X)** | The input columns used to make predictions | tenure, MonthlyCharges, Contract, etc. (19 columns) |
| **Target (y)** | The column we're trying to predict | `Churn` (Yes/No → 1/0) |

### Train/Test Split

- We split data into **80% training** and **20% testing**.
- The model **learns patterns** from the training set.
- We **evaluate** it on the test set (data it has never seen) to see how well it generalizes.
- `stratify=y` ensures both sets have the **same proportion** of churners (~26.5%).

### One-Hot Encoding

Converts categories into binary columns. Example:

| Contract | → | Contract_One year | Contract_Two year |
|----------|---|-------------------|-------------------|
| Month-to-month | | 0 | 0 |
| One year | | 1 | 0 |
| Two year | | 0 | 1 |

(`drop_first=True` drops "Month-to-month" to avoid redundancy — this is the **dummy variable trap**)

### Label Encoding

For columns with only 2 values: converts them to 0 and 1.
Example: `gender` → Male=1, Female=0.

### Standard Scaling

Makes numerical features have **mean = 0** and **standard deviation = 1**. This is important because Logistic Regression is sensitive to feature magnitudes — without scaling, `TotalCharges` (values like 1000–8000) would dominate `tenure` (values like 1–72).

### Data Leakage

A critical concept: you must **fit preprocessing (scaling, imputation) ONLY on training data**, then apply (transform) it to test data. If you fit on all data first, the model "peeks" at test data during training, giving unrealistically high scores.

---

## 3. The Dataset

| Property | Value |
|----------|-------|
| **Name** | IBM Telco Customer Churn Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **File** | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Columns** | 21 (20 features + 1 target) |
| **Target variable** | `Churn` (Yes / No) |
| **Class balance** | ~73.5% No, ~26.5% Yes (**imbalanced**) |

### Column Groups

| Group | Columns | Type |
|-------|---------|------|
| **ID** | `customerID` | Dropped (useless for prediction) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Categorical (binary) |
| **Account** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | Mix of numerical & categorical |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Categorical |
| **Target** | `Churn` | Binary (Yes/No) |

### Known Data Issues

- `TotalCharges` is stored as **text** (object type) in the CSV because some rows have blank spaces instead of numbers.
- 11 rows have missing `TotalCharges` — these are new customers with `tenure = 0`.

---

## 4. Project Architecture — File Map

```
data_mining_project/
├── data/
│   ├── raw/                              ← Original CSV (untouched)
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/                        ← Cleaned CSV after preprocessing
│       └── telco_churn_preprocessed.csv
├── reports/
│   ├── figures/                          ← 5 PNG chart images
│   │   ├── churn_distribution.png
│   │   ├── churn_rate_by_contract.png
│   │   ├── monthly_charges_by_churn.png
│   │   ├── confusion_matrix.png
│   │   └── top_model_coefficients.png
│   ├── project_report.md                 ← Full written report
│   └── presentation_outline.md           ← Slide-by-slide outline
├── src/
│   ├── data_exploration.py               ← Step 1: Look at the data
│   ├── data_preprocessing.py             ← Step 2: Clean the data
│   ├── model_training.py                 ← Step 3: Train & evaluate model
│   ├── app.py                            ← Streamlit web dashboard
│   └── run_in_vscode.py                  ← All-in-one script for VS Code
├── requirements.txt                      ← Python libraries needed
├── README.md                             ← Project documentation
└── WALKTHROUGH.md                        ← 👈 You are here
```

---

## 5. Step-by-Step Code Walkthrough

### 5.1 — `src/data_exploration.py` (30 lines)

**Purpose:** First look at the data — understand its shape, types, and missing values.

```python
df = pd.read_csv(data_path)        # Load CSV into a DataFrame
df.info()                           # Show column names, types, non-null counts
df.head()                           # Show first 5 rows
df.describe()                       # Statistical summary (mean, std, min, max)
df.isnull().sum()                   # Count missing values per column
```

**What you'd learn from running this:**

- `TotalCharges` shows up as `object` type (should be numeric) → needs fixing.
- No columns show NaN initially (the blanks are hidden as empty strings).
- 7,043 rows, 21 columns.

---

### 5.2 — `src/data_preprocessing.py` (76 lines)

**Purpose:** Clean and transform data so the ML model can use it.

**Step-by-step what happens:**

| Step | Code | What it does |
|------|------|-------------|
| 1 | `pd.to_numeric(df['TotalCharges'], errors='coerce')` | Converts text to numbers; blank strings become `NaN` |
| 2 | `df['TotalCharges'].fillna(median)` | Fills those 11 NaN values with the median (1397.475) |
| 3 | `df.drop('customerID')` | Removes the ID column |
| 4 | `LabelEncoder()` on binary columns | Converts Yes/No, Male/Female → 1/0 |
| 5 | `pd.get_dummies(drop_first=True)` on multi-category columns | One-Hot Encoding |
| 6 | `LabelEncoder()` on `Churn` | Target: Yes→1, No→0 |
| 7 | Saves to `processed/telco_churn_preprocessed.csv` | Stores cleaned data |

> **⚠️ Note:** This standalone script encodes on the **entire** dataset before splitting — technically a minor form of data leakage. The `model_training.py` and `app.py` files fix this by using a **Pipeline** that fits only on training data.

---

### 5.3 — `src/model_training.py` (79 lines)

**Purpose:** The core ML file — trains and evaluates the model with proper leakage-safe preprocessing.

**The Pipeline Architecture:**

```
                              ┌──────────────────────────────────────────┐
                              │     ColumnTransformer (Preprocessor)     │
                              ├──────────────────────────────────────────┤
  Raw Features X ────────────▶│                                          │
                              │  tenure, MonthlyCharges, TotalCharges    │
                              │  ──▶ Median Imputer ──▶ StandardScaler  │
                              │                                          │
                              │  gender, Contract, PaymentMethod, etc.   │──────▶  Logistic Regression ──▶ Predictions (0 or 1)
                              │  ──▶ OneHotEncoder (drop_first)          │
                              │                                          │
                              │  SeniorCitizen                           │
                              │  ──▶ Passthrough (already 0/1)           │
                              └──────────────────────────────────────────┘
```

**Key code explained:**

```python
# 1. Load raw data and fix TotalCharges type
df = pd.read_csv(raw_data_path)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 2. Separate features (X) and target (y)
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

# 3. Split: 80% train, 20% test (stratified = same churn ratio in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('continuous', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Fill NaN with median
        ('scaler', StandardScaler())                    # Scale to mean=0, std=1
    ]), ['tenure', 'MonthlyCharges', 'TotalCharges']),

    ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'),
     categorical_cols),  # One-hot encode text columns

    ('binary_numeric', 'passthrough', ['SeniorCitizen'])  # Already 0/1, leave as-is
])

# 5. Full model = preprocessor + classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
])

# 6. Train (fit) on training data only
model.fit(X_train, y_train)

# 7. Predict on test data
y_pred = model.predict(X_test)
```

**Why use a Pipeline?**

- **No data leakage:** scaling/imputation learns from training data only.
- **Clean code:** one `model.fit()` does everything.
- **Reproducible:** same transformations applied consistently.

**Training parameters and why:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `test_size=0.2` | 20% for testing | Standard split ratio |
| `random_state=42` | Fixed seed | Makes results reproducible |
| `stratify=y` | Preserve class ratios | Important for imbalanced data |
| `solver='lbfgs'` | Optimization algorithm | Default, works well for small-medium datasets |
| `max_iter=1000` | Max optimization steps | Ensures convergence |

---

### 5.4 — `src/run_in_vscode.py` (216 lines)

**Purpose:** All-in-one script that runs in the terminal (VS Code). Does everything: loads data, trains model, prints results, **and saves 5 chart images** to `reports/figures/`.

This is essentially `model_training.py` + chart generation combined. The charts it creates:

1. `churn_distribution.png` — Bar chart of how many customers churned vs didn't
2. `churn_rate_by_contract.png` — Churn % by contract type
3. `monthly_charges_by_churn.png` — Box plot of monthly charges
4. `confusion_matrix.png` — Heatmap of predictions vs reality
5. `top_model_coefficients.png` — Most influential features

**How to run:**

```bash
python3 src/run_in_vscode.py
```

---

### 5.5 — `src/app.py` (275 lines)

**Purpose:** Interactive web dashboard built with **Streamlit**. This is what's live at https://auramining.streamlit.app

**Dashboard sections:**

1. **Raw Data Overview** — Shows the original 7,043-row table
2. **Preprocessed Data** — Shows data after cleaning
3. **Churn Distribution** — Bar chart + metrics
4. **Feature Distributions** — Interactive dropdown to explore any feature
5. **Churn Rate by Contract** — Key business insight chart
6. **Monthly Charges by Churn** — Box plot comparison
7. **Model Metrics** — Accuracy, Precision, Recall, F1
8. **Confusion Matrix** — Visual heatmap
9. **Top Coefficients** — Which features matter most

**Key Streamlit features used:**

| Feature | Purpose |
|---------|---------|
| `@st.cache_data` | Caches data loading so it doesn't reload on every interaction |
| `@st.cache_resource` | Caches the trained model |
| `st.selectbox()` | Interactive dropdown for feature selection |
| `st.metric()` | Nice metric display cards |
| `st.columns()` | Side-by-side layout |

---

## 6. The Model — Logistic Regression Explained

### In plain English

Logistic Regression answers: **"What's the probability this customer will churn?"**

It draws a "decision boundary" — customers on one side are predicted to churn, those on the other side aren't.

### How it works (simplified)

1. Take all features (tenure, charges, contract type, etc.)
2. Multiply each by a **weight (coefficient)** the model learned
3. Sum them up: `z = w₁·tenure + w₂·charges + w₃·contract + ... + bias`
4. Pass through the **sigmoid function**: `probability = 1 / (1 + e^(-z))`
5. If probability > 0.5 → predict **Churn (1)**; otherwise → **No Churn (0)**

### The sigmoid function

```
         1.0 |          ___________
             |        /
         0.5 |------/------------ (decision threshold)
             |    /
         0.0 |___/
             ────────────────────→ z
```

### Why Logistic Regression for this project?

| | |
|---|---|
| ✅ **Interpretable** | Coefficients tell you exactly which features increase/decrease churn risk |
| ✅ **Fast** | Trains in milliseconds on this dataset |
| ✅ **Good baseline** | Gives 80%+ accuracy, solid starting point |
| ✅ **Binary classification** | Exactly what churn prediction needs |
| ❌ **Limitation** | Can't capture complex non-linear relationships (Random Forest, XGBoost can) |

---

## 7. Results & What They Mean

### Performance Metrics

| Metric | Value | Plain English |
|--------|-------|--------------|
| **Accuracy** | **80.55%** | Out of all predictions, 80.55% were correct |
| **Precision** | **65.72%** | When model says "will churn," it's right 65.7% of the time |
| **Recall** | **55.88%** | Model catches 55.9% of all actual churners |
| **F1-Score** | **60.40%** | Harmonic mean of precision & recall (balanced measure) |

### Confusion Matrix Breakdown

```
                      Predicted
                  No Churn  │  Churn
                 ───────────┼──────────
Actual No Churn │   926     │   109        ← 926 correct, 109 false alarms
       Churn    │   165     │   209        ← 165 missed!, 209 caught
```

| Cell | Count | Meaning |
|------|-------|---------|
| **True Negative (TN)** | 926 | Correctly predicted "won't churn" ✅ |
| **False Positive (FP)** | 109 | Said "will churn" but they didn't ❌ (false alarm) |
| **False Negative (FN)** | 165 | Said "won't churn" but they DID ❌ (missed churner!) |
| **True Positive (TP)** | 209 | Correctly predicted "will churn" ✅ |

> **⚠️ The 165 False Negatives are the biggest concern.** These are customers who churned but the model didn't catch them. In business terms, these are the customers you'd lose without intervention.

### How metrics are calculated

```
Accuracy  = (TP + TN) / Total           = (209 + 926) / 1409 = 80.55%
Precision = TP / (TP + FP)              = 209 / (209 + 109)  = 65.72%
Recall    = TP / (TP + FN)              = 209 / (209 + 165)  = 55.88%
F1        = 2 × (Precision × Recall) / (Precision + Recall)  = 60.40%
```

---

## 8. Charts Explained (with images)

### Chart 1: Churn Distribution

<p align="center">
  <img src="reports/figures/churn_distribution.png" alt="Churn Distribution" width="500"/>
</p>

**What it shows:** ~5,174 customers did NOT churn vs ~1,869 who DID churn.

**Key takeaway:** The dataset is **imbalanced** — roughly 73% No vs 27% Yes. This means the model has more examples of "No churn" to learn from, which is why it's better at predicting non-churners.

---

### Chart 2: Churn Rate by Contract Type

<p align="center">
  <img src="reports/figures/churn_rate_by_contract.png" alt="Churn Rate by Contract" width="550"/>
</p>

**What it shows:** Month-to-month customers churn at **~42%**, One-year at **~11%**, Two-year at **~3%**.

**Key takeaway:** **Contract type is the #1 predictor of churn.** Customers locked into longer contracts are far less likely to leave. This is the most actionable business insight — encourage customers to sign longer contracts.

---

### Chart 3: Monthly Charges by Churn

<p align="center">
  <img src="reports/figures/monthly_charges_by_churn.png" alt="Monthly Charges by Churn" width="550"/>
</p>

**What it shows:** Churned customers tend to have **higher monthly charges** (median ~$80) compared to non-churned customers (~$65).

**Key takeaway:** Price-sensitive customers leave. Consider targeted pricing or discounts for high-charge customers at risk of churning.

---

### Chart 4: Confusion Matrix

<p align="center">
  <img src="reports/figures/confusion_matrix.png" alt="Confusion Matrix" width="480"/>
</p>

**What it shows:** A heatmap version of the TP/TN/FP/FN table from Section 7.

**How to read it:**
- **Darker blue = higher count.** The top-left (926 True Negatives) is the darkest because it's the largest value.
- **Diagonal cells (top-left, bottom-right)** = correct predictions.
- **Off-diagonal cells** = errors.

**Key takeaway:** The model is much better at predicting non-churners (926 correct) than churners (209 correct). This is because the dataset is imbalanced — 73% of the data is "No churn."

---

### Chart 5: Top Model Coefficients

<p align="center">
  <img src="reports/figures/top_model_coefficients.png" alt="Top Model Coefficients" width="600"/>
</p>

**What it shows:** The features with the largest impact on churn prediction.

**How to read it:**
- **Orange bars (positive coefficient)** → **Increases** churn probability
- **Green bars (negative coefficient)** → **Decreases** churn probability

**Key insights from the coefficients:**

| Feature | Direction | Business Interpretation |
|---------|-----------|------------------------|
| Fiber optic internet | ↑ Increases churn | Possibly due to higher cost or service quality issues |
| Two-year contract | ↓ Decreases churn | Customers are locked in, less likely to leave |
| Tenure | ↓ Decreases churn | Longer-tenured customers are more loyal |
| Electronic check payment | ↑ Increases churn | Less committed payment method (not auto-pay) |
| Monthly charges | ↑ Increases churn | Higher bills = more incentive to switch providers |

---

## 9. The Streamlit Dashboard

**Live URL:** https://auramining.streamlit.app

The dashboard (`src/app.py`) is an interactive web app that lets anyone explore the data and results without running any code. It:

1. Loads the raw CSV
2. Shows both raw and preprocessed data tables
3. Plots interactive charts (user can select which feature to visualize)
4. Trains the Logistic Regression model in real-time
5. Displays all performance metrics and the confusion matrix
6. Shows the top model coefficients

**To run it locally:**

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

---

## 10. Presentation Q&A Cheat Sheet

Here are the most likely questions you'll be asked, with confident answers:

---

### Q: "What data mining technique did you use?"

> **A:** Classification — specifically **Logistic Regression**. Classification is used when the output is a category (churn: yes or no), as opposed to regression which predicts a continuous number.

---

### Q: "Why Logistic Regression and not Random Forest or another model?"

> **A:** Logistic Regression was chosen as a **baseline** because it's interpretable — you can see exactly which features drive churn through the coefficients. It's also computationally efficient and well-suited for binary classification. Future work would include trying ensemble methods like Random Forest or XGBoost for potentially better accuracy.

---

### Q: "What is the accuracy of your model?"

> **A:** **80.55%** on the test set. But accuracy alone can be misleading with imbalanced data. The F1-score of 60.4% gives a more balanced picture, and the recall of 55.9% tells us we're catching about 56% of actual churners.

---

### Q: "What is data leakage and how did you handle it?"

> **A:** Data leakage happens when information from the test set influences the training process. We prevented it by using a **scikit-learn Pipeline** — the imputer and scaler are fitted only on training data, then applied to test data. This ensures the model has never "seen" test data statistics during training.

---

### Q: "What preprocessing did you do?"

> **A:** Five steps: (1) Converted TotalCharges from text to numeric, (2) Imputed 11 missing values with the median, (3) Dropped the customerID column, (4) Encoded categorical variables — Label Encoding for binary columns, One-Hot Encoding for multi-category columns, and (5) Scaled numerical features using StandardScaler.

---

### Q: "What is the confusion matrix?"

> **A:** It's a 2×2 table showing how many predictions were correct vs incorrect. Our model correctly identified 926 non-churners and 209 churners, but missed 165 actual churners (false negatives) and falsely flagged 109 non-churners as churners (false positives).

---

### Q: "What does precision vs recall mean?"

> **A:** **Precision** answers: "Of all customers I predicted would churn, how many actually did?" (65.7%). **Recall** answers: "Of all customers who actually churned, how many did I catch?" (55.9%). In churn prediction, **recall is more important** because missing a churner (false negative) means losing a customer.

---

### Q: "What are the most important features?"

> **A:** Based on the model coefficients: (1) **Contract type** — month-to-month customers churn most, (2) **Internet service type** — fiber optic users churn more, (3) **Tenure** — longer-tenured customers are more loyal, (4) **Payment method** — electronic check users churn more, (5) **Monthly charges** — higher charges correlate with higher churn.

---

### Q: "How would you improve the model?"

> **A:** Four main approaches: (1) **Handle class imbalance** using SMOTE to generate synthetic churner samples, (2) **Try advanced models** like Random Forest or XGBoost, (3) **Hyperparameter tuning** with GridSearchCV, and (4) **Feature engineering** — create new features like charge-per-month-of-tenure ratio.

---

### Q: "What is StandardScaler?"

> **A:** It transforms each feature to have mean=0 and standard deviation=1 using the formula: `z = (x - mean) / std`. This prevents features with large values (like TotalCharges in thousands) from dominating features with small values (like tenure 1–72).

---

### Q: "What is One-Hot Encoding and why `drop_first=True`?"

> **A:** One-Hot Encoding creates a new binary column for each category. `drop_first=True` drops one category to avoid the **dummy variable trap** — where one column is perfectly predictable from the others, causing multicollinearity issues in Logistic Regression.

---

### Q: "What tools/libraries did you use?"

> **A:** Python with pandas (data manipulation), scikit-learn (ML model and preprocessing), matplotlib and seaborn (visualization), Streamlit (web dashboard), and NumPy (numerical operations).

---

### Q: "What is `stratify` in `train_test_split`?"

> **A:** `stratify=y` ensures the train and test sets have the **same proportion** of churners (~26.5%). Without it, random splitting might put most churners in one set, giving misleading results.

---

> **💡 Pro presentation tip:** When showing the confusion matrix, point to the **165 False Negatives** and say: *"These are the customers we'd lose — improving recall to catch more of them is the key area for future work."* This shows you understand the business implications, not just the numbers.
