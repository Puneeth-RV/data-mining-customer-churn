import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_model(X_train):
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    passthrough_cols = ["SeniorCitizen"]
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous_cols,
            ),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("binary_numeric", "passthrough", passthrough_cols),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)),
        ]
    )


def feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame(
        {
            "Feature": preprocessor.get_feature_names_out(),
            "Coefficient": classifier.coef_[0],
        }
    )
    importance["Absolute Coefficient"] = np.abs(importance["Coefficient"])
    importance["Feature"] = (
        importance["Feature"]
        .str.replace(r"^(continuous|categorical|binary_numeric)__", "", regex=True)
        .str.replace("_", " ", regex=False)
    )
    return importance.sort_values("Absolute Coefficient", ascending=False)


def save_charts(df, y_test, y_pred, model):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    churn_counts = df["Churn"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax, color="#2a9d8f")
    ax.set_title("Distribution of Customer Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_distribution.png", dpi=200)
    plt.close(fig)

    contract_churn = (
        df.assign(ChurnFlag=df["Churn"].map({"No": 0, "Yes": 1}))
        .groupby("Contract", as_index=False)["ChurnFlag"]
        .mean()
    )
    contract_churn["Churn Rate (%)"] = contract_churn["ChurnFlag"] * 100
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(data=contract_churn, x="Contract", y="Churn Rate (%)", ax=ax, color="#457b9d")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_rate_by_contract.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", ax=ax, palette="Set2", legend=False)
    ax.set_title("Monthly Charges Distribution by Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_charges_by_churn.png", dpi=200)
    plt.close(fig)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    top_features = feature_importance(model).head(10).sort_values("Coefficient")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = np.where(top_features["Coefficient"] >= 0, "#d95f02", "#1b9e77")
    ax.barh(top_features["Feature"], top_features["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_model_coefficients.png", dpi=200)
    plt.close(fig)


def main():
    print("Customer Churn Prediction - VS Code Runner")
    print("=" * 48)

    df = load_data()
    print(f"Dataset loaded: {RAW_DATA_PATH}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nTarget distribution:")
    print(df["Churn"].value_counts().to_string())

    X = df.drop(["customerID", "Churn"], axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X_train)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print("\nTraining parameters:")
    print("test_size=0.2, random_state=42, stratify=y, solver='lbfgs', max_iter=1000")

    print("\nModel metrics:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nTop model coefficients:")
    print(feature_importance(model).head(10)[["Feature", "Coefficient"]].round(3).to_string(index=False))

    save_charts(df, y_test, y_pred, model)
    print(f"\nCharts saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
