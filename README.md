# Data Mining Mini-Project: Customer Churn Prediction

This repository contains a Data Mining mini-project focused on predicting customer churn in a telecommunications company. Implemented in Python, it uses a Classification approach and features an interactive Streamlit application to visualize data and model results.

## Project Structure

*   **`data/`**: Stores datasets.
    *   `raw/`: Original dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
    *   `processed/`: Cleaned and preprocessed dataset (`telco_churn_preprocessed.csv`).
*   **`src/`**: Python scripts for the project pipeline and Streamlit app.
    *   `data_exploration.py`: Initial data loading and exploratory analysis.
    *   `data_preprocessing.py`: Data cleaning, encoding, and scaling.
    *   `model_training.py`: Data splitting, model training (Logistic Regression), and evaluation.
    *   `app.py`: Interactive Streamlit dashboard.
*   **`reports/`**: Project documentation.
    *   `project_report.md`: Detailed written report.
    *   `presentation_outline.md`: PowerPoint presentation summary.

## Project Flow

The project follows a standard data mining pipeline, visualized below using a Mermaid diagram:

```mermaid
graph TD
    A[Raw Data<br>(WA_Fn-UseC...)] --> B{Data Exploration<br>(src/data_exploration.py)}
    B --> C{Data Preprocessing<br>(src/data_preprocessing.py)}
    C --> D{Model Training & Eval<br>(src/model_training.py)}
    D --> E(Interactive Streamlit Dashboard<br>(src/app.py))
    subgraph Streamlit Features
        E --> F[Data Overviews]
        E --> G[Distributions]
        E --> H[Model Metrics & CM]
    end
```

## Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Puneeth-RV/data-mining-customer-churn.git
    cd data-mining-customer-churn/data_mining_project
    ```
2.  **Install Dependencies:**
    ```bash
    python3 -m pip install pandas scikit-learn numpy streamlit matplotlib seaborn
    ```
3.  **Run the Pipeline (Optional):**
    For batch processing and to ensure `telco_churn_preprocessed.csv` is generated:
    ```bash
    python3 src/data_exploration.py
    python3 src/data_preprocessing.py
    python3 src/model_training.py
    ```
4.  **Run the Streamlit Application:**
    To launch the interactive dashboard in your web browser:
    ```bash
    python3 -m streamlit run src/app.py
    ```

## Project Overview

This project successfully implements a Logistic Regression model for customer churn prediction. Detailed methodology, results, and future scope are in `reports/project_report.md`. The Streamlit dashboard offers a visual, interactive exploration of data and model performance.
