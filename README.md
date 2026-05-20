# Data Mining Mini-Project: Customer Churn Prediction

This repository contains the code, data, and reports for a Data Mining mini-project focused on predicting customer churn in a telecommunications company using a Classification approach. The project is implemented in Python and includes an interactive Streamlit application to visualize the data and model results.

## Project Structure

The project is organized into the following directories:

*   **`data/`**: Stores datasets.
    *   **`raw/`**: Contains the original, unaltered dataset.
        *   `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The raw Telco Customer Churn dataset.
    *   **`processed/`**: Stores cleaned and preprocessed datasets ready for model training.
        *   `telco_churn_preprocessed.csv`: The dataset after cleaning, encoding, and scaling.
*   **`src/`**: Contains all Python scripts that constitute the project's pipeline and the Streamlit application.
    *   **`data_exploration.py`**: Performs initial data loading and exploratory data analysis (EDA) to understand the dataset's structure, identify data types, and check for missing values.
    *   **`data_preprocessing.py`**: Handles data cleaning, missing value imputation, encoding of categorical features, and feature scaling, transforming the raw data into a format suitable for machine learning.
    *   **`model_training.py`**: Splits the preprocessed data into training and testing sets, trains a classification model (Logistic Regression in this case), and evaluates its performance using various metrics.
    *   **`app.py`**: The Streamlit application that provides an interactive dashboard to visualize the raw and processed data, feature distributions, and the trained model's performance.
*   **`reports/`**: Contains project documentation and presentation materials.
    *   **`project_report.md`**: A detailed written report of the mini-project, covering problem definition, methodology, implementation, results, and future scope.
    *   **`presentation_outline.md`**: An outline for a PowerPoint presentation summarizing the project.
*   **`README.md`**: This file, providing an overview of the project.

## Project Flow

The project follows a typical data mining pipeline:

1.  **Data Acquisition:** The raw `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset is obtained.
2.  **Data Exploration:** The `data_exploration.py` script is used to understand the initial characteristics of the dataset.
3.  **Data Preprocessing:** The `data_preprocessing.py` script cleans the raw data, handles missing values, converts categorical features into numerical formats (using Label Encoding and One-Hot Encoding), and scales numerical features. The resulting preprocessed data is saved as `telco_churn_preprocessed.csv`.
4.  **Model Training and Evaluation:** The `model_training.py` script loads the preprocessed data, splits it into training and testing sets, trains a Logistic Regression model, and evaluates its performance using metrics like accuracy, precision, recall, and F1-score.
5.  **Interactive Visualization:** The `app.py` Streamlit application integrates the data loading, preprocessing, and model evaluation steps to present an interactive web dashboard. This dashboard allows for visual exploration of the data and the model's results.

## Setup and Running the Project

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Puneeth-RV/data-mining-customer-churn.git
    cd data-mining-customer-churn/data_mining_project
    ```
2.  **Install Dependencies:**
    Ensure you have Python installed. Then, install all the required libraries using pip:
    ```bash
    python3 -m pip install pandas scikit-learn numpy streamlit matplotlib seaborn
    ```
3.  **Run the Pipeline (Optional, for batch processing):**
    You can execute the core data processing and model training scripts sequentially. Note that `data_exploration.py` primarily prints output to the console.
    ```bash
    python3 src/data_exploration.py
    python3 src/data_preprocessing.py
    python3 src/model_training.py
    ```
    (Running these scripts will generate the `telco_churn_preprocessed.csv` file, which is then used by the `model_training.py` and `app.py`.)

4.  **Run the Streamlit Application (Interactive Dashboard):**
    To launch the interactive frontend:
    ```bash
    python3 -m streamlit run src/app.py
    ```
    This command will open a new tab in your web browser displaying the dashboard.

## Project Overview

The project aims to predict customer churn using the Telco Customer Churn dataset. A Logistic Regression model was implemented, and its performance was evaluated using various metrics. The detailed methodology, results, and future scope are documented in the `reports/project_report.md`. The Streamlit dashboard provides a user-friendly interface to explore the dataset and the model's performance visually.
