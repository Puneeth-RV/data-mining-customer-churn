# Data Mining Mini-Project: Customer Churn Prediction

This repository contains the code and reports for a Data Mining mini-project focused on predicting customer churn in a telecommunications company using a Classification approach.

## Project Structure

*   `data/`: Contains raw and processed datasets.
    *   `raw/`: Original downloaded dataset.
        *   `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The raw Telco Customer Churn dataset.
    *   `processed/`: Cleaned and preprocessed datasets.
        *   `telco_churn_preprocessed.csv`: Dataset after cleaning, encoding, and scaling.
*   `src/`: Contains Python scripts for different stages of the project pipeline.
    *   `data_exploration.py`: Script for initial data loading and exploratory data analysis.
    *   `data_preprocessing.py`: Script for handling missing values, encoding categorical features, and feature scaling.
    *   `model_training.py`: Script for splitting data, training the classification model (Logistic Regression), and evaluating its performance.
*   `reports/`: Contains project documentation and presentation materials.
    *   `project_report.md`: Detailed written report of the mini-project.
    *   `presentation_outline.md`: Outline for the PowerPoint presentation.

## Setup and Running the Project

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>/data_mining_project
    ```
2.  **Install Dependencies:**
    Ensure you have Python installed. Then install the required libraries:
    ```bash
    pip install pandas scikit-learn numpy
    ```
3.  **Run the pipeline:**
    Execute the scripts in the following order:
    ```bash
    python src/data_exploration.py
    python src/data_preprocessing.py
    python src/model_training.py
    ```
    (Note: `data_exploration.py` is for initial understanding and doesn't produce an output file necessary for subsequent steps.)

## Project Overview

The project aims to predict customer churn using the Telco Customer Churn dataset. A Logistic Regression model was implemented, and its performance was evaluated using various metrics. The detailed methodology, results, and future scope are documented in the `reports/project_report.md`.
