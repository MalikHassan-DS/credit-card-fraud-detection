#  Credit Card Fraud Detection

A Machine Learning project for detecting potentially fraudulent credit card transactions using a **Random Forest classifier** and an interactive **Streamlit web application**.

# Project Overview

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small proportion of all transactions.

This project explores multiple machine learning approaches, handles class imbalance, evaluates models using appropriate classification metrics, tunes the classification threshold, and deploys the final Random Forest model through a Streamlit application.

# Objectives

* Prepare and clean the transaction dataset
* Perform Exploratory Data Analysis (EDA)
* Analyze class imbalance
* Train and compare multiple machine learning approaches
* Handle class imbalance using class weighting and SMOTE
* Evaluate models using appropriate classification metrics
* Analyze ROC-AUC and PR-AUC
* Tune the classification threshold
* Select and save the final model
* Build a Streamlit application for transaction prediction

# Dataset

The dataset contains:

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `Time`     | Time elapsed between transactions |
| `V1`–`V28` | Anonymized principal components   |
| `Amount`   | Transaction amount                |
| `Class`    | Target variable                   |

### Target Classes

* `0` = Normal transaction
* `1` = Fraudulent transaction

# Dataset Cleaning

# Original Dataset

* **284,807 transactions**
* **31 columns**
* **1,081 duplicate rows**

# After Duplicate Removal

* **283,726 transactions**
* **31 columns**
* **0 missing values**

# Class Distribution After Cleaning

* Normal: **283,253**
* Fraud: **473**

This demonstrates the severe class imbalance present in the dataset.

# Exploratory Data Analysis

The project includes analysis of:

* Class distribution
* Transaction amount
* Transaction time
* Correlation with the target
* Important features associated with fraud

# Average Transaction Amount

| Transaction Type | Average Amount |
| ---------------- | -------------: |
| Normal           |          88.29 |
| Fraud            |         122.21 |

# Machine Learning Models

Several approaches were evaluated:

1. Logistic Regression
2. Class-Weighted Logistic Regression
3. SMOTE + Logistic Regression
4. Random Forest

# Model Comparison

| Model                              |   Accuracy |  Precision |     Recall |   F1 Score |
| ---------------------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression                |     99.91% |     84.62% |     57.89% |     68.75% |
| Class-Weighted Logistic Regression |     97.53% |      5.64% |     87.37% |     10.59% |
| SMOTE Logistic Regression          |     97.37% |      5.30% |     87.37% |       ~10% |
| **Random Forest**                  | **99.95%** | **86.52%** | **81.05%** | **83.70%** |

# Final Model

The final selected model is a **Random Forest Classifier**.

It was selected because it provided a strong balance between precision and recall for the minority fraud class.

# Final Model Performance

| Metric    |       Score |
| --------- | ----------: |
| Accuracy  | **99.947%** |
| Precision |  **86.52%** |
| Recall    |  **81.05%** |
| F1 Score  |  **83.70%** |
| ROC-AUC   |  **93.91%** |
| PR-AUC    |  **80.12%** |

# Threshold Tuning

Instead of relying only on the default probability threshold of `0.50`, multiple thresholds were evaluated.

The selected threshold was:

**0.20**

At this threshold:

* Precision: **86.52%**
* Recall: **81.05%**
* F1 Score: **83.70%**

The threshold of `0.20` provided the best F1 score among the evaluated thresholds and offered a better balance between detecting fraudulent transactions and limiting false positives.

# Feature Importance

The most important features identified by the Random Forest model included:

1. V14
2. V10
3. V12
4. V17
5. V4
6. V3
7. V11
8. V16
9. V2
10. V9

# SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** was tested to address the severe class imbalance.

# Before SMOTE

* Normal: **226,602**
* Fraud: **378**

# After SMOTE

* Normal: **226,602**
* Fraud: **226,602**

SMOTE improved fraud recall but produced a large number of false positives in this experiment, so it was not selected as the final approach.

# Streamlit Application

A Streamlit web application was developed to provide an interactive interface for fraud detection.

The application allows users to enter:

* `Time`
* `V1`–`V28`
* Transaction `Amount`

The Random Forest model then calculates the **fraud probability** and applies the selected **0.20 decision threshold**.

The application classifies the transaction as:

*  Normal Transaction
*  Potential Fraudulent Transaction

# Project Structure

text
Machine_Learning_project/
│
├── app.py
├── fraud_detection.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── fraud_detection_random_forest.pkl
└── threshold.pkl

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib
* Streamlit
* Jupyter Notebook
* GitHub

# Running the Application Locally

# 1. Clone the repository

bash
git clone https://github.com/MalikHassan-DS/credit-card-fraud-detection.git


# 2. Navigate to the project directory

bash
cd credit-card-fraud-detection


# 3. Install dependencies

bash
pip install -r requirements.txt


# 4. Run the Streamlit application

bash
python -m streamlit run app.py

The application will open in your browser.

# Key Results

The final Random Forest model achieved:

* **99.947% accuracy**
* **86.52% precision**
* **81.05% recall**
* **83.70% F1 score**
* **93.91% ROC-AUC**
* **80.12% PR-AUC**

The project demonstrates an end-to-end machine learning workflow, from data cleaning and exploratory analysis to model development, evaluation, threshold optimization, model persistence, and deployment through Streamlit.

##  Author

**Malik Hassan**

GitHub: **MalikHassan-DS**
