#  Credit Card Fraud Detection

A Machine Learning project for detecting potentially fraudulent credit card transactions using a Random Forest classifier and a Streamlit web application.

# Project Overview

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small proportion of all transactions.

This project explores different machine learning approaches and selects a final Random Forest model based on precision, recall, F1-score, ROC-AUC, PR-AUC, and threshold analysis.

# Objectives

 and prepare the transaction dataset
 Perform exploratory data analysis (EDA)
 Analyze class imbalance
 Train and compare multiple machine learning approaches
 Handle class imbalance using class weighting and SMOTE
 Evaluate models using appropriate classification metrics
 Tune the classification threshold
 Select and save the final model
 Build a Streamlit application for transaction prediction

# Dataset

The dataset contains:

 `Time` — time elapsed between transactions
 `V1` to `V28` — anonymized principal components
 `Amount` — transaction amount
 `Class` — target variable

Target classes:

 `0` = Normal transaction
 `1` = Fraudulent transaction

# Dataset Cleaning

Original dataset:

 284,807 transactions
 31 columns
 1,081 duplicate rows

After duplicate removal:

 283,726 transactions
 31 columns
 0 missing values

Class distribution after cleaning:

 Normal: 283,253
 Fraud: 473

This shows a severe class imbalance.

# Exploratory Data Analysis

The project includes analysis of:

 Class distribution
 Transaction amount
 Transaction time
 Correlation with the target
 Important features associated with fraud

The average transaction amount was:

| Transaction Type | Average Amount |
|---|---:|
| Normal | 88.29 |
| Fraud | 122.21 |

# Machine Learning Models

Several approaches were evaluated:

1. Logistic Regression
2. Class-Weighted Logistic Regression
3. SMOTE + Logistic Regression
4. Random Forest

# Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 99.91% | 84.62% | 57.89% | 68.75% |
| Class-Weighted Logistic Regression | 97.53% | 5.64% | 87.37% | 10.59% |
| SMOTE Logistic Regression | 97.37% | 5.30% | 87.37% | ~10% |
| Random Forest | 99.95% | 86.52% | 81.05% | 83.70% |

# Final Model

The final selected model is:

**Random Forest Classifier**

The model was selected because it provided a strong balance between precision and recall for the minority fraud class.

# Final Performance

 Accuracy: **99.947%**
 Precision: **86.52%**
 Recall: **81.05%**
 F1 Score: **83.70%**
 ROC-AUC: **93.91%**
 PR-AUC: **80.12%**

# Threshold Tuning

Instead of relying only on the default 0.50 probability threshold, multiple thresholds were evaluated.

The selected threshold was:

**0.20**

At this threshold:

 Precision: **86.52%**
 Recall: **81.05%**
 F1 Score: **83.70%**

This provides a better balance between detecting fraudulent transactions and limiting false positives.

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

SMOTE (Synthetic Minority Over-sampling Technique) was tested to address the severe class imbalance.

Before SMOTE:

 Normal: 226,602
 Fraud: 378

After SMOTE:

 Normal: 226,602
 Fraud: 226,602

SMOTE improved fraud recall but produced a large number of false positives in this experiment, so it was not selected as the final approach.

# Streamlit Application

A Streamlit web application was developed to provide an interactive interface for fraud detection.

The application allows users to enter:

 Time
 V1–V28
 Transaction Amount

The application then calculates the fraud probability and classifies the transaction as either:

 Normal Transaction
 Potential Fraudulent Transaction

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
├── models/
│   ├── fraud_detection_random_forest.pkl
│   └── threshold.pkl
│
└── data/