#  Credit Card Fraud Detection

A Machine Learning project for detecting potentially fraudulent credit card transactions using a **Random Forest classifier** and an interactive **Streamlit web application**.

# Project Overview

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small proportion of all transactions.

This project explores multiple machine learning approaches, handles class imbalance, evaluates models using appropriate classification metrics, tunes the classification threshold, and deploys the final Random Forest model through a Streamlit application.

# Live Demo

Try the deployed Streamlit application:

**[ Open Credit Card Fraud Detection App](https://credit-card-fraud-detection-8rysmugbp6iuwagtjkw4wp.streamlit.app/)**

# Objectives

- Prepare and clean the transaction dataset
- Perform Exploratory Data Analysis (EDA)
- Analyze class imbalance
- Train and compare multiple machine learning approaches
- Handle class imbalance using class weighting and SMOTE
- Evaluate models using appropriate classification metrics
- Analyze ROC-AUC and PR-AUC
- Tune the classification threshold
- Select and save the final model
- Build and deploy a Streamlit application for transaction prediction

# Dataset

The dataset contains the following features:

| Feature | Description |
|---|---|
| `Time` | Time elapsed between transactions |
| `V1`–`V28` | Anonymized principal components |
| `Amount` | Transaction amount |
| `Class` | Target variable |

### Target Classes

 `0` = Normal transaction
 `1` = Fraudulent transaction

# Dataset Cleaning

### Original Dataset

 **284,807 transactions**
 **31 columns**
 **1,081 duplicate rows**

### After Duplicate Removal

 **283,726 transactions**
 **31 columns**
 **0 missing values**

### Class Distribution After Cleaning

 Normal: **283,253**
 Fraud: **473**

This demonstrates the severe class imbalance present in the dataset.
# Exploratory Data Analysis

The project includes analysis of:

 Class distribution
 Transaction amount
 Transaction time
 Correlation with the target
 Important features associated with fraud

### Average Transaction Amount

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
| **Random Forest** | **99.95%** | **86.52%** | **81.05%** | **83.70%** |

# Final Model

The final selected model is a **Random Forest Classifier**.

It was selected because it provided a strong balance between precision and recall for the minority fraud class.

# Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | **99.947%** |
| Precision | **86.52%** |
| Recall | **81.05%** |
| F1 Score | **83.70%** |
| ROC-AUC | **93.91%** |
| PR-AUC | **80.12%** |

# Threshold Tuning

Instead of relying only on the default probability threshold of `0.50`, multiple thresholds were evaluated.

The selected decision threshold was:

**0.20**

At this threshold:

 Precision: **86.52%**
 Recall: **81.05%**
 F1 Score: **83.70%**

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

### Before SMOTE

 Normal: **226,602**
 Fraud: **378**

### After SMOTE

 Normal: **226,602**
 Fraud: **226,602**

SMOTE improved fraud recall but produced a large number of false positives in this experiment, so it was not selected as the final approach.

# Streamlit Application

An interactive Streamlit web application was developed to provide an interface for fraud detection.

The application uses the trained Random Forest model and the selected **0.20 decision threshold**.

### Application Features

The application provides:

 Random Forest model information
 Decision threshold information
 Test transaction count
 **Load Normal Transaction** button
 **Load Fraud Transaction** button
 Automatic loading of transaction values
 Transaction information
 V1–V28 feature values
 Transaction amount
 Fraud probability
 Predicted transaction class
 Actual vs predicted comparison
 Correct/incorrect prediction status

### Test Transaction Workflow

Instead of manually entering all V1–V28 values, the application loads transactions directly from:

`test_transactions.csv`

Users can select:

- 🟢 **Load Normal Transaction**
- 🔴 **Load Fraud Transaction**

The selected transaction is then automatically populated in the application.

After clicking **Analyze Transaction**, the Random Forest model calculates the fraud probability and compares it with the `0.20` decision threshold.

The application then displays:

 Fraud probability
 Decision threshold
 Predicted class
 Actual class
 Whether the prediction is correct

# Example Application Test

A fraud transaction from the test dataset was successfully evaluated by the deployed application.

Example result:

 Actual Label: **Fraud (1)**
 Fraud Probability: **63.00%**
 Decision Threshold: **20.00%**
 Predicted Label: **Fraud (1)**
 Result: **Model Prediction is CORRECT**

This is an example application test and is separate from the overall model evaluation metrics reported above.

# Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── fraud_detection.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── fraud_detection_random_forest.pkl
├── threshold.pkl
└── test_transactions.csv
