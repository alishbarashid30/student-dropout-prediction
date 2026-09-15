# Student Dropout Prediction

## Project Overview

This project uses Machine Learning to predict student dropout risk based on academic, demographic, and enrollment-related information.

A Logistic Regression classification model was trained and integrated into a Gradio application that provides a dropout probability and risk category.

## Problem Statement

Student dropout is an important issue for educational institutions. Early identification of students who may be at risk can help institutions provide appropriate academic support.

The goal of this project is to develop a machine learning system that predicts whether a student is at risk of dropping out.

## Dataset

The dataset contains 4424 student records and 37 columns.

Important information includes:

- Age at enrollment
- Admission grade
- Previous qualification
- Curricular unit performance
- Tuition fee status
- Scholarship status
- Debtor status
- Academic and demographic information

The original target contains three categories:

- Dropout
- Graduate
- Enrolled

For this project, the target was converted into binary classification:

- 1 = Dropout
- 0 = Not Dropout

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset.
2. Parsed the CSV using the semicolon separator.
3. Created a binary Dropout target.
4. Removed the original Target column.
5. Converted categorical features using one-hot encoding.
6. Split the data into training and testing sets.
7. Applied StandardScaler to the features.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Dataset structure
- Numerical features
- Categorical features
- Target distribution
- Academic performance
- Relationships between features and dropout risk

The analysis helped identify important patterns and features that could contribute to dropout prediction.

## Machine Learning Model

### Logistic Regression

Logistic Regression was selected because Student Dropout Prediction is a classification problem.

The model was trained on the preprocessed training data.

## Training

The dataset was divided into:

- 80% training data
- 20% testing data

Feature scaling was performed using StandardScaler.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

False positives and false negatives were also analyzed.

## Student Risk Prediction Application

A Gradio application was developed to allow users to enter student information.

The application provides:

- Dropout probability
- Risk category

### Risk Categories

- Low Risk: below 30%
- Medium Risk: 30% to below 60%
- High Risk: 60% or above

## Application Workflow

Student Information
↓
Data Preprocessing
↓
Feature Scaling
↓
Logistic Regression
↓
Dropout Probability
↓
Risk Category

## Potential Use

This project can be used as an educational early-warning system.

Educational institutions could use the prediction results to identify students who may need additional academic support.

The model should be used as a supportive tool rather than as the sole basis for decisions about students.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gradio
- Google Colab
- GitHub

## Conclusion

This project demonstrates a complete Machine Learning workflow, from dataset collection and preprocessing to exploratory analysis, model training, evaluation, prediction, and application deployment.

The final application demonstrates how Machine Learning can be integrated into an educational student-risk prediction system.

## Author

Alishba Rashid

BS Artificial Intelligence
