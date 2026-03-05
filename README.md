# Salary Prediction using Ridge Regression

## Project Overview

This project builds a machine learning model that predicts an individual's salary based on demographic and career-related features. The goal was to implement a full end-to-end regression pipeline using real-world data and evaluate how well the model generalizes to unseen examples.

## Dataset

The dataset contains information about individuals including:

* Age
* Years of Experience
* Gender
* Education Level
* Salary (target variable)

Missing values were removed before training the model.

## Methodology

The following machine learning workflow was implemented:

1. Data cleaning and preprocessing
2. One-hot encoding for categorical variables (Gender, Education Level)
3. Train/test split to evaluate generalization
4. Feature scaling using StandardScaler
5. Ridge Regression to control overfitting
6. Hyperparameter tuning using RidgeCV with 5-fold cross-validation
7. Model evaluation using Root Mean Squared Error (RMSE)

## Results

The final model achieved:

* Train RMSE ≈ $15,000
* Test RMSE ≈ $15,000

The similar training and testing errors indicate the model generalizes well without significant overfitting.

## Model Interpretation

Feature coefficients were analyzed to understand which variables most influence salary predictions. Age and years of experience were the strongest predictors, while education level also showed a meaningful positive impact.

## Example Prediction

The notebook allows users to input their own features (age, experience, education, and gender) and receive a predicted salary range based on the model’s typical prediction error.

## Key Skills Demonstrated

* Data preprocessing and feature engineering
* Handling categorical variables
* Regularized linear regression (Ridge)
* Cross-validation for hyperparameter tuning
* Model evaluation and interpretation
* Machine learning workflow implementation in Python

## Tools Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

