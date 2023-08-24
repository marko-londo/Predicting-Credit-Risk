# Loan Prediction Analysis

In this repository, we train two machine learning models on a loan dataset to predict loan risk. The primary focus is on understanding which machine learning model - Logistic Regression or Random Forest Classifier - performs better given the dataset's characteristics.

## Dataset Overview

The dataset contains numerous features related to loans, such as `loan_amnt`, `int_rate`, `home_ownership`, `annual_inc`, and many others. The target variable is `loan_status`, indicating the risk associated with the loan.

## Workflow

1. **Data Preprocessing**: 
   - Dropped target column.
   - Encoded categorical variables using One-Hot Encoding.
   
2. **Data Scaling**:
   - Used `StandardScaler` from `sklearn` to scale the dataset, ensuring that all features have zero mean and unit variance.
   
3. **Model Training & Evaluation**:
   - **Logistic Regression**:
     - Trained on both the original and scaled data.
     - Evaluated model accuracy on test data.
   - **Random Forest Classifier**:
     - Trained with hyperparameters: `n_estimators`, `max_depth`, and `min_samples_split`.
     - Evaluated model accuracy on test data.

## Results

- The Random Forest Classifier outperformed the Logistic Regression model in
  this scenario, both with scaled and un-scaled data.
- My belief was that the Logistic Regression might perform better due
  to possible linear relationships in the data, however the dataset may have
  been more complex than I initially thought.
- Post-scaling, both models witnessed performance enhancements. Specifically, while the Random Forest model saw a minimal uptick in performance, the Logistic Regression model's accuracy surged by 16% on the test dataset.
- The improvements, especially in the Logistic Regression model, were anticipated. By ensuring all features operate on a consistent scale, algorithms can more effectively learn the underlying patterns, leading to better predictions.
