# Module 12 Report Template

## Overview of the Analysis

In this analysis, we aimed to build machine learning models to predict the creditworthiness of borrowers using financial data. The purpose of this analysis was to develop models that can assist in assessing the risk associated with loans, specifically distinguishing between healthy loans (label 0) and high-risk loans (label 1).

The dataset contained various financial information related to borrowers, and the primary goal was to predict the loan_status variable, which indicates whether a loan is healthy (0) or high-risk (1). 

We began by exploring the dataset, checking the distribution of loan statuses, and identifying the variables needed for prediction. Now i am going to describe all the process:

1.-Data Preprocessing:

  Data Loading: The process started by loading the dataset (lending_data.csv) into a Pandas DataFrame.
  Data Understanding: An initial exploration of the data to understand the features and target variable (loan_status).
  Data Cleaning: Handling missing values and ensuring data consistency and quality.
  Feature Selection: Identifying the relevant features for building the predictive model.

2.-Data Splitting:
  The data was split into training and testing sets using the train_test_split function, allowing us to train the model on one subset and evaluate its performance on another.

3.-Model Building (Original Data):
  Logistic Regression: A logistic regression model was instantiated and trained using the original training data (X_train and y_train).
  Model Evaluation: The performance of the model was assessed using metrics such as balanced accuracy, precision, recall, and a confusion matrix.

4.-Resampling:
  To address class imbalance in the training data (a common issue in credit risk analysis), we used the RandomOverSampler from the imbalanced-learn library. This technique oversampled the minority class (high-risk loans) to ensure an equal number of data points for both classes.

5.-Model Building (Resampled Data):
  A new logistic regression model was instantiated and trained using the resampled training data (X_train_resampled and y_train_resampled).

6.-Model Evaluation (Resampled Data):
  The performance of the resampled model was assessed using the same metrics (balanced accuracy, precision, recall) and compared with the original model.

7.-Result Analysis:
  The results of both models were compared, focusing on their balanced accuracy, precision, and recall scores.
  The analysis considered the significance of accurately predicting high-risk loans in the context of credit risk assessment.

In this analysis, we employed the following methods and techniques:

1.-Logistic Regression:
  Logistic regression is a classification algorithm used to predict binary outcomes, making it suitable for credit risk assessment where the goal is to distinguish between healthy loans and high-risk loans.
  We used the LogisticRegression module from scikit-learn to train a logistic regression model with the original data and again with resampled data.

2.-RandomOverSampler:
  To address the issue of class imbalance in the original training data, we used the RandomOverSampler from the imbalanced-learn library.
  Random oversampling is a resampling technique that duplicates examples from the minority class (in this case, high-risk loans) to balance the class distribution, ensuring that both classes have an equal number of data points.
  This resampling method was applied to the training data to improve the model's performance in identifying high-risk loans.

## Results

* Machine Learning Model 1:
  * Balanced Accuracy Score: 0.9521
  * Precision (Healthy Loans - 0): 1.00
  * Precision (High-Risk Loans - 1): 0.86
  * Recall (Healthy Loans - 0): 1.00
  * Recall (High-Risk Loans - 1): 0.91.

* Machine Learning Model 2:
  * Balanced Accuracy Score: 0.9942
  * Precision (Healthy Loans - 0): 1.00
  * Precision (High-Risk Loans - 1): 0.85
  * Recall (Healthy Loans - 0): 0.99
  * Recall (High-Risk Loans - 1): 0.99

## Summary

The machine learning models were evaluated based on balanced accuracy, precision, and recall scores. The results are as follows:

Machine Learning Model 2 (Trained with Resampled Data) outperforms Model 1 (Trained with Original Data) in all metrics, including balanced accuracy, precision, and recall. Model 2 achieves an extremely high balanced accuracy score of 0.9942, indicating exceptional performance in distinguishing between healthy and high-risk loans.

Performance may depend on the specific problem we are trying to solve. In a credit risk assessment context, it is crucial to predict high-risk loans (label 1) accurately. Model 2 excels in this regard, with a high precision and recall for high-risk loans, making it highly suitable for identifying loans with a higher likelihood of default.

In summary, Model 2, trained with resampled data, is the recommended choice for assessing the creditworthiness of borrowers. It demonstrates superior performance in identifying high-risk loans, which is of utmost importance in this context. Resampling the data to address class imbalance significantly improved the model's performance.
