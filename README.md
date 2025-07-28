# Econometric Machine Learning for Healthcare Data

## Overview

This repository demonstrates how econometric methods and machine learning models can be applied to healthcare data to predict patient outcomes. The dataset used contains various healthcare parameters such as age, BMI, blood pressure, diabetes, cholesterol levels, smoking status, and medication use. The goal is to predict the risk of cardiovascular disease (e.g., `outcome` variable).

We apply **Ordinary Least Squares (OLS)** regression as an econometric method and **Random Forest Regression** as a machine learning model to predict patient outcomes based on the provided healthcare parameters.

## Contents

1. **Data Preprocessing**: This step cleans the dataset, handling missing values and converting categorical variables into numeric representations.
2. **Econometric Model (OLS)**: We apply a linear regression model to understand the relationship between healthcare parameters and the outcome.
3. **Machine Learning Model (Random Forest)**: We apply a Random Forest Regressor to model the dataset and predict the outcome.
4. **Evaluation**: We evaluate both models using **Mean Squared Error (MSE)** and **R-squared** metrics. We also perform **cross-validation** on the Random Forest model.

## Features Used in the Dataset

* **age**: Patient's age (numeric)
* **bmi**: Body Mass Index (numeric)
* **blood\_pressure**: Systolic blood pressure (numeric)
* **diabetes**: Whether the patient has diabetes (binary: 0: No, 1: Yes)
* **cholesterol**: Cholesterol level (numeric)
* **smoking\_status**: Whether the patient is a smoker (binary: 0: No, 1: Yes)
* **family\_history**: Whether the patient has family history of cardiovascular disease (binary: 0: No, 1: Yes)
* **medication**: Whether the patient is on hypertension medication (binary: 0: No, 1: Yes)
* **outcome**: Target variable indicating cardiovascular disease risk (0: Low risk, 1: High risk)

## Setup

1. Clone the repository or download the code from the provided link.
2. Ensure the dataset is loaded and available for the `pd.read_csv()` function in the script.

## Required Libraries

To run the code, install the necessary Python packages by running:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

You may also need `nltk` for text processing (if applicable for other data-related tasks in your project):

```bash
pip install nltk
```

## How to Run

1. Download your healthcare dataset and place it in the folder path specified in the code (or modify the path as needed).
2. Run the script to preprocess the data, train the models, and evaluate them.

```bash
python run_econometric_ml_healthcare.py
```

## Code Breakdown

### 1. **Data Preprocessing**:

* The script processes categorical data, such as diabetes, smoking status, family history, and medication, converting them to binary values.
* Missing values are handled by filling them with the mean of the respective columns.

### 2. **Econometric Model: OLS Regression**:

* OLS regression is used to explore the linear relationships between the features and the outcome variable. The model summary provides key statistics like R-squared, coefficients, and p-values.

### 3. **Machine Learning Model: Random Forest Regressor**:

* A Random Forest Regressor is trained on the scaled features and used to predict the outcome (cardiovascular disease risk). The model is evaluated using MSE and R-squared.

### 4. **Model Evaluation**:

* Both models (OLS and Random Forest) are evaluated using Mean Squared Error (MSE) and R-squared, which are printed for comparison.
* The Random Forest model is also cross-validated using K-fold cross-validation to provide more reliable performance metrics.

### 5. **Best Model Selection**:

* Based on the evaluation metrics, the best-performing model is selected for making predictions. In most cases, the Random Forest model is likely to outperform OLS due to its non-linear nature.

## Example Output

The code will output the following results:

* OLS Model Evaluation Metrics (MSE, R-squared)
* Random Forest Model Evaluation Metrics (MSE, R-squared)
* Cross-validation scores for the Random Forest model
* A comparison between the two models to determine which performs better for predicting cardiovascular disease risk.

## Reference

The techniques and methods applied here are inspired by the following repository:

* [ML-for-Econometrics by A. Strittmatter](https://github.com/ASrittmatter/ML-for-Econometrics)

## Future Enhancements

* **Hyperparameter Tuning**: Implement hyperparameter optimization techniques (e.g., GridSearchCV, RandomizedSearchCV) for improving model performance.
* **Advanced Machine Learning Models**: Consider other machine learning models such as XGBoost, LightGBM, or Neural Networks for even better performance.
* **Feature Engineering**: Enhance feature selection and engineering techniques to better model complex relationships in the data.
