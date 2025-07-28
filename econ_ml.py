import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('./data/healthcare_data.csv')

# columns
# 'age': Patient's age
# 'bmi': Body Mass Index (numeric)
# 'blood_pressure': Systolic blood pressure (numeric)
# 'diabetes': Whether the patient has diabetes (binary: 0: No, 1: Yes)
# 'cholesterol': Cholesterol level (numeric)
# 'smoking_status': Whether the patient is a smoker (binary: 0: No, 1: Yes)
# 'family_history': Whether the patient has family history of cardiovascular disease (binary: 0: No, 1: Yes)
# 'medication': Whether the patient is on hypertension medication (binary: 0: No, 1: Yes)
# 'outcome': Target variable indicating cardiovascular disease risk (0: Low risk, 1: High risk)

# Data preprocessing
# Convert categorical columns (if any) to numeric (diabetes, smoking_status, family_history, medication)
data['diabetes'] = data['diabetes'].map({'No': 0, 'Yes': 1})
data['smoking_status'] = data['smoking_status'].map({'Non-smoker': 0, 'Smoker': 1})
data['family_history'] = data['family_history'].map({'No': 0, 'Yes': 1})
data['medication'] = data['medication'].map({'No': 0, 'Yes': 1})


data.fillna(data.mean(), inplace=True)

# Feature selection
X = data[['age', 'bmi', 'blood_pressure', 'diabetes', 'cholesterol', 'smoking_status', 'family_history', 'medication']]
y = data['outcome']  # Target variable (e.g., cardiovascular disease risk)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features for ML models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Econometric Model: OLS (Ordinary Least Squares)
X_train_ols = sm.add_constant(X_train)  # Adding constant term for the intercept
model_ols = sm.OLS(y_train, X_train_ols).fit()  # Fit OLS model
print(model_ols.summary())  # Displaying the summary of the OLS model

#  Machine Learning Model: Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predicting with Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)

# 3. Evaluate both models

# OLS Model Evaluation
X_test_ols = sm.add_constant(X_test)  # Add constant for the test set
y_pred_ols = model_ols.predict(X_test_ols)

# OLS Model Metrics
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)
print(f"\nOLS Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_ols:.4f}")
print(f"R-squared: {r2_ols:.4f}")

# Random Forest Model Evaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"\nRandom Forest Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"R-squared: {r2_rf:.4f}")

# Cross-validation for Random Forest
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"\nCross-validation scores (MSE): {cv_scores}")
print(f"Average Cross-validation MSE: {np.mean(-cv_scores):.4f}")

#  best model
if r2_rf > r2_ols:
    print("\nRandom Forest performs better in predicting cardiovascular disease risk.")
else:
    print("\nOLS model performs better in predicting cardiovascular disease risk.")
