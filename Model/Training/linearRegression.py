import pandas as pd
import numpy as np
import pickle
import os

# Path to the dataset
data_path = "/home/midstan/Documents/Health Insurance Premium/Model/Converting Dataset for Training/Dataset/insurance_converted.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Prepare features (X) and target (y)
# Assuming all features are numerical based on sample data (sex and smoker as 0/1, region as 1-4)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].values
y = df['expenses'].values

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using the normal equation.
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Add bias term (intercept column of ones)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal equation: theta = (X^T X)^-1 X^T y
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]
        return self

    def predict(self, X):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Predictions: X_b * theta
        return X_b.dot(np.r_[self.intercept_, self.coef_])

# Instantiate the model
lr_model = LinearRegressionScratch()

# Train the model on the full dataset
lr_model.fit(X, y)

# Compute R² score manually
y_pred = lr_model.predict(X)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

# Save the trained model to a file for later use
model_path = 'lr_insurance_model_scratch.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(lr_model, f)

print(f"Model saved to {model_path}")
print(f"Training R² score: {r2_score:.4f}")
print(f"Intercept: {lr_model.intercept_:.2f}")
print(f"Coefficients: {lr_model.coef_}")