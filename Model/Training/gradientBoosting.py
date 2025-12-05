import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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

class GradientBoostingRegressor:
    """
    A simple implementation of Gradient Boosting for Regression.
    - Base model: Mean of the target values.
    - Weak learners: Decision Trees fitted to negative gradients (residuals).
    - Stops early if mean absolute residual is below a tolerance.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, tol=1.0):
        """
        :param n_estimators: Maximum number of trees.
        :param learning_rate: Shrinkage factor for each tree's contribution.
        :param max_depth: Maximum depth of each decision tree.
        :param tol: Tolerance for mean absolute residual to stop early.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.tol = tol  # Tolerance for residuals (adjusted for expenses scale ~thousands)
        self.trees = []  # List to store decision trees
        self.initial_prediction = None  # Mean as base prediction

    def fit(self, X, y):
        # Initialize predictions with the mean (base model)
        self.initial_prediction = np.mean(y)
        predictions = np.full_like(y, self.initial_prediction, dtype=np.float64)
        
        # Compute initial residuals
        residuals = y - predictions
        
        # Build trees iteratively
        for i in range(self.n_estimators):
            # Fit a decision tree to the current residuals (negative gradient for regression)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            
            # Predict residuals with the tree
            tree_pred = tree.predict(X)
            
            # Update predictions: add learning_rate * tree prediction
            predictions += self.learning_rate * tree_pred
            
            # Update residuals
            residuals = y - predictions
            
            # Store the tree
            self.trees.append(tree)
            
            # Early stopping: check if mean absolute residual is below tolerance
            mean_abs_residual = np.mean(np.abs(residuals))
            if mean_abs_residual < self.tol:
                print(f"Early stopping at iteration {i+1}: mean abs residual = {mean_abs_residual:.2f}")
                break
        
        print(f"Training completed. Final mean abs residual: {np.mean(np.abs(residuals)):.2f}")
        print(f"Number of trees built: {len(self.trees)}")
        return self

    def predict(self, X):
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)
        
        # Add contributions from all trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions

# Instantiate the model (adjust hyperparameters as needed)
gb_model = GradientBoostingRegressor(
    n_estimators=100000, 
    learning_rate=0.1, 
    max_depth=3, 
    tol=1.0  # Stop if mean abs residual < 1 (small relative to expenses)
)

# Train the model on the full dataset
gb_model.fit(X, y)

# Save the trained model to a file for later use
model_path = 'gb_insurance_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(gb_model, f)

print(f"Model saved to {model_path}")

# Example: How to load and predict after training
# (Uncomment to test if you have sample input data)

# Load the model
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Example new input (must match feature order and types)
new_X = np.array([[18,1,33.8,1,0,3]])  # Sample from your data
prediction = loaded_model.predict(new_X)
print(f"Predicted expenses: {prediction[0]:.2f}")
