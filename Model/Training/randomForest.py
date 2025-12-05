import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, List

# Path to the dataset
data_path = "/home/midstan/Documents/Health Insurance Premium/Model/Converting Dataset for Training/Dataset/insurance_converted.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Prepare features (X) and target (y)
# Assuming all features are numerical based on sample data (sex and smoker as 0/1, region as 1-4)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].values
y = df['expenses'].values

class Node:
    """
    Node in the decision tree.
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf node value (mean of targets)

def mse(y: np.ndarray) -> float:
    """
    Mean Squared Error for a set of targets.
    """
    return np.mean((y - np.mean(y)) ** 2)

def best_split(X: np.ndarray, y: np.ndarray, feature_idxs: List[int]) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
    """
    Find the best split: feature and threshold minimizing MSE.
    """
    best_idx, best_thresh, best_mse, best_left, best_right = None, None, float('inf'), None, None
    current_mse = mse(y)
    
    for idx in feature_idxs:
        thresholds = np.unique(X[:, idx])
        for thresh in thresholds:
            left_mask = X[:, idx] <= thresh
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            left_y, right_y = y[left_mask], y[right_mask]
            weighted_mse = (len(left_y) * mse(left_y) + len(right_y) * mse(right_y)) / len(y)
            if weighted_mse < best_mse:
                best_idx = idx
                best_thresh = thresh
                best_mse = weighted_mse
                best_left = left_y
                best_right = right_y
    
    return best_idx, best_thresh, best_mse, best_left, best_right

def build_tree(X: np.ndarray, y: np.ndarray, feature_idxs: List[int], max_depth: int = None, min_samples_split: int = 2, depth: int = 0) -> Node:
    """
    Recursively build the decision tree.
    """
    if len(y) < min_samples_split or (max_depth is not None and depth >= max_depth):
        return Node(value=np.mean(y))
    
    idx, thresh, _, _, _ = best_split(X, y, feature_idxs)
    if idx is None:
        return Node(value=np.mean(y))
    
    left_mask = X[:, idx] <= thresh
    right_mask = ~left_mask
    
    left = build_tree(X[left_mask], y[left_mask], feature_idxs, max_depth, min_samples_split, depth + 1)
    right = build_tree(X[right_mask], y[right_mask], feature_idxs, max_depth, min_samples_split, depth + 1)
    
    return Node(idx, thresh, left, right)

def predict_tree(node: Node, x: np.ndarray) -> float:
    """
    Predict with a single tree.
    """
    if node.value is not None:
        return node.value
    if x[node.feature_idx] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor from scratch (CART-like, MSE criterion).
    """
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root = None
        self.feature_idxs = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        self.n_features = X.shape[1]
        self.feature_idxs = list(range(self.n_features))
        self.root = build_tree(X, y, self.feature_idxs, self.max_depth, self.min_samples_split)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([predict_tree(self.root, x) for x in X])

class RandomForestRegressorScratch:
    """
    Random Forest Regressor from scratch: Ensemble of decision trees with bagging and random features.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 3, min_samples_split: int = 2, max_features: str = 'sqrt', random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _get_feature_idxs(self, n_features: int) -> List[int]:
        if self.max_features == 'sqrt':
            return np.random.choice(n_features, max(1, int(np.sqrt(n_features))), replace=False).tolist()
        elif self.max_features == 'log2':
            return np.random.choice(n_features, max(1, int(np.log2(n_features))), replace=False).tolist()
        else:
            raise ValueError("max_features must be 'sqrt' or 'log2'")

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        for _ in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X, y)
            feature_idxs = self._get_feature_idxs(n_features)
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=np.random.randint(0, 1000)  # Vary seed per tree
            )
            tree.feature_idxs = feature_idxs  # Assign random features
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.feature_subsets.append(feature_idxs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

# Instantiate the model (hyperparameters to roughly match original: 100 trees, depth 3, sqrt features)
rf_model = RandomForestRegressorScratch(
    n_estimators=100,
    max_depth=3,
    min_samples_split=2,
    max_features='sqrt',
    random_state=42
)

# Train the model on the full dataset
rf_model.fit(X, y)

# Compute R² score manually
y_pred = rf_model.predict(X)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

# Save the trained model to a file for later use
model_path = 'rf_insurance_model_scratch.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"Model saved to {model_path}")
print(f"Training R² score: {r2_score:.4f}")
print(f"Number of trees: {len(rf_model.trees)}")