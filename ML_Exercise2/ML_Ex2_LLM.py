import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import KFold

# Step 1: Define a function to split the dataset
def split_data(X, y, feature_index, threshold):
    left_mask = X.iloc[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# Step 2: Define the mean squared error (MSE) to evaluate splits
def mse(y):
    return np.mean((y - np.mean(y))**2) if len(y) > 0 else 0

def mse_split(y_left, y_right):
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    return (n_left / n_total) * mse(y_left) + (n_right / n_total) * mse(y_right)

# Step 3: Build a decision tree
class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth #The maximum depth of each decision tree.
        self.min_samples_split = min_samples_split #The minimum number of samples required to split a node.
        self.tree = None

    def fit(self, X, y, depth=0):
        # Stop conditions: max depth or too few samples
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        # Find the best split
        best_feature, best_threshold, best_mse = None, None, float("inf")
        for feature_index in range(X.shape[1]):  # Loop over features
            thresholds = np.unique(X.iloc[:, feature_index])  # Get unique values for thresholds
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                current_mse = mse_split(y_left, y_right)
                if current_mse < best_mse:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_mse = current_mse

        # Stop if no split improves MSE
        if best_feature is None:
            return np.mean(y)

        # Split data and recurse
        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.fit(X_left, y_left, depth + 1),
            'right': self.fit(X_right, y_right, depth + 1),
        }

    def predict_one(self, tree, x):
        if not isinstance(tree, dict):
            return tree
        feature = tree['feature']
        threshold = tree['threshold']
        if x.iloc[feature] <= threshold:
            return self.predict_one(tree['left'], x)
        else:
            return self.predict_one(tree['right'], x)

    def predict(self, X):
        return np.array([self.predict_one(self.tree, x) for _, x in X.iterrows()])

# Step 4: Define the Random Forest Regressor
class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators #The number of decision trees in the forest
        self.max_depth = max_depth #The maximum depth of each decision tree.
        self.min_samples_split = min_samples_split #The minimum number of samples required to split a node.
        self.max_features = max_features #The number of features to consider when looking for the best split at each node.
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True) 
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_estimators):
            print(f"Starting tree {i + 1} out of {self.n_estimators}...")  # Progress update
            start_time = time.time()
            X_sample, y_sample = self.bootstrap_sample(X, y)
            max_features = self.max_features or X.shape[1]
            features = np.random.choice(range(X.shape[1]), max_features, replace=False)

            # Subset X_sample to selected features
            X_sample_subset = X_sample.iloc[:, features]
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            print(f"Training tree {i + 1}...")  # Update during tree training
            tree.tree = tree.fit(X_sample_subset, y_sample)

            # Store the tree and the selected feature indices
            self.trees.append((tree, features))
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Finished tree {i + 1}. Time taken: {elapsed_time:.2f} seconds.")  # Completion update

    def predict(self, X):
        predictions = []
        for tree, features in self.trees:
            X_subset = X.iloc[:, features]
            predictions.append(tree.predict(X_subset))
        return np.mean(predictions, axis=0)


def cross_validate_with_kfold(model, X, y, k=5):
    """
    Perform k-fold cross-validation using sklearn's KFold.
    Args:
        model: The machine learning model to train.
        X: Features (numpy array).
        y: Targets (numpy array).
        k: Number of folds.
    Returns:
        metrics: A dictionary with average performance metrics over the folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics_list = []

    fold = 0
    for train_index, val_index in kf.split(X):
        fold += 1
        print(f"Starting Fold {fold}/{k}...")

        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Validate the model
        y_val_pred = model.predict(X_val)

        # Compute metrics for this fold
        fold_metrics = {
            'RMSE': rmse(y_val, y_val_pred),
            'MRE': mre(y_val, y_val_pred),
            'Correlation': correlation(y_val, y_val_pred),
            'R^2': r2(y_val, y_val_pred),
            'SMAPE': smape(y_val, y_val_pred)
        }
        metrics_list.append(fold_metrics)

    # Average the metrics over all folds
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    return avg_metrics

# Performance Metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mre(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


# Step 5: Test the implementation
if __name__ == "__main__":
    os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise 2")

    import sys
    log_file = open("LMM_ML_Ex2.txt", "w")
    sys.stdout = log_file

    ### CT DATASET ###
    CT_train = pd.read_csv("CT_train.csv")
    CT_test = pd.read_csv("CT_test.csv")
    # Split data into features and targets
    X_train = CT_train.drop(columns='critical_temp')
    y_train = CT_train['critical_temp']
    X_test = CT_test.drop(columns='critical_temp')
    y_test = CT_test['critical_temp']
    # Train the Random Forest Regressor
    start = time.time()
    rf = RandomForestRegressor(n_estimators=5)
    rf.fit(X_train, y_train)
    end = time.time()
    print(f"Total Time to Train Random Forest: {end - start:.4f} seconds")
    # Make predictions
    y_pred = rf.predict(X_test)
    # Evaluate the model using custom performance metrics
    print("Performance Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_pred):.4f}")
    print(f"Mean Relative Error (MRE): {mre(y_test, y_pred):.4f}")
    print(f"Correlation: {correlation(y_test, y_pred):.4f}")
    print(f"R^2 Score: {r2(y_test, y_pred):.4f}")
    print(f"SMAPE: {smape(y_test, y_pred):.4f}")


    ### MPG DATASET ###
    MPG_train = pd.read_csv("MPG_train.csv")
    MPG_test = pd.read_csv("MPG_test.csv")
    X_train = MPG_train.drop(columns='mpg')
    y_train = MPG_train['mpg']
    X_test = MPG_test.drop(columns='mpg')
    y_test = MPG_test['mpg']

    start = time.time()
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    end = time.time()
    print(f"Total Time to Train Random Forest: {end - start:.4f} seconds")

    y_pred = rf.predict(X_test)
    # Make predictions
    y_pred = rf.predict(X_test)
    # Evaluate the model using custom performance metrics
    print("Performance Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_pred):.4f}")
    print(f"Mean Relative Error (MRE): {mre(y_test, y_pred):.4f}")
    print(f"Correlation: {correlation(y_test, y_pred):.4f}")
    print(f"R^2 Score: {r2(y_test, y_pred):.4f}")
    print(f"SMAPE: {smape(y_test, y_pred):.4f}")

    log_file.close()