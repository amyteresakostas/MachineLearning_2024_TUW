import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import os

# Step 1: Define a function to split the dataset
def split_data(X, y, feature_index, threshold):
    left_mask = X.iloc[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# Step 2: Define the mean squared error (MSE) to evaluate splits
def mse(y):
    return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0

def mse_split(y_left, y_right):
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    return (n_left / n_total) * mse(y_left) + (n_right / n_total) * mse(y_right)

# Step 3: Build a decision tree
class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth #Maximum depth of each decision tree
        self.min_samples_split = min_samples_split # Minimum number of samples required to split a node
        self.min_samples_leaf = min_samples_leaf  # Minimum samples in each leaf
        self.tree = None

    def fit(self, X, y, depth=0):
        # Stop recursion if max depth is reached or insufficient samples for a valid split
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        # Check if there are enough samples in both the left and right split to satisfy min_samples_leaf
        best_feature, best_threshold, best_mse = None, None, float("inf")
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X.iloc[:, feature_index])
            for threshold in thresholds:
                left_mask = X.iloc[:, feature_index] <= threshold
                right_mask = ~left_mask
                y_left, y_right = y[left_mask], y[right_mask]

                # Ensure that both the left and right splits satisfy min_samples_leaf
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                current_mse = (len(y_left) * mse(y_left) + len(y_right) * mse(y_right)) / len(y)
                if current_mse < best_mse:
                    best_feature, best_threshold, best_mse = feature_index, threshold, current_mse

        # If no valid split is found, return the mean of the target
        if best_feature is None:
            return np.mean(y)

        # Perform the split based on the best feature and threshold
        left_mask = X.iloc[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.fit(X_left, y_left, depth + 1),
            'right': self.fit(X_right, y_right, depth + 1),
        }

    def predict_one(self, tree, x): # Predict a single data point.
        if not isinstance(tree, dict):
            return tree
        if x.iloc[tree['feature']] <= tree['threshold']:
            return self.predict_one(tree['left'], x)
        else:
            return self.predict_one(tree['right'], x)

    def predict(self, X): # Predict for an entire dataset.
        return np.array([self.predict_one(self.tree, x) for _, x in X.iterrows()])

# Step 4: Define the Random Forest Regressor
class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf  # Added parameter for min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []

    def bootstrap_sample(self, X, y):
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return X.iloc[indices], y.iloc[indices]

    def train_tree(self, X, y, tree_idx):
        print(f"Starting tree {tree_idx + 1} out of {self.n_estimators}...")
        start_time = time.time()

        # Bootstrap sample
        X_sample, y_sample = self.bootstrap_sample(X, y)

        # Feature subset selection
        max_features = self.max_features or X.shape[1]
        selected_features = np.random.choice(X.shape[1], max_features, replace=False)
        X_sample_subset = X_sample.iloc[:, selected_features]

        # Train a decision tree with the new parameter for min_samples_leaf
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf  # Pass min_samples_leaf to each tree
        )
        tree.tree = tree.fit(X_sample_subset, y_sample)

        print(f"Finished tree {tree_idx + 1}. Time taken: {time.time() - start_time:.2f} seconds.")
        return tree, selected_features

    def fit(self, X, y):
        start_time = time.time()
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_tree)(X, y, tree_idx) for tree_idx in range(self.n_estimators)
        )
        print("")
        print(f"Total Time to Train Random Forest: {time.time() - start_time:.4f} seconds")

    def predict(self, X):
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X.iloc[:, features]) for tree, features in self.trees
        )
        return np.mean(predictions, axis=0)

# Performance Metrics
def mse(y):
    return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0

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

# Cross-Validation
def cross_validate_with_kfold(model, X, y, k=5, n_jobs=-1):
    print(f"Performing {k}-Fold Cross-Validation...")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(model, X, y, train_idx, val_idx, fold_idx)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X))
    )
    #print("Cross-validation completed.")
    return pd.DataFrame(results).mean().to_dict()

def train_and_evaluate(model, X, y, train_idx, val_idx, fold_idx):
    print(f"Starting Fold {fold_idx + 1}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Fold {fold_idx+1} training completed in {end_time - start_time:.4f} seconds.")
    y_val_pred = model.predict(X_val)
    return {
        'RMSE': rmse(y_val, y_val_pred),
        'MRE': mre(y_val, y_val_pred),
        'Correlation': correlation(y_val, y_val_pred),
        'R^2': r2(y_val, y_val_pred),
        'SMAPE': smape(y_val, y_val_pred),
    }

class StreamToLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.console = sys.stdout  # Save the original stdout (console)

    def write(self, message):
        # Write the message to both the console and the log file
        if message != "\n":  # Avoid empty newline writes
            self.console.write(message)  # Print to the console
            if self.log_file and not self.log_file.closed:
                self.log_file.write(message)  # Write to the log file
        else:
            # Ensure that newline characters are also written properly
            self.console.write("\n")
            if self.log_file and not self.log_file.closed:
                self.log_file.write("\n")

    def flush(self):
        # This is needed for Python 3 compatibility
        self.console.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        # Close the log file safely
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

# Step 5: Test the implementation
if __name__ == "__main__":
    os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise 2")

    import sys
    #log_file = open("ML_Ex2_LLM_withParallelization.txt", "w")
    #sys.stdout = StreamToLogger(log_file)

    ##### CT DATASET #####
    print("-------------------------------------------------")
    print("CT Dataset")

    CT_train = pd.read_csv("CT_train.csv")
    CT_test = pd.read_csv("CT_test.csv")
    # Split data into features and targets
    X_train = CT_train.drop(columns='critical_temp')
    y_train = CT_train['critical_temp']
    X_test = CT_test.drop(columns='critical_temp')
    y_test = CT_test['critical_temp']

    ### WITHOUT CV ###
    print("------------------------------------")
    print("Without CV")
    rf = RandomForestRegressor(n_estimators=5)
    rf.fit(X_train, y_train)
    # Make predictions
    y_pred = rf.predict(X_test)
    # Evaluate the model using custom performance metrics
    print("Performance Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_pred):.4f}")
    print(f"Mean Relative Error (MRE): {mre(y_test, y_pred):.4f}")
    print(f"Correlation: {correlation(y_test, y_pred):.4f}")
    print(f"R^2 Score: {r2(y_test, y_pred):.4f}")
    print(f"SMAPE: {smape(y_test, y_pred):.4f}")

    ### WITH CROSS VALIDATION ###
    print("------------------------------------")
    print("With CV")
    rf = RandomForestRegressor(n_estimators=4, max_depth=5)
    k = 3
    start = time.time()
    avg_metrics = cross_validate_with_kfold(rf, X_train, y_train, k=k)
    print(avg_metrics)
    end = time.time()
    print("")
    print(f"Total Time for Cross-Validation: {end - start:.4f} seconds")
    print("Performance Metrics:")
    print(avg_metrics)


    ##### MPG DATASET #####
    print("-------------------------------------------------")
    print("MPG Dataset")
    MPG_train = pd.read_csv("MPG_train.csv")
    MPG_test = pd.read_csv("MPG_test.csv")
    X_train = MPG_train.drop(columns='mpg')
    y_train = MPG_train['mpg']
    X_test = MPG_test.drop(columns='mpg')
    y_test = MPG_test['mpg']

    ### WITHOUT CV ###
    print("------------------------------------")
    print("Without CV")
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    # Make predictions
    y_pred = rf.predict(X_test)
    # Evaluate the model using custom performance metrics
    print("Performance Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_pred):.4f}")
    print(f"Mean Relative Error (MRE): {mre(y_test, y_pred):.4f}")
    print(f"Correlation: {correlation(y_test, y_pred):.4f}")
    print(f"R^2 Score: {r2(y_test, y_pred):.4f}")
    print(f"SMAPE: {smape(y_test, y_pred):.4f}")

    ### WITH CROSS VALIDATION ###
    print("------------------------------------")
    print("With CV")
    rf = RandomForestRegressor(n_estimators=10, max_depth=5)
    k = 5
    start = time.time()
    avg_metrics = cross_validate_with_kfold(rf, X_train, y_train, k=k)
    end = time.time()
    print("")
    print(f"Total Time for Cross-Validation: {end - start:.4f} seconds")
    print("Performance Metrics:")
    print(avg_metrics)

    #sys.stdout.close()