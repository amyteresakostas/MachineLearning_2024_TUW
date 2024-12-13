import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold
import time
import pandas as pd
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import sys
from itertools import product

# Step 1: Define functions for the metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
def mre(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / (ss_total + 1e-10))
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
def correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

# Step 2: Build a decision tree
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = np.random.default_rng(random_state)  # Random number generator
        self.tree = None

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        # Limit the number of features considered for splitting
        if self.max_features is None:
            feature_indices = np.arange(n)
        else:
            feature_indices = self.random_state.choice(n, size=min(self.max_features, n), replace=False)

        best_gain = 0
        split_idx, split_val = None, None
        current_impurity = np.var(y)

        for col in feature_indices:
            thresholds = np.unique(X[:, col])
            for threshold in thresholds:
                left_mask = X[:, col] <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue

                left_impurity = np.var(y[left_mask]) if len(y[left_mask]) > 0 else 0
                right_impurity = np.var(y[right_mask]) if len(y[right_mask]) > 0 else 0
                weighted_impurity = (len(y[left_mask]) * left_impurity + len(y[right_mask]) * right_impurity) / m

                gain = current_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    split_idx = col
                    split_val = threshold

        return split_idx, split_val

    def _build_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        split_idx, split_val = self._best_split(X, y)
        if split_idx is None:
            return np.mean(y)

        left_mask = X[:, split_idx] <= split_val
        right_mask = ~left_mask

        return {
            'split_idx': split_idx,
            'split_val': split_val,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        if row[tree['split_idx']] <= tree['split_val']:
            return self._predict_row(row, tree['left'])
        else:
            return self._predict_row(row, tree['right'])

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])

# Step 3: Define the Random Forest Regressor
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None, n_jobs=8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []

    def _bootstrap_sample(self, X, y):
        indices = np.random.choice(len(X), len(X), replace=True)
        return X.iloc[indices], y.iloc[indices]

    def _train_tree(self, bootstrap_sample):
        X_sample, y_sample = bootstrap_sample
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features  # Pass max_features to each tree
        )
        tree.fit(X_sample.values, y_sample.values)  # Convert to numpy for decision tree
        return tree

    def fit(self, X, y):
        with Pool(processes=self.n_jobs) as pool:
            bootstrap_samples = [self._bootstrap_sample(X, y) for _ in range(self.n_estimators)]
            self.trees = pool.map(self._train_tree, bootstrap_samples)

    def predict(self, X):
        predictions = np.array([tree.predict(X.values) for tree in self.trees])
        return np.mean(predictions, axis=0)


# Step 4: Define Crossvalidation Function
def cross_validate_random_forest(X, y, n_folds=5, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None, n_jobs=8):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_metrics = []
    total_train_time = 0

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,  # Pass max_features to RandomForest
            n_jobs=n_jobs
        )
        start_time = time.time()
        rf.fit(X_train, y_train_cv)
        train_time = time.time() - start_time
        total_train_time += train_time

        y_pred_cv = rf.predict(X_test)
        metrics = {
            'rmse': rmse(y_test_cv, y_pred_cv),
            'mre': mre(y_test_cv, y_pred_cv),
            'r_squared': r_squared(y_test_cv, y_pred_cv),
            'smape': smape(y_test_cv, y_pred_cv),
            'correlation': correlation(y_test_cv, y_pred_cv)
        }
        all_metrics.append(metrics)

    avg_metrics = {metric: np.mean([fold[metric] for fold in all_metrics]) for metric in all_metrics[0]}
    avg_metrics['train_time'] = total_train_time

    return avg_metrics, total_train_time



# Step 5: Test the implementation
if __name__ == "__main__":
    ### Load Datasets ###
    os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise 2")
    # MPG dataset
    df_MPG_test = pd.read_csv('MPG_train.csv', sep=",")
    df_MPG_train = pd.read_csv('MPG_test.csv', sep=",")
    X_train_MPG = df_MPG_train.drop(columns="mpg", axis=1)
    y_train_MPG = df_MPG_train["mpg"]
    X_test_MPG = df_MPG_test.drop(columns="mpg", axis=1)
    y_test_MPG = df_MPG_test["mpg"]

    # Superconductor dataset
    df_CT_test = pd.read_csv('CT_test.csv', sep=",")
    df_CT_train = pd.read_csv('CT_train.csv', sep=",")
    X_train_CT = df_CT_train.drop(columns="critical_temp", axis=1)
    y_train_CT = df_CT_train["critical_temp"]
    X_test_CT = df_CT_test.drop(columns="critical_temp", axis=1)
    y_test_CT = df_CT_test["critical_temp"]

    def grid_search_random_forest(X, y, param_grid, n_folds=5):
        param_combinations = list(product(
            param_grid['n_estimators'],
            param_grid['max_depth'],
            param_grid['min_samples_split'],
            param_grid['max_features']
        ))
        results = []
        start_time = time.time()  # Start tracking time
        for params in param_combinations:
            print(f"Testing parameters: {dict(zip(param_grid.keys(), params))}")
            metrics, train_time = cross_validate_random_forest(
                X, y,
                n_folds=n_folds,
                n_estimators=params[0],
                max_depth=params[1],
                min_samples_split=params[2],
                max_features=params[3]
            )
            results.append({
                'n_estimators': params[0],
                'max_depth': params[1],
                'min_samples_split': params[2],
                'max_features': params[3],
                **metrics,
                'train_time': train_time
            })
        total_time = time.time() - start_time  # Calculate total elapsed time
        results_df = pd.DataFrame(results)
        # Sort by RMSE for best hyperparameter selection
        best_row = results_df.loc[results_df['rmse'].idxmin()]
        best_parameters = {
            'n_estimators': best_row['n_estimators'],
            'max_depth': best_row['max_depth'],
            'min_samples_split': best_row['min_samples_split'],
            'max_features': best_row['max_features']
        }
        best_rmse = best_row['rmse']
        # Print the results DataFrame and the best parameters
        print(results_df.to_string(index=False))  # Ensures all columns print side-by-side
        print('Best hyperparameters are:', best_parameters)
        print('Best score is:', best_rmse)
        print(f'Total grid search time: {total_time:.2f} seconds')
        return results_df, best_parameters, best_rmse, total_time

    print("------------------------------------")
    print("MPG Dataset")
    gr_space_MPG = {
        'n_estimators': [50, 100],
        'max_depth': [None, 20],
        'min_samples_split': [5, 30],
        'max_features': [int(np.sqrt(X_train_MPG.shape[1])), int(np.log(X_train_MPG.shape[1]))]
    }
    grid_results_MPG = grid_search_random_forest(X_train_MPG, y_train_MPG, gr_space_MPG, n_folds=5)
    results_df_MPG, best_parameters_MPG, best_rmse_MPG, total_time_MPG = grid_results_MPG
    #print(results_df_MPG.sort_values(by='rmse'))

    print("------------------------------------")
    print("CT Dataset")
    gr_space_CT = {
        'n_estimators': [50, 100],
        'max_depth': [None, 20],
        'min_samples_split': [5, 30],
        'max_features': [int(np.sqrt(X_train_CT.shape[1])), int(np.log(X_train_CT.shape[1]))]
    }
    grid_results_CT = grid_search_random_forest(X_train_CT, y_train_CT, gr_space_CT, n_folds=5)
    results_df_CT, best_parameters_CT, best_rmse_CT, total_time_CT = grid_results_CT
    # print(results_df_CT.sort_values(by='rmse'))

    ### FIT BEST MODEL ON TEST DATASETS ###
    print("------------------------------------")
    print("MPG: TEST DATASET")
    _, best_parameters_MPG, _, _ = grid_results_MPG
    best_model_MPG = RandomForest(
        n_estimators=int(best_parameters_MPG['n_estimators']),  # Ensure this is an int
        max_depth=None if pd.isna(best_parameters_MPG['max_depth']) else int(best_parameters_MPG['max_depth']),
        min_samples_split=int(best_parameters_MPG['min_samples_split']),  # Ensure this is an int
        max_features=None if pd.isna(best_parameters_MPG['max_features']) else int(best_parameters_MPG['max_features']),
        n_jobs=8
    )
    best_model_MPG.fit(X_train_MPG, y_train_MPG)
    y_pred_test_MPG = best_model_MPG.predict(X_test_MPG)
    test_metrics_MPG = {
        'rmse': rmse(y_test_MPG, y_pred_test_MPG),
        'mre': mre(y_test_MPG, y_pred_test_MPG),
        'r_squared': r_squared(y_test_MPG, y_pred_test_MPG),
        'smape': smape(y_test_MPG, y_pred_test_MPG),
        'correlation': correlation(y_test_MPG, y_pred_test_MPG)
    }
    print("Test Dataset Metrics:")
    for metric, value in test_metrics_MPG.items():
        print(f"{metric}: {value:.4f}")

    print("------------------------------------")
    print("CT: TEST DATASET")
    _, best_parameters_CT, _, _ = grid_results_CT
    best_model_CT = RandomForest(
        n_estimators=int(best_parameters_CT['n_estimators']),  # Ensure this is an int
        max_depth=None if pd.isna(best_parameters_CT['max_depth']) else int(best_parameters_CT['max_depth']),
        min_samples_split=int(best_parameters_CT['min_samples_split']),  # Ensure this is an int
        max_features=None if pd.isna(best_parameters_CT['max_features']) else int(best_parameters_CT['max_features']),
        n_jobs=8
    )
    best_model_CT.fit(X_train_CT, y_train_CT)
    y_pred_test_CT = best_model_CT.predict(X_test_CT)
    test_metrics_CT = {
        'rmse': rmse(y_test_CT, y_pred_test_CT),
        'mre': mre(y_test_CT, y_pred_test_CT),
        'r_squared': r_squared(y_test_CT, y_pred_test_CT),
        'smape': smape(y_test_CT, y_pred_test_CT),
        'correlation': correlation(y_test_CT, y_pred_test_CT)
    }
    print("Test Dataset Metrics:")
    for metric, value in test_metrics_CT.items():
        print(f"{metric}: {value:.4f}")


"""
if __name__ == "__main__":
    os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise 2")
    os.makedirs("plot", exist_ok=True)
    
    #log_file = open("ML_Ex2_Existing_RF_performanceMeasures.txt", "w")
    #sys.stdout = log_file
    num_cpus = os.cpu_count()
    print(f"Number of CPUs: {num_cpus}")

    print("------------------------------------")
    print("MPG Dataset")
    MPG_train = pd.read_csv("MPG_train.csv")
    MPG_test = pd.read_csv("MPG_test.csv")
    X_train = MPG_train.drop('mpg', axis=1);
    y_train = MPG_train['mpg']
    X_test = MPG_test.drop('mpg', axis=1);
    y_test = MPG_test['mpg']


    ### WITHOUT CV ###
    print("----------------------------------------------------------------------------------------------")
    print("MPG - Without CV:")
    rf = RandomForest(n_estimators=100, max_depth=5, min_samples_split=2, n_jobs=num_cpus, max_features=30)
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = rf.predict(X_test)
    metrics = {
        'rmse': rmse(y_test, y_pred),
        'mre': mre(y_test, y_pred),
        'r_squared': r_squared(y_test, y_pred),
        'smape': smape(y_test, y_pred),
        'correlation': correlation(y_test, y_pred)
    }
    print("Metrics:", metrics)
    print("Training Time:", train_time, "seconds")

    ### WITH CV ###
    print("----------------------------------------------------------------------------------------------")
    print("CT - With CV, Without Scaling:")
    metrics, train_time = cross_validate_random_forest(X_train, y_train, n_folds=5,
                                                       n_estimators=100, max_depth=5,
                                                       min_samples_split=2, n_jobs=8, max_features=30)
    print("Cross-Validation Metrics:", metrics)
    print("Training Time:", train_time, "seconds")

    print("------------------------------------")
    print("CT Dataset")
    MPG_train = pd.read_csv("CT_train.csv")
    MPG_test = pd.read_csv("CT_test.csv")
    X_train = MPG_train.drop('critical_temp', axis=1);
    y_train = MPG_train['critical_temp']
    X_test = MPG_test.drop('critical_temp', axis=1);
    y_test = MPG_test['critical_temp']

    ### WITHOUT CV ###
    print("----------------------------------------------------------------------------------------------")
    print("CT - Without CV:")
    rf = RandomForest(n_estimators=50, max_depth=5, min_samples_split=2, n_jobs=num_cpus, max_features=30)
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = rf.predict(X_test)
    metrics = {
        'rmse': rmse(y_test, y_pred),
        'mre': mre(y_test, y_pred),
        'r_squared': r_squared(y_test, y_pred),
        'smape': smape(y_test, y_pred),
        'correlation': correlation(y_test, y_pred)
    }
    print("Metrics:", metrics)
    print("Training Time:", train_time, "seconds")

    ### WITH CV ###
    print("----------------------------------------------------------------------------------------------")
    print("CT - With CV, Without Scaling:")
    metrics, train_time = cross_validate_random_forest(X_train, y_train, n_folds=5,
                                                       n_estimators=50, max_depth=5,
                                                       min_samples_split=2, n_jobs=8, max_features=30)
    print("Cross-Validation Metrics:", metrics)
    print("Training Time:", train_time, "seconds")


    #log_file.close()
"""