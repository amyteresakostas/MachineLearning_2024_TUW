import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import time
from sklearn.model_selection import ParameterGrid
import time

class DecisionTree():

    def __init__(self, max_depth=None, min_criterion=None, min_sample_split=None, max_features=None, loss='mse', random_state=None):
        self.feature = None
        self.threshold = None
        self.gain = None
        self.left = None
        self.right = None
        self.value = None
        self.depth = 0
        self.n_samples = None

        self.max_depth = max_depth
        self.min_criterion = min_criterion
        self.min_sample_split = min_sample_split
        self.max_features = max_features
        self.loss = loss
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Recursively grow the decision tree."""
        self.n_samples = len(target)
        self.value = target.mean()

        if self.n_samples < self.min_sample_split or (self.max_depth is not None and self.depth >= self.max_depth):
            return

        best_feature, best_threshold, best_reduction = self._find_best_split(features, target)

        if best_reduction < self.min_criterion:
            return  # Stop if the improvement is too small

        self.feature = best_feature
        self.threshold = best_threshold
        self.gain = best_reduction

        left_idx = features[self.feature] <= self.threshold
        right_idx = features[self.feature] > self.threshold

        self.left = DecisionTree(self.max_depth, self.min_criterion, self.min_sample_split, self.max_features,
                                 self.loss)
        self.left.depth = self.depth + 1
        self.left.fit(features[left_idx], target[left_idx])

        self.right = DecisionTree(self.max_depth, self.min_criterion, self.min_sample_split, self.max_features,
                                  self.loss)
        self.right.depth = self.depth + 1
        self.right.fit(features[right_idx], target[right_idx])

    def _find_best_split(self, features, target):
        best_reduction = 0.0
        best_feature = None
        best_threshold = None

        impurity_node = self._calc_mse(target)

        # Select a random subset of features if max_features is set
        feature_subset = features.columns
        if self.max_features is not None:
            # feature_subset = random.sample(list(features.columns), min(self.max_features, len(features.columns)))
            feature_subset = self.rng.choice(list(features.columns), size=min(self.max_features, len(features.columns)), replace=False)
        for col in feature_subset:
            sorted_values = np.sort(features[col].unique())
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2.0
            for threshold in thresholds:
                target_left = target[features[col] <= threshold]
                target_right = target[features[col] > threshold]

                # Skip empty splits
                if len(target_left) == 0 or len(target_right) == 0:
                    continue

                impurity_left = self._calc_mse(target_left)
                impurity_right = self._calc_mse(target_right)

                n_left = len(target_left) / self.n_samples
                n_right = len(target_right) / self.n_samples

                weighted_impurity = n_left * impurity_left + n_right * impurity_right
                impurity_reduction = impurity_node - weighted_impurity

                if impurity_reduction > best_reduction:
                    best_reduction = impurity_reduction
                    best_feature = col
                    best_threshold = threshold

        return best_feature, best_threshold, best_reduction

    def _calc_mse(self, target: pd.Series):
        """Calculate the mean squared error (MSE) for a node."""
        if len(target) == 0:
            return 0
        return np.mean((target - target.mean()) ** 2)

    def _predict(self, sample: pd.Series):
        """Predict for a single sample."""
        if self.feature is not None:
            if sample[self.feature] <= self.threshold:
                return self.left._predict(sample)
            else:
                return self.right._predict(sample)
        return self.value

    def predict(self, features: pd.DataFrame):
        """Predict for all samples in the dataset."""
        return np.array([self._predict(sample) for _, sample in features.iterrows()])


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, loss='mse',random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.max_features = max_features
        self.trees = []

    def get_params(self, deep=True):
        """Get the parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
            'loss': self.loss,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def make_decision_tree_model(self):
        tree = DecisionTree(max_depth=self.max_depth, min_criterion=0.01,
                            min_sample_split=self.min_samples_split, max_features=self.max_features,
                            loss='mse', random_state=self.random_state)
        return tree

    def bootstraping(self, X, y):
        if X.shape[0] == y.shape[0]:
            # indices of boostrap samples with replacement used
            # inds = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            inds = self.rng.integers(0, X.shape[0], size=X.shape[0])
            # indices of out-of-bag selection samples
            out_of_bag_inds = list(set(X.index) - set(inds))
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                return X.iloc[inds], y.iloc[inds]
            else:
                return X[inds], y[inds]
        else:
            print('x_train, y_train datasets have not the same number of rows!')

    def fit(self, X, y):
        self.trees = []
        from joblib import Parallel, delayed
        import time

        def train_decision_tree(X, y, max_depth, min_samples_split, tree_index):
            tree_seed = self.rng.integers(0, 1e6)
            #(f"Starting tree {tree_index + 1}...")
            start_time = time.time()
            tree = DecisionTree(
                max_depth=max_depth, min_criterion=0.01,
                min_sample_split=min_samples_split, max_features=self.max_features,
                loss='mse', random_state=tree_seed
            )
            X_s, y_s = self.bootstraping(X, y)
            tree.fit(X_s, y_s)
            elapsed_time = time.time() - start_time
            #print(f"Tree {tree_index + 1} is done and took {elapsed_time:.2f} seconds.")
            return tree

        self.trees = Parallel(n_jobs=-1)(
            delayed(train_decision_tree)(X, y, self.max_depth, self.min_samples_split, i) for i in
            range(self.n_estimators)
        )

    """
    def predict(self, X):
        if not self.trees:
            print('The tree list is empty, you must train the model before making any prediction')
            return None
        # recursively call the predict function
        predictions = []
        tree_predictions_mean = []
        for tree in self.trees:
            pred = tree.predict(X)
            predictions.append(pred.reshape(-1, 1))
        # Mean value of ensemble predictions
        tree_predictions_mean = np.mean(np.concatenate(predictions, axis=1), axis=1)
        return tree_predictions_mean
    """

    def predict(self, X):
        if not self.trees:
            print('The tree list is empty, you must train the model before making any prediction')
            return None

        predictions = np.mean([tree.predict(X) for tree in self.trees], axis=0)
        return predictions

#############################################################
########################## TESTING ##########################
#############################################################

### Load Datasets ###
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


### Hyperparameter tuning with crossvalidation ###
def correlation(y_true, y_pred):
    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]
def smape(y_true, y_pred):
    return (100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))

def hyperparameter_tuning(X, y):
    scores = {}
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [5, 20],
        'max_features': [int(np.sqrt(X.shape[1])), int(np.log(X.shape[1]))]
    }
    parameter_combinations = list(ParameterGrid(param_grid))

    best_score = float('inf')
    best_params = None

    print("Starting hyperparameter tuning...\n")

    for params in parameter_combinations:
        print(f"Testing parameters: {params}")
        start_time = time.time()

        # Create and configure the RandomForest model
        model = RandomForest(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            max_features=params['max_features'],
            random_state=42
        )

        # Perform 5-fold cross-validation
        fold_rmse_scores = []
        fold_smape_scores = []
        fold_correlations = []
        fold_r_squared = []

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            print(f"  Starting fold {fold + 1}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Calculate RMSE
            fold_rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            fold_rmse_scores.append(fold_rmse)

            # Calculate SMAPE
            fold_smape = smape(y_val, y_pred)  # Use the smape function you defined
            fold_smape_scores.append(fold_smape)

            # Calculate Correlation
            fold_corr = correlation(y_val, y_pred)
            fold_correlations.append(fold_corr)

            # Calculate R-squared
            ss_residual = np.sum((y_val - y_pred) ** 2)
            ss_total = np.sum((y_val - np.mean(y_val)) ** 2)
            fold_r2 = 1 - (ss_residual / ss_total)
            fold_r_squared.append(fold_r2)

        mean_rmse = np.mean(fold_rmse_scores)
        mean_smape = np.mean(fold_smape_scores)
        mean_corr = np.mean(fold_correlations)
        mean_r2 = np.mean(fold_r_squared)
        elapsed_time = time.time() - start_time

        # Save scores
        scores[str(params)] = {"rmse": mean_rmse, "smape": mean_smape, "correlation": mean_corr, "r2": mean_r2,
                               "fit_time": elapsed_time}

        print(f"Results for parameters {params}:")
        print(f"  RMSE: {mean_rmse:.4f}")
        print(f"  SMAPE: {mean_smape:.4f}")
        print(f"  Correlation: {mean_corr:.4f}")  # Print Correlation
        print(f"  R-squared: {mean_r2:.4f}")  # Print R-squared
        print(f"  Time Taken: {elapsed_time:.2f} seconds\n")

        # Update best score and parameters
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    print("\nHyperparameter tuning complete.")
    print("Best parameters:", best_params)
    print("Best RMSE:", best_score)

    return scores, best_params, best_score


### Results ###

print("MPG DATASET")
scores_MPG, best_params_MPG, best_score_MPG = hyperparameter_tuning(X_train_MPG, y_train_MPG)
#print_hyperparam_results(scores_MPG, best_params_MPG, best_score_MPG)
print("")
print("CT DATASET")
scores_CT, best_params_CT, best_score_CT = hyperparameter_tuning(X_train_CT, y_train_CT)
#print_hyperparam_results(scores_CT, best_params_CT, best_score_CT)


### Predict with best parameters on Test set ###
best_model_MPG = RandomForest(**best_params_MPG, random_state=42)
best_model_MPG.fit(X_train_MPG, y_train_MPG)
y_pred_MPG = best_model_MPG.predict(X_test_MPG)
print("RMSE for MPG on Test Set:", root_mean_squared_error(y_test_MPG, y_pred_MPG))

best_model_CT = RandomForest(**best_params_CT, random_state=42)
best_model_CT.fit(X_train_CT, y_train_CT)
y_pred_CT = best_model_CT.predict(X_test_CT)
print("RMSE for CT on Test Set:", root_mean_squared_error(y_test_CT, y_pred_CT))
