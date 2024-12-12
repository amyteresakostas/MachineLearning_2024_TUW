import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


class DecisionTree():

    def __init__(self, max_depth=None, min_criterion=None, min_sample_split=None, max_features=None, loss='mse',
                 random_state=None):
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

        # Stopping criteria
        if self.n_samples < self.min_sample_split or (self.max_depth is not None and self.depth >= self.max_depth):
            return

        best_feature, best_threshold, best_reduction = self._find_best_split(features, target)

        # Stopping criteria
        if best_reduction < self.min_criterion:
            return

        self.feature = best_feature
        self.threshold = best_threshold
        self.gain = best_reduction

        # Split data and grow child nodes
        left_idx = features[self.feature] <= self.threshold
        right_idx = features[self.feature] > self.threshold

        self.left = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=self.min_criterion,
            min_sample_split=self.min_sample_split,
            random_state=self.rng.integers(0, 1e6)
        )
        self.left.depth = self.depth + 1
        self.left.fit(features[left_idx], target[left_idx])

        self.right = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=self.min_criterion,
            min_sample_split=self.min_sample_split,
            random_state=self.rng.integers(0, 1e6)
        )
        self.right.depth = self.depth + 1
        self.right.fit(features[right_idx], target[right_idx])

    def _find_best_split(self, features, target):
        best_reduction = 0.0
        best_feature = None
        best_threshold = None

        if self.loss == 'mse':
            impurity_node = self._calc_mse(target)
        elif self.loss == 'entropy':
            impurity_node = self._calc_entropy(target)
        else:
            raise ValueError("Invalid loss function. Choose either 'mse' or 'entropy'.")

        # Select a random subset of features if max_features is set
        feature_subset = features.columns
        if self.max_features is not None:
            # feature_subset = random.sample(list(features.columns), min(self.max_features, len(features.columns)))
            feature_subset = self.rng.choice(list(features.columns), size=min(self.max_features, len(features.columns)),
                                             replace=False)
        for col in feature_subset:
            sorted_values = np.sort(features[col].unique())
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2.0

            for threshold in thresholds:
                target_left = target[features[col] <= threshold]
                target_right = target[features[col] > threshold]

                # Skip empty splits
                if len(target_left) == 0 or len(target_right) == 0:
                    continue

                if self.loss == 'mse':
                    impurity_left = self._calc_mse(target_left)
                    impurity_right = self._calc_mse(target_right)
                elif self.loss == 'entropy':
                    impurity_left = self._calc_entropy(target_left)
                    impurity_right = self._calc_entropy(target_right)

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

    def _calc_entropy(self, target: pd.Series):
        """Calculate the entropy for a node."""
        class_counts = target.value_counts()
        probabilities = class_counts / len(target)

        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

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

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, loss='mse',
                 random_state=None):

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

        def train_decision_tree(X, y, max_depth, min_samples_split):
            tree_seed = self.rng.integers(0, 1e6)
            tree = DecisionTree(max_depth=max_depth, min_criterion=0.01,
                                min_sample_split=min_samples_split, max_features=self.max_features,
                                loss='mse', random_state=tree_seed)
            X_s, y_s = self.bootstraping(X, y)
            tree.fit(X_s, y_s)
            return tree

        # parallelize processing to speed up the fitting
        self.trees = Parallel(n_jobs=-1)(
            delayed(train_decision_tree)(X, y, self.max_depth, self.min_samples_split) for _ in
            range(self.n_estimators))

        '''
       for estimator in range(self.n_estimators):
          print('Estimator number {}'.format(estimator))
          tree = self.make_decision_tree_model() 
               
          X_s, y_s = self.bootstraping(X,y)
          print("End of bootstrap")  
          tree.fit(X_s, y_s) 
          self.trees.append(tree)
       '''

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


#############################################################
########################## TESTING ##########################
#############################################################

### Load Datasets ###
# MPG dataset
df_MPG_test = pd.read_csv('data/MPG_train.csv', sep=",")
df_MPG_train = pd.read_csv('data/MPG_test.csv', sep=",")

X_train_MPG = df_MPG_train.drop(columns="mpg", axis=1)
y_train_MPG = df_MPG_train["mpg"]

X_test_MPG = df_MPG_test.drop(columns="mpg", axis=1)
y_test_MPG = df_MPG_test["mpg"]

# Superconductor dataset
df_CT_test = pd.read_csv('data/CT_test.csv', sep=",")
df_CT_train = pd.read_csv('data/CT_train.csv', sep=",")

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
    rf_grid = RandomForest(random_state=42)

    gr_space = {
        'n_estimators': [50, 100],
        'max_depth': [None, 20],
        'min_samples_split': [5, 30],
        'max_features': [int(np.sqrt(X.shape[1])), int(np.log(X.shape[1]))]
    }

    scoring = {
        "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
        "smape": make_scorer(smape, greater_is_better=False),
        "correlation": make_scorer(correlation)
    }

    grid = GridSearchCV(rf_grid, gr_space, cv=5, scoring=scoring, refit='rmse')
    model_grid = grid.fit(X, y)

    for params, mean_rmse, mean_r2, mean_smape, mean_correlation, mean_fit_time in zip(
            model_grid.cv_results_['params'],
            model_grid.cv_results_['mean_test_rmse'],
            model_grid.cv_results_['mean_test_r2'],
            model_grid.cv_results_['mean_test_smape'],
            model_grid.cv_results_['mean_test_correlation'],
            model_grid.cv_results_['mean_fit_time']):
        param_str = str(params)
        scores[param_str] = {
            "mean_test_rmse": mean_rmse,
            "mean_test_r2": mean_r2,
            "mean_test_smape": mean_smape,
            "mean_test_correlation": mean_correlation,
            "mean_fit_time": mean_fit_time
        }

    return scores, model_grid.best_params_, model_grid.best_score_


def print_hyperparam_results(param_effect_scores, best_params, best_score):
    print(f"\nHyperparameter Tuning Results")
    print(f"{'Parameters'} | {'RMSE'} | {'R2'} | {'SMAPE'} | {'Correlation'} | {'Fit Time'}")
    for params, metrics in param_effect_scores.items():
        print(
            f"{params} | {-metrics['mean_test_rmse']:.4f}   | {metrics['mean_test_r2']:.4f}   | {-metrics['mean_test_smape']:.4f}   | {metrics['mean_test_correlation']:.4f} | {metrics['mean_fit_time']:.4f}")
    print("\n")
    print('Best hyperparameters are:', best_params)
    print('Best score is:', best_score)


### Results ###
scores_MPG, best_params_MPG, best_score_MPG = hyperparameter_tuning(X_train_MPG, y_train_MPG)
print_hyperparam_results(scores_MPG, best_params_MPG, best_score_MPG)

scores_CT, best_params_CT, best_score_CT = hyperparameter_tuning(X_train_CT, y_train_CT)
print_hyperparam_results(scores_CT, best_params_CT, best_score_CT)

### Predict with best parameters on Test set ###
best_model_MPG = RandomForest(**best_params_MPG, random_state=42)
best_model_MPG.fit(X_train_MPG, y_train_MPG)
y_pred_MPG = best_model_MPG.predict(X_test_MPG)
print("RMSE for MPG on Test Set:", root_mean_squared_error(y_test_MPG, y_pred_MPG))

best_model_CT = RandomForest(**best_params_CT, random_state=42)
best_model_CT.fit(X_train_CT, y_train_CT)
y_pred_CT = best_model_CT.predict(X_test_CT)
print("RMSE for CT on Test Set:", root_mean_squared_error(y_test_CT, y_pred_CT))
