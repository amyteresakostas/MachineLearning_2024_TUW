import numpy as np
import pandas as pd


class DecisionTree():
    '''
    How the algorithm works:
    1. We'll start with all examples at the root node then:
    2. We'll calculate MSE/Variance reduction for splitting on all possible features and pick the one with the lowest value
    3. Then we'll split the data according to the selected feature
    4. We'll repeat this process until stopping criteria are met
    '''

    def __init__(self, max_depth=4, min_criterion=0.05, min_sample_split=2):
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

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Recursively grow the decision tree."""
        self.n_samples = len(target)
        self.value = target.mean()  # Predicted value for leaf nodes

        # Stopping criteria
        if self.n_samples < self.min_sample_split or self.depth >= self.max_depth:
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
            min_sample_split=self.min_sample_split
        )
        self.left.depth = self.depth + 1
        self.left.fit(features[left_idx], target[left_idx])

        self.right = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=self.min_criterion,
            min_sample_split=self.min_sample_split
        )
        self.right.depth = self.depth + 1
        self.right.fit(features[right_idx], target[right_idx])

    def _find_best_split(self, features, target):
        best_reduction = 0.0
        best_feature = None
        best_threshold = None

        # Calculate MSE of the current node
        mse_node = self._calc_mse(target)

        for col in features.columns:
            sorted_values = np.sort(features[col].unique())
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2.0  # Calculate thresholds once

            for threshold in thresholds:
                target_left = target[features[col] <= threshold]
                target_right = target[features[col] > threshold]

                # Skip empty splits
                if len(target_left) == 0 or len(target_right) == 0:
                    continue

                mse_left = self._calc_mse(target_left)
                mse_right = self._calc_mse(target_right)
                n_left = len(target_left) / self.n_samples
                n_right = len(target_right) / self.n_samples

                mse_reduction = mse_node - (n_left * mse_left + n_right * mse_right)

                if mse_reduction > best_reduction:
                    best_reduction = mse_reduction
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


#############################################################
########################## TESTING ##########################
#############################################################

df = pd.read_csv('data/Superconductor.csv', sep=",")
print(df)

X = df.iloc[:, :-75]  # exclude a lot of features just for testing purposes
y = df.iloc[:, -1]

# Split into train and test sets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and predict using own implementation
tree = DecisionTree(max_depth=5, min_criterion=0.01, min_sample_split=5)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

# Fit and predict using scikit-learn DecisionTreeRegressor
sklearn_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_predictions = sklearn_tree.predict(X_test)

# Compare Mean Squared Errors
mse = mean_squared_error(y_test, predictions)
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)

print(f"Decision Tree MSE: {mse}")
print(f"Scikit-learn Decision Tree MSE: {sklearn_mse}")
