import numpy as np
import pandas as pd
from collections import Counter

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



class RandomForest:


    def __init__(self, n_estimators=100, max_depth = None, min_samples_split=2, loss='mse'):
    
           self.n_estimators = n_estimators
           self.max_depth = max_depth 
           self.min_samples_split = min_samples_split 
           self.loss = loss 
      
           self.trees = [] 
           
    
    def make_decision_tree_model(self):
    
       tree = DecisionTree(max_depth=self.max_depth, min_criterion=0.01, min_sample_split=self.min_samples_split)
       # default loss is MSE 
       
       return tree 
    
    def bootstraping(self, X,y):
    
       if X.shape[0] == y.shape[0]:
       
          #indices of boostrap samples with replacement used
          inds = np.random.choice(X.shape[0], size= X.shape[0], replace=True)
       
          #indices of out-of-bag selection samples 
          out_of_bag_inds = list(set(X.index) - set(inds))
         
 
          if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
       
             return X.iloc[inds], y.iloc[inds]
          else:
    
             return X[inds], y[inds]
       else:
          print('x_train, y_train datasets have not the same number of rows!')
       
    
    def fit(self, X, y):
    
       # X = x_train
       # y = y_train 
       
       self.trees = []
       
       from joblib import Parallel, delayed
       

       def train_decision_tree(X, y, max_depth, min_samples_split):
          tree = DecisionTree(max_depth=max_depth, min_criterion=0.01, min_sample_split=min_samples_split)
          X_s, y_s = self.bootstraping(X, y)
          tree.fit(X_s, y_s)
          return tree

       # parallelize processing to speed up the fitting 
       self.trees = Parallel(n_jobs=-1)(delayed(train_decision_tree)(X, y, self.max_depth, self.min_samples_split) for _ in range(self.n_estimators))
       
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
    
       # X = X_test 
       
       if not self.trees:
          print('The tree list is empty, you must train the model before making any prediction')
          return None
       
       #recursively call the predict function 
       predictions = [] 
       tree_predictions_mean = []       
       for tree in self.trees:
         pred = tree.predict(X)
         predictions.append(pred.reshape(-1,1))
         
       #Mean value of ensemble predictions 
       tree_predictions_mean = np.mean(np.concatenate(predictions, axis=1), axis=1)
       
       return tree_predictions_mean
       
       
    
      


#############################################################
########################## TESTING ##########################
#############################################################

df = pd.read_csv('data/SuperConductor.csv', sep=",")

df = df.sample(frac=0.1, random_state=42)

X = df.drop(columns="critical_temp", axis=1)
y = df["critical_temp"] 
X = df.iloc[:, :10]
#print(X)

#X = df.iloc[:, :-75]  # exclude a lot of features just for testing purposes
#y = df.iloc[:, -1]

# Split into train and test sets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.metrics import mean_squared_error
import pickle 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and predict using own implementation


tree = RandomForest(n_estimators=50, max_depth = 30, min_samples_split=30)
tree.fit(X_train, y_train)
filename = 'finalized_model.pkl'
pickle.dump(tree, open(filename, 'wb'))

# if you want to load the tree instance from the pickle file 
#f = open(filename, 'rb')
#tree = pickle.load(f) 
#f.close()
 

predictions = tree.predict(X_test)
mse = mean_squared_error(y_test, predictions)


sklearn_tree = RandomForestRegressor(n_estimators=5, max_depth=30, min_samples_split=30) 
sklearn_tree.fit(X_train, y_train)
sklearn_predictions = sklearn_tree.predict(X_test)
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)

print(f"Decision Tree MSE: {mse}")
print(f"Scikit-learn Decision Tree MSE: {sklearn_mse}")
