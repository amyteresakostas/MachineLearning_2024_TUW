import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from voting_classification import votings
from multiprocessing import Process
import warnings
import time
warnings.filterwarnings('ignore')


import logging

# Configure logging
logging.basicConfig(
    filename='Exercise1.log',       # Log output file
    filemode='w',
    level=logging.INFO,        # Log level: INFO, DEBUG, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)


logging.getLogger('numexpr').setLevel(logging.ERROR) 

warnings.filterwarnings("ignore", category=UserWarning, module="numexpr")  



    
    
    

class SVM_classification:

  def __init__(self):
  
     pass
     
  
  def log_message(self, message):
    logging.info(message)

  
  def load_datasets(self):
    
    #votes = votings()
    #voting_data = votes.run_votings()
    
    voting_data = pd.read_csv("voting_imputed.csv")
    
    rta_data = pd.read_csv("RTA_cleaned.csv")
    
    machine_data = pd.read_csv("Machine_cleaned.csv")
    
    review_data = pd.read_csv("amazon_review_ID.shuf.lrn.csv")
    
    
    return voting_data, rta_data, machine_data , review_data 
    
    
  def split_dataset(self, df, valid_split, output_class):

     X = df.drop(columns=[output_class])
     try:
       y = df[output_class].astype(int)
     except:
       y = df[output_class]

     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_split, random_state=42)

     return X_train, y_train, X_valid, y_valid  
    
    
  def data_scaling(self, X_train, X_valid):


    # standarization
    X_train_scaled = scale(X_train) 
    X_valid_scaled = scale(X_valid)

    return X_train_scaled, X_valid_scaled  
    

  def run_with_timeout(self, func, timeout, *args):
    kill_condition = False 
    process = Process(target=func, args=args)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        print("Function terminated after {} s.".format(timeout))
        kill_condition = True 
        
    return kill_condition


  def train_SVM(self, X_train, y_train, k, c, g, d): 

    # k = kernel, c = Regularization, g = gamma, d = degree
    
    if k=='rbf':
       svm_model = SVC(kernel=k, C=c, gamma=g,  probability=True, random_state=42)
    
    elif k=='poly':   
        svm_model = SVC(kernel=k, C=c, gamma=g, degree = d,  probability=True, random_state=42)
        
    else: #linear
       svm_model = SVC(kernel=k, C=c, probability=True, random_state=42)
       
    
    # Run the function with a 2 min timeout
    
    if self.run_with_timeout(svm_model.fit, 360, X_train, y_train):

        return None 
        
    else:
       start_time = time.time()

       svm_model.fit(X_train, y_train)
       
       end_time = time.time()
       
       print("Train time for {}, {}, {}, {} is:".format(k, c, g, d))
       print(end_time - start_time)

       return svm_model 

     
  def predict(self, trained_model, X_valid):
    
      
  
    y_pred = trained_model.predict(X_valid)
  
       
    y_pred_proba = trained_model.predict_proba(X_valid)[:, 1]

    return y_pred, y_pred_proba 
    
   

  def compare_metrics(self, y_true, y_pred, y_pred_proba):
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
    #    'roc_auc': roc_auc_score(y_true, y_pred_proba)

    }
    return scores
    
   
  def print_scores(self, scores: dict):
  
    print("\nScores:\n=======")
    for metric_name, metric_value in scores.items():
        print(f"{metric_name}: {metric_value}") 
    
  

  def hyperparameter_tuning(self, X_train, y_train, X_valid, y_valid): 


    parameters = [['linear', 0.1, '', ''], ['linear', 1.0, '', ''], ['linear', 100.0, '', ''],
    ['rbf', 0.1, 0.01, ''], ['rbf', 1.0, 0.01, ''], ['rbf', 100.0, 0.01, ''],
    ['rbf', 0.1, 1.0, ''], ['rbf', 1.0, 1.0, ''], ['rbf', 100.0, 1.0, ''],
    ['rbf', 0.1, 1000, ''], ['rbf', 1.0, 1000, ''], ['rbf', 100.0, 1000, ''],
    ['poly', 0.1, 0.01, 3], ['poly', 1.0, 0.01, 3], ['poly', 100.0, 0.01, 3],
    ['poly', 0.1, 1.0, 3], ['poly', 1.0, 1.0, 3], ['poly', 100.0, 1.0, 3],
    ['poly', 0.1, 1000, 5], ['poly', 1.0, 1000, 5], ['poly', 100.0, 1000, 5],
    ['poly', 0.1, 1.0, 2], ['poly', 1.0, 1.0, 2], ['poly', 100.0, 1.0, 2],
    ['poly', 0.1, 1.0, 5], ['poly', 1.0, 1.0, 5], ['poly', 100.0, 1.0, 5]]
    
    for par in parameters:
    
       trained_model = self.train_SVM(X_train, y_train, par[0], par[1], par[2],  par[3])
       y_pred, y_pred_proba = self.predict(trained_model, X_valid)

       # Evaluation
       print('For kernel {}, C {}, gamma {} and degree {}'.format(par[0], par[1], par[2], par[3]))
       metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
       print(metrics_scores)
     
     
   
  def GridSearchCV(self, X, y):
  
    from sklearn.metrics import classification_report
    param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}]
   # 'degree': [2, 3, 5]  # Only relevant for 'poly'
   
   
    grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X, y)

    # Best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    
    return grid_search
 
 
  
  def cross_validation_metrics(self, model, X_train, Y_train):
  
        scoring = {
            "accuracy": 'accuracy',
            "precision": make_scorer(precision_score, average='weighted', zero_division=0),
            "recall": make_scorer(recall_score, average='weighted', zero_division=0),
            "f1": make_scorer(f1_score, average='weighted')
        }
        metrics = {metric: cross_val_score(SVC(kernel='linear'), X_train, Y_train, cv=10, scoring=scorer, n_jobs = -1).mean() for metric, scorer in
                   scoring.items()}
         
         
        return metrics

 
  def oversampling(self, X, y):
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        return X_resampled, y_resampled
        
        

  def dimension_reduction(self, trained_model, X, y):
  
  
     corr_matrix = X.corr().abs()
     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
     to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
     X_reduced = X.drop(columns=to_drop)
     
     metrics = self.cross_validation_metrics(trained_model, X, y)
     print("Metrics after dimension reduction:")
     print(metrics)

 

  def compare_holdout_to_cross_validation(self, X_train, y_train, X_valid, y_valid, k, c, gamma, d):
  
    trained_m = self.train_SVM(X_train, y_train, k, c, gamma, d)
    y_pred, y_pred_proba = self.predict(trained_m, X_valid)
    
    print("Hold out method for best parameters give: ")
  
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    


 
  def apply_classifier(self):
  
    voting_data, rta_data, machine_data, review_data = self.load_datasets()
     
  
    #### Voting dataset ####
     
    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = self.split_dataset(voting_data, valid_split, 'class')
    
    self.log_message("")
    self.log_message("Performance of Support Vector Machines classifier on dataset voting")    
    self.log_message("######################################################")
    
    ### Test classifier on unscaled data ####
    
    trained_model = self.train_SVM(X_train, y_train, 'linear', 1.0, '', '') # no gamma, degree for linear
    
    y_pred, y_pred_proba = self.predict(trained_model, X_valid)
    
    print("Unscaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ### Test classifier on scaled data ####
    
    X_train_scaled, X_valid_scaled = self.data_scaling(X_train, X_valid)
    trained_model_scaled = self.train_SVM(X_train_scaled, y_train, 'linear', 1.0, '', '')
    y_pred, y_pred_proba = self.predict(trained_model_scaled, X_valid_scaled)
    
    print("Scaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ## Cross validation method for scaled data
   
    metrics = self.cross_validation_metrics(trained_model, X_train, y_train)
    print('10-fold cross validation for linear kernel and C=1 and scaled data') 
    print(metrics)
    
    # Parameter tuning manually
    
    self.hyperparameter_tuning(X_train_scaled, y_train, X_valid_scaled, y_valid)
    
    #Find best parameter from given ranges with GridsearcCV
    grid_search = self.GridSearchCV(X_train_scaled, y_train)
    
    #Compare holdout to cross validation for best estimate
    best_kernel = grid_search.best_estimator_.kernel 
   
    if best_kernel =='linear':
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C, '','')
    else:
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C,grid_search.best_estimator_.gamma, '')
        
    
    
    #### Machine failed ####

     
    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = self.split_dataset(machine_data, valid_split, 'fail')
    
    self.log_message("")
    self.log_message("Performance of Support Vector Machines classifier on dataset machine failure")    
    self.log_message("######################################################")
    
    ### Test classifier on unscaled data ####
    
    trained_model = self.train_SVM(X_train, y_train, 'linear', 1.0, '', '') # no gamma, degree for linear
    
    y_pred, y_pred_proba = self.predict(trained_model, X_valid)
    
    print("Unscaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ### Test classifier on scaled data ####
    
    X_train_scaled, X_valid_scaled = self.data_scaling(X_train, X_valid)
    trained_model_scaled = self.train_SVM(X_train_scaled, y_train, 'linear', 1.0, '', '')
    y_pred, y_pred_proba = self.predict(trained_model_scaled, X_valid_scaled)
    
    print("Scaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ## Cross validation method for scaled data
   
    metrics = self.cross_validation_metrics(trained_model, X_train, y_train)
    print('10-fold cross validation for linear kernel and C=1 and scaled data') 
    print(metrics)
    
    # Parameter tuning manually
    
    self.hyperparameter_tuning(X_train_scaled, y_train, X_valid_scaled, y_valid)
    
    #Find best parameter from given ranges with GridsearcCV
    grid_search = self.GridSearchCV(X_train_scaled, y_train)
    
    #Compare holdout to cross validation for best estimate
    best_kernel = grid_search.best_estimator_.kernel 
   
    if best_kernel =='linear':
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C, '','')
    else:
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C,grid_search.best_estimator_.gamma, '')
     
     
   
   
    #### RTA dataset ####
    for col in rta_data.columns:
      label_encoder = LabelEncoder()
      # Encode only non-NaN values
      non_nan_mask = rta_data[col].notna()
      rta_data.loc[non_nan_mask, col] = label_encoder.fit_transform(rta_data.loc[non_nan_mask, col]).astype(int)
      
      
    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = self.split_dataset(rta_data, valid_split, 'Accident_severity')
    
    self.log_message("")
    self.log_message("Performance of Support Vector Machines classifier on dataset RTA")    
    self.log_message("######################################################")
    
    ### Test classifier on unscaled data ####
    
    trained_model = self.train_SVM(X_train, y_train, 'linear', 1.0, '', '') # no gamma, degree for linear
    
    y_pred, y_pred_proba = self.predict(trained_model, X_valid)
    
    print("Unscaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ### Test classifier on scaled data ####
    
    X_train_scaled, X_valid_scaled = self.data_scaling(X_train, X_valid)
    trained_model_scaled = self.train_SVM(X_train_scaled, y_train, 'linear', 1.0, '', '')
    y_pred, y_pred_proba = self.predict(trained_model_scaled, X_valid_scaled)
    
    print("Scaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ## Cross validation method for scaled data
   
    metrics = self.cross_validation_metrics(trained_model, X_train, y_train)
    print('10-fold cross validation for linear kernel and C=1 and scaled data') 
    print(metrics)
    
    # Parameter tuning manually
    
    self.hyperparameter_tuning(X_train_scaled, y_train, X_valid_scaled, y_valid)
    
    #Find best parameter from given ranges with GridsearcCV
    grid_search = self.GridSearchCV(X_train_scaled, y_train)
    
    #Compare holdout to cross validation for best estimate
    best_kernel = grid_search.best_estimator_.kernel 
   
    if best_kernel =='linear':
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C, '','')
    else:
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C,grid_search.best_estimator_.gamma, '')    
   
  
    X_org = rta_data.drop(columns=['Accident_severity'])
    y_org = rta_data['Accident_severity']
    label_encoder2 = LabelEncoder()

    # Oversampling
    y_encoded = label_encoder.fit_transform(y_org)
    X_sampled, y_sampled = self.oversampling(X_org, y_encoded)
    X_sampled_scaled, X_valid_scaled = self.data_scaling(X_sampled, X_valid)
    metrics = self.cross_validation_metrics(trained_model, X_sampled_scaled, y_sampled)
    
    print("Metrics after oversampling:")
    print(metrics)
    
    
    self.dimension_reduction(trained_model, X_org, y_encoded)

    
    
    #### Reviews dataset ####
    for col in review_data.columns:
      label_encoder = LabelEncoder()
      # Encode only non-NaN values
      non_nan_mask = review_data[col].notna()
      review_data.loc[non_nan_mask, col] = label_encoder.fit_transform(review_data.loc[non_nan_mask, col]).astype(int)
      
      
    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = self.split_dataset(review_data, valid_split, 'Class')
    
    self.log_message("")
    self.log_message("Performance of Support Vector Machines classifier on dataset amazon reviews")    
    self.log_message("######################################################")
   
    ### Test classifier on unscaled data ####
    
    trained_model = self.train_SVM(X_train, y_train, 'linear', 1.0, '', '') # no gamma, degree for linear
   
    y_pred, y_pred_proba = self.predict(trained_model, X_valid)
    
    print("Unscaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ### Test classifier on scaled data ####
    
    X_train_scaled, X_valid_scaled = self.data_scaling(X_train, X_valid)
    trained_model_scaled = self.train_SVM(X_train_scaled, y_train, 'linear', 1.0, '', '')
    y_pred, y_pred_proba = self.predict(trained_model_scaled, X_valid_scaled)
    
    print("Scaled data for linear kernel with C=1")
    print("##############")
    metrics_scores = self.compare_metrics(y_valid, y_pred, y_pred_proba)
    print(metrics_scores)
    
    ## Cross validation method for scaled data
   
    metrics = self.cross_validation_metrics(trained_model, X_train_scaled, y_train)
    print('10-fold cross validation for linear kernel and C=1 and scaled data') 
    print(metrics)
    
    # Parameter tuning manually
    
    self.hyperparameter_tuning(X_train_scaled, y_train, X_valid_scaled, y_valid)
    
    #Find best parameter from given ranges with GridsearcCV
    grid_search = self.GridSearchCV(X_train_scaled, y_train)
    
    #Compare holdout to cross validation for best estimate
    best_kernel = grid_search.best_estimator_.kernel 
   
    if best_kernel =='linear':
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C, '','')
    else:
       self.compare_holdout_to_cross_validation(X_train_scaled, y_train, X_valid_scaled, y_valid, best_kernel, grid_search.best_estimator_.C,grid_search.best_estimator_.gamma, '')    
   
    
    #Oversampling 
    X_org = review_data.drop(columns=['Class'])
    y_org = review_data['Class']
    label_encoder2 = LabelEncoder()

    
    y_encoded = label_encoder.fit_transform(y_org)
    X_sampled, y_sampled = self.oversampling(X_org, y_encoded)
    X_sampled_scaled, X_valid_scaled = self.data_scaling(X_sampled, X_valid)
    
    metrics = self.cross_validation_metrics(trained_model, X_sampled_scaled, y_sampled)
    print("Metrics after oversampling:")
    print(metrics)
    
    #Dimension reduction
    self.dimension_reduction(trained_model, X_org, y_encoded)
    

