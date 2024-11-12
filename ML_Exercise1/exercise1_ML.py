import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from RTA_preprocessing import RTA
from voting_classification import votings
from reviews_classification import reviews 
from MachineFailureData_Preprocessing import machine_failure
from machine_learning_tools import ML_tools

import warnings
warnings.filterwarnings('ignore')


import logging

# Configure logging
logging.basicConfig(
    filename='Exercise1.log',       # Log output file
    filemode='w',
    level=logging.INFO,        # Log level: INFO, DEBUG, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)


logging.getLogger('numexpr').setLevel(logging.ERROR)  # Only log errors from NumExpr

warnings.filterwarnings("ignore", category=UserWarning, module="numexpr")  # Ignore UserWarnings in NumExpr





def log_message(message):
    logging.info(message)
    
    
    
def get_datasets():

     votes = votings()
     votes_dataset = votes.run_votings()
     
     rta = RTA()
     rta_dataset=pd.read_csv("RTA_cleaned.csv")
     #rta_dataset = rta.run_rta()

     m_failed = machine_failure()
    # m_failed_dataset = m_failed.run_machine_failure()
     m_failed_dataset=pd.read_csv("Machine_cleaned.csv")
     
     amazon_reviews = reviews()
     reviews_dataset = amazon_reviews.run_reviews()


     return votes_dataset, rta_dataset, m_failed_dataset, reviews_dataset
    

def analyze_votes_dataset(votes_dataset):

    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = ML_tools().split_dataset(votes_dataset, valid_split, 'class')
    
    ###Random Forest ######
    
    log_message("These are the results from RandomForest classifier on dataset voting")    
    # Model fitting
    trained_model = ML_tools().train_RandomForestClassifier(X_train, y_train)
    y_pred, y_pred_proba = ML_tools().predict(trained_model, X_valid)

    # Evaluation
    metrics_scores = ML_tools().compare_metrics(y_valid, y_pred, y_pred_proba)
    log_message(metrics_scores)
    ML_tools().print_scores(metrics_scores)
    

    
    ###Support Vector Machine ######
    log_message("")
    log_message("These are the results from Support Vector Machine classifier on dataset voting")    
    # Model fitting
    parameters = [['linear', 0.1, 'auto'], ['linear', 1.0, 'auto'], ['linear', 100.0, 'auto'], ['linear', 0.1, 'scale'],
    ['linear', 1.0, 'scale'], ['linear', 100.0, 'scale'], ['rbf', 0.1, 'auto'], ['rbf', 1.0, 'auto'], ['rbf', 100.0, 'auto'],
    ['rbf', 0.1, 'scale'], ['rbf', 1.0, 'scale'], ['rbf', 100.0, 'scale'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto']]
    for par in parameters:
    
       trained_model = ML_tools().train_SVM(X_train, y_train, par[0], par[1], par[2])
       y_pred, y_pred_proba = ML_tools().predict(trained_model, X_valid)

       # Evaluation
       log_message('For kernel {}, C {} and gamma {}'.format(par[0], par[1], par[2]))
       metrics_scores = ML_tools().compare_metrics(y_valid, y_pred, y_pred_proba)
       log_message(metrics_scores)
       ML_tools().print_scores(metrics_scores)
   
    
    
    

def analyze_rta_dataset(rta_dataset):


    for col in rta_dataset.columns:
      label_encoder = LabelEncoder()
      # Encode only non-NaN values
      non_nan_mask = rta_dataset[col].notna()
      rta_dataset.loc[non_nan_mask, col] = label_encoder.fit_transform(rta_dataset.loc[non_nan_mask, col]).astype(int)


    valid_split = 0.2
  
    X_train, y_train, X_valid, y_valid = ML_tools().split_dataset(rta_dataset, valid_split, 'Accident_severity')  
    
    ###Support Vector Machine ######
    log_message("")
    log_message("These are the results from Support Vector Machine classifier on dataset road traffic accidents")    
    # Model fitting
    parameters = [['linear', 0.1, 'auto'], ['linear', 1.0, 'auto'], ['linear', 100.0, 'auto'], ['linear', 0.1, 'scale'],
    ['linear', 1.0, 'scale'], ['linear', 100.0, 'scale'], ['rbf', 0.1, 'auto'], ['rbf', 1.0, 'auto'], ['rbf', 100.0, 'auto'],
    ['rbf', 0.1, 'scale'], ['rbf', 1.0, 'scale'], ['rbf', 100.0, 'scale'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto']]
    for par in parameters:
    
       trained_model = ML_tools().train_SVM(X_train, y_train, par[0], par[1], par[2])
       y_pred, y_pred_proba = ML_tools().predict(trained_model, X_valid)

       # Evaluation
       log_message('For kernel {}, C {} and gamma {}'.format(par[0], par[1], par[2]))
       metrics_scores = ML_tools().compare_metrics(y_valid, y_pred, y_pred_proba)
       log_message(metrics_scores)
       ML_tools().print_scores(metrics_scores)
       
       
       

def analyze_machine_failed_dataset(m_failed_dataset):


    for col in m_failed_dataset.columns:
      label_encoder = LabelEncoder()
      # Encode only non-NaN values
      non_nan_mask = m_failed_dataset[col].notna()
      m_failed_dataset.loc[non_nan_mask, col] = label_encoder.fit_transform(m_failed_dataset.loc[non_nan_mask, col]).astype(int)
      
    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = ML_tools().split_dataset(m_failed_dataset, valid_split, 'fail')  
    
    ###Support Vector Machine ######
    log_message("")
    log_message("These are the results from Support Vector Machine classifier on dataset machine failure")    
    # Model fitting
    parameters = [['linear', 0.1, 'auto'], ['linear', 1.0, 'auto'], ['linear', 100.0, 'auto'], ['linear', 0.1, 'scale'],
    ['linear', 1.0, 'scale'], ['linear', 100.0, 'scale'], ['rbf', 0.1, 'auto'], ['rbf', 1.0, 'auto'], ['rbf', 100.0, 'auto'],
    ['rbf', 0.1, 'scale'], ['rbf', 1.0, 'scale'], ['rbf', 100.0, 'scale'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto']]
    for par in parameters:
    
       trained_model = ML_tools().train_SVM(X_train, y_train, par[0], par[1], par[2])
       y_pred, y_pred_proba = ML_tools().predict(trained_model, X_valid)

       # Evaluation
       log_message('For kernel {}, C {} and gamma {}'.format(par[0], par[1], par[2]))
       metrics_scores = ML_tools().compare_metrics(y_valid, y_pred, y_pred_proba)
       log_message(metrics_scores)
       ML_tools().print_scores(metrics_scores)

def analyze_amazon_reviews_dataset(reviews_dataset):
    
    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = ML_tools().split_dataset(reviews_dataset, valid_split, 'Class')
    
    ###Random Forest ######
    
    log_message("These are the results from RandomForest classifier on dataset voting")    
    # Model fitting

    

    
    ###Support Vector Machine ######
    log_message("")
    log_message("These are the results from Support Vector Machine classifier on dataset voting")    
    # Model fitting
    parameters = [['linear', 0.1, 'auto'], ['linear', 1.0, 'auto'], ['linear', 100.0, 'auto'], ['linear', 0.1, 'scale'],
    ['linear', 1.0, 'scale'], ['linear', 100.0, 'scale'], ['rbf', 0.1, 'auto'], ['rbf', 1.0, 'auto'], ['rbf', 100.0, 'auto'],
    ['rbf', 0.1, 'scale'], ['rbf', 1.0, 'scale'], ['rbf', 100.0, 'scale'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto'],
    ['poly', 0.1, 'auto'], ['poly', 1.0, 'auto'], ['poly', 100.0, 'auto']]
    for par in parameters:
    
       trained_model = ML_tools().train_SVM(X_train, y_train, par[0], par[1], par[2])
       y_pred, y_pred_proba = ML_tools().predict(trained_model, X_valid)

       # Evaluation
       log_message('For kernel {}, C {} and gamma {}'.format(par[0], par[1], par[2]))
       metrics_scores = ML_tools().compare_metrics(y_valid, y_pred, y_pred_proba)
       log_message(metrics_scores)
       ML_tools().print_scores(metrics_scores)


if __name__ == '__main__':
  
     votes_dataset, rta_dataset, m_failed_dataset, reviews_dataset =  get_datasets()
     
     analyze_votes_dataset(votes_dataset)
     
     analyze_rta_dataset(rta_dataset)
     analyze_machine_failed_dataset(m_failed_dataset)
     
     analyze_amazon_reviews_dataset(reviews_dataset)
     
     
    
    