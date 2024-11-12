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


class ML_tools:


  def __init__(self):
  
     pass
     
     
     
  def split_dataset(self, df, valid_size, output_class):
    
    X = df.drop(columns=[output_class])
    try:
       y = df[output_class].astype(int)
    except:
       y = df[output_class]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=42)

    return X_train, y_train, X_valid, y_valid
    
    
  
  def data_scaling(self, X_train, X_valid):


    # standarization
    X_train_scaled = scale(X_train) 
    X_valid_scaled = scale(X_valid)

    return X_train_scaled, X_valid_scaled  
    
   
  def train_RandomForestClassifier(self, X_train, y_train):
    trained_model = RandomForestClassifier()
    trained_model.fit(X_train, y_train)

    return trained_model


  def train_SVM(self, X_train, y_train, k, c, g): 

    # Model fitting
    if k!='linear':
       svm_model = SVC(kernel=k, C=c, gamma=g,  probability=True, random_state=42)
    else:
       svm_model = SVC(kernel=k, C=c, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    return svm_model 
  
  
  def do_GridSearchCV(self, dataset):
  
  
     pass
  
  
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