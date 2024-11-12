#code for developing a machine learning model for classification task

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


def exlude_features_with_zero(df): 


    list_with_zero_columns = []
    for col in df.columns:
     if col!="Class":
        if np.percentile(df[col], 25) == np.percentile(df[col], 50) == np.percentile(df[col], 75)==0:
            list_with_zero_columns.append(col)

    df = df.drop(columns = list_with_zero_columns, axis=1)
    
    return df




def data_scaling(X_train, X_test):


    # standarization
    X_train_scaled = scale(X_train) 
    X_test_scaled = scale(X_test)

    return X_train_scaled, X_test_scaled

    
  
    

    
def apply_svm_model(X_train_scaled, y_train, X_test_scaled, y_test):

    
    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)


    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)



def split_data_apply_model(df): 

   X = df.drop("Class", axis=1) 
   y = df["Class"] 

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   X_train_scaled, X_test_scaled = data_scaling(X_train, X_test)
   
   apply_svm_model(X_train_scaled, y_train, X_test_scaled, y_test)
   
   


if __name__ == '__main__':


    df = pd.read_csv('amazon_review_ID.shuf.lrn.csv')
    print("#####################################")
    print(df.columns)
    print("Columns with missing values return True, if False then no columns has empty cells!")
    print(df.isnull().values.any())
    
    print("#########################")
    print("Datatypes included in dataframe shown in the following dictionary:") # due to large number of columns
    x = df.columns.to_series().groupby(df.dtypes).groups
    print(x.items())
    
    print("#########################")
    print("A statistical description of dataframe") # here we can see that there are outliers since max is much larger than mean + 3*std for many columns
    print(df.describe())

    split_data_apply_model(df)


  