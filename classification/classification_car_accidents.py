#code for developing a machine learning model for classification task

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")


plt.rc('xtick', labelsize=8 )  # x tick labels fontsize
plt.rc('ytick', labelsize=8 )  # y tick labels fontsize



def plot_attributes(df):


  directory = "plots_classification_accidents"
  
  if not os.path.exists(directory):
    os.makedirs(directory)

  list_with_interval_columns = ["Age_band_of_driver", "Driving_experience", "Service_year_of_vehicle", "Age_band_of_casualty"]
  #columns which include ratio data, individual processing to order the intervals


  fig, axis = plt.subplots(figsize = (10,6))
  for col in df.columns:
  
   if col=="Time":
   
    df_time = df.copy()
    df_time[col] = pd.to_datetime(df[col].astype(str), format='%H:%M:%S', errors='coerce')

    plt.hist([t.hour + t.minute/60. for t in df_time[col]], bins = 24, color ="blue", edgecolor = "black", align='mid')
   
   elif col=="Day_of_week":
     #order strings based on weekdays
     days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
     df[col] = pd.Categorical(df[col], categories=days_order, ordered=True)
     sns.countplot(x  = col, data = df)
   
   elif col in list_with_interval_columns:
     
     df_int = df.copy()
     # create an additional column where you assign a number based on the interval and use this number to order the dataframe column
     # useful only for better visualization of the histogram
     df_int["index"] = np.where(df_int[col].str.split('-').str.len()>1, df_int[col].str.split('-').str[0], 1000)
     df_int["index"] = np.where((df_int[col].str.split('Under').str.len()>1) | (df_int[col].str.split('Below').str.len()>1), 0, df_int["index"])
     df_int["index"] = np.where((df_int[col].str.split('Over').str.len()>1) | (df_int[col].str.split('Above').str.len()>1), 100, df_int["index"])
     
     df_int["index"] = pd.to_numeric(df_int["index"])
     df_int = df_int.sort_values(by='index').drop(columns='index').reset_index(drop=True)
  
     
     sns.countplot(x = col, data = df_int)

   
   else:
     # for rest of nominal or ordinal data
     sns.countplot(x  = col, data = df)
   

   plt.ylabel('Occurence', fontsize = 10)
   plt.xlabel(col, fontsize = 10)

   if len(df[col].unique())>5 and df[col].apply(pd.to_numeric, errors='coerce').isnull().values.any() :
   #rotate xticks by 90 if column features more than 5 categories and the data of the column is not numerical
    if col!="Time":
       plt.xticks(rotation=90)
 
   else:
       plt.xticks(rotation=0)
       
   plt.tight_layout()
   
   
   plt.savefig(directory + os.sep + '{}.png'.format(col)) #save as .pdf if image resolution is not good enough for latex
   plt.clf()



if __name__ == '__main__':


    df = pd.read_csv('RTA dataset.csv')
    print("Columns with empty cells return True")
    print(df.isnull().any())
    
    print("#########################")
    print("Type of data included in each column")
    print(df.dtypes)
    
    print("#########################")
    print("List of column headers")
    print(df.columns)
    
    plot_attributes(df)