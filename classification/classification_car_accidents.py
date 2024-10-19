#code for developing a machine learning model for classification task

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")




def plot_attributes(df):


  directory = "plots_classification_accidents"
  
  if not os.path.exists(directory):
    os.makedirs(directory)

  list_with_interval_columns = ["Age_band_of_driver", "Driving_experience", "Service_year_of_vehicle", "Age_band_of_casualty"]
  dataframes  = {} # store copied dataframes, actually not necessary
  ind = 0 # index for dataframes
  
  for i in range(len(list_with_interval_columns)):
      dataframes[f'df_copy_{i}'] = df.copy()


  fig, axis = plt.subplots(figsize = (10,4))
  for col in df.columns:
 
   if col=="Time":
   
    df_time = df.copy()
    df_time[col] = pd.to_datetime(df[col].astype(str), format='%H:%M:%S', errors='coerce')

    plt.hist([t.hour + t.minute/60. for t in df_time[col]], bins = 24, color='skyblue', edgecolor = "black", align='mid')
   
   elif col=="Day_of_week":
     #order strings based on weekdays
     days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
     df[col] = pd.Categorical(df[col], categories=days_order, ordered=True)

     df[col].value_counts(sort = False).plot(kind='bar', color='skyblue', legend=False) #.reset_index()

   
   elif col in list_with_interval_columns:
       
     # create an additional column where you assign a number based on the interval and use this number to order the dataframe column
     # useful only for better visualization of the histogram
     dataframes[f'df_copy_{ind}']["index"] = np.where(df[col].str.split('-').str.len()>1, df[col].str.split('-').str[0], 1000)
     dataframes[f'df_copy_{ind}']["index"] = np.where((df[col].str.split('Under').str.len()>1) | (df[col].str.split('Below').str.len()>1), 0, dataframes[f'df_copy_{ind}']["index"])
     dataframes[f'df_copy_{ind}']["index"] = np.where((df[col].str.split('Over').str.len()>1) | (df[col].str.split('Above').str.len()>1), 100, dataframes[f'df_copy_{ind}']["index"])
     
     dataframes[f'df_copy_{ind}']["index"] = pd.to_numeric(dataframes[f'df_copy_{ind}']["index"])
   
     dataframes[f'df_copy_{ind}'] =  dataframes[f'df_copy_{ind}'].sort_values(by='index').drop(columns='index').reset_index(drop=True)
     
     dataframes[f'df_copy_{ind}'][col].value_counts(sort=False).reindex(dataframes[f'df_copy_{ind}'][col].unique()).plot(kind='bar', color='skyblue', legend=False)

     
     ind +=1
 
   
   else:
     # for rest of nominal or ordinal data
    
     df[col].value_counts().plot(kind='bar', color='skyblue', legend=False)
     


   plt.xlabel(col)
   plt.ylabel("Frequency")


   if len(df[col].unique())>5 and df[col].apply(pd.to_numeric, errors='coerce').isnull().values.any() :
   #rotate xticks by 90 if column features more than 5 categories and the data of the column is not numerical
    if col!="Time":
       plt.xticks(rotation=90, ha="right")
       
   else:
       plt.xticks(rotation=45, ha="right")
       

   plt.title(col)
  
   plt.savefig(directory + os.sep + '{}.png'.format(col), bbox_inches='tight') #save as .pdf if image resolution is not good enough for latex
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