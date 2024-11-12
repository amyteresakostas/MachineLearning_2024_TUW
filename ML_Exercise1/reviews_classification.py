import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


class reviews():

  def __init__(self):
  
     super().__init__()


  def plot_class(self, df):
  
    top10 = df['Class'].value_counts().head(10)
    top10 = top10.reset_index()
    top10.columns = ['Class', 'Frequency']
    plt.figure(figsize=(6, 4))
    sns.barplot(x=top10['Class'], y=top10['Frequency'], color="red")
    plt.ylabel("Frequency")
    plt.title("Barplot of 10 most frequent values of Class")
    plt.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("reviews_class.png")
    
    
  def plot_correlation(self, df):
  
    
    first_10 = df.iloc[:, :11]
    first_10 = first_10.drop("ID", axis=1)
   
    
    
    column_variance = df.drop(["Class", "ID"], axis =1).var() 
    top10_variance = df[column_variance.nlargest(10).index ]

    
    
    random_columns = df.drop(["Class", "ID"], axis =1).sample(axis=1, n=10) 

    
    df_dict = {'first_10':first_10, 'top_10_variance':top10_variance, '10_random':random_columns }
    for key, dataframe in df_dict.items():
      
      label_encoder = LabelEncoder()
      dataframe['Class'] = df['Class']
      dataframe['Class'] = label_encoder.fit_transform(dataframe['Class']).astype(int)
      column_names = dataframe.columns.tolist()
      # Correlation matrix
      correlations = dataframe.corr(method='kendall')

      # Plot figsize
      fig, ax = plt.subplots(figsize=(10, 10))
      # Generate Color Map
      colormap = sns.diverging_palette(220, 10, as_cmap=True)
      # Generate Heat Map, allow annotations and place floats in map
      sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
      ax.set_xticklabels(
        column_names,
        rotation=45,
        horizontalalignment='right'
    )
      ax.set_yticklabels(column_names)
      plt.title('Correlation map of {} columns'.format(key))
      plt.tight_layout()
      plt.savefig('correlation_map_{}.png'.format(key))
      plt.clf()
  
  



  def run_reviews(self):
   
   
    print("\033[92mAnalyzing Amazon reviews dataset\033[0m")
    df = pd.read_csv('reviews/amazon_review_ID.shuf.lrn.csv')
   
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
    self.plot_class(df)
    self.plot_correlation(df)
    
    return df 