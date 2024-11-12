import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class machine_failure():

  def __init__(self):
  
     super().__init__()


  def plot_failed(self, fail_counts):
  
      color = "lightblue"
      plt.figure(figsize=(6, 4))
      sns.barplot(x=fail_counts.index, y=fail_counts.values, color=color)
      plt.ylim(0, 0.6)
      plt.title("Machine Failure")
      plt.xlabel("Fail")
      plt.ylabel("Proportion")
      plt.savefig("plots/MachineFailure.png", bbox_inches="tight")
      #plt.show()

   
  def barplots(self, indep): 
         
         color = "lightblue"
         # Bar plots
         bar_vars = ["tempMode", "AQ", "USS", "CS", "VOC", "IP", "Temperature"]
         for var in bar_vars:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=indep[var].value_counts(normalize=True).index,
                y=indep[var].value_counts(normalize=True).values,
                color=color, legend=False)
            plt.title(var)
            plt.xlabel(var)
            plt.ylabel("Proportion")
            plt.savefig(f"plots/{var}_barplot.png", bbox_inches="tight")
            #plt.show()

         # Histograms
         hist_vars = ["footfall", "RP", "Temperature"]
         for var in hist_vars:
           plt.figure(figsize=(6, 4))
           sns.histplot(indep[var], kde=True, color=color)
           plt.title(var)
           plt.xlabel(var)
           plt.ylabel("Frequency")
           plt.savefig(f"plots/{var}_histogram.png", bbox_inches="tight")
           #plt.show()
   

  def plot_correlation(self, dataset):
  
      ### Correlation ###
      cor_matrix = dataset.corr(method="pearson")
      plt.figure(figsize=(8, 6))
      #sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
      sns.heatmap(
        cor_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8})
      plt.title("Correlation Matrix")
      plt.savefig("plots/correlation_matrix.png", bbox_inches="tight")
      #plt.show()
      
      

  def run_machine_failure(self):
 
     color = "lightblue"
     os.makedirs("plots", exist_ok=True)

     ### Load dataset ###
     dataset = pd.read_csv("data.csv")
     print(dataset.head())
     print(dataset.shape)  # 944 individuals with 10 variables
     print(dataset.isna().sum())  # There are no missing values in any column
     print(dataset.dtypes)  # All variables are be integers

     # Dependent variable analysis
     fail_counts = dataset['fail'].value_counts(normalize=True)
     print(fail_counts)
     
     self.plot_failed(fail_counts)


     ### Independent variables ###
     indep = dataset.iloc[:, :-1]

     # Summary statistics
     print(indep.describe())

     # Number of Unique values
     unique_counts = indep.nunique()
     print(unique_counts)

     # Frequency tables and proportions
     for col in indep.columns:
        counts = indep[col].value_counts()
        proportions = counts / len(indep) * 100
        print(f"Counts for {col}:\n{counts}\n")
        print(f"Proportions for {col}:\n{proportions.round(2)}%\n")


     self.barplots(ind)
     self.plot_correlation(dataset)

     ### Outlier detection ###
     for col in indep.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=indep[col], color=color)  # Use 'y' to make the boxplot vertical
        plt.title(col)
        plt.ylabel(col)
        plt.savefig(f"plots/{col}_boxplot.png", bbox_inches="tight")
        #plt.show()

     # Save the cleaned dataset to a new CSV file
     dataset.to_csv("Machine_cleaned.csv", index=False)
     
     return dataset