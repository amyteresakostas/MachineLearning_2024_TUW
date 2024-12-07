import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def preprocessing(dataset, dat_name, target):
    print("Dimensions: ", dataset.shape)
    print("Datatypes: ", dataset.dtypes.unique())
    print("Missing values:", dataset.isna().sum().sum())
    print("Missing values total:", dataset.isna().sum().sum())

    if dataset[target].isna().sum() > 0:
        dataset = dataset.dropna(subset=[target]).reset_index(drop=True)
        print("Target had missing values. These rows were removed.")
        print("Dimensions: ", dataset.shape)

    ### NUMERICAL FEATURES ###
    dataset_num = dataset.select_dtypes(include=['number'])
    summary_table = pd.DataFrame({
        'min': dataset_num.min(),
        'median': dataset_num.median(),
        'mean': dataset_num.mean(),
        'max': dataset_num.max(),
        'variance': dataset_num.var(),
        'stdv': dataset_num.std(),
        'variance_norm': (dataset_num.quantile(0.75) - dataset_num.quantile(0.25)) / (dataset_num.quantile(0.75) + dataset_num.quantile(0.25)) / 2,
        'CV': dataset_num.std() / dataset_num.mean()
    }).round(4)
    print("Summary Table:")
    print(summary_table)

    ### NON NUMERICAL FEATURES ###
    dataset_notNum = dataset.drop(columns = dataset_num.columns)
    for col in dataset_notNum.columns:
        prop = dataset_notNum[col].value_counts(normalize=True)
        if len(prop) < 20:
            print(prop)
        else:
            print(len(prop))

    ### PLOTS ###
    for col in dataset_num.columns:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.histplot(dataset_num[col], kde=False, color='lightblue', ax=axs[0])
        axs[0].set_title(f'Histogram of {col}')
        sns.boxplot(x=dataset_num[col], color='lightblue', ax=axs[1])
        axs[1].set_title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.savefig(f"plots/{dat_name}_{col}.png", bbox_inches="tight")
        plt.close(fig)

    for col in dataset_notNum.columns:
        prop = dataset_notNum[col].value_counts(normalize=True)
        df = prop.reset_index()
        df.columns = ['category', 'proportion']
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x='category', y='proportion')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Proportion")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"plots/{dat_name}_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()

    ### CORRELATION ###
    corr_matrix = dataset_num.corr()
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title(f"{dat_name} Correlation Matrix")
    plt.savefig(f"plots/{dat_name}_Correlation.png", dpi=300)
    plt.close(fig)

    return dataset

os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise 2")
os.makedirs("plots", exist_ok=True)

log_file = open("Preprocessing_ML_Ex2.txt", "w")
sys.stdout = log_file

"""
########## SUPER CONDUCTOR CRITICAL TEMPERATURE ##########
na_values = ["NA", "", "NULL", "unknown", "Unknown", "na"]
CT = pd.read_csv("Superconductor.csv", na_values=na_values)
preprocessing(CT, "CT", "critical_temp")
CT_train, CT_test = train_test_split(CT, test_size=0.2, random_state=42)

### PERFORM PREPROCESSING ON THE TRAINING DATSET ###

### DIMENSIONALITY REDUCTION ###
X = CT_train
corr_matrix = X.iloc[:, :-1].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
CT_train = X.drop(columns=to_drop)
print(f"Reduced features from {X.shape[1] - 1} to {CT_train.shape[1]} after correlation analysis.")
print(to_drop)
CT_test = CT_test.drop(columns=to_drop)

CT_train.to_csv("CT_train.csv", index=False)
CT_test.to_csv("CT_test.csv", index=False)
"""


########## AUTO MPG ##########
na_values = ["NA", "", "NULL", "unknown", "Unknown", "na"]
MPG = pd.read_table("Auto.txt", na_values=na_values) #--> missing values in 2 columns (including target)
print("Unique car-name: ", len(MPG['car_name'].unique()))
MPG['make'] = MPG['car_name'].str.split().str[0]
MPG = MPG.drop(columns = 'car_name')  #--> each car has unique name
print("Unique Make: ", len(MPG['make'].unique()))
MPG = preprocessing(MPG, "MPG", 'mpg')
print("Versuch: ", MPG.shape)

### SPLIT THE DATA ###
MPG_train, MPG_test = train_test_split(MPG, test_size=0.2, random_state=42)
print("Missing values - train:", MPG_train.isna().sum())
print("Missing values - test:", MPG_test.isna().sum())
median_hp = MPG_train['horsepower'].median()
print(median_hp)
MPG_train['horsepower'] = MPG_train['horsepower'].fillna(median_hp)
MPG_test['horsepower'] = MPG_test['horsepower'].fillna(median_hp)
print("Missing values - train:", MPG_train.isna().sum().sum())
print("Missing values - test:", MPG_test.isna().sum().sum())

MPG_train.to_csv("MPG_train.csv", index=False)
MPG_test.to_csv("MPG_test.csv", index=False)

log_file.close()
