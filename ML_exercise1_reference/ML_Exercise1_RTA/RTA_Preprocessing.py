import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy.stats as stats
import os
from statsmodels.stats.contingency_tables import mcnemar

# Set working directory (optional)
os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/RTA")
color = "lightblue"
os.makedirs("plots", exist_ok=True)

# Load data with specified NA values
na_values = ["NA", "", "NULL", "unknown", "Unknown", "na"]
dataset = pd.read_csv("RTA Dataset.csv", na_values=na_values)

print(dataset.head())
print(dataset.describe())
print("Dataset dimensions:", dataset.shape)
missing_values = dataset.isna().sum()
print("Missing values per column:\n", missing_values[missing_values > 0])
print(dataset.dtypes)

# Change datatype
def extract_hour(time_str):
    if pd.isna(time_str):
        return np.nan
    try:
        hour = int(time_str.split(":")[0])
        return hour
    except ValueError:
        return np.nan
dataset['hour'] = dataset['Time'].apply(extract_hour)
age_band_order = ["Under 18", "18-30", "31-50", "Over 51"]
edu_level_order = ["Illiterate", "Writing & Reading", "Elementary school",
                   "Junior high school", "High school", "Above high school"]
driving_experience_order = ["No License", "Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr"]

dataset['Age_band_of_driver'] = pd.Categorical(dataset['Age_band_of_driver'], categories=age_band_order, ordered=True)
dataset['Educational_level'] = pd.Categorical(dataset['Educational_level'], categories=edu_level_order, ordered=True)
dataset['Driving_experience'] = pd.Categorical(dataset['Driving_experience'], categories=driving_experience_order,
                                               ordered=True)
dataset['Number_of_vehicles_involved'] = pd.to_numeric(dataset['Number_of_vehicles_involved'], errors='coerce')
dataset['Number_of_casualties'] = pd.to_numeric(dataset['Number_of_casualties'], errors='coerce')

# Remove unnecessary columns based on criteria
columns_to_drop = ['Service_year_of_vehicle', 'Age_band_of_casualty', 'Casualty_severity',
                   'Work_of_casuality', 'Fitness_of_casuality', 'Defect_of_vehicle', 'Time']
dataset_reduced = dataset.drop(columns=columns_to_drop)
# Drop rows with >11 missing values
data = dataset_reduced.dropna(thresh=dataset_reduced.shape[1] - 11)
print("Reduced dataset dimensions:", dataset_reduced.shape)

# Convert categorical columns to numeric for Spearman correlation
correlation_data = data[['Age_band_of_driver', 'Educational_level', 'Driving_experience',
                                    'Number_of_vehicles_involved', 'Number_of_casualties', 'hour']].apply(
    lambda x: x.cat.codes if x.dtype.name == 'category' else x)

# Spearman correlation matrix
cor_matrix = correlation_data.corr(method='spearman')
sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", square=True)
plt.title('Spearman Correlation Matrix')
plt.savefig("plots/correlation_matrix_num.png", bbox_inches="tight")
#plt.show()

# Cramér's V for categorical pairs
def cramers_v(x, y):
    # Generate the confusion matrix
    confusion_matrix = pd.crosstab(x, y)
    # Calculate the chi-square test statistic
    chi2, _, _, _ = stats.chi2_contingency(confusion_matrix, correction=False)
    # Calculate total number of observations
    n = confusion_matrix.sum().sum()
    # Minimum dimension (k - 1)
    min_dim = min(confusion_matrix.shape) - 1
    # Calculate Cramér's V
    return np.sqrt(chi2 / (n * min_dim))

character_or_factor_columns = data.select_dtypes(include=['category', 'object']).columns
cramersV_results = pd.DataFrame(index=character_or_factor_columns, columns=character_or_factor_columns)
# Populate Cramér's V values between pairs of categorical columns
for col1 in character_or_factor_columns:
    for col2 in character_or_factor_columns:
        if col1 != col2:
            cramersV_results.loc[col1, col2] = cramers_v(data[col1], data[col2])

# Convert results to float for heatmap
cramersV_results = cramersV_results.astype(float)

# Plot heatmap of Cramér's V correlation matrix
sns.heatmap(cramersV_results, annot=True, cmap="coolwarm", square=True, annot_kws={"size": 8}, fmt=".2f")
plt.title("Cramér's V Correlation Matrix")
plt.savefig("plots/correlation_matrix_char.png", bbox_inches="tight")
#plt.show()

# Imputation based on missing data
n = len(data)
tab = round(data.isna().sum() / n * 100, 2)
print(tab[tab > 0])

# Impute missing values in Weather_conditions based on Road_surface_conditions
# Check the relationship between Road_surface_conditions and Weather_conditions
print(pd.crosstab(data['Road_surface_conditions'], data['Weather_conditions']))
print(pd.value_counts(data['Weather_conditions']))
# Imputation rules:
# if Dry --> Normal
# if Flood over 3cm. deep --> Raining
# if Snow --> Snow
# if Wet or damp --> Raining
for i in range(n):
    if pd.isna(data.loc[i, 'Weather_conditions']):  # Only check when Weather_conditions is NA
        if data.loc[i, 'Road_surface_conditions'] == 'Dry':
            data.loc[i, 'Weather_conditions'] = 'Normal'
        elif data.loc[i, 'Road_surface_conditions'] == 'Flood over 3cm. deep':
            data.loc[i, 'Weather_conditions'] = 'Raining'
        elif data.loc[i, 'Road_surface_conditions'] == 'Snow':
            data.loc[i, 'Weather_conditions'] = 'Snow'
        elif data.loc[i, 'Road_surface_conditions'] == 'Wet or damp':
            data.loc[i, 'Weather_conditions'] = 'Raining'
        else:
            data.loc[i, 'Weather_conditions'] = 'Normal'

import pandas as pd
import numpy as np

# Impute Sex_of_driver, Vehicle_driver_relation, Owner_of_vehicle, Area_accident_occured,
# Lanes_or_Medians, Road_allignment, Road_surface_type, Type_of_collision, Vehicle_movement
# and Cause_of_accident with mode
data['Sex_of_driver'] = data['Sex_of_driver'].fillna(data['Sex_of_driver'].mode()[0])
data['Vehicle_driver_relation'] = data['Vehicle_driver_relation'].fillna(data['Vehicle_driver_relation'].mode()[0])
data['Owner_of_vehicle'] = data['Owner_of_vehicle'].fillna(data['Owner_of_vehicle'].mode()[0])
data['Area_accident_occured'] = data['Area_accident_occured'].fillna(data['Area_accident_occured'].mode()[0])
data['Lanes_or_Medians'] = data['Lanes_or_Medians'].fillna(data['Lanes_or_Medians'].mode()[0])
data['Road_allignment'] = data['Road_allignment'].fillna(data['Road_allignment'].mode()[0])
data['Road_surface_type'] = data['Road_surface_type'].fillna(data['Road_surface_type'].mode()[0])
data['Type_of_collision'] = data['Type_of_collision'].fillna(data['Type_of_collision'].mode()[0])
data['Vehicle_movement'] = data['Vehicle_movement'].fillna(data['Vehicle_movement'].mode()[0])
data['Cause_of_accident'] = data['Cause_of_accident'].fillna(data['Cause_of_accident'].mode()[0])
data['Driving_experience'] = data['Driving_experience'].fillna(data['Driving_experience'].mode()[0])
data['Type_of_vehicle'] = data['Type_of_vehicle'].fillna(data['Type_of_vehicle'].mode()[0])

# Impute based on proportion
x = np.round(pd.value_counts(data['Types_of_Junction'], normalize=True) * len(data), 0)
vec = np.repeat(np.arange(len(x)), x)
np.random.shuffle(vec)

for i in range(len(data)):
    if pd.isna(data['Types_of_Junction'][i]):
        data.loc[i, 'Types_of_Junction'] = x.index[vec[i]]

# Impute missing values in Educational_level based on the distribution of Types_of_Junction
x = np.round(pd.value_counts(data['Educational_level'], normalize=True) * len(data), 0)
vec = np.repeat(np.arange(len(x)), x)
np.random.shuffle(vec)

for i in range(len(data)):
    if pd.isna(data['Educational_level'][i]):
        data.loc[i, 'Educational_level'] = x.index[vec[i]]

# Impute missing values in Age_band_of_driver based on the hour
for i in range(len(data)):
    if pd.isna(data['Age_band_of_driver'][i]):
        if data['hour'][i] <= 8:
            data.loc[i, 'Age_band_of_driver'] = '18-30'
        elif data['hour'][i] <= 10:
            data.loc[i, 'Age_band_of_driver'] = '31-50'
        elif data['hour'][i] <= 14:
            data.loc[i, 'Age_band_of_driver'] = '18-30'
        elif data['hour'][i] <= 17:
            data.loc[i, 'Age_band_of_driver'] = '31-50'
        elif data['hour'][i] <= 24:
            data.loc[i, 'Age_band_of_driver'] = '18-30'
        else:
            data.loc[i, 'Age_band_of_driver'] = '18-30'

# Impute missing values in Casualty_class based on Pedestrian_movement
x = np.round(pd.value_counts(data['Casualty_class'], normalize=True) * len(data), 0)
vec = np.repeat(np.arange(len(x)), x)
np.random.shuffle(vec)

for i in range(len(data)):
    if pd.isna(data['Casualty_class'][i]):
        if data['Road_surface_conditions'][i] == 'Not a Pedestrian' and vec[i] == 0:
            data.loc[i, 'Casualty_class'] = 'Driver or rider'
        elif data['Road_surface_conditions'][i] == 'Not a Pedestrian' and vec[i] == 1:
            data.loc[i, 'Casualty_class'] = 'Passenger'
        elif data['Road_surface_conditions'][i] == 'Not a Pedestrian' and vec[i] == 2:
            data.loc[i, 'Casualty_class'] = 'Pedestrian'
        else:
            data.loc[i, 'Casualty_class'] = 'Pedestrian'

# Impute missing values in Educational_level based on Vehicle_driver_relation
for i in range(len(data)):
    if pd.isna(data['Educational_level'][i]) and data['Vehicle_driver_relation'][i] == 'Employee':
        data.loc[i, 'Educational_level'] = 'Junior high school'
    if pd.isna(data['Educational_level'][i]) and data['Vehicle_driver_relation'][i] == 'Other':
        data.loc[i, 'Educational_level'] = 'Junior high school'
    if pd.isna(data['Educational_level'][i]) and data['Vehicle_driver_relation'][i] == 'Owner':
        data.loc[i, 'Educational_level'] = 'Junior high school'

# Impute Age_band_of_driver with the help of hour
for i in range(len(data)):
    if pd.isna(data['Age_band_of_driver'][i]):
        if data['hour'][i] <= 8:
            data.loc[i, 'Age_band_of_driver'] = '18-30'
        elif data['hour'][i] <= 10:
            data.loc[i, 'Age_band_of_driver'] = '31-50'
        elif data['hour'][i] <= 14:
            data.loc[i, 'Age_band_of_driver'] = '18-30'
        elif data['hour'][i] <= 17:
            data.loc[i, 'Age_band_of_driver'] = '31-50'
        elif data['hour'][i] <= 24:
            data.loc[i, 'Age_band_of_driver'] = '18-30'
        else:
            data.loc[i, 'Age_band_of_driver'] = '18-30'


# Save cleaned data
#data.to_csv("RTA_cleaned.csv", index=False)

# Data Analysis (Bar plots for categorical data and histograms and boxplots for numerical data)
# Data Analysis: Displaying dimensions and head of the data
print(f"Data dimensions: {data.shape}")
print(f"Head of the data:\n{data.head()}")

bar_vars = data.columns
hist_vars = ["hour", "Number_of_vehicles_involved", "Number_of_casualties"]

for var in bar_vars:
    prop_data = data[var].value_counts(normalize=True)
    df = prop_data.reset_index()
    df.columns = ['category', 'proportion']
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='category', y='proportion')
    plt.title(f"{var}")
    plt.xlabel(var)
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"plots/plots_{var}_barplot.png", dpi=300, bbox_inches='tight')

for var in hist_vars:
    # Create histogram for numeric variables
    plt.figure(figsize=(8, 6))
    sns.histplot(data[var], kde=False, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {var}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"plots/plots_{var}_histogram.png", dpi=300, bbox_inches='tight')

    # Create boxplot for numeric variables
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, y=var, color='lightblue', flierprops=dict(markerfacecolor='red', marker='o'))
    plt.title(f"Boxplot of {var}")
    plt.ylabel(var)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"plots/plots_{var}_boxplot.png", dpi=300, bbox_inches='tight')


