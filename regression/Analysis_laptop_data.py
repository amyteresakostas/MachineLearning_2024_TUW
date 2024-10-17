import pandas as pd
import matplotlib.pyplot as plt
import os

# Set working directory


# Load the dataset
laptop = pd.read_csv("laptop_price - dataset.csv")


# Display the first few rows and the dimensions
print(laptop.head())
print(laptop.shape)  # (1275, 15)

# Rename columns
laptop.columns = list(laptop.columns[:7]) + ['CPU_Frequency[GHz]', 'Ram[GB]'] + list(laptop.columns[9:13]) + [
    'Weight[kg]', 'Price[Euro]']

# Check data types
print(laptop.dtypes)

# Count of data types
print(laptop.dtypes.value_counts())  # Count of each data type

##### Analysis character variables #####

# Initialize an empty dictionary to store frequency tables
table_dict = {}

# Loop through each column in the laptop dataset
for column in laptop.columns:
    if laptop[column].dtype == 'object':  # Check if the column is of type 'object' (string)
        table_dict[column] = laptop[column].value_counts()

# Print the lengths of the frequency tables
for column in table_dict:
    print(f"{column}: {len(table_dict[column])}")

### Barplots ###

# Create a subfolder for plots
os.makedirs("Plots_Laptop", exist_ok=True)

for column, freq_table in table_dict.items():
    # Get the top 10 elements
    top_elements = freq_table.nlargest(10)  # Get top 10 most frequent

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    top_elements.plot(kind='bar', color='skyblue')
    plt.title(f"Top 10: {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout

    # Save the plot
    plt.savefig(f"Plots_Laptop/Plot_character_{column}_top_elements_plot.png", bbox_inches='tight')
    plt.close()  # Close the plot to free memory

##### Analysis integer variable #####

# Analysis of Ram[GB]
print(laptop['Ram[GB]'].value_counts())
print(laptop['Ram[GB]'].min(), laptop['Ram[GB]'].max())  # Range
print(laptop['Ram[GB]'].quantile([0.25, 0.5, 0.75]))  # Quantiles
print(laptop['Ram[GB]'].mean())  # Mean

# Create frequency table for RAM
ram_freq_df = laptop['Ram[GB]'].value_counts().reset_index()
ram_freq_df.columns = ['Ram[GB]', 'Freq']

# Convert 'Ram[GB]' to string type to ensure it is treated as categorical
ram_freq_df['Ram[GB]'] = ram_freq_df['Ram[GB]'].astype(str)

plt.figure(figsize=(10, 6))
ram_freq_df.plot(kind='bar', color='skyblue', legend=False)
plt.title("RAM[GB]")
plt.xlabel("RAM[GB]")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Adjust layout

# Save the plot
plt.savefig(f"Plots_Laptop/Plot_integer_RAM[GB].png", bbox_inches='tight')
plt.close()

##### Analysis double variable #####

# Identify double (float) columns
double_cols = laptop.select_dtypes(include=['float64']).columns

# Display summary statistics for double columns
print(laptop[double_cols].describe())

### Histograms ###

for column in double_cols:
    # Create histogram plot for the current column
    plt.figure(figsize=(10, 6))
    plt.hist(laptop[column], color='skyblue', edgecolor='black', bins=30)  # Adjust number of bins as needed
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"Plots_Laptop/Plot_double_{column}_histogram_plot.png", bbox_inches='tight')
    plt.close()
