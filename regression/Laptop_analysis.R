library("ggplot2")

laptop <- read.csv("laptop_price - dataset.csv")

head(laptop)
dim(laptop) #1275 samples, 15 features
colnames(laptop)
colnames(laptop) <- c(colnames(laptop)[1:7], 'CPU_Frequency[GHz]', 'Ram[GB]', colnames(laptop)[10:13], 'Weight[kg]', 'Price[Euro]')
colnames(laptop)

classes <- sapply(laptop, typeof)
table(classes) #10 character, 4 double, 1 integer


##### Analysis character variables #####

# Initialize an empty list to store the tables
table_list <- list()

# Loop through each column in the laptop dataset
for (i in 1:ncol(laptop)) {
  if (is.character(laptop[, i])) {
    # Store the frequency table for the character column in the list
    table_list[[colnames(laptop)[i]]] <- table(laptop[, i])
  }
}

length(table_list[["Company"]]) #19
length(table_list[["Product"]]) #618 --> a lot of variable, maybe we can summarize a little bit
length(table_list[["TypeName"]]) #6
length(table_list[["ScreenResolution"]]) #40
length(table_list[["CPU_Company"]]) #3
length(table_list[["CPU_Type"]]) #93
length(table_list[["Memory"]]) #39
length(table_list[["GPU_Company"]]) #4
length(table_list[["GPU_Type"]]) #106
length(table_list[["OpSys"]]) #9

### Barplots ###

for (column in names(table_list)) {
  # Get the frequency table
  freq_table <- table_list[[column]]
  
  # Convert the table to a data frame for ggplot
  freq_df <- as.data.frame(freq_table)
  
  # Get the top 10 elements
  top_elements <- head(freq_df[order(-freq_df$Freq), ], 10)  # Sort and get the top 10
  
  # Create the bar plot
  p <- ggplot(top_elements, aes(x = reorder(Var1, -Freq), y = Freq)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(title = paste("Top 10: ", column), x = column, y = "Frequency") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
  name <- paste0("Plots_Laptop/", "Plot_character_", column, "_top_elements_plot.png")
  ggsave(name, plot = p, bg = "white")
}


##### Analysis integer variable #####

table(laptop$`Ram[GB]`)
range(laptop$`Ram[GB]`)
quantile(laptop$`Ram[GB]`, c(0.25, 0.5, 0.75))
mean(laptop$`Ram[GB]`)

ram_freq_df <- as.data.frame(table(laptop$`Ram[GB]`))
colnames(ram_freq_df) <- c("Ram[GB]", "Freq")
p <- ggplot(ram_freq_df, aes(x = reorder(`Ram[GB]`, -Freq), y = Freq)) +  # Reorder for descending frequency
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "RAM[GB]", x = "RAM (GB)", y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels
name <- paste0("Plots_Laptop/", "Plot_integer_", "RAM[GB].png")
ggsave(name, plot = p, bg = "white")


##### Analysis double variable #####

### Price is the target value ###

double <- vector()
for (i in 1:ncol(laptop)) {
  if (is.double(laptop[, i])) {
    double <- c(double, i)
  }
}

summary(laptop[, double])

### Histograms ###

columns <- colnames(laptop[, double])

for (column in columns) {
  column_sym <- sym(column)  # Convert column name to a symbol
  
  # Create histogram plot for the current column
  p <- ggplot(laptop, aes(x = !!column_sym)) +  # Use !! to unquote the symbol
    geom_histogram(color = "black", fill = "skyblue", bins = 30) +  # Adjust the number of bins as needed
    labs(title = paste("Histogram of", column), x = column, y = "Frequency") +  # Set title and labels
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Generate a file name for saving the plot
  name <- paste0("Plot_double_", column, "_histogram_plot.png")  # Fixed naming convention
  ggsave(name, plot = p, bg = "white")  # Save the plot
}

