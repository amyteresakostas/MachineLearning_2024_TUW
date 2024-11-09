setwd("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1")
library(corrplot)

dataset <- read.csv("data.csv")
head(dataset)

dim(dataset) # 944 individuals with 10 variables
apply(is.na(dataset), 2, sum) # There are no missing values in the entire datset
apply(dataset, 2, class) # all variables are integer


# dependet variable #
dataset$fail
table(dataset$fail)
prop.table(table(dataset$fail))
barplot(prop.table(table(dataset$fail)), ylim = c(0, 0.6))


# independent variables #
indep = dataset[, 1:length(dataset)-1]

summary(indep)
apply(indep, 2, sd)

apply(dataset, 2, function(x) {length(unique(x))}) 
apply(indep, 2, unique)

apply(indep, 2, table)
apply(indep, 2, function(x) {round(prop.table(table(x))*100, 2)})

barplot(prop.table(table(indep$tempMode)))
barplot(prop.table(table(indep$AQ)))
barplot(prop.table(table(indep$USS)))
barplot(prop.table(table(indep$CS)))
barplot(prop.table(table(indep$VOC)))
barplot(prop.table(table(indep$IP)))
barplot(prop.table(table(indep$Temperature)))
hist(indep$footfall)
hist(indep$RP)
hist(indep$Temperature)

cor_matrix <- cor(dataset, method = "pearson")
corrplot(cor_matrix, 
                   method = "square",              
                   addCoef.col = "black",         
                   number.cex = 0.7,              
                   col = colorRampPalette(c("red", "white", "blue"))(200),  
                   tl.col = "black",             
                   tl.srt = 45,                   
                   type = "full") 

#some outliers, but they all make sense, so I won't remove any
colnames = colnames(indep)
for(i in 1:length(indep)) {
  boxplot(indep[, i], main = colnames[i])
}

write.csv(dataset,"Machine_cleaned.csv", row.names = FALSE)
