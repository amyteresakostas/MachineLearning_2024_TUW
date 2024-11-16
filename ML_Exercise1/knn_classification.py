import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay, make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, \
    matthews_corrcoef
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
import sys
from sklearn.model_selection import GridSearchCV

os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1")
os.makedirs("plots", exist_ok=True)
log_file = open("output.txt", "w")
sys.stdout = log_file


########## AMAZON REVIEW DATASET ##########
print("----------------------------------------------------------------------------------------------")
print("--------------------------------------- AMAZON REVIEW ----------------------------------------")
print("----------------------------------------------------------------------------------------------")
review_data = pd.read_csv("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/Reviews/amazon_review_ID.shuf.lrn.csv")
review_data = review_data.drop(columns='ID')

# Split into training and testing sets (80/20 split)
train_data, test_data = train_test_split(review_data, test_size=0.2, random_state=42)
X_train = train_data.drop('Class', axis=1); y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1); y_test = test_data['Class']

### We first tried to do the classification on the datset without scaling, without crossvalidation and k=3
print("----------------------------------------------------------------------------------------------")
print("Baseline - KNN without Scaling")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
## Evaluate performance ##
# Confusion Matrix
conf_matrix1 = confusion_matrix(y_test, y_pred1)
print("RTA - Without Scaling: ")
print(conf_matrix1)
disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix1)
disp1.plot(cmap='Blues')
plt.title('Confusion Matrix - No Scaling')
plt.savefig(f"plots/rta_cm_withoutScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision (weighted):", precision_score(y_test, y_pred1, average='weighted', zero_division=1))
print("Recall (weighted):", recall_score(y_test, y_pred1, average='weighted', zero_division=1))
print("F1 Score (weighted):", f1_score(y_test, y_pred1, average='weighted', zero_division=1))

### KNN with Scaling
print("----------------------------------------------------------------------------------------------")
print("KNN with Scaling")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred2 = knn.predict(X_test_scaled)
## Evaluate performance ##
# Confusion matrix
conf_matrix2 = confusion_matrix(y_test, y_pred2)
print("RTA - With Scaling: ")
print(conf_matrix2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix2)
disp2.plot(cmap='Blues')
plt.savefig(f"plots/rta_cm_withScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision (weighted):", precision_score(y_test, y_pred2, average='weighted', zero_division=1))
print("Recall (weighted):", recall_score(y_test, y_pred2, average='weighted', zero_division=1))
print("F1 Score (weighted):", f1_score(y_test, y_pred2, average='weighted', zero_division=1))

### KNN with Cross Validation (5-fold) and without scaling since this performed better
print("----------------------------------------------------------------------------------------------")
print("KNN with Cross Validation (5-fold)")
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1': make_scorer(f1_score, average='weighted', zero_division=1)
}
cv_results = cross_validate(knn, X_train, y_train, cv=5, scoring=scoring)
print(f"Cross-validation Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

### Hyperparameter Tuning - Find Optimal K using Cross-validation
print("----------------------------------------------------------------------------------------------")
print("Hyperparameter Tuning - Optimal k")
"""
k_values = range(1, 30)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_results = cross_validate(knn, X_train, y_train, cv=5, scoring=scoring)
    accuracy_scores.append(cv_results['test_accuracy'].mean())
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracy_scores)
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
#plt.savefig("plots/rta_optimalK.png", bbox_inches="tight")
optimal_k = k_values[np.argmax(accuracy_scores)]
print("Optimal k:", optimal_k)
"""
optimal_k = 1
print("Optimal k:", optimal_k)

### Dimensionality Reduction
print("----------------------------------------------------------------------------------------------")
print("Dimensionality Reduction")
X = review_data.drop(columns='Class')
y = review_data['Class']
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X = X.drop(columns=to_drop)
print(f"Reduced features from {review_data.shape[1] - 1} to {X.shape[1]} after correlation analysis.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=optimal_k)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1': make_scorer(f1_score, average='weighted', zero_division=1)
}
cv_results = cross_validate(knn, X_train, y_train, cv=5, scoring=scoring)
print(f"Cross-validation Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

### Oversampling using RandomOverSampler to address class imbalance
print("----------------------------------------------------------------------------------------------")
print("Oversampling using RandomOverSampler")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_train_resampled)}")
# Re-train with resampled data
knn = KNeighborsClassifier(n_neighbors=optimal_k)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1': make_scorer(f1_score, average='weighted', zero_division=1)
}
cv_results = cross_validate(knn, X_train_resampled, y_train_resampled, cv=5, scoring=scoring)
print(f"Cross-validation Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

#knn = KNeighborsClassifier(n_neighbors=optimal_k)
#knn.fit(X_train_resampled, y_train_resampled)
#y_pred_resampled = knn.predict(X_test)
# Evaluate performance on resampled data
#conf_matrix_resampled = confusion_matrix(y_test, y_pred_resampled)
#print("Confusion Matrix - Oversampling:")
#print(conf_matrix_resampled)
#disp_resampled = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_resampled)
#disp_resampled.plot(cmap='Blues')
#plt.title('Confusion Matrix - Oversampling')
#plt.savefig("plots/rta_cm_oversampling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score for oversampled data
#print("Accuracy:", accuracy_score(y_test, y_pred_resampled))
#print("Precision (weighted):", precision_score(y_test, y_pred_resampled, average='weighted'))
#print("Recall (weighted):", recall_score(y_test, y_pred_resampled, average='weighted'))
#print("F1 Score (weighted):", f1_score(y_test, y_pred_resampled, average='weighted'))

### Final Model with Optimal k
print("----------------------------------------------------------------------------------------------")
print(f"Final Model with Optimal k={optimal_k}")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cv_results = cross_validate(final_knn, X_train, y_train, cv=5, scoring=scoring)
print(f"Cross-validation Results with optimal k:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")
# Train and test final model
final_knn.fit(X_train, y_train)
final_predictions = final_knn.predict(X_test)
plt.title(f'Final Confusion Matrix (k={optimal_k})')
final_conf_matrix = confusion_matrix(y_test, final_predictions)
print(final_conf_matrix)
disp_final = ConfusionMatrixDisplay(confusion_matrix=final_conf_matrix)
disp_final.plot(cmap='Blues')
plt.savefig("plots/machine_cm_finalModel.png", bbox_inches="tight")
print("Final Model Evaluation - optimal k:")
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
print("Precision (weighted):", precision_score(y_test, final_predictions, average='weighted', zero_division=1))
print("Recall (weighted):", recall_score(y_test, final_predictions, average='weighted', zero_division=1))
print("F1 Score (weighted):", f1_score(y_test, final_predictions, average='weighted', zero_division=1))
plt.savefig("plots/review_finalModel.png", bbox_inches="tight")



########## ROAD TRAFFIC ACCIDENT DATASET ##########
print("----------------------------------------------------------------------------------------------")
print("----------------------------------- ROAD TRAFFIC ACCIDENT ------------------------------------")
print("----------------------------------------------------------------------------------------------")
rta_data = pd.read_csv("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/RTA/RTA_cleaned.csv")
rta_data['Accident_severity'] = rta_data['Accident_severity'].astype('category')

### Split into training and testing sets (80/20 split) ###
train_data, test_data = train_test_split(rta_data, test_size=0.2, random_state=42)
X_train = train_data.drop('Accident_severity', axis=1); y_train = train_data['Accident_severity']
X_test = test_data.drop('Accident_severity', axis=1); y_test = test_data['Accident_severity']
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

### We first tried to do the classification on the datset without scaling, without crossvalidation and k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
## Evaluate performance ##
# Confusion Matrix
conf_matrix1 = confusion_matrix(y_test, y_pred1)
print("RTA - Without Scaling: ")
print(conf_matrix1)
disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix1)
disp1.plot(cmap='Blues')
plt.title('Confusion Matrix - No Scaling')
plt.savefig(f"plots/rta_cm_withoutScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision (weighted):", precision_score(y_test, y_pred1, average='weighted'))
print("Recall (weighted):", recall_score(y_test, y_pred1, average='weighted'))
print("F1 Score (weighted):", f1_score(y_test, y_pred1, average='weighted'))


### Since knn-algorithms are very sensitive to scaling, we added scaling to see how much the resuts would change.
### We used the StandardScaler, we sill don't use cross validation and have k = 3
print("----------------------------------------------------------------------------------------------")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred2 = knn.predict(X_test_scaled)
## Evaluate performance ##
# Confusion matrix
conf_matrix2 = confusion_matrix(y_test, y_pred2)
print("RTA - With Scaling: ")
print(conf_matrix2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix2)
disp2.plot(cmap='Blues')
plt.savefig(f"plots/rta_cm_withScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision (weighted):", precision_score(y_test, y_pred2, average='weighted'))
print("Recall (weighted):", recall_score(y_test, y_pred2, average='weighted'))
print("F1 Score (weighted):", f1_score(y_test, y_pred2, average='weighted'))

### We added a 10-fold cross validation to see if we would get an even more accurate result
print("----------------------------------------------------------------------------------------------")
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=10)
print("RTA - With Cross Validation: ")
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}
cv_results = cross_validate(knn, X_train_scaled, y_train, cv=10, scoring=scoring)
print(f"Cross-validation Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

### Hyperparameter tuning (finding optimal k)
print("----------------------------------------------------------------------------------------------")
k_values = range(1, 30)
accuracy_scores = []
#for k in k_values:
    #knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(X_train_scaled, y_train)
    #y_pred = knn.predict(X_test_scaled)
    #accuracy_scores.append(accuracy_score(y_test, y_pred))
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted', zero_division=1),
        'recall': make_scorer(recall_score, average='weighted', zero_division=1),
        'f1': make_scorer(f1_score, average='weighted', zero_division=1)
    }
    cv_results = cross_validate(knn, X_train_scaled, y_train, cv=10, scoring=scoring)
    accuracy_scores.append(cv_results['test_accuracy'].mean())
# Create a new figure for the accuracy plot
plt.figure(figsize=(8, 6))  # Optional: to set the figure size
plt.plot(k_values, accuracy_scores)  # Line plot with markers
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)  # Optional: to add grid lines to the plot
plt.savefig(f"plots/rta_optimalK.png", bbox_inches="tight")
optimal_k = k_values[np.argmax(accuracy_scores)]
print("Optimal k:", optimal_k)

### Final model with optimal k - Crossvalidation
print("----------------------------------------------------------------------------------------------")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1': make_scorer(f1_score, average='weighted', zero_division=1)
}
cv_results = cross_validate(final_knn, X_train_scaled, y_train, cv=10, scoring=scoring)
print(f"Cross-validation Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

### Final model with optimal k
print("----------------------------------------------------------------------------------------------")
final_knn.fit(X_train_scaled, y_train)
final_predictions = final_knn.predict(X_test_scaled)
plt.title(f'Final Confusion Matrix (k={optimal_k})')
final_conf_matrix = confusion_matrix(y_test, final_predictions)
print(final_conf_matrix)
disp_final = ConfusionMatrixDisplay(confusion_matrix=final_conf_matrix)
disp_final.plot(cmap='Blues')
plt.savefig(f"plots/machine_cm_finalModel.png", bbox_inches="tight")
print("Machine Failure - optimal k - Final model: ")
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
print("Precision (weighted):", precision_score(y_test, final_predictions, average='weighted', zero_division=1))
print("Recall (weighted):", recall_score(y_test, final_predictions, average='weighted', zero_division=1))
print("F1 Score (weighted):", f1_score(y_test, final_predictions, average='weighted', zero_division=1))
plt.savefig(f"plots/rta_finalModel.png", bbox_inches="tight")

### We want to try oversampling for the data, since the second and especially the third category are underrepresented
print("----------------------------------------------------------------------------------------------")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_train_resampled)}")
#smote = SMOTE(random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
#print(f"Original class distribution: {Counter(y_train)}")
#print(f"Resampled class distribution: {Counter(y_train_resampled)}")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_resampled, y_train_resampled)
y_pred = knn.predict(X_test_scaled)
## Evaluate performance ##
# Confusion Matrix
conf_matrix1 = confusion_matrix(y_test, y_pred)
print("RTA - Oversampling: ")
print(conf_matrix1)
disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix1)
disp1.plot(cmap='Blues')
plt.title('Confusion Matrix - No Scaling')
plt.savefig(f"plots/rta_cm_withoutScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(y_test, y_pred, average='weighted'))
print("Recall (weighted):", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))

print("----------------------------------------------------------------------------------------------")
print("RTA - Oversampling - Cross Validation: ")
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn, X_train_resampled, y_train_resampled, cv=10)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}
cv_results = cross_validate(knn, X_train_resampled, y_train_resampled, cv=10, scoring=scoring)
print(f"Cross-validation Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

print("----------------------------------------------------------------------------------------------")
print("RTA - Oversampling - Optimal k: ")
k_values = range(1, 30)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted', zero_division=1),
        'recall': make_scorer(recall_score, average='weighted', zero_division=1),
        'f1': make_scorer(f1_score, average='weighted', zero_division=1)
    }
    cv_results = cross_validate(knn, X_train_resampled, y_train_resampled, cv=10, scoring=scoring)
    accuracy_scores.append(cv_results['test_accuracy'].mean())
# Create a new figure for the accuracy plot
plt.figure(figsize=(8, 6))  # Optional: to set the figure size
plt.plot(k_values, accuracy_scores)  # Line plot with markers
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)  # Optional: to add grid lines to the plot
plt.savefig(f"plots/rta_optimalK.png", bbox_inches="tight")
optimal_k = k_values[np.argmax(accuracy_scores)]
print("Optimal k:", optimal_k)

### Final model with optimal k - Crossvalidation
print("----------------------------------------------------------------------------------------------")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1': make_scorer(f1_score, average='weighted', zero_division=1)
}
cv_results = cross_validate(final_knn, X_train_resampled, y_train_resampled, cv=10, scoring=scoring)
print(f"Cross-validation Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

### Final model with optimal k
print("----------------------------------------------------------------------------------------------")
final_knn.fit(X_train_resampled, y_train_resampled)
final_predictions = final_knn.predict(X_test_scaled)
plt.title(f'Final Confusion Matrix (k={optimal_k})')
final_conf_matrix = confusion_matrix(y_test, final_predictions)
print(final_conf_matrix)
disp_final = ConfusionMatrixDisplay(confusion_matrix=final_conf_matrix)
disp_final.plot(cmap='Blues')
plt.savefig(f"plots/machine_cm_finalModel.png", bbox_inches="tight")
print("Machine Failure - optimal k - Final model: ")
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
print("Precision (weighted):", precision_score(y_test, final_predictions, average='weighted', zero_division=1))
print("Recall (weighted):", recall_score(y_test, final_predictions, average='weighted', zero_division=1))
print("F1 Score (weighted):", f1_score(y_test, final_predictions, average='weighted', zero_division=1))
plt.savefig(f"plots/rta_finalModel.png", bbox_inches="tight")
#log_file.close()

########## MACHINE FAILURE DATASET ##########
print("----------------------------------------------------------------------------------------------")
print("-------------------------------------- MACHINE FAILURE ---------------------------------------")
print("----------------------------------------------------------------------------------------------")
machine_data = pd.read_csv("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/MachineFailure/Machine_cleaned.csv")
machine_data['fail'] = machine_data['fail'].astype('category')

### Split into training and testing sets (80/20 split) ###
train_data, test_data = train_test_split(machine_data, test_size=0.2, random_state=42)
X_train = train_data.drop('fail', axis=1); y_train = train_data['fail']
X_test = test_data.drop('fail', axis=1); y_test = test_data['fail']

### We first tried to do the classification on the datset without scaling, without crossvalidation and k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
## Evaluate performance ##
# Confusion Matrix
conf_matrix1 = confusion_matrix(y_test, y_pred1)
print("Machine Failure - Without Scaling: ")
print(conf_matrix1)
disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix1)
disp1.plot(cmap='Blues')
plt.title('Confusion Matrix - No Scaling')
plt.savefig(f"plots/machine_cm_withoutScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1, pos_label=1))
print("Recall:", recall_score(y_test, y_pred1, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred1, pos_label=1))
# AUC-ROC, AUC-PR, MCC
roc_auc = roc_auc_score(y_test, y_pred1)
fpr, tpr, _ = roc_curve(y_test, y_pred1)
print("AUC-ROC:", roc_auc)
precision, recall, _ = precision_recall_curve(y_test, y_pred1)
auc_pr = average_precision_score(y_test, y_pred1)
print("AUC-PR:", auc_pr)
mcc = matthews_corrcoef(y_test, y_pred1)
print("MCC:", mcc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}'); plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (No Scaling)'); plt.legend(loc='lower right')
plt.savefig(f"plots/machine_ROC_withoutScaling.png", bbox_inches="tight")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (No Scaling)'); plt.legend(loc='lower left')
plt.savefig(f"plots/machine_PR_withoutScaling.png", bbox_inches="tight")

### Since knn-algorithms are very sensitive to scaling, we added scaling to see how much the resuts would change.
### We used the StandardScaler, we sill don't use cross validation and have k = 3
print("----------------------------------------------------------------------------------------------")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred2 = knn.predict(X_test_scaled)
## Evaluate performance ##
# Confusion matrix
conf_matrix2 = confusion_matrix(y_test, y_pred2)
print("Machine Failure - With Scaling: ")
print(conf_matrix2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix2)
disp2.plot(cmap='Blues')
plt.savefig(f"plots/machine_cm_withScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2, pos_label=1))
print("Recall:", recall_score(y_test, y_pred2, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred2, pos_label=1))
# AUC-ROC, AUC-PR, MCC
roc_auc = roc_auc_score(y_test, y_pred2)
fpr, tpr, _ = roc_curve(y_test, y_pred2)
print("AUC-ROC:", roc_auc)
precision, recall, _ = precision_recall_curve(y_test, y_pred2)
auc_pr = average_precision_score(y_test, y_pred2)
print("AUC-PR:", auc_pr)
mcc = matthews_corrcoef(y_test, y_pred2)
print("MCC:", mcc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}'); plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (with Scaling)'); plt.legend(loc='lower right')
plt.savefig(f"plots/machine_ROC_withScaling.png", bbox_inches="tight")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (with Scaling)'); plt.legend(loc='lower left')
plt.savefig(f"plots/machine_PR_withScaling.png", bbox_inches="tight")

### We added a 10-fold cross validation to see if we would get an even more accurate result
print("----------------------------------------------------------------------------------------------")
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=10)
print("Machine Failure - With Cross Validation: ")
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(knn, X_train_scaled, y_train, cv=10, scoring=scoring)
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

### Hyperparameter tuning (finding optimal k)
print("----------------------------------------------------------------------------------------------")
k_values = range(1, 30)
accuracy_scores = []
#for k in k_values:
    #knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(X_train_scaled, y_train)
    #y_pred = knn.predict(X_test_scaled)
    #accuracy_scores.append(accuracy_score(y_test, y_pred))
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')
    accuracy_scores.append(cv_scores.mean())
# Create a new figure for the accuracy plot
plt.figure(figsize=(8, 6))  # Optional: to set the figure size
plt.plot(k_values, accuracy_scores)  # Line plot with markers
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)  # Optional: to add grid lines to the plot
plt.savefig(f"plots/machine_optimalK.png", bbox_inches="tight")
optimal_k = k_values[np.argmax(accuracy_scores)]
print("Optimal k:", optimal_k)

### Final model with optimal k - Crossvalidation
print("----------------------------------------------------------------------------------------------")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cv_results = cross_validate(final_knn, X_train_scaled, y_train, cv=10,
                            scoring=['accuracy', 'precision', 'recall', 'f1'],
                            return_train_score=False)
print(f"Machine Failure - With optimal k={optimal_k} (Cross-validation): ")
print(f"Average Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Average Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Average Recall: {cv_results['test_recall'].mean():.4f}")
print(f"Average F1 Score: {cv_results['test_f1'].mean():.4f}")

### Final model with optimal k
print("----------------------------------------------------------------------------------------------")
final_knn.fit(X_train_scaled, y_train)
final_predictions = final_knn.predict(X_test_scaled)
plt.title(f'Final Confusion Matrix (k={optimal_k})')
final_conf_matrix = confusion_matrix(y_test, final_predictions)
print(final_conf_matrix)
disp_final = ConfusionMatrixDisplay(confusion_matrix=final_conf_matrix)
disp_final.plot(cmap='Blues')
plt.savefig(f"plots/machine_cm_finalModel.png", bbox_inches="tight")
print("Machine Failure - optimal k - Final model: ")
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
print("Final Model Precision:", precision_score(y_test, final_predictions, pos_label=1))
print("Final Model Recall:", recall_score(y_test, final_predictions, pos_label=1))
print("Final Model F1 Score:", f1_score(y_test, final_predictions, pos_label=1))
plt.savefig(f"plots/machine_finalModel.png", bbox_inches="tight")
# AUC-ROC, AUC-PR, MCC
roc_auc = roc_auc_score(y_test, final_predictions)
fpr, tpr, _ = roc_curve(y_test, final_predictions)
print("AUC-ROC:", roc_auc)
precision, recall, _ = precision_recall_curve(y_test, final_predictions)
auc_pr = average_precision_score(y_test, final_predictions)
print("AUC-PR:", auc_pr)
mcc = matthews_corrcoef(y_test, final_predictions)
print("MCC:", mcc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}'); plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (No Scaling)'); plt.legend(loc='lower right')
plt.savefig(f"plots/machine_ROC_finalModel.png", bbox_inches="tight")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (No Scaling)'); plt.legend(loc='lower left')
plt.savefig(f"plots/machine_PR_finalModel.png", bbox_inches="tight")



########## CONGRESSIONAL VOTING ##########
print("----------------------------------------------------------------------------------------------")
print("------------------------------------------- VOTING -------------------------------------------")
print("----------------------------------------------------------------------------------------------")
voting_data = pd.read_csv("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/Voting/voting_imputed.csv")
voting_data['class'] = voting_data['class'].astype('category')

### Split into training and testing sets (80/20 split) ###
train_data, test_data = train_test_split(voting_data, test_size=0.2, random_state=42)
X_train = train_data.drop('class', axis=1); y_train = train_data['class']
X_test = test_data.drop('class', axis=1); y_test = test_data['class']
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

### We first tried to do the classification on the datset without scaling, without crossvalidation and k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
## Evaluate performance ##
# Confusion Matrix
conf_matrix1 = confusion_matrix(y_test, y_pred1)
print("Voting - Without Scaling: ")
print(conf_matrix1)
disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix1)
disp1.plot(cmap='Blues')
plt.title('Confusion Matrix - No Scaling')
plt.savefig(f"plots/voting_cm_withoutScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1, pos_label=1))
print("Recall:", recall_score(y_test, y_pred1, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred1, pos_label=1))
# AUC-ROC, AUC-PR, MCC
roc_auc = roc_auc_score(y_test, y_pred1)
fpr, tpr, _ = roc_curve(y_test, y_pred1)
print("AUC-ROC:", roc_auc)
precision, recall, _ = precision_recall_curve(y_test, y_pred1)
auc_pr = average_precision_score(y_test, y_pred1)
print("AUC-PR:", auc_pr)
mcc = matthews_corrcoef(y_test, y_pred1)
print("MCC:", mcc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}'); plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (No Scaling)'); plt.legend(loc='lower right')
plt.savefig(f"plots/voting_ROC_withoutScaling.png", bbox_inches="tight")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (No Scaling)'); plt.legend(loc='lower left')
plt.savefig(f"plots/voting_PR_withoutScaling.png", bbox_inches="tight")

### Since knn-algorithms are very sensitive to scaling, we added scaling to see how much the resuts would change.
### We used the StandardScaler, we sill don't use cross validation and have k = 3
print("----------------------------------------------------------------------------------------------")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred2 = knn.predict(X_test_scaled)
## Evaluate performance ##
# Confusion matrix
conf_matrix2 = confusion_matrix(y_test, y_pred2)
print("Voting - With Scaling: ")
print(conf_matrix2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix2)
disp2.plot(cmap='Blues')
plt.savefig(f"plots/machine_cm_withScaling.png", bbox_inches="tight")
# Accuracy, Precision, Recall, F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2, pos_label=1))
print("Recall:", recall_score(y_test, y_pred2, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred2, pos_label=1))
# AUC-ROC, AUC-PR, MCC
roc_auc = roc_auc_score(y_test, y_pred2)
fpr, tpr, _ = roc_curve(y_test, y_pred2)
print("AUC-ROC:", roc_auc)
precision, recall, _ = precision_recall_curve(y_test, y_pred2)
auc_pr = average_precision_score(y_test, y_pred2)
print("AUC-PR:", auc_pr)
mcc = matthews_corrcoef(y_test, y_pred2)
print("MCC:", mcc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}'); plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (with Scaling)'); plt.legend(loc='lower right')
plt.savefig(f"plots/voting_ROC_withScaling.png", bbox_inches="tight")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (with Scaling)'); plt.legend(loc='lower left')
plt.savefig(f"plots/voting_PR_withScaling.png", bbox_inches="tight")

### We added a 10-fold cross validation to see if we would get an even more accurate result
print("----------------------------------------------------------------------------------------------")
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=10)
print("Voting - With Cross Validation: ")
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(knn, X_train_scaled, y_train, cv=10, scoring=scoring)
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall: {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")

### Hyperparameter tuning (finding optimal k)
print("----------------------------------------------------------------------------------------------")
k_values = range(1, 30)
accuracy_scores = []
#for k in k_values:
    #knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(X_train_scaled, y_train)
    #y_pred = knn.predict(X_test_scaled)
    #accuracy_scores.append(accuracy_score(y_test, y_pred))
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')
    accuracy_scores.append(cv_scores.mean())
# Create a new figure for the accuracy plot
plt.figure(figsize=(8, 6))  # Optional: to set the figure size
plt.plot(k_values, accuracy_scores)  # Line plot with markers
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)  # Optional: to add grid lines to the plot
plt.savefig(f"plots/voting_optimalK.png", bbox_inches="tight")
optimal_k = k_values[np.argmax(accuracy_scores)]
print("Optimal k:", optimal_k)

### Final model with optimal k - Crossvalidation
print("----------------------------------------------------------------------------------------------")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cv_results = cross_validate(final_knn, X_train_scaled, y_train, cv=10,
                            scoring=['accuracy', 'precision', 'recall', 'f1'],
                            return_train_score=False)
print(f"Machine Failure - With optimal k={optimal_k} (Cross-validation): ")
print(f"Average Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Average Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Average Recall: {cv_results['test_recall'].mean():.4f}")
print(f"Average F1 Score: {cv_results['test_f1'].mean():.4f}")

### Final model with optimal k
print("----------------------------------------------------------------------------------------------")
final_knn.fit(X_train_scaled, y_train)
final_predictions = final_knn.predict(X_test_scaled)
plt.title(f'Final Confusion Matrix (k={optimal_k})')
final_conf_matrix = confusion_matrix(y_test, final_predictions)
print(final_conf_matrix)
disp_final = ConfusionMatrixDisplay(confusion_matrix=final_conf_matrix)
disp_final.plot(cmap='Blues')
plt.savefig(f"plots/voting_cm_finalModel.png", bbox_inches="tight")
print("Voting - optimal k - Final model: ")
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
print("Final Model Precision:", precision_score(y_test, final_predictions, pos_label=1))
print("Final Model Recall:", recall_score(y_test, final_predictions, pos_label=1))
print("Final Model F1 Score:", f1_score(y_test, final_predictions, pos_label=1))
plt.savefig(f"plots/voting_finalModel.png", bbox_inches="tight")
# AUC-ROC, AUC-PR, MCC
roc_auc = roc_auc_score(y_test, final_predictions)
fpr, tpr, _ = roc_curve(y_test, final_predictions)
print("AUC-ROC:", roc_auc)
precision, recall, _ = precision_recall_curve(y_test, final_predictions)
auc_pr = average_precision_score(y_test, final_predictions)
print("AUC-PR:", auc_pr)
mcc = matthews_corrcoef(y_test, final_predictions)
print("MCC:", mcc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}'); plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (No Scaling)'); plt.legend(loc='lower right')
plt.savefig(f"plots/voting_ROC_finalModel.png", bbox_inches="tight")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (No Scaling)'); plt.legend(loc='lower left')
plt.savefig(f"plots/voting_PR_finalModel.png", bbox_inches="tight")
log_file.close()


