import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay, make_scorer, roc_auc_score, roc_curve, precision_recall_curve, \
    average_precision_score,matthews_corrcoef
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import sys
import time

def holdout_model(model, X_train, y_train, X_test, y_test, title, save_path=None):
    """Fit model and print metrics."""
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='Blues')
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.title(title)
    plt.close()

    if(y_test.nunique() == 2):
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, pos_label=1))
        print("Recall:", recall_score(y_test, y_pred, pos_label=1))
        print("F1 Score:", f1_score(y_test, y_pred, pos_label=1))
        print("Time: ", end - start)
    else:
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision (weighted):", precision_score(y_test, y_pred, average='weighted', zero_division=1))
        print("Recall (weighted):", recall_score(y_test, y_pred, average='weighted', zero_division=1))
        print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted', zero_division=1))
        print("Time: ", end-start)

def cross_validate_model(model, X, y, cv):
    """Perform cross-validation and print metrics."""
    if y_test.nunique() != 2:
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=1),
            'recall': make_scorer(recall_score, average='weighted', zero_division=1),
            'f1': make_scorer(f1_score, average='weighted', zero_division=1)
        }
        start = time.time()
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        end = time.time()
        print(f"Accuracy: {results['test_accuracy'].mean():.4f}")
        print(f"Precision: {results['test_precision'].mean():.4f}")
        print(f"Recall: {results['test_recall'].mean():.4f}")
        print(f"F1 Score: {results['test_f1'].mean():.4f}")
        print("Time: ", end-start)
        return results
    else:
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        start = time.time()
        cv_results = cross_validate(knn, X_train_scaled, y_train, cv=cv, scoring=scoring)
        end = time.time()
        print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
        print(f"Precision: {cv_results['test_precision'].mean():.4f}")
        print(f"Recall: {cv_results['test_recall'].mean():.4f}")
        print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")
        print("Time:", end - start)
        return cv_results

def scale_data(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def find_optimal_k(min, max, X_train, y_train, cv, save_path=None):
    """Find the optimal k and plot accuracy against k values."""
    k_values = range(min, max)
    accuracy_scores = []
    start_k = time.time()
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=1),
            'recall': make_scorer(recall_score, average='weighted', zero_division=1),
            'f1': make_scorer(f1_score, average='weighted', zero_division=1)
        }
        cv_results = cross_validate(knn, X_train, y_train, cv=cv, scoring=scoring)
        accuracy_scores.append(cv_results['test_accuracy'].mean())
    end_k = time.time()
    optimal_k = k_values[np.argmax(accuracy_scores)]
    print("Optimal k:", optimal_k)
    print("Time: ", {end_k - start_k})
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.title('Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return optimal_k

os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1")
os.makedirs("plots", exist_ok=True)
log_file = open("knnClassification_performanceMeasures.txt", "w")
sys.stdout = log_file

print("K NEAREST NEIGHBOR CLASSIFICATION")
print("")
print("")


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

###  Baseline - KNN without scaling
print("Baseline - KNN without Scaling")
knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(knn, X_train, y_train, X_test, y_test, "Baseline_No_Scaling", save_path="plots/amazon_cm_withoutScaling.png")

### KNN with scaling
print("----------------------------------------------------------------------------------------------")
print("KNN with Scaling")
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
scaled_knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(scaled_knn, X_train_scaled, y_train, X_test_scaled, y_test, "With_Scaling", save_path="plots/amazon_cm_withScaling.png")

### KNN with cross-validation
print("----------------------------------------------------------------------------------------------")
print("KNN with Cross Validation (10-fold)")
cross_validate_model(scaled_knn, X_train, y_train, cv=5)

### Hyperparameter tuning: find optimal k
print("----------------------------------------------------------------------------------------------")
print("Hyperparameter Tuning: find optimal k")
optimal_k = find_optimal_k(1, 30, X_train_scaled, y_train, cv=5, save_path="plots/amazon_optimalK.png")

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
cross_validate_model(knn, X_train, y_train, cv=5)

# Oversampling using RandomOverSampler
print("----------------------------------------------------------------------------------------------")
print("Oversampling using RandomOverSampler")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
#print(f"Original class distribution: {y_train.value_counts()}")
#print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts()}")

# Model with oversampled data
print("----------------------------------------------------------------------------------------------")
print("KNN with Oversampling")
oversampled_knn = KNeighborsClassifier(n_neighbors=optimal_k)
holdout_model(oversampled_knn, X_train_resampled, y_train_resampled, X_test, y_test, "Oversampled", save_path="plots/amazon_cm_oversampled.png")

# Cross-validation with oversampled data
print("----------------------------------------------------------------------------------------------")
print("Oversampling - Cross Validation")
cross_validate_model(oversampled_knn, X_train_resampled, y_train_resampled, cv=5)

print("----------------------------------------------------------------------------------------------")
print("Final model")
holdout_model(oversampled_knn, X_train_resampled, y_train_resampled, X_test, y_test, "Oversampled", save_path="plots/amazon_cm_oversampled.png")



########## ROAD TRAFFIC ACCIDENT DATASET ##########
print("----------------------------------------------------------------------------------------------")
print("----------------------------------- ROAD TRAFFIC ACCIDENT ------------------------------------")
print("----------------------------------------------------------------------------------------------")
rta_data = pd.read_csv("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/RTA/RTA_cleaned.csv")
rta_data['Accident_severity'] = rta_data['Accident_severity'].astype('category')

### Split into training and testing sets (80/20 split)
train_data, test_data = train_test_split(rta_data, test_size=0.2, random_state=42)
X_train = train_data.drop('Accident_severity', axis=1); y_train = train_data['Accident_severity']
X_test = test_data.drop('Accident_severity', axis=1); y_test = test_data['Accident_severity']
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Baseline - KNN without scaling
print("Baseline - KNN without Scaling")
baseline_knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(baseline_knn, X_train, y_train, X_test, y_test, "Baseline_No_Scaling", save_path="plots/rta_cm_withoutScaling.png")

# KNN with scaling
print("----------------------------------------------------------------------------------------------")
print("KNN with Scaling")
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
scaled_knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(scaled_knn, X_train_scaled, y_train, X_test_scaled, y_test, "With_Scaling", save_path="plots/rta_cm_withScaling.png")

# KNN with cross-validation
print("----------------------------------------------------------------------------------------------")
print("KNN with Cross Validation (10-fold)")
cross_validate_model(scaled_knn, X_train_scaled, y_train, cv=10)

# Hyperparameter tuning: find optimal k
print("----------------------------------------------------------------------------------------------")
print("Hyperparameter Tuning: find optimal k")
optimal_k = find_optimal_k(1, 30, X_train_scaled, y_train, cv=10, save_path="plots/rta_optimalK.png")
knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(knn, X_train_scaled, y_train, cv=10)

# Oversampling using RandomOverSampler
print("----------------------------------------------------------------------------------------------")
print("Oversampling using RandomOverSampler")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
print(f"Original class distribution: {y_train.value_counts()}")
print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts()}")

# Model with oversampled data
print("----------------------------------------------------------------------------------------------")
print("KNN with Oversampling")
oversampled_knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(oversampled_knn, X_train_resampled, y_train_resampled, X_test_scaled, y_test, "Oversampled", save_path="plots/rta_cm_oversampled.png")

# Cross-validation with oversampled data
print("----------------------------------------------------------------------------------------------")
print("Oversampling - Cross Validation")
cross_validate_model(oversampled_knn, X_train_resampled, y_train_resampled, cv=10)

# Find optimal k with oversampled data
print("----------------------------------------------------------------------------------------------")
print("Find optimal k with oversampled data")
optimal_k = find_optimal_k(1, 30, X_train_resampled, y_train_resampled, cv=10, save_path="plots/rta_optimalK_oversampled.png")

### Final model with optimal k - Crossvalidation
print("----------------------------------------------------------------------------------------------")
("Final model Crossvalidation")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(final_knn, X_train_resampled, y_train_resampled, cv=10)

### Final model with optimal k
print("----------------------------------------------------------------------------------------------")
print("Final Model")
final_knn.fit(X_train_resampled, y_train_resampled)
holdout_model(final_knn, X_train_resampled, y_train_resampled, X_test_scaled, y_test, "Final Model", save_path="plots/rta_cm_oversampled.png")



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
print("Machine Failure - Without Scaling:")
knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(knn, X_train, y_train, X_test, y_test, title="Confusion Matrix - No Scaling")

# Perform KNN classification with scaling
print("----------------------------------------------------------------------------------------------")
print("Machine Failure - With Scaling:")
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, title="Confusion Matrix - With Scaling")

# Perform 10-fold cross-validation with scaling and k=3
print("----------------------------------------------------------------------------------------------")
print("Machine Failure - With Cross Validation:")
knn = KNeighborsClassifier(n_neighbors=3)
cross_validate_model(knn, X_train_scaled, y_train, cv=10)

# Hyperparameter tuning (finding optimal k)
print("----------------------------------------------------------------------------------------------")
print("Finding optimal k:")
optimal_k = find_optimal_k(1, 30, X_train_scaled, y_train, cv = 10, save_path="plots/machine_optimalK.png")

# Final model with optimal k using cross-validation
print("----------------------------------------------------------------------------------------------")
print(f"Machine Failure - With optimal k={optimal_k} (Cross-validation):")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(final_knn, X_train_scaled, y_train, cv=10)

# Final model with optimal k
print("----------------------------------------------------------------------------------------------")
print(f"Machine Failure - Final Model with k={optimal_k}:")
holdout_model(final_knn, X_train_scaled, y_train, X_test_scaled, y_test, title=f"Final Confusion Matrix (k={optimal_k})", save_path="plots/machine_finalModel.png")


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
print("Voting - Without Scaling:")
knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(knn, X_train, y_train, X_test, y_test, title="Confusion Matrix - No Scaling", save_path="plots/voting_cm_withoutScaling.png")

# Perform KNN classification with scaling
print("----------------------------------------------------------------------------------------------")
print("Voting - With Scaling:")
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
holdout_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, title="Confusion Matrix - With Scaling", save_path="plots/voting_cm_withScaling.png")

# Perform 10-fold cross-validation with scaling and k=3
print("----------------------------------------------------------------------------------------------")
print("Voting - With Cross Validation:")
cross_validate_model(knn, X_train, y_train, cv=10)

# Hyperparameter tuning (finding optimal k)
print("----------------------------------------------------------------------------------------------")
print("Finding optimal k:")
optimal_k = find_optimal_k(1, 30, X_train, y_train, cv=10, save_path="plots/voting_optimalK.png")

# Final model with optimal k using cross-validation
print("----------------------------------------------------------------------------------------------")
print(f"Voting - With optimal k={optimal_k} (Cross-validation):")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(final_knn, X_train, y_train, cv=10)

# Final model with optimal k
print("----------------------------------------------------------------------------------------------")
print(f"Voting - Final Model with k={optimal_k}:")
holdout_model(final_knn, X_train, y_train, X_test, y_test, title=f"Final Confusion Matrix (k={optimal_k})", save_path="plots/voting_finalModel.png")

log_file.close()