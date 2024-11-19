import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, make_scorer
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import sys
import time
from sklearn.model_selection import GridSearchCV

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

def cross_validate_model(model, X, y, cv, parameters = True):
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
        if parameters == True:
            print(f"Accuracy: {results['test_accuracy'].mean():.4f}")
            print(f"Precision: {results['test_precision'].mean():.4f}")
            print(f"Recall: {results['test_recall'].mean():.4f}")
            print(f"F1 Score: {results['test_f1'].mean():.4f}")
            print("Time: ", end-start)
        return results
    else:
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        start = time.time()
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        end = time.time()
        if parameters == True:
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
        cv_results = cross_validate_model(knn, X_train, y_train, cv=cv, parameters=False)
        accuracy_scores.append(cv_results['test_accuracy'].mean())
    end_k = time.time()
    optimal_k = k_values[np.argmax(accuracy_scores)]
    print("Optimal k:", optimal_k)
    print("Time: ", {end_k - start_k})
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.title('Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return optimal_k

def find_optimal_weight(k, X_train, y_train, cv):
    """Find the optimal weight"""
    accuracy_scores = []
    start_k = time.time()
    weights = ['uniform', 'distance']
    for weight in weights:
        knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
        cv_results = cross_validate_model(knn, X_train, y_train, cv=cv, parameters=False)
        accuracy_scores.append(cv_results['test_accuracy'].mean())
    end_k = time.time()
    optimal_weight = weights[np.argmax(accuracy_scores)]
    print("Optimal weight:", optimal_weight)
    print("Time: ", {end_k - start_k})
    return optimal_weight


def find_optimal_metric(k, X_train, y_train, cv):
    """Find the optimal distance metric"""
    accuracy_scores = []
    start_k = time.time()
    metrics = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        cv_results = cross_validate_model(knn, X_train, y_train, cv=cv, parameters=False)
        accuracy_scores.append(cv_results['test_accuracy'].mean())
    end_k = time.time()
    optimal_metric = metrics[np.argmax(accuracy_scores)]
    print("Optimal metric:", optimal_metric)
    print("Time: ", {end_k - start_k})
    return optimal_metric

def find_optimal_algorithm(k, X_train, y_train, cv):
    """Find the optimal algorithm"""
    accuracy_scores = []
    start_k = time.time()
    algorithms = ['ball_tree', 'kd_tree', 'brute']
    for algorithm in algorithms:
        knn = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm)
        cv_results = cross_validate_model(knn, X_train, y_train, cv=cv, parameters=False)
        accuracy_scores.append(cv_results['test_accuracy'].mean())
    end_k = time.time()
    optimal_algorithm = algorithms[np.argmax(accuracy_scores)]
    print("Optimal algorithm:", optimal_algorithm)
    print("Time: ", {end_k - start_k})
    return optimal_algorithm

os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1")
os.makedirs("plots", exist_ok=True)
log_file = open("knnClassification_performanceMeasures.txt", "w")
sys.stdout = log_file

print("K NEAREST NEIGHBOR CLASSIFICATION")
print("")
print("")


##################################### CONGRESSIONAL VOTING DATSET #####################################
print("----------------------------------------------------------------------------------------------")
print("----------------------------------- CONGRESSIONAL VOTING -------------------------------------")
print("----------------------------------------------------------------------------------------------")

voting_data = pd.read_csv("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/Voting/voting_imputed.csv")
voting_data['class'] = voting_data['class'].astype('category')

### Split into training and testing sets (80/20 split) ###
train_data, test_data = train_test_split(voting_data, test_size=0.2, random_state=42)
X_train = train_data.drop('class', axis=1); y_train = train_data['class']
X_test = test_data.drop('class', axis=1); y_test = test_data['class']

### We first tried to do the classification on the datset without scaling, without crossvalidation and k=3
print("Voting - Without Scaling:")
knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(knn, X_train, y_train, X_test, y_test, title="Confusion Matrix - No Scaling", save_path="plots/voting_cm_withoutScaling.png")

"""
# Perform KNN classification with scaling and k=3
print("----------------------------------------------------------------------------------------------")
print("Voting - With Scaling:")
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
holdout_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, title="Confusion Matrix - With Scaling", save_path="plots/voting_cm_withScaling.png")
"""

# Perform 10-fold cross-validation
print("----------------------------------------------------------------------------------------------")
print("Voting - With Cross Validation:")
cross_validate_model(knn, X_train, y_train, cv=10)

# Finding optimal k
print("----------------------------------------------------------------------------------------------")
print("Finding optimal k:")
optimal_k = find_optimal_k(1, 30, X_train, y_train, cv=10, save_path="plots/voting_optimalK.png")
print(f"Voting - With optimal k={optimal_k} (Cross-validation):")
k_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(k_knn, X_train, y_train, cv=10)

# Finding optimal weight
print("----------------------------------------------------------------------------------------------")
print("Finding optimal weight:")
optimal_weight = find_optimal_weight(optimal_k, X_train, y_train, cv=10)
weight_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights=optimal_weight)
cross_validate_model(weight_knn, X_train, y_train, cv=10)

# Finding optimal distance metric
print("----------------------------------------------------------------------------------------------")
print("Finding optimal metric:")
optimal_metric = find_optimal_metric(optimal_k, X_train, y_train, cv=10)
metric_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=optimal_metric)
cross_validate_model(metric_knn, X_train, y_train, cv=10)

# Finding optimal algorithm
print("----------------------------------------------------------------------------------------------")
print("Finding optimal algorithm:")
optimal_algorithm = find_optimal_algorithm(optimal_k, X_train, y_train, cv=10)
algorithm_knn = KNeighborsClassifier(n_neighbors=optimal_k, algorithm=optimal_algorithm)
cross_validate_model(algorithm_knn, X_train, y_train, cv=10)

# Hyperparameter tuning (finding the best parameter combinations)
print("----------------------------------------------------------------------------------------------")
print("Hyperparameter tuning (finding best parameter combinations):")
param_grid = {
    'n_neighbors': range(1, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}

# Define scoring dictionary
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': make_scorer(precision_score, average='macro', zero_division=1),
    'recall_weighted': make_scorer(recall_score, average='macro', zero_division=1),
    'f1_weighted': make_scorer(f1_score, average='macro', zero_division=1)
}
start = time.time()
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation precision: {grid_search.cv_results_['mean_test_precision_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation recall: {grid_search.cv_results_['mean_test_recall_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation F1 Score: {grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]:.4f}")
print("Time for Hypertuning: ", time.time()-start)


####################################### MACHINE FAILURE DATASET #######################################
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
holdout_model(knn, X_train, y_train, X_test, y_test, title="Confusion Matrix - No Scaling", save_path="plots/machine_cm_withoutScaling.png")

# Perform KNN classification with scaling and k=3
print("----------------------------------------------------------------------------------------------")
print("Machine Failure - With Scaling:")
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, title="Confusion Matrix - With Scaling", save_path="plots/machine_cm_withScaling.png")

# Perform 10-fold cross-validation without scaling and k=3
print("----------------------------------------------------------------------------------------------")
print("Machine Failure - With Cross Validation, unscaled:")
knn = KNeighborsClassifier(n_neighbors=3)
cross_validate_model(knn, X_train, y_train, cv=10)


# Perform 10-fold cross-validation with scaling and k=3
print("----------------------------------------------------------------------------------------------")
print("Machine Failure - With Cross Validation, scaled:")
knn = KNeighborsClassifier(n_neighbors=3)
cross_validate_model(knn, X_train_scaled, y_train, cv=10)

# Finding optimal k
print("----------------------------------------------------------------------------------------------")
print("Finding optimal k:")
optimal_k = find_optimal_k(1, 30, X_train_scaled, y_train, cv=10, save_path="plots/voting_optimalK.png")
print(f"Voting - With optimal k={optimal_k} (Cross-validation):")
k_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(k_knn, X_train_scaled, y_train, cv=10)

# Finding optimal weight
print("----------------------------------------------------------------------------------------------")
print("Finding optimal weight:")
optimal_weight = find_optimal_weight(optimal_k, X_train_scaled, y_train, cv=10)
weight_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights=optimal_weight)
cross_validate_model(weight_knn, X_train_scaled, y_train, cv=10)

# Finding optimal distance metric
print("----------------------------------------------------------------------------------------------")
print("Finding optimal metric:")
optimal_metric = find_optimal_metric(optimal_k, X_train_scaled, y_train, cv=10)
metric_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=optimal_metric)
cross_validate_model(metric_knn, X_train_scaled, y_train, cv=10)

# Finding optimal algorithm
print("----------------------------------------------------------------------------------------------")
print("Finding optimal algorithm:")
optimal_algorithm = find_optimal_algorithm(optimal_k, X_train_scaled, y_train, cv=10)
algorithm_knn = KNeighborsClassifier(n_neighbors=optimal_k, algorithm=optimal_algorithm)
cross_validate_model(algorithm_knn, X_train_scaled, y_train, cv=10)

# Hyperparameter tuning (finding the best parameter combinations)
print("----------------------------------------------------------------------------------------------")
print("Hyperparameter tuning (finding best parameter combinations):")
param_grid = {
    'n_neighbors': range(1, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=1)
}
start = time.time()
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation precision: {grid_search.cv_results_['mean_test_precision_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation recall: {grid_search.cv_results_['mean_test_recall_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation F1 Score: {grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]:.4f}")
print("Time for Hypertuning: ", time.time()-start)


####################################### ROAD TRAFFIC ACCIDENT DATASET #######################################
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
print("KNN with Cross Validation (10-fold), unscaled")
cross_validate_model(scaled_knn, X_train, y_train, cv=10)

# KNN with cross-validation
print("----------------------------------------------------------------------------------------------")
print("KNN with Cross Validation (10-fold), scaled")
cross_validate_model(scaled_knn, X_train_scaled, y_train, cv=10)


# Finding optimal k
optimal_k = find_optimal_k(1, 30, X_train, y_train, cv=10, save_path="plots/voting_optimalK.png")

# Oversampling using RandomOverSampler
print("----------------------------------------------------------------------------------------------")
print("Oversampling using RandomOverSampler")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
#print(f"Original class distribution: {y_train.value_counts()}")
#print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts()}")
oversampled_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(oversampled_knn, X_train_resampled, y_train_resampled, cv=10)

# Finding optimal k
print("----------------------------------------------------------------------------------------------")
print("# Finding optimal k")
optimal_k = find_optimal_k(1, 15, X_train_resampled, y_train_resampled, cv=10, save_path="plots/rta_optimalK_oversampled.png")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(final_knn, X_train_resampled, y_train_resampled, cv=10)

# Finding optimal weight
print("----------------------------------------------------------------------------------------------")
print("Finding optimal weight:")
optimal_weight = find_optimal_weight(optimal_k, X_train_resampled, y_train_resampled, cv=10)
weight_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights=optimal_weight)
cross_validate_model(weight_knn, X_train_resampled, y_train_resampled, cv=10)

# Finding optimal distance metric
print("----------------------------------------------------------------------------------------------")
print("Finding optimal metric:")
optimal_metric = find_optimal_metric(optimal_k, X_train_resampled, y_train_resampled, cv=10)
metric_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=optimal_metric)
cross_validate_model(metric_knn, X_train_resampled, y_train_resampled, cv=10)

# Finding optimal algorithm
print("----------------------------------------------------------------------------------------------")
print("Finding optimal algorithm:")
optimal_algorithm = find_optimal_algorithm(optimal_k, X_train_resampled, y_train_resampled, cv=10)
algorithm_knn = KNeighborsClassifier(n_neighbors=optimal_k, algorithm=optimal_algorithm)
cross_validate_model(algorithm_knn, X_train_resampled, y_train_resampled, cv=10)


# Hyperparameter tuning (finding the best parameter combinations)
print("----------------------------------------------------------------------------------------------")
print("Hyperparameter tuning (finding best parameter combinations):")
param_grid = {
    'n_neighbors': range(1,25),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'chebyshev', 'euclidean', 'manhattan']
}
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=1)
}
start = time.time()
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring=scoring, refit='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled,)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation precision: {grid_search.cv_results_['mean_test_precision_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation recall: {grid_search.cv_results_['mean_test_recall_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation F1 Score: {grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]:.4f}")
print("Time for Hypertuning: ", time.time()-start)


######################################## AMAZON REWIEV DATASET ########################################
print("----------------------------------------------------------------------------------------------")
print("--------------------------------------- AMAZON REWIEV ----------------------------------------")
print("----------------------------------------------------------------------------------------------")
review_data = pd.read_csv("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise1/Reviews/amazon_review_ID.shuf.lrn.csv")
review_data = review_data.drop(columns='ID')
review_data['Class'] = review_data['Class'].astype('category')

# Split into training and testing sets (80/20 split)
train_data, test_data = train_test_split(review_data, test_size=0.2, random_state=42)
X_train = train_data.drop('Class', axis=1); y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1); y_test = test_data['Class']

### KNN without scaling - holdout
print("Review - Without Scaling:")
knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(knn, X_train, y_train, X_test, y_test, "Baseline_No_Scaling", save_path="plots/amazon_cm_withoutScaling.png")

### KNN with scaling - holdout
print("----------------------------------------------------------------------------------------------")
print("KNN with Scaling")
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
scaled_knn = KNeighborsClassifier(n_neighbors=3)
holdout_model(scaled_knn, X_train_scaled, y_train, X_test_scaled, y_test, "With_Scaling", save_path="plots/amazon_cm_withScaling.png")
#optimal_k = 1

### Perform 5-fold cross-validation
print("----------------------------------------------------------------------------------------------")
print("Machine Failure - With Cross Validation - unscaled data:")
cross_validate_model(knn, X_train, y_train, cv=5)

### Perform 5-fold cross-validation
print("----------------------------------------------------------------------------------------------")
print("Machine Failure - With Cross Validation - scaled data:")
cross_validate_model(knn, X_train_scaled, y_train, cv=5)

# Finding optimal k
print("----------------------------------------------------------------------------------------------")
print("Finding optimal k:")
optimal_k = find_optimal_k(1, 15, X_train, y_train, cv=5, save_path="plots/amazon_optimalK.png")
print("optimal k: ", optimal_k)

### Dimensionality Reduction
print("----------------------------------------------------------------------------------------------")
print("Dimensionality Reduction")
start = time.time()
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
print(time.time() - start)

### Oversampling using RandomOverSampler
print("----------------------------------------------------------------------------------------------")
print("Oversampling using RandomOverSampler")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
#print(f"Original class distribution: {y_train.value_counts()}")
#print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts()}")
oversampled_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(oversampled_knn, X_train_resampled, y_train_resampled, cv=5)

# Finding optimal k
print("----------------------------------------------------------------------------------------------")
print("Finding optimal k:")
optimal_k = find_optimal_k(1, 15, X_train_resampled, y_train_resampled, cv=5, save_path="plots/amazon_optimalK.png")
print(f"Voting - With optimal k={optimal_k} (Cross-validation):")
k_knn = KNeighborsClassifier(n_neighbors=optimal_k)
cross_validate_model(k_knn, X_train_resampled, y_train_resampled, cv=5)

# Finding optimal weight
print("----------------------------------------------------------------------------------------------")
print("Finding optimal weight:")
optimal_weight = find_optimal_weight(optimal_k, X_train_resampled, y_train_resampled, cv=5)
weight_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights=optimal_weight)
cross_validate_model(weight_knn, X_train_resampled, y_train_resampled, cv=5)

# Finding optimal distance metric
print("----------------------------------------------------------------------------------------------")
print("Finding optimal metric:")
optimal_metric = find_optimal_metric(optimal_k, X_train_resampled, y_train_resampled, cv=5)
metric_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=optimal_metric)
cross_validate_model(metric_knn, X_train_resampled, y_train_resampled, cv=5)

# Finding optimal algorithm
print("----------------------------------------------------------------------------------------------")
print("Finding optimal algorithm:")
optimal_algorithm = find_optimal_algorithm(optimal_k, X_train_resampled, y_train_resampled, cv=5)
algorithm_knn = KNeighborsClassifier(n_neighbors=optimal_k, algorithm=optimal_algorithm)
cross_validate_model(algorithm_knn, X_train_resampled, y_train_resampled, cv=5)

# Hyperparameter tuning (finding the best parameter combinations)
print("----------------------------------------------------------------------------------------------")
print("Hyperparameter tuning (finding best parameter combinations):")
param_grid = {
    'n_neighbors': range(1, 15),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=1)
}
start = time.time()
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation precision: {grid_search.cv_results_['mean_test_precision_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation recall: {grid_search.cv_results_['mean_test_recall_weighted'][grid_search.best_index_]:.4f}")
print(f"Best cross-validation F1 Score: {grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]:.4f}")
print("Time for Hypertuning: ", time.time()-start)

log_file.close()
