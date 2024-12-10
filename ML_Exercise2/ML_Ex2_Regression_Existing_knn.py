import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, mean_squared_error
import sys
import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def rmse(y_true, y_pred):
    return (np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mre(y_true, y_pred):
    return (np.mean(np.abs((y_true - y_pred) / y_true)))
def correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
def smape(y_true, y_pred):
    return (100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))

def holdout_model(model, X_train, y_train, X_test, y_test, title, save_path=None):
    start_fit = time.time()
    model.fit(X_train, y_train)
    end_fit = time.time()
    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()

    rmse_val = rmse(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    smape_val = smape(y_test, y_pred)
    mre_val = mre(y_test, y_pred)
    corr_val = correlation(y_test, y_pred)
    print("Fitting time: ", end_fit-start_fit)
    print("Prediction time: ", end_pred - start_pred)
    print("RMSE: ", rmse_val)
    print("R squared: ", r2_val)
    print("Smape: ", smape_val)
    print("MRE: ", mre_val)
    print("Correlation: ", corr_val)

    if save_path:
        plt.scatter(y_test, y_pred, color='red')
        plt.title(title)
        plt.xlabel('true')
        plt.ylabel('predicted')
        #plt.show()
        plt.savefig(save_path, bbox_inches="tight")

    return rmse_val, smape_val, r2_val, corr_val, mre_val

def cross_validate_model(model, X, y, cv, printer = True):
    scoring = {
        'rmse': make_scorer(rmse, greater_is_better=False),
        'r2': make_scorer(r2),
        'smape': make_scorer(smape, greater_is_better=False),
        'mre': make_scorer(mre, greater_is_better=False),
        'corr': make_scorer(correlation)
    }

    start = time.time()
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    end = time.time()

    if printer == True:
        print("Crossvalidation Time:", end-start)
        print("RMSE:", -np.mean(cv_results['test_rmse']))

        print("SMAPE:", -np.mean(cv_results['test_smape']))
        print("R²:", np.mean(cv_results['test_r2']))
        print("Correlation:", np.mean(cv_results['test_corr']))
        print("MRE:", -np.mean(cv_results['test_mre']))

    return cv_results

def find_optimal_k(min, max, step, X_train, y_train, cv, save_path=None):
    k_values = range(min, max, step)
    r2_scores = []
    start_k = time.time()
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        cv_results = cross_validate_model(knn, X_train, y_train, cv=cv, printer=False)
        r2_scores.append(cv_results['test_r2'].mean())
    end_k = time.time()
    optimal_k = k_values[np.argmax(r2_scores)]
    print("Optimal k:", optimal_k)
    print("Time: ", {end_k - start_k})
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, r2_scores, marker='o')
    plt.title('RMSE vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('RMSE')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #plt.show()
    return r2_scores

def find_optimal_weight(k, X_train, y_train, cv):
    """Find the optimal weight"""
    r2_scores = []
    start_k = time.time()
    weights = ['uniform', 'distance']
    for weight in weights:
        knn = KNeighborsRegressor(n_neighbors=k, weights=weight)
        cv_results = cross_validate_model(knn, X_train, y_train, cv=cv, printer=False)
        r2_scores.append(cv_results['test_r2'].mean())
    end_k = time.time()
    optimal_weight = weights[np.argmax(r2_scores)]
    print("Optimal weight:", optimal_weight)
    print("Time: ", {end_k - start_k})
    return r2_scores

def find_optimal_distance(k, X_train, y_train, cv):
    r2_scores = []
    start_k = time.time()
    distances = ['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine']
    for distance in distances:
        knn = KNeighborsRegressor(n_neighbors=k, metric=distance)
        cv_results = cross_validate_model(knn, X_train, y_train, cv=cv, printer=False)
        r2_scores.append(cv_results['test_r2'].mean())
    end_k = time.time()
    optimal_distance = distances[np.argmax(r2_scores)]
    print("Optimal distance:", optimal_distance)
    print("Time: ", {end_k - start_k})
    return r2_scores

os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise 2")
os.makedirs("plot", exist_ok=True)

log_file = open("ML_Ex2_Existing_KNN_performanceMeasures.txt", "w")
sys.stdout = log_file

print("------------------------------------")
print("MPG Dataset")
MPG_train = pd.read_csv("MPG_train.csv")
MPG_test = pd.read_csv("MPG_test.csv")
MPG_train = pd.get_dummies(MPG_train)
MPG_test = pd.get_dummies(MPG_test)
X_train = MPG_train.drop('mpg', axis=1); y_train = MPG_train['mpg']
X_test = MPG_test.drop('mpg', axis=1); y_test = MPG_test['mpg']
X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_train = X_train.fillna(False).astype(int)
X_test = X_test.fillna(False).astype(int)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

### WITHOUT CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Without CV, Without Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
holdout_model(knn, X_train, y_train, X_test, y_test, title="KNN - No cv, no scaling", save_path="plots/mgp_noScaling.png")

### WITHOUT CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Without CV, With Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
holdout_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, title="KNN - No cv, scaling", save_path="plots/mgp_Scaling.png")

### WITH CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("MPG - With CV, Without Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
cross_validate_model(knn, X_train, y_train, 5)

### WITH CV, WITH SCALING ###
print("----------------------------------------------------------------------------------------------")
print("MPG - With CV, With Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
cross_validate_model(knn, X_train_scaled, y_train, 5)

### PARAMETER K ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Parameter k:")
find_optimal_k(1, 50, 1, X_train_scaled, y_train, 5, save_path=None)

### PARAMETER WEIGHT ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Parameter weight:")
weight = find_optimal_weight(6, X_train_scaled, y_train, 5)
print(weight)

### PARAMETER DISTANCE ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Parameter weight:")
distance = find_optimal_distance(6, X_train_scaled, y_train, 5)
print(distance)

########################################################################################################################
########################################################################################################################
print("")
print("")
print("")
print("------------------------------------")
print("CT Dataset")
CT_train = pd.read_csv("CT_train.csv")
CT_test = pd.read_csv("CT_test.csv")
X_train = CT_train.drop('critical_temp', axis=1); y_train = CT_train['critical_temp']
X_test = CT_test.drop('critical_temp', axis=1); y_test = CT_test['critical_temp']
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

### WITHOUT CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("CT - Without CV, Without Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
holdout_model(knn, X_train, y_train, X_test, y_test, title="KNN - No cv, no scaling", save_path="plots/ct_noScaling.png")

### WITHOUT CV, WITH SCALING ###
print("----------------------------------------------------------------------------------------------")
print("CT - Without CV, With Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
holdout_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, title="KNN - No cv, scaling", save_path="plots/ct_Scaling.png")

### WITH CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("CT - With CV, Without Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
cross_validate_model(knn, X_train, y_train, 5)

### WITH CV, WITH SCALING ###
print("----------------------------------------------------------------------------------------------")
print("CT - With CV, With Scaling:")
knn = KNeighborsRegressor(n_neighbors=5)
cross_validate_model(knn, X_train_scaled, y_train, 5)

### PARAMETER K ###
print("----------------------------------------------------------------------------------------------")
print("CT - Parameter k:")
find_optimal_k(1, 25, 1, X_train_scaled, y_train, 5, save_path=None)

### PARAMETER WEIGHT ###
print("----------------------------------------------------------------------------------------------")
print("CT - Parameter weight:")
weight = find_optimal_weight(6, X_train_scaled, y_train, 5)
print(weight)

### PARAMETER DISTANCE ###
print("----------------------------------------------------------------------------------------------")
print("CT - Parameter weight:")
distance = find_optimal_distance(6, X_train_scaled, y_train, 5)
print(distance)

log_file.close()