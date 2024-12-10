import os
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import pearsonr
import sys
import matplotlib.pyplot as plt

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

    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    smape_val = smape(y_test, y_pred)
    mre_val = mre(y_test, y_pred)
    corr_val = correlation(y_test, y_pred)
    print("Fitting time: ", end_fit-start_fit)
    print("Prediction time: ", end_pred - start_pred)
    print("MSE: ", mse_val)
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

    return mse_val, smape_val, r2_val, corr_val, mre_val

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
        print("SMAPE:", -np.mean(cv_results['test_smape']))
        print("RMSE:", -np.mean(cv_results['test_rmse']))
        print("R²:", np.mean(cv_results['test_r2']))
        print("Correlation:", np.mean(cv_results['test_corr']))
        print("MRE:", -np.mean(cv_results['test_mre']))

    return cv_results

def find_optimal_max_depth(max_depth, X_train, y_train, cv):
    r2_scores = []
    start_k = time.time()
    for depth in max_depth:
        rf = RandomForestRegressor(max_depth=depth, random_state=42, n_jobs=-1)
        cv_results = cross_validate_model(rf, X_train, y_train, cv=cv, printer=False)
        r2_scores.append(cv_results['test_r2'].mean())
    end_k = time.time()
    optimal_max_depth = max_depth[np.argmax(r2_scores)]
    print("Optimal maximum depth:", optimal_max_depth)
    print("Time: ", {end_k - start_k})
    return r2_scores

def find_optimal_n_estimator(n_estimator, X_train, y_train, cv):
    r2_scores = []
    start_k = time.time()
    for estimator in n_estimator:
        rf = RandomForestRegressor(n_estimators=estimator, random_state=42, n_jobs=-1)
        cv_results = cross_validate_model(rf, X_train, y_train, cv=cv, printer=False)
        r2_scores.append(cv_results['test_r2'].mean())
    end_k = time.time()
    optimal_n_estimator = n_estimator[np.argmax(r2_scores)]
    print("Optimal number of estimators:", optimal_n_estimator)
    print("Time: ", {end_k - start_k})
    return r2_scores

def find_optimal_min_samples_leaf(min_samples_leaf, X_train, y_train, cv):
    r2_scores = []
    start_k = time.time()
    for leaf in min_samples_leaf:
        rf = RandomForestRegressor(min_samples_leaf=leaf, random_state=42, n_jobs=-1)
        cv_results = cross_validate_model(rf, X_train, y_train, cv=cv, printer=False)
        r2_scores.append(cv_results['test_r2'].mean())
    end_k = time.time()
    optimal_min_samples_leaf = min_samples_leaf[np.argmax(r2_scores)]
    print("Optimal number of minimum samples leaf:", optimal_min_samples_leaf)
    print("Time: ", {end_k - start_k})
    return r2_scores

def find_optimal_min_samples_split(min_samples_split, X_train, y_train, cv):
    r2_scores = []
    start_k = time.time()
    for split in min_samples_split:
        rf = RandomForestRegressor(min_samples_leaf=split, random_state=42, n_jobs=-1)
        cv_results = cross_validate_model(rf, X_train, y_train, cv=cv, printer=False)
        r2_scores.append(cv_results['test_r2'].mean())
    end_k = time.time()
    optimal_min_samples_split = min_samples_split[np.argmax(r2_scores)]
    print("Optimal number of minimum samples split:", optimal_min_samples_split)
    print("Time: ", {end_k - start_k})
    return r2_scores

os.chdir("C:/Users/ameli/OneDrive/Studium/TU Wien/WS2024/ML/Exercise 2")
os.makedirs("plot", exist_ok=True)

log_file = open("ML_Ex2_Existing_RF_performanceMeasures.txt", "w")
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
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
holdout_model(rf, X_train, y_train, X_test, y_test, title="KNN - No cv, no scaling", save_path="plots/ct_noScaling.png")

### WITHOUT CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Without CV, With Scaling:")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
holdout_model(rf, X_train_scaled, y_train, X_test_scaled, y_test, title="KNN - No cv, scaling", save_path="plots/ct_Scaling.png")

### WITH CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("MPG - With CV, Without Scaling:")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
cross_validate_model(rf, X_train, y_train, 5)

### WITH CV, WITH SCALING ###
print("----------------------------------------------------------------------------------------------")
print("MPG - With CV, With Scaling:")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
cross_validate_model(rf, X_train_scaled, y_train, 5)

### PARAMETER MAX_DEPTH ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Parameter max_depth:")
find_optimal_max_depth([10, 15, 20, None], X_train_scaled, y_train, 5)

### PARAMETER N_ESTIMATOR ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Parameter n_estimator:")
weight = find_optimal_n_estimator([10, 80, 200, 400], X_train_scaled, y_train, 5)
print(weight)

### PARAMETER MIN_SAMPLES_LEAF ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Parameter min_samples_leaf:")
distance = find_optimal_min_samples_leaf([1, 5, 10], X_train_scaled, y_train, 5)
print(distance)

### PARAMETER MIN_SAMPLES_SPLIT ###
print("----------------------------------------------------------------------------------------------")
print("MPG - Parameter min_samples_split:")
distance = find_optimal_min_samples_split([2, 5, 10], X_train_scaled, y_train, 5)
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
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
holdout_model(rf, X_train, y_train, X_test, y_test, title="KNN - No cv, no scaling", save_path="plots/ct_noScaling.png")

### WITHOUT CV, WITH SCALING ###
print("----------------------------------------------------------------------------------------------")
print("CT - Without CV, With Scaling:")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
holdout_model(rf, X_train_scaled, y_train, X_test_scaled, y_test, title="KNN - No cv, scaling", save_path="plots/ct_Scaling.png")

### WITH CV, WITHOUT SCALING ###
print("----------------------------------------------------------------------------------------------")
print("CT - With CV, Without Scaling:")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
cross_validate_model(rf, X_train, y_train, 5)

### WITH CV, WITH SCALING ###
print("----------------------------------------------------------------------------------------------")
print("CT - With CV, With Scaling:")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
cross_validate_model(rf, X_train_scaled, y_train, 5)

### PARAMETER MAX_DEPTH ###
print("----------------------------------------------------------------------------------------------")
print("CT - Parameter max_depth:")
find_optimal_max_depth([10, 15, 20, None], X_train_scaled, y_train, 5)

### PARAMETER N_ESTIMATOR ###
print("----------------------------------------------------------------------------------------------")
print("CT - Parameter n_estimator:")
weight = find_optimal_n_estimator([10, 80, 200, 400], X_train_scaled, y_train, 5)
print(weight)

### PARAMETER MIN_SAMPLES_LEAF ###
print("----------------------------------------------------------------------------------------------")
print("CT - Parameter min_samples_leaf:")
distance = find_optimal_min_samples_leaf([1, 5, 10], X_train_scaled, y_train, 5)
print(distance)

### PARAMETER MIN_SAMPLES_SPLIT ###
print("----------------------------------------------------------------------------------------------")
print("CT - Parameter min_samples_split:")
distance = find_optimal_min_samples_split([2, 5, 10], X_train_scaled, y_train, 5)
print(distance)
