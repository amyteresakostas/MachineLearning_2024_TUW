import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import pickle
from imblearn.over_sampling import RandomOverSampler


class DataProcessor:
    """
        Handles data splitting for creation of training/validation sets.
    """

    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    def create_dataset(self, valid_size):
        """
            Create train and validation sets from training data.
        """
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size=valid_size, random_state=42)

        return X_train, y_train, X_valid, y_valid


class ModelTrainer:
    """
        Manages model training, evaluation, and hyperparameter tuning.
    """

    def __init__(self, model_type=RandomForestClassifier, random_state=123):
        self.model = model_type(random_state=random_state)

    def oversampling(self, X, y):
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        return X_resampled, y_resampled

    def train_and_evaluate(self, X_train, y_train, X_valid, y_valid, model = None, scale = False):
        """
           Trains and evaluates a model on validation data with optional scaling and balancing.
        """
        if model is None:
            model = self.model

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_valid)

        metrics = {
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred, average='weighted', zero_division=1),
            "recall": recall_score(y_valid, y_pred, average='weighted', zero_division=1),
            "f1_score": f1_score(y_valid, y_pred, average='weighted'),
            "training_time": train_time
        }

        return metrics

    def cross_val_metrics(self, X, y):
        """
            Computes cross-validation metrics.
        """
        scoring = {
            "accuracy": 'accuracy',
            "precision": make_scorer(precision_score, average='weighted', zero_division=1),
            "recall": make_scorer(recall_score, average='weighted', zero_division=1),
            "f1": make_scorer(f1_score, average='weighted')
        }
        metrics = {metric: cross_val_score(self.model, X, y, cv=10, scoring=scorer).mean() for metric, scorer in
                   scoring.items()}

        return metrics


    def hyperparameter_tuning(self, X, y):
        """
            Performs grid search for hyperparameter optimization.
        """
        scores = {}
        rf_grid = RandomForestClassifier(random_state=123)
        gr_space = {
            'max_depth': [None, 10, 20],
            'n_estimators': [100, 200, 400],
            'min_samples_leaf': [1, 5, 10],
            'min_samples_split': [2, 5, 10]
        }

        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=1),
            'recall': make_scorer(recall_score, average='weighted', zero_division=1),
            'f1': make_scorer(f1_score, average='weighted')
        }

        grid = GridSearchCV(rf_grid, gr_space, cv=10, scoring=scoring, refit='accuracy')
        model_grid = grid.fit(X, y)

        for params, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_fit_time in zip(
                model_grid.cv_results_['params'],
                model_grid.cv_results_['mean_test_accuracy'],
                model_grid.cv_results_['mean_test_precision'],
                model_grid.cv_results_['mean_test_recall'],
                model_grid.cv_results_['mean_test_f1'],
                model_grid.cv_results_['mean_fit_time']):
            param_str = str(params)
            scores[param_str] = {
                "mean_test_accuracy": mean_accuracy,
                "mean_test_precision": mean_precision,
                "mean_test_recall": mean_recall,
                "mean_test_f1": mean_f1,
                "mean_fit_time": mean_fit_time
            }

        return scores, model_grid.best_params_, model_grid.best_score_


# Printing functions
def print_results(results, dataset_name):
    print(f"\n{'=' * 30}\n{dataset_name} Results:\n{'=' * 30}")
    for key, metrics in results.items():  # Iterate over the nested dictionary
        print(f"\n{key} Metrics:")
        for metric, value in metrics.items():  # Iterate over the metrics dictionary
            print(f"{metric}: {value:.4f}")
    print("\n")


def print_hyperparam_results(param_effect_scores, best_params, best_score):
    print(f"\nHyperparameter Tuning Results")
    print(f"{'Parameters'} | {'Accuracy'} | {'Precision'} | {'Recall'} | {'F1 Score'} | {'Fit Time'}")
    for params, metrics in param_effect_scores.items():
        print(
            f"{params} | {metrics['mean_test_accuracy']:.4f}   | {metrics['mean_test_precision']:.4f}   | {metrics['mean_test_recall']:.4f}   | {metrics['mean_test_f1']:.4f} | {metrics['mean_fit_time']:.4f}")
    print("\n")
    print('Best hyperparameters are:', best_params)
    print('Best score is:', best_score)


def executor(df, target_column, dataset_name):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    processor = DataProcessor(df, target_column)

    X_train, y_train, X_valid, y_valid = processor.create_dataset(0.2)
    results = {}
    # Model training and holdout evaluation
    model_trainer = ModelTrainer()

    holdout_metrics = model_trainer.train_and_evaluate(X_train, y_train, X_valid, y_valid)
    results['Holdout'] = holdout_metrics

    # Cross-validation metrics
    cross_val_scores = model_trainer.cross_val_metrics(X, y)
    results['Cross Validation'] = cross_val_scores

    # Cross-validation with resampling
    if dataset_name in ['Amazon Reviews', 'Road Traffic Accidents']:
        X_resampled, y_resampled = model_trainer.oversampling(X, y)
        cross_val_scores_resampling = model_trainer.cross_val_metrics(X_resampled, y_resampled)
        results['Cross Validation oversampling'] = cross_val_scores_resampling

    # Dimensionality reduction
    if dataset_name == 'Amazon Reviews':
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        X_reduced = X.drop(columns=to_drop)
        cross_val_scores_reduction = model_trainer.cross_val_metrics(X_reduced, y)
        results['Cross Validation dim reduction'] = cross_val_scores_reduction

    print_results(results, dataset_name)

    # Hyperparameter tuning
    param_effect_scores, best_params, best_score = model_trainer.hyperparameter_tuning(X, y)
    print_hyperparam_results(param_effect_scores, best_params, best_score)

    # Train the best model
    best_model = RandomForestClassifier(**best_params, random_state=123)
    best_model.fit(X, y)

    return best_model


# Create prediction file for Kaggle competition
def predict_on_testset(model, test_file, output_file, label_encoder):
    """
        Predict outcomes on the test dataset using the given model.
    """
    df_test = pd.read_csv(test_file, sep = ",")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(label_encoder, 'rb') as f:
        encoder = pickle.load(f)

    with open(output_file, 'w') as f:
        f.write('ID,"class"\n')

        for _, row in df_test.iterrows():
            row_id = row['ID']
            features = row.drop('ID').values.reshape(1, -1)
            prediction = model.predict(features)[0]
            prediction_decoded = encoder.inverse_transform([prediction])[0]

            f.write(f"{row_id},{prediction_decoded}\n")


if __name__ == '__main__':
    # Load datasets
    df_train_voting = pd.read_csv('../data/congressionalVoting/voting_cleaned.csv', sep=",")
    df_train_machine = pd.read_csv('../data/machine/Machine_cleaned.csv', sep=",")
    df_train_rta = pd.read_csv('../data/RTA/RTA_encoded.csv', sep=",")
    df_train_reviews = pd.read_csv('../data/reviews/reviews_cleaned.csv', sep=",")

    # Load datasets and run executor
    datasets = [
        (df_train_reviews, 'Class', 'Amazon Reviews'),
        (df_train_machine, 'fail', 'Machine Failure'),
        (df_train_rta, 'Accident_severity', 'Road Traffic Accidents'),
        (df_train_voting, 'class', 'Congressional Voting')
    ]

    best_models = {}

    for df, target, name in datasets:
        best_model = executor(df, target, name)
        best_models[name] = best_model

    # Predict best models on test set for Kaggle competition
    predict_on_testset(best_models['Congressional Voting'], '../data/congressionalVoting/voting_cleaned_test.csv', '../predictions/predictions_voting_group08.csv', '../label_encoder_voting.pkl')
    predict_on_testset(best_models['Amazon Reviews'], '../data/reviews/amazon_review_ID.shuf.tes.csv', '../predictions/predictions_review_group08.csv', '../label_encoder_review.pkl')