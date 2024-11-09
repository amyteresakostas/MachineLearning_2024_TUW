import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def barplot_target(target, save_path='plots/barplot_target.png'):
    # Barplot of target variable
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    class_count = target.value_counts()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_count.index, y=class_count.values)
    plt.title('Barplot Class')
    plt.xlabel('Party Member')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def correlation_matrix(df, save_path='plots/corr_matrix.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    column_names = df.columns.tolist()
    # Correlation matrix
    correlations = df.corr(method='kendall')

    # Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
    ax.set_xticklabels(
        column_names,
        rotation=45,
        horizontalalignment='right'
    )
    ax.set_yticklabels(column_names)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def stacked_barplot(df, save_path='plots/stacked_barplot.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vote_counts = df.apply(lambda x: x.value_counts(dropna=False))
    vote_counts = vote_counts.T

    # Plotting with Matplotlib
    vote_counts.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Add labels and title
    plt.xlabel('Issues')
    plt.ylabel('Vote Counts')
    plt.title('Vote Counts for Each Issue')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Vote Type', labels=['No(n)', 'Yes (y)', 'Unknown'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def impute_missing(columns_to_impute):
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_data = imputer.fit_transform(columns_to_impute)

    training_imputed = pd.DataFrame(imputed_data, columns=columns_to_impute.columns)
    training_imputed['class'] = training['class']  # Add 'class' back

    training_imputed = training_imputed[['class'] + list(columns_to_impute.columns)]
    training_imputed[columns_to_impute.columns] = training_imputed[columns_to_impute.columns]

    return (training_imputed)


def create_dataset(df, valid_size):
    X = df.drop(columns=['class'])
    y = df['class'].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=12007762)

    return X_train, y_train, X_valid, y_valid


def train_model(model_type, X_train, y_train):
    trained_model = model_type()
    trained_model.fit(X_train, y_train)

    return trained_model


def predict(trained_model, X_valid):
    y_pred = trained_model.predict(X_valid)
    y_pred_proba = trained_model.predict_proba(X_valid)[:, 1]

    return y_pred, y_pred_proba


def compare_metrics(y_true, y_pred, y_pred_proba):
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)

    }
    return scores


def print_scores(scores: dict):
    print("\nScores:\n=======")
    for metric_name, metric_value in scores.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == '__main__':
    ### DATA PREPROCESSING ###

    # Load data
    training = pd.read_csv('Congressional_Voting/CongressionalVotingID.shuf.lrn.csv', sep=",")
    test = pd.read_csv('Congressional_Voting/CongressionalVotingID.shuf.tes.csv', sep=",")

    # Drop ID column
    training = training.drop(columns=['ID'])
    test = test.drop(columns=['ID'])

    # Encoding
    training = training.replace('unknown', np.nan)
    for col in training.columns:
        label_encoder = LabelEncoder()
        # Encode only non-NaN values
        non_nan_mask = training[col].notna()
        training.loc[non_nan_mask, col] = label_encoder.fit_transform(training.loc[non_nan_mask, col]).astype(int)

    # target:               features:
    # 0 = democrat          0 = no
    # 1 = republican        1 = yes

    # Exploratory Analysis
    barplot_target(training['class'])
    correlation_matrix(training)
    stacked_barplot(training)

    # Missing values
    for column in training.columns:
        print(f"Value counts for '{column}':")
        print(training[column].value_counts(dropna=False).sort_index())
        print("\n" + "-" * 30 + "\n")

    # 1st step: Exclude observations where half of columns are unknown
    na_count = training.isna().sum(axis=1)
    threshold = training[1:].shape[1] / 2

    training = training[na_count <= threshold].reset_index(drop=True)  # 2 rows were excluded

    # 2nd step: Impute based on most frequent records
    columns_to_impute = training.drop(columns=['class'])
    training_imputed = impute_missing(columns_to_impute)

    na_count = training_imputed.isna().sum(axis=0)
    print(na_count)

    ### CLASSIFICATION ###

    # Create train and validate set
    valid_split = 0.2
    X_train, y_train, X_valid, y_valid = create_dataset(training_imputed, valid_size=valid_split)

    # Model fitting
    trained_model = train_model(RandomForestClassifier, X_train, y_train)
    y_pred, y_pred_proba = predict(trained_model, X_valid)

    # Evalutaion
    metrics_scores = compare_metrics(y_valid, y_pred, y_pred_proba)
    print_scores(metrics_scores)