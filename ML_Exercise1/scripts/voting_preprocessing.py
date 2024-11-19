import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
import pickle


def barplot_target(target, save_path='../plots/barplot_target_votings.png'):
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


def correlation_matrix(df, save_path='../plots/corr_matrix_votings.png'):
    # Correlation plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    column_names = df.columns.tolist()
    correlations = df.corr(method='kendall')

    fig, ax = plt.subplots(figsize=(10, 10))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
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


def stacked_barplot(df, save_path='../plots/stacked_barplot.png'):
    # Plot of Vote counts
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vote_counts = df.apply(lambda x: x.value_counts(dropna=False))
    vote_counts = vote_counts.T

    vote_counts.plot(kind='bar', stacked=True, figsize=(10, 6))

    plt.xlabel('Issues')
    plt.ylabel('Vote Counts')
    plt.title('Vote Counts for Each Issue')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Vote Type', labels=['No(n)', 'Yes (y)', 'Unknown'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def encode_data(df):
    """
    Encode non-NaN values in the dataset using LabelEncoder.
    """
    for col in df.columns:
        label_encoder = LabelEncoder()
        non_nan_mask = df[col].notna()
        df.loc[non_nan_mask, col] = label_encoder.fit_transform(df.loc[non_nan_mask, col]).astype(int)

        with open('../label_encoder_voting.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
    return df


def impute_missing_values(df, strategy='most_frequent'):
    """
    Impute missing values in the dataset based on the specified strategy.
    """
    imputer = SimpleImputer(strategy=strategy)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


def preprocess_data(df):
    """
    General preprocessing function for training or test data.
    """
    df = encode_data(df.replace('unknown', np.nan))
    df = impute_missing_values(df)
    return df


if __name__ == '__main__':
    # Load data
    training = pd.read_csv('../data/congressionalVoting/CongressionalVotingID.shuf.lrn.csv', sep=",")
    test = pd.read_csv('../data/congressionalVoting/CongressionalVotingID.shuf.tes.csv', sep=",", header=0)

    # Preprocess Training Data
    training_preprocessed = preprocess_data(training.drop(columns='ID'))

    # Exploratory Analysis
    barplot_target(training_preprocessed['class'])
    correlation_matrix(training_preprocessed)
    stacked_barplot(training_preprocessed)

    # Preprocess Test Data
    test_ids = test['ID']  # Keep IDs separately
    test_preprocessed = preprocess_data(test.drop(columns='ID'))

    # Save Preprocessed Data
    training_preprocessed.to_csv("../data/congressionalVoting/voting_cleaned.csv", index=False)
    test_preprocessed.insert(0, 'ID', test_ids)  # Add ID column back to test data
    test_preprocessed.to_csv("../data/congressionalVoting/voting_cleaned_test.csv", index=False)

    # target:               features:
    # 0 = democrat          0 = no
    # 1 = republican        1 = yes
