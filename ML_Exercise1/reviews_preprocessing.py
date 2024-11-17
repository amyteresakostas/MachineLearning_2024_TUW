import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
import pickle


def exclude_features_with_zero(df):
    list_with_zero_columns = []
    for col in df.columns:
        if col != "Class":
            if np.percentile(df[col], 25) == np.percentile(df[col], 50) == np.percentile(df[col], 75) == 0:
                list_with_zero_columns.append(col)

    df = df.drop(columns=list_with_zero_columns, axis=1)

    return df

def data_scaling(X_train, X_test):
    # standarization
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)

    return X_train_scaled, X_test_scaled


if __name__ == '__main__':
    # Load data
    training = pd.read_csv('../data/reviews/amazon_review_ID.shuf.lrn.csv', sep=",")
    test = pd.read_csv('../data/reviews/amazon_review_ID.shuf.tes.csv', sep=",")

    # Encoding
    label_encoder = LabelEncoder()
    training['Class'] = label_encoder.fit_transform(training['Class']).astype(int)

    training.drop(columns='ID', inplace=True)
    training.to_csv('../data/reviews/reviews_cleaned.csv', index=False)

    with open('../label_encoder_review.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
