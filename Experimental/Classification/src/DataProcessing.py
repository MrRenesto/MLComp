from typing import Tuple, Any
import matplotlib.pyplot as plt

import numpy as np
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_raw_data(file='..\\Data\\train_features.csv'):

    features = pd.read_csv(file)
    return features


def get_train_and_test_data(featurefile='..\\Data\\train_features.csv', labelfile='..\\Data\\train_label.csv') -> tuple[
    Any, Any, Any, Any]:
    """

    :type featurefile: basestring
    :type labelfile: basestring
    """
    # Load your data from CSV files
    features = pd.read_csv(featurefile)
    labels = pd.read_csv(labelfile)

    features = drop_features(features)

    features = preprocess_training_data(features.values)

    # Perform data preprocessing
    # Handle missing values, encoding, scaling, etc.

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels.values, test_size=0.2, random_state=42, shuffle=False)

    column_to_remove = 0  # Replace with the index or name of the column you want to remove
    #X_test = np.delete(X_test, column_to_remove, axis=1)
    #X_train = np.delete(X_train, column_to_remove, axis=1)

    return X_train, X_test, y_train, y_test


def drop_features(features):
    if 'Id' in features:
        features = features.drop('Id', axis=1)

    if 'feature_30' in features:
        features = features.drop('feature_30', axis=1)

    if 'feature_3' in features:
        features = features.drop('feature_13', axis=1)

    if 'feature_1' in features:
        features = features.drop('feature_1', axis=1)

    if 'feature_14' in features:
        features = features.drop('feature_14', axis=1)

    return features

def get_validation_data(validation_file='..\\Data\\test_features.csv'):
    validation_data = pd.read_csv(validation_file)
    X_val = drop_features(validation_data)
    # 2. Extract the features from the validation data
    return validation_data, X_val.values


def preprocess_training_data(X: Any) -> Any:

    # Create a list to store the minimum and maximum values for each feature
    min_max_values = []

    # Loop through each feature in X
    X_cleaned = X.copy()  # Make a copy of the original data to avoid modifying it directly

    for feature_index in range(X.shape[1]):
        feature_values = X[:, feature_index]

        # Calculate the lower whisker value (minimum)
        min_boxplot_value = np.percentile(feature_values, 25) - 1.5 * (
                    np.percentile(feature_values, 75) - np.percentile(feature_values, 25))

        # Calculate the upper whisker value (maximum)
        max_boxplot_value = np.percentile(feature_values, 75) + 1.5 * (
                    np.percentile(feature_values, 75) - np.percentile(feature_values, 25))

        min_max_values.append((min_boxplot_value, max_boxplot_value))

        # Detect and replace outliers below the lower whisker with the minimum value
        X_cleaned[X_cleaned[:, feature_index] < min_boxplot_value, feature_index] = min_boxplot_value

        # Detect and replace outliers above the upper whisker with the maximum value
        X_cleaned[X_cleaned[:, feature_index] > max_boxplot_value, feature_index] = max_boxplot_value

    '''
    cleaned_data_X = X[outliers == 1]
    cleaned_data_Y = Y[outliers == 1]
    '''

    return preprocess(X_cleaned)


def preprocess(X: Any) -> Any:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
