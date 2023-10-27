from typing import Tuple, Any

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

    if 'Id' in features:
        features = features.drop('Id', axis=1)

    features, labels = preprocess_training_data(features.values, labels.values)

    # Perform data preprocessing
    # Handle missing values, encoding, scaling, etc.

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=False)

    column_to_remove = 0  # Replace with the index or name of the column you want to remove
    #X_test = np.delete(X_test, column_to_remove, axis=1)
    #X_train = np.delete(X_train, column_to_remove, axis=1)

    return X_train, X_test, y_train, y_test


def get_validation_data(validation_file='..\\Data\\test_features.csv'):

    validation_data = pd.read_csv(validation_file)

    # 2. Extract the features from the validation data
    X_val = validation_data.drop('Id', axis=1)
    return validation_data, X_val.values


def preprocess_training_data(X: Any, Y: Any) -> Tuple[Any,Any]:
    # Create an Isolation Forest model
    clf = IsolationForest(contamination=0.05, random_state=36)  # You can adjust the contamination parameter

    # Fit the model to your data
    clf.fit(X)  # X is your dataset
    # Predict outliers
    outliers = clf.predict(X)    # 'outliers' will contain -1 for outliers and 1 for inliers (normal data points)
    cleaned_data_X = X[outliers == 1]
    cleaned_data_Y = Y[outliers == 1]

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(outliers)

    return preprocess(cleaned_data_X), cleaned_data_Y


def preprocess(X: Any) -> Any:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
