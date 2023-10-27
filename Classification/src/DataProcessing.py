from typing import Tuple, Any
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

    # Perform data preprocessing
    # Handle missing values, encoding, scaling, etc.

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=63)

    X_test = X_test.drop('Id', axis=1)
    X_train = X_train.drop('Id', axis=1)

    return X_train, X_test, y_train, y_test


def get_validation_data(validation_file='..\\Data\\test_features.csv'):

    validation_data = pd.read_csv(validation_file)

    # 2. Extract the features from the validation data
    X_val = validation_data.drop('Id', axis=1)
    return validation_data, X_val.values


def preprocess_training_data(X: Any, Y: Any) -> Tuple[Any,Any]:
    # Create an Isolation Forest model
    clf = IsolationForest(contamination=0.05)  # You can adjust the contamination parameter

    # Fit the model to your data
    clf.fit(X)  # X is your dataset
    # Predict outliers
    outliers = clf.predict(X)
    # 'outliers' will contain -1 for outliers and 1 for inliers (normal data points)
    # You can remove the outliers from your dataset like this
    cleaned_data_X = X[outliers == 1]
    cleaned_data_Y = Y[outliers == 1]
    return preprocess(cleaned_data_X), cleaned_data_Y


def preprocess(X: Any) -> Any:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
