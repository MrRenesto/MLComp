from typing import Tuple, Any

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


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




def preprocess(X: Any) -> Any:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
