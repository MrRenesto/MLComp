import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from smac import HyperparameterOptimizationFacade, Scenario

from Classification.src.ResultHandler import upload_result


def loaddata():
    # Load your data
    features_df = pd.read_csv('../../Data/train_features.csv')
    labels_df = pd.read_csv('../../Data/train_label.csv')

    data = pd.merge(features_df, labels_df, on='Id')
    data = data.drop(['Id', 'feature_2'], axis=1)

    X = data.drop('label', axis=1)
    y = data['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)
    return X_train, X_test, y_train, y_test


def hyperparamtuning(classifierP, configSpace, train_method, dir, X_train, X_test, y_train, y_test):


    # Define your binary classification model
    clf = classifierP

    # Define the hyperparameter search space
    cs = configSpace
    # Add other hyperparameters as needed
    # Scenario object specifying the optimization environment
    scenario = Scenario(cs, deterministic=True, n_trials=200, output_directory=dir)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train_method)
    incumbent = smac.optimize()

    best_hyperparameters = dict(incumbent)
    print("Best Hyperparameters:", best_hyperparameters)

    clf.set_params(**best_hyperparameters)
    clf.fit(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("Test Accuracy with Best Hyperparameters:", test_accuracy)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    print("Classification Report:\n", report)
    print("Best Hyperparameters:", best_hyperparameters)


    labels_df = pd.read_csv('../../Data/test_features.csv')

    labels_df2 = labels_df.drop('Id', axis=1)
    labels_df2 = labels_df2.drop('feature_2', axis=1)

    scaler = StandardScaler()
    labels_df2 = scaler.fit_transform(labels_df2)

    labels_df_pred = clf.predict(labels_df2)

    upload_result(labels_df, labels_df_pred, "LGBM F1 Report: ")
    return report

'''
X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    classifier = RandomForestClassifier(n_estimators=config['n_estimators'], max_depth=config['max_depth'], random_state=seed)
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    return 1 - np.mean(scores)


# example how to use
clf = RandomForestClassifier()

import smac
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('n_estimators', 10, 200))
cs.add_hyperparameter(UniformIntegerHyperparameter('max_depth', 1, 32))

hyperparamtuning(clf, cs, train, './smac_randomforest', X_train, X_test, y_train, y_test )
'''