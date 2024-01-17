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
    scenario = Scenario(cs, deterministic=True, n_trials=500, output_directory=dir)

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

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import f1_score

    # Assuming 'clf' is your classifier, 'X' is your feature matrix, and 'y' is your target variable

    # Define the number of folds for cross-validation (e.g., 5-fold)
    num_folds = 5

    # Set up a StratifiedKFold for classification tasks
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Initialize an empty list to store cross-validation F1 macro scores
    cv_f1_macro_scores = []

    # Combine feature matrices X_test and X_train
    X = np.concatenate([X_test, X_train], axis=0)

    # Combine target variables y_test and y_train
    y = np.concatenate([y_test, y_train], axis=0)
    # Loop through each fold
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Copy the best hyperparameters to a new dictionary for this fold
        best_hyperparameters_fold = dict(incumbent)

        # Set the classifier's parameters to the best hyperparameters for this fold
        # clf.set_params(**best_hyperparameters_fold)

        # Train the classifier on the training fold
        clf.fit(X_train_fold, y_train_fold)

        # Predict on the validation fold
        y_pred_val = clf.predict(X_val_fold)

        # Calculate the F1 macro score for this fold
        f1_macro_fold = f1_score(y_val_fold, y_pred_val, average='macro')

        # Append the F1 macro score to the list of cross-validation scores
        cv_f1_macro_scores.append(f1_macro_fold)

    # Calculate and print the mean F1 macro score
    mean_cv_f1_macro_score = sum(cv_f1_macro_scores) / len(cv_f1_macro_scores)
    print("Mean Cross-Validation F1 Macro Score:", mean_cv_f1_macro_score)

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