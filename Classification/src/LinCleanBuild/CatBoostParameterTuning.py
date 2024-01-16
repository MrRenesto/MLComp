import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from smac import HyperparameterOptimizationFacade, Scenario

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata
from Classification.src.ResultHandler import upload_result


X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    classifier = CatBoostClassifier(num_trees=config['num_trees'], learning_rate=config['learning_rate'], random_state=seed)
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    return 1 - np.mean(scores)


# example how to use
clf = CatBoostClassifier()

import smac
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('num_trees', 10, 200))
cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 0.001, 1))

hyperparamtuning(clf, cs, train, './smac_catboost', X_train, X_test, y_train, y_test)