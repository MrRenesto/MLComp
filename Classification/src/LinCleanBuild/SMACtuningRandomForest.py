import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from smac import HyperparameterOptimizationFacade, Scenario

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning
from Classification.src.ResultHandler import upload_result

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

hyperparamtuning(clf, cs, train_method=train)