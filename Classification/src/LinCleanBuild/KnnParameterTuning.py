import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from smac import HyperparameterOptimizationFacade, Scenario

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata
from Classification.src.ResultHandler import upload_result


X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    classifier = KNeighborsClassifier(n_neighbors=config['n_neighbors'],
                                      weights=config['weights'],
                                      metric=config['metric']
                                    )
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', classifier)])

    scorer = make_scorer(f1_score, average='macro')
    # Use cross_val_score with the specified scorer
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring=scorer)
    #scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    #return 1 - np.mean(scores)
    return 1 - np.mean(scores)


# example how to use
clf = KNeighborsClassifier()

import smac
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('n_neighbors', 1, 100))
cs.add_hyperparameter(CategoricalHyperparameter('weights', ['uniform','distance']))
cs.add_hyperparameter(CategoricalHyperparameter('metric', ['minkowski','euclidean','manhattan']))



hyperparamtuning(clf, cs, train, './smac_Knn_smote', X_train, X_test, y_train, y_test, 500)
