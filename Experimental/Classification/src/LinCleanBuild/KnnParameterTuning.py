import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata

X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    classifier = KNeighborsClassifier(n_neighbors=config['n_neighbors'],
                                      weights=config['weights'],
                                      metric=config['metric']
                                    )
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', classifier)])

    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring=scorer)
    return 1 - np.mean(scores)


# example how to use
clf = KNeighborsClassifier()

from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('n_neighbors', 1, 100))
cs.add_hyperparameter(CategoricalHyperparameter('weights', ['uniform','distance']))
cs.add_hyperparameter(CategoricalHyperparameter('metric', ['minkowski','euclidean','manhattan']))



hyperparamtuning(clf, cs, train, './smac_Knn_smote', X_train, X_test, y_train, y_test, 500)
