import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata

X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    classifier = BaggingClassifier(
        n_estimators=config['n_estimators'],
        max_samples=config['max_samples'],
        random_state=seed
    )
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', classifier)])

    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
    return 1 - np.mean(scores)

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('n_estimators', 10, 200))
cs.add_hyperparameter(UniformIntegerHyperparameter('max_samples', 2, len(X_train)))

# example how to use
clf = BaggingClassifier()

hyperparamtuning(clf, cs, train, './smac_bag', X_train, X_test, y_train, y_test, 50)
