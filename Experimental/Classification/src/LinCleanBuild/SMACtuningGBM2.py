import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, UniformIntegerHyperparameter, UniformFloatHyperparameter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata

X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    param = {'n_estimators': config['n_estimators'],
             'max_depth': config['max_depth'],
             'learning_rate': config['learning_rate'],
             'subsample': config['subsample'],
             'min_samples_split': config['min_samples_split'],
             'min_samples_leaf': config['min_samples_leaf']
             }

    classifier = GradientBoostingClassifier(random_state=seed)
    classifier.set_params(**param)
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', classifier)])

    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
    return 1 - np.mean(scores)

# Define the hyperparameter search space for GradientBoostingClassifier
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('n_estimators', 10, 200))
cs.add_hyperparameter(UniformIntegerHyperparameter('max_depth', 1, 32))
cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 0.01, 0.2))
cs.add_hyperparameter(UniformFloatHyperparameter('subsample', 0.5, 1.0))
cs.add_hyperparameter(UniformIntegerHyperparameter('min_samples_split', 2, 20))
cs.add_hyperparameter(UniformIntegerHyperparameter('min_samples_leaf', 1, 10))

# Perform hyperparameter tuning
hyperparamtuning(GradientBoostingClassifier(), cs, train, './smac_gb_smote', X_train, X_test, y_train, y_test, 50)

#Best Hyperparameters: {'learning_rate': 0.0868414542196764, 'max_depth': 11, 'min_samples_leaf': 9, 'min_samples_split': 20, 'n_estimators': 199, 'subsample': 0.9318711362027603}

