import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata

X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    param = {'num_leaves': config['num_leaves'], 'objective': 'binary',
             'n_estimators': config['n_estimators'],
             'max_depth': config['max_depth'],
             'bagging_fraction': config['bagging_fraction'],
             'feature_fraction': config['feature_fraction'],
             'min_data_in_leaf': config['min_data_in_leaf']
             }

    classifier = LGBMClassifier(random_state=seed)
    classifier.set_params(**param)
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', classifier)])

    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
    return 1 - np.mean(scores)


# example how to use
clf = LGBMClassifier()

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('n_estimators', 10, 200))
cs.add_hyperparameter(UniformIntegerHyperparameter('max_depth', 1, 32))
cs.add_hyperparameter(UniformFloatHyperparameter('bagging_fraction', 0.1, 1))
cs.add_hyperparameter(UniformIntegerHyperparameter('bagging_frequency', 5, 10))
cs.add_hyperparameter(UniformFloatHyperparameter('feature_fraction', 0.1, 1.0))
cs.add_hyperparameter(UniformIntegerHyperparameter('num_leaves', 10, 1500))
cs.add_hyperparameter(UniformIntegerHyperparameter('min_data_in_leaf', 10, 100))

hyperparamtuning(clf, cs, train, './smac_lgbm_smote', X_train, X_test, y_train, y_test, 500)

# {'bagging_fraction': 0.2485409575822156, 'bagging_frequency': 10, 'feature_fraction': 0.6894900157985461, 'max_depth': 32, 'min_data_in_leaf': 13, 'n_estimators': 167, 'num_leaves': 851}
# {'bagging_fraction': 0.39542184201392894, 'bagging_frequency': 8, 'feature_fraction': 0.809330431307996, 'max_depth': 17, 'min_data_in_leaf': 15, 'n_estimators': 174, 'num_leaves': 1455}
#
# Best Hyperparameters: {'bagging_fraction': 0.6724286944225192, 'bagging_frequency': 7, 'feature_fraction': 0.5101102797721143, 'max_depth': 32, 'min_data_in_leaf': 14, 'n_estimators': 120, 'num_leaves': 37}