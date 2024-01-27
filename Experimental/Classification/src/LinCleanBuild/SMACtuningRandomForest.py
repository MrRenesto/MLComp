import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata

X_train, X_test, y_train, y_test = loaddata()
def train(config: Configuration, seed: int = 0) -> float:
    classifier = RandomForestClassifier(n_estimators=config['n_estimators'],
                                       # max_depth=config['max_depth'],
                                       # max_features=config['max_features'],
                                       # min_samples_split=config['min_samples_split'],
                                       # min_samples_leaf=config['min_samples_leaf'],
                                        random_state=seed)
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', classifier)])

    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
    return 1 - np.mean(scores)


# example how to use
clf = RandomForestClassifier()

from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('n_estimators', 10, 200))
#cs.add_hyperparameter(UniformIntegerHyperparameter('max_depth', 1, 32))
#cs.add_hyperparameter(UniformIntegerHyperparameter('max_features', 8, 20))
#cs.add_hyperparameter(UniformIntegerHyperparameter('min_samples_split', 2, 10))
#cs.add_hyperparameter(UniformIntegerHyperparameter('min_samples_leaf', 1, 10))

hyperparamtuning(clf, cs, train,'./smac_randomforest',X_train, X_test, y_train, y_test, 50)

# {'max_depth': 24, 'max_features': 15, 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 172}
# {'max_depth': 26, 'max_features': 8, 'min_samples_leaf': 3, 'min_samples_split': 7, 'n_estimators': 193}