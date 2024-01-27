import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

from Classification.src.LinCleanBuild.SMACtuning import hyperparamtuning, loaddata

X_train, X_test, y_train, y_test = loaddata()

def train(config: Configuration, seed: int = 0) -> float:
    classifier = CatBoostClassifier(num_trees=config['num_trees'], learning_rate=config['learning_rate'],
                                    depth=config['depth'],
                                    l2_leaf_reg=config['l2_leaf_reg'],
                                    random_state=seed)
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', classifier)])

    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
    return 1 - np.mean(scores)


# example how to use
clf = CatBoostClassifier()

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

# Define the hyperparameter search space
cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('num_trees', 10, 200))
cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 0.001, 1))
cs.add_hyperparameter(UniformIntegerHyperparameter('depth', 4, 10))
cs.add_hyperparameter(UniformIntegerHyperparameter('l2_leaf_reg', 1, 10))

hyperparamtuning(clf, cs, train, './smac_catboost_smote', X_train, X_test, y_train, y_test)

# {'depth': 6, 'l2_leaf_reg': 2, 'learning_rate': 0.18244088617530532, 'num_trees': 144} macro avg       0.82      0.83      0.82       156
# {'depth': 8, 'l2_leaf_reg': 10, 'learning_rate': 0.5717205031242358, 'num_trees': 121} macro avg       0.79      0.80      0.79       156
# Best Hyperparameters: {'depth': 9, 'l2_leaf_reg': 8, 'learning_rate': 0.10112666042498882, 'num_trees': 185}