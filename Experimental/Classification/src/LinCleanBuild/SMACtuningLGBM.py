import lightgbm
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from smac import HyperparameterOptimizationFacade, Scenario

from Excecutable.Classification.ResultHandler import upload_result

# Load your data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

data = pd.merge(features_df, labels_df, on='Id')
data = data.drop(['Id', 'feature_2'], axis=1)

X = data.drop('label', axis=1)
y = data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=929)

# Define your binary classification model
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
# Add other hyperparameters as needed


def train(config: Configuration, seed: int = 0) -> float:
    param = {'num_leaves': config['num_leaves'], 'objective': 'binary',
             'n_estimators': config['n_estimators'],
             'max_depth': config['max_depth'],
             'bagging_fraction': config['bagging_fraction'],
             'feature_fraction': config['feature_fraction'],
             'min_data_in_leaf': config['min_data_in_leaf']
             }

    classifier = LGBMClassifier(n_estimators=config['n_estimators'], max_depth=config['max_depth'],
                                num_leaves=config['num_leaves'], feature_fraction=config['feature_fraction'], random_state=seed)
    # scores = cross_val_score(param, classifier, X_test, y_test, cv=5, scoring='f1_macro')

    # combined_array = np.column_stack((X_test, y_test))
    train_data = lightgbm.Dataset(X_test, label=y_test)
    cv = lightgbm.cv(param, train_data, nfold=5, metrics='f1_macro')
    return 1


# Scenario object specifying the optimization environment
scenario = Scenario(cs, deterministic=True, n_trials=1000)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()

best_hyperparameters = dict(incumbent)
print("Best Hyperparameters:", best_hyperparameters)

clf.set_params(**best_hyperparameters)
clf.fit(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print("Test Accuracy with Best Hyperparameters:", test_accuracy)


report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", report)
print("Best Hyperparameters:", best_hyperparameters)


labels_df = pd.read_csv('../../Data/test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)

scaler = StandardScaler()
labels_df2 = scaler.fit_transform(labels_df2)

labels_df_pred = clf.predict(labels_df2)

upload_result(labels_df, labels_df_pred, "LGBM F1 Report: " + str(report))