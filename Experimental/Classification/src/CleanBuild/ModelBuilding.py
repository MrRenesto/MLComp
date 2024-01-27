from datetime import datetime
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Load data
features_df = pd.read_csv('../../Data/ModelBuilding_train_data.csv')
test_data = pd.read_csv('../../Data/ModelBuilding_test_data.csv')

# Split features and labels
X = features_df.drop('label', axis=1)
y = features_df['label']

# Create LGBMClassifier
classifier = LGBMClassifier(learning_rate=0.02, n_estimators=600, silent=True, nthread=1)

# Define hyperparameter search space
params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7, 10],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7, 0.8, 1.0],
    'subsample': [0.6, 0.8, 1.0],
    'num_leaves': [30, 50, 80, 100],
    'bagging_fraction': [0.8, 0.9, 1],
    'max_bin': [20, 40, 60, 80, 90],
    'min_data_in_leaf': [20, 40, 60, 80],
    'min_sum_hessian_in_leaf': [0, 20, 40, 60, 80, 100],
}

# Define cross-validation strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

# Initialize RandomizedSearchCV
clf = RandomizedSearchCV(
    classifier, param_distributions=params, n_iter=300, scoring='f1_macro',
    cv=skf.split(X, y), verbose=3, n_jobs=4, random_state=34
)

# Fit RandomizedSearchCV to find the best hyperparameters
clf.fit(X, y)

# Get the model with the best parameters
best_model = clf.best_estimator_

# Fit the final model with the best hyperparameters
best_model.fit(X, y)

# Print the best hyperparameters
print("Best Hyperparameters:", clf.best_params_)

# Evaluate the model on the test set
test_features = test_data.drop('label', axis=1)
test_labels = test_data['label']
test_score = best_model.score(test_features, test_labels)
print('Test Score:', test_score)

# Save the model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'models/LGBMClassifier_{timestamp}.pkl'
joblib.dump(best_model, filename)
