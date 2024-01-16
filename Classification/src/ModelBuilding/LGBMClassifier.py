import pandas as pd
from numpy.random import randint, uniform
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
from lightgbm import LGBMClassifier
from Classification.src.ResultHandler import *

from sklearn.utils import resample
from scipy.sparse import coo_matrix

# Load feature data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
data = data.drop('feature_2', axis=1)

X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
y = data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

from skopt import BayesSearchCV
'''
rs_params = {
        'n_estimators': (500, 600),
        'learning_rate': (0.1, 0.2),
        'num_leaves': (10, 20)
}

param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 7),
    'subsample': uniform(0.8, 0.2),
    'colsample_bytree': uniform(0.8, 0.2),
    'min_child_samples': randint(5, 20)
}

random_search = RandomizedSearchCV(estimator=LGBMClassifier(),
                                   param_distributions=rs_params,
                                   scoring='f1_macro',
                                   cv=5,
                                   n_iter=10,  # Number of random combinations to try
                                   verbose=2,
                                   n_jobs=-1)


# Fit the grid search to the data
random_search.fit(X, y)

# Print the best parameters and corresponding F1 score
print("Best Parameters: ", random_search.best_params_)
print("Best F1 Score: ", random_search.best_score_)
'''
rf_classifier = LGBMClassifier()

# You can also calculate and print the average F1 score across all classes
#f1_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='f1_macro')
#print("Average F1 Score: " + str(f1_scores))

randomstate=420

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomstate)

#X_sparse = coo_matrix(X_train)
#X_train, X_sparse, y_train = resample(X_train, X_sparse, y_train)

rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", report)

labels_df = pd.read_csv('../../Data/test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)

#scaler = StandardScaler()
labels_df2 = scaler.fit_transform(labels_df2)

labels_df_pred = rf_classifier.predict(labels_df2)

upload_result(labels_df, labels_df_pred, "Randomstate = " + str(randomstate) + "RandomForest F1 Report: " + str(report))
