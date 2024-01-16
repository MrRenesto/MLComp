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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import resample
from scipy.sparse import coo_matrix

# Load feature data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
data = data.drop('feature_2', axis=1)

X = data.drop('label', axis=1)
y = data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# define pipeline
steps = [('over', SMOTE()), ('model', LGBMClassifier())]
pipeline = Pipeline(steps=steps)

import random
# Define hyperparameter distributions for RandomizedSearchCV
param_dist = {
        'model__bagging_fraction': [random.uniform(0.1, 1.0) for _ in range(10)],
        'model__bagging_frequency': [random.randint(5, 10) for _ in range(10)],
        'model__feature_fraction': [random.uniform(0.1, 1.0) for _ in range(10)]
,        'model__max_depth': [random.randint(10, 15) for _ in range(10)],
        'model__min_data_in_leaf': [random.randint(70, 120) for _ in range(10)],
        'model__num_leaves': [random.randint(100, 1500) for _ in range(10)]
}

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10,
                                   scoring='f1_macro', cv=10, n_jobs=-1, random_state=0)

# Fit the model to the data
random_search.fit(X, y)



# Print the best parameters and corresponding score
print("Best Parameters:", random_search.best_params_)
print("Best f1_macro Score:", random_search.best_score_)

'''
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=0)
scores = cross_val_score(pipeline, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
print("Model", 'f1_macro', " mean=", scores.mean(), "stddev=", scores.std())
'''



labels_df = pd.read_csv('../../Data/test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)

scaler = StandardScaler()
labels_df2 = scaler.fit_transform(labels_df2)

labels_df_pred = random_search.predict(labels_df2)

upload_result(labels_df, labels_df_pred, "LGBM F1 Report: " + str("test"))
