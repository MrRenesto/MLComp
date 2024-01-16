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
import numpy as np
from catboost import CatBoostClassifier, Pool

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

rf_classifier = CatBoostClassifier(iterations=10,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)

randomstate=420
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomstate)

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
