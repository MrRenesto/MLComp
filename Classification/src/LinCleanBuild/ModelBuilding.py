import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from numpy.random import randint, uniform
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
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
def buildmodel(classifier, oversample, scale):

    # Load feature data
    features_df = pd.read_csv('../../Data/train_features.csv')
    labels_df = pd.read_csv('../../Data/train_label.csv')

    # Merge dataframes based on 'Id' column
    data = pd.merge(features_df, labels_df, on='Id')

    data = data.drop('Id', axis=1)
    data = data.drop('feature_2', axis=1)

    X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
    y = data['label']

    if(scale):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    randomstate = 420
    if(oversample):
        steps = [('over', SMOTE()),('model', classifier)]
        rf_classifier = Pipeline(steps=steps)
    else:
        rf_classifier = classifier

    scorer = make_scorer(f1_score, average='macro')
    # Use cross_val_score with the specified scorer
    scores = cross_val_score(rf_classifier, X, y, cv=5, scoring=scorer)

    rf_classifier.fit(X, y)
    print(scores)

    print("Mean F1_macro:", np.mean(scores))
    print("Standard Deviation of F1_macro:", np.std(scores))

    labels_df = pd.read_csv('../../Data/test_features.csv')

    labels_df2 = labels_df.drop('Id', axis=1)
    labels_df2 = labels_df2.drop('feature_2', axis=1)

    if(scale):
        scaler = StandardScaler()
        labels_df2 = scaler.fit_transform(labels_df2)

    labels_df_pred = rf_classifier.predict(labels_df2)

    upload_result(labels_df, labels_df_pred, "Randomstate = " + str(randomstate) + "Catboost F1 Report: " + str(scores))
