import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np

from src.ResultHandler import *

# Load feature data
features_df = pd.read_csv('..\\..\\Data\\train_features.csv')
labels_df = pd.read_csv('..\\..\\Data\\train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
data = data.drop('feature_2', axis=1)

X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
y = data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf_classifier = KNeighborsClassifier(n_neighbors=3)
rf_classifier.fit(X_train, y_train)

# You can also calculate and print the average F1 score across all classes
f1_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='f1_macro')
print("Average F1 Score: " + str(f1_scores))

y_pred = rf_classifier.predict(X_test)

report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", report)

labels_df = pd.read_csv('..\\..\\Data\\test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)

labels_df2 = scaler.fit_transform(labels_df2)

labels_df_pred = rf_classifier.predict(labels_df2)

upload_result(labels_df, labels_df_pred, "RandomForest F1 Report: " + str(report))
