import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
from Classification.src.ResultHandler import *
# import pycaret classification and init setup
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment

# Load feature data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
# data = data.drop('feature_2', axis=1)
# data = data.drop('feature_20', axis=1)
# data = data.drop('feature_12', axis=1)

X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


exp = ClassificationExperiment()
type(exp)

exp.setup(data, target='label', session_id=345, normalize=True, normalize_method='minmax')
result = exp.compare_models(sort='f1')

exp.plot_model(plot='feature', estimator=result)
print(result)
