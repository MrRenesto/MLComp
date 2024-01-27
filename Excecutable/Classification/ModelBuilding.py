import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from Excecutable.Classification.ResultHandler import upload_result


# This File implements my Model Building.
# This method first evaluates the Classifier with Corss Validation on the Training Data
# you can activate Oversample Scaling and Outlier Capping as Preprocessing
# In the end it creates a Model over thw hole Training Data and predicts Values for the Test Data
def buildmodel(classifier, oversample, scale, outlier=False):

    # Load feature data
    features_df = pd.read_csv('Data/train_features.csv')
    labels_df = pd.read_csv('Data/train_label.csv')

    # Merge dataframes based on 'Id' column
    data = pd.merge(features_df, labels_df, on='Id')

    data = data.drop('Id', axis=1)
    data = data.drop('feature_2', axis=1)

    X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
    y = data['label']

    # Function to replace outliers with min and max values
    def replace_outliers_with_min_max(column):
        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Replace values below lower_bound with min, and above upper_bound with max
        column[column < lower_bound] = np.min(column)
        column[column > upper_bound] = np.max(column)
        return column

    if outlier:
        # Apply the outlier replacement logic to each column in X
        X = X.apply(replace_outliers_with_min_max, axis=0)

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

    labels_df = pd.read_csv('Data/test_features.csv')

    labels_df2 = labels_df.drop('Id', axis=1)
    labels_df2 = labels_df2.drop('feature_2', axis=1)

    if outlier:
        # Apply the outlier replacement logic to each column in X
        labels_df2 = labels_df2.apply(replace_outliers_with_min_max, axis=0)

    if(scale):
        scaler = StandardScaler()
        labels_df2 = scaler.fit_transform(labels_df2)

    labels_df_pred = rf_classifier.predict(labels_df2)

    upload_result(labels_df, labels_df_pred, "F1 Scores : " + str(scores))
