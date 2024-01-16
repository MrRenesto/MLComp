import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.utils import resample
from scipy.sparse import coo_matrix

from lightgbm import LGBMClassifier
from Classification.src.ResultHandler import *
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def handle_outliers(column):
    Q1 = X[column].quantile(0.25)
    Q3 = X[column].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap values exceeding the bounds
    X[column] = X[column].apply(lambda x: min(upper_bound, max(lower_bound, x)))


# Load feature data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
# data = data.drop('feature_2', axis=1)
# data = data.drop('feature_21', axis=1)
# data = data.drop('feature_10', axis=1)
# data = data.drop('feature_3', axis=1)
# data = data.drop('feature_23', axis=1)


X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
y = data['label']

# for column in X.columns:
#    handle_outliers(column)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_sparse = coo_matrix(X_train)
X_train, X_sparse, y_train = resample(X_train, X_sparse, y_train)

random_state = 4269

listAlgo = [LGBMClassifier(random_state=random_state),
            GradientBoostingClassifier(random_state=random_state),
            ExtraTreesClassifier(random_state=random_state),
            KNeighborsClassifier(),
            RandomForestClassifier(random_state=random_state),
            LinearSVC(random_state=random_state),
            SVC(random_state=random_state),
            LogisticRegression(random_state=random_state),
            AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=random_state), random_state=random_state),
            BaggingClassifier(random_state=random_state),
            MLPClassifier(random_state=random_state, max_iter=200),
            DummyClassifier(strategy="uniform")]

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Algorithm', 'F1 Macro', 'Precision Macro', 'Recall Macro', 'Accuracy'])

# Loop through each algorithm
for algo in listAlgo:
    # Perform cross-validation
    scores = cross_validate(algo, X_train, y_train, cv=5, scoring=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy'])

    algo.fit(X_train, y_train)
    predict = algo.predict(X_test)

    f1_macro = f1_score(y_test, predict, average='macro')
    # Extract and calculate average scores
    avg_f1 = scores['test_f1_macro'].mean()
    avg_precision = scores['test_precision_macro'].mean()
    avg_recall = scores['test_recall_macro'].mean()
    avg_accuracy = scores['test_accuracy'].mean()

    # Append results to the DataFrame
    results_df = results_df._append({'Algorithm': type(algo).__name__,
                                    'F1 Macro': avg_f1,
                                    'Precision Macro': avg_precision,
                                    'Recall Macro': avg_recall,
                                    'Accuracy': avg_accuracy,
                                     'F1 Test Data: ': f1_macro}, ignore_index=True)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Print the results
print(results_df)

results_df.to_csv('classification_results.csv', index=False)



