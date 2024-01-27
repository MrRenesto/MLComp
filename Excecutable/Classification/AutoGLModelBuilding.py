import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from autogluon.tabular import TabularPredictor

from Excecutable.Classification.ResultHandler import upload_result

#This File Includes my AutoML with AutoGlon and my second best entry
# Public Leaderbord: 0.8087
# Local Score: 0.824
# Press Y at the end to create Prediction. Do you want to build Model and Submit Results? Y/N:


# Load feature data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
data = data.drop('feature_2', axis=1)

# AutoGluon training
predictor = (TabularPredictor(label='label', path='ag_models',
             eval_metric='f1_macro')
             .fit(train_data=data, time_limit=1800,
                  presets='best_quality'))


# Access the leaderboard to get results of each model
leaderboard = predictor.leaderboard(extra_info=True)

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print the results
print("AutoGluon Cross-Validation Leaderboard:")
print(leaderboard)

# Load and preprocess test data
test_features_df = pd.read_csv('../../Data/test_features.csv')
test_features = test_features_df.drop('Id', axis=1)
test_features = test_features.drop('feature_2', axis=1)

# AutoGluon prediction on the test set
test_predictions = predictor.predict(test_features)

# Upload the result
upload_result(test_features_df, test_predictions, "AutoGluon")