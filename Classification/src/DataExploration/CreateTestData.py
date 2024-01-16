from sklearn.model_selection import train_test_split
import pandas as pd

# Read the features and labels
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

# Drop 'Id' column
data = data.drop('Id', axis=1)

# Separate features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the data into 90% training and 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create dataframes for training and test sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the data into new CSV files
train_data.to_csv('../../Data/ModelBuilding_train_data.csv', index=False)
test_data.to_csv('../../Data/ModelBuilding_test_data.csv', index=False)