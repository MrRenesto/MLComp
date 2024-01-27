import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from Excecutable.Classification.ResultHandler import upload_result

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

random_state = 69
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Define multiple models
model1 = RandomForestClassifier(random_state=random_state)
model2 = GradientBoostingClassifier(random_state=random_state)
model3 = LGBMClassifier(random_state=random_state)

# Create a Voting Classifier
ensemble_model = VotingClassifier(estimators=[('rf', model1), ('gb', model2), ('svm', model3)], voting='soft')

# Evaluate the ensemble model using cross-validation
cv_results = cross_val_score(ensemble_model, X, y, cv=5, scoring='f1_macro')

print("Cross-Validation Results:")
print("Mean f1_macro: {:.2f}".format(cv_results.mean()))
print("Standard Deviation: {:.2f}".format(cv_results.std()))

# Train the ensemble model on the entire training set
ensemble_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble_model.predict(X_test)

# Evaluate the ensemble model on the test set
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report on Test Set:\n", report)


labels_df = pd.read_csv('../../Data/test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)

#scaler = StandardScaler()
labels_df2 = scaler.fit_transform(labels_df2)

labels_df_pred = ensemble_model.predict(labels_df2)

upload_result(labels_df, labels_df_pred, "ensemble_model F1 Report: " + str(report))