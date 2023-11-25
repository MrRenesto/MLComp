import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
from src.ResultHandler import *

# Load feature data
features_df = pd.read_csv('..\\..\\Data\\train_features.csv')
labels_df = pd.read_csv('..\\..\\Data\\train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
data = data.drop('feature_2', axis=1)
# data = data.drop('feature_20', axis=1)
# data = data.drop('feature_12', axis=1)

X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")


report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", report)

# Print feature importances
feature_importances = pd.DataFrame(rf_classifier.feature_importances_, index=X.columns, columns=['Importance'])
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importances)


labels_df = pd.read_csv('..\\..\\Data\\test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)
# labels_df2 = labels_df2.drop('feature_20', axis=1)
# labels_df2 = labels_df2.drop('feature_12', axis=1)

labels_df_pred = rf_classifier.predict(labels_df2)

upload_result(labels_df, labels_df_pred, "RandomForest F1 Local: " + str(f1))