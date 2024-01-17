from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


# Load feature data
from sklearn.preprocessing import StandardScaler

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

# Create a binary classification model (Random Forest in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize RFECV with the classifier, the number of features to select at each step, and cross-validation strategy
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')

# Fit RFECV to the data
rfecv.fit(X, y)

# Plot the number of features versus cross-validated score
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross-Validated Score (Accuracy)')
plt.title('RFECV - Number of Features vs. Cross-Validated Score')
plt.show()

# Get the optimal number of features
optimal_num_features = rfecv.n_features_

# Display the selected features
selected_features = pd.DataFrame({'Feature': range(X.shape[1]), 'Selected': rfecv.support_, 'Ranking': rfecv.ranking_})
print('Selected Features:')
print(selected_features)
