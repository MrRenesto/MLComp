import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Read the data
from sklearn.preprocessing import StandardScaler

features_df = pd.read_csv('../../Data/ModelBuilding_train_data.csv')

# Split data into features (X) and target variable (y)
X = features_df.drop('label', axis=1)
y = features_df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the scaled array back to a pandas DataFrame
X = pd.DataFrame(X, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a LightGBM classifier
classifier = RandomForestClassifier() # learning_rate=0.01, min_child_weight=1, max_depth=15, colsample_bytree=0.7, subsample=0.6, num_leaves=80, n_estimators=600, silent=True, nthread=1)

# Fit the classifier on the original features
classifier.fit(X_train, y_train)

# Evaluate the accuracy on the original features
y_pred = classifier.predict(X_test)
original_accuracy = f1_score(y_test, y_pred)

# Randomly select a subset of features
random_subset = X_train.sample(frac=0.5, axis=1, random_state=42)  # You can adjust the fraction as needed

best_accuracy = 0.0
best_subset = None

# Loop through different fractions of features in the random subset
for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for i in range(10):
        # Select a subset of features
        current_subset = random_subset.sample(frac=fraction, axis=1)

        # Fit the classifier on the current subset
        classifier.fit(current_subset, y_train)

        # Transform the test set to keep the same subset of features
        X_test_current_subset = X_test[current_subset.columns]

        # Evaluate the accuracy on the current subset of features
        y_pred_current_subset = classifier.predict(X_test_current_subset)
        current_accuracy = f1_score(y_test, y_pred_current_subset)

        print(f'f1_score on {fraction * 100:.0f}% randomly selected features: {current_accuracy:.4f}')

        # Update the best subset if the current accuracy is better
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_subset = current_subset

print(f'f1_score on original features: {original_accuracy:.4f}')
print(f'\nBest f1_score: {best_accuracy:.4f}')
print('Best subset of features:')
print(best_subset.columns.tolist())
