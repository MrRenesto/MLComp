
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

X = pd.read_csv('../Data/train_features.csv')
y = pd.read_csv('../Data/train_label.csv')

if 'Id' in X:
    X = X.drop('Id', axis=1)


if 'Id' in y:
    y = y.drop('Id', axis=1)


model = RandomForestRegressor()
model.fit(X, y.values.ravel())  # X is your feature data, y is your target
feature_importance = model.feature_importances_

# Create a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort the DataFrame by feature importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Format the importance values to two decimal points
feature_importance_df['Importance'] = feature_importance_df['Importance'].map('{:.6f}'.format)

# Print the sorted DataFrame
print(feature_importance_df)