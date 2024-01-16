import pandas as pd
from sklearn.metrics import f1_score
import joblib

# Load the test data
test_data = pd.read_csv('../../Data/ModelBuilding_test_data.csv')
predict_data = pd.read_csv('../../Data/test_features.csv')

# Separate features and labels
# X_test = test_data.drop('label', axis=1)
X_test = test_data[['feature_24', 'feature_12', 'feature_23', 'feature_17', 'feature_16', 'feature_29', 'feature_9', 'feature_15', 'feature_27', 'feature_4']]
y_test = test_data['label']

predict_data = predict_data.drop('Id', axis=1)

filename = "LGBMClassifier_20231212_155131"

# Load the trained model
loaded_model = joblib.load('models/' + filename + '.pkl')

# Make predictions on the test data
y_pred = loaded_model.predict(X_test)

# Calculate the F1 macro score
f1_macro = f1_score(y_test, y_pred, average='macro')

print('F1 Macro Score on Test Data:', f1_macro)

prediction = loaded_model.predict(predict_data[['feature_24', 'feature_12', 'feature_23', 'feature_17', 'feature_16', 'feature_29', 'feature_9', 'feature_15', 'feature_27', 'feature_4']])
# Generate sequential IDs starting from 0
ids = range(len(prediction))

# Create a DataFrame with 'Id' and 'Label' columns
result_df = pd.DataFrame({'Id': ids, 'Label': prediction})

# Save the DataFrame to a CSV file
result_df.to_csv('prediction/predictions' + filename + '.csv', index=False)

# Print the result DataFrame
print(result_df)