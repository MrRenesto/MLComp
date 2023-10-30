# Step 1: Load and Preprocess Data
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from ResultHandler import *




# Load your data from CSV files
features = pd.read_csv('..\\Data\\train_features.csv')
labels = pd.read_csv('..\\Data\\train_label.csv')

# Perform data preprocessing
# Handle missing values, encoding, scaling, etc.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=63)

# Step 2: Choose a Machine Learning Algorithm
from sklearn.ensemble import RandomForestClassifier

# Choose a classification algorithm
clf = RandomForestClassifier(n_estimators=100,max_features=10)

# Step 3: Train the Model
clf.fit(X_train, y_train)

# Step 4: Evaluate the Model

# Make predictions on the test set
y_pred = clf.predict(X_test)

y_test_array = y_test['label'].to_numpy()

# Now, y_test_array contains the labels
print(y_test_array)

# Extract predicted labels as a NumPy array
y_pred_labels = y_pred[:, 1]

# Now, y_pred_array contains the predicted labels
print(y_pred_labels)

# Calculate accuracy for binary classification
accuracy = accuracy_score(y_test_array, y_pred_labels)

print(f"Accuracy: {accuracy:.2f}")

# Print a classification report with more metrics
print(classification_report(y_test_array, y_pred_labels))


validation_data = pd.read_csv('..\\Data\\test_features.csv')

# 2. Extract the features from the validation data
X_val = validation_data  # All columns except the 'Id' column

# 3. Make predictions using your trained model
# Replace 'your_model' with the actual variable that holds your trained model
y_pred_val = clf.predict(X_val)
y_pred_val = [pred[1] for pred in y_pred_val]

# 4. Create a DataFrame with the predictions and the 'Id' column
predictions_df = pd.DataFrame({'Id': validation_data['Id'], 'Predicted_Label': y_pred_val})

# 5. Save the predictions to a CSV file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f'predictions_validation_{current_datetime}.csv'

# Save the DataFrame to the CSV file
predictions_df.to_csv(file_name, index=False)
