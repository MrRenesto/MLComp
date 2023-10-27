# Step 1: Load and Preprocess Data
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report


# Load your data from CSV files
features = pd.read_csv('..\\Data\\train_features.csv')
labels = pd.read_csv('..\\Data\\train_label.csv')

# Perform data preprocessing
# Handle missing values, encoding, scaling, etc.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=63)


X_test = X_test.drop('Id', axis=1)
X_train = X_train.drop('Id', axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Choose a Machine Learning Algorithm
from sklearn.neighbors  import KNeighborsClassifier


# Create an empty list to store accuracy scores
accuracy_scores_test = []
accuracy_scores_train = []

# Define a range of values for n_neighbors to try
n_neighbors_values = range(1, 101)
y_test_array = y_test['label'].to_numpy()
y_train_array = y_train['label'].to_numpy()

for n in n_neighbors_values:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    accuracy = accuracy_score(y_test_array, y_pred[:, 1])
    accuracy_scores_test.append(accuracy)

    accuracy2 = accuracy_score(y_train_array, y_pred_train[:, 1])
    accuracy_scores_train.append(accuracy2)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_values, accuracy_scores_test, marker='.')
plt.plot(n_neighbors_values, accuracy_scores_train, marker='.')
plt.title('n_neighbors vs. Accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


"""
validation_data = pd.read_csv('..\\Data\\test_features.csv')

# 2. Extract the features from the validation data
X_val = validation_data  # All columns except the 'Id' column
X_val = X_val.drop('Id', axis=1)

# 3. Make predictions using your trained model
# Replace 'your_model' with the actual variable that holds your trained model
y_pred_val = clf.predict(X_val.values)
y_pred_val = [pred[1] for pred in y_pred_val]

# 4. Create a DataFrame with the predictions and the 'Id' column
predictions_df = pd.DataFrame({'Id': validation_data['Id'], 'Predicted_Label': y_pred_val})

# 5. Save the predictions to a CSV file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f'predictions_validation_{current_datetime}.csv'

# Save the DataFrame to the CSV file
predictions_df.to_csv(file_name, index=False)

"""