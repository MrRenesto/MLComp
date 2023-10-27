# Step 1: Load and Preprocess Data
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from DataProcessing import *


X_train, X_test, y_train, y_test = get_train_and_test_data()

X_test = preprocess(X_test)
X_train = preprocess(X_train)

# Step 2: Choose a Machine Learning Algorithm
from sklearn.neighbors  import KNeighborsClassifier


# Create an empty list to store accuracy scores
accuracy_scores_test = []
accuracy_scores_train = []

# Define a range of values for n_neighbors to try
n_neighbors_values = range(1, 101)
y_test_array = y_test['label'].to_numpy()
y_train_array = y_train['label'].to_numpy()

k_values = [i for i in range (1,21)]
scores = []

from sklearn.model_selection import cross_val_score

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train_array, cv=10)
    scores.append(np.mean(score))
"""
import matplotlib.pyplot as plt

plt.plot(k_values,scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.grid(True)
plt.show()
"""

validation_data, X_val = get_validation_data()
X_val = preprocess(X_val)

# 3. Make predictions using your trained model
# Replace 'your_model' with the actual variable that holds your trained model

best_index = np.argmax(scores)
best_k = k_values[best_index]

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

y_pred_val_test = knn.predict(X_test)
y_pred_labels = y_pred_val_test[:, 1]


# Calculate accuracy for binary classification
accuracy = accuracy_score(y_test_array, y_pred_labels)

print(f"Accuracy: {accuracy:.2f}")


y_pred_val = knn.predict(X_val)
y_pred_val = [pred[1] for pred in y_pred_val]

# 4. Create a DataFrame with the predictions and the 'Id' column
predictions_df = pd.DataFrame({'Id': validation_data['Id'], 'Predicted_Label': y_pred_val})

# 5. Save the predictions to a CSV file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f'predictions_validation_{current_datetime}.csv'

# Save the DataFrame to the CSV file
predictions_df.to_csv("Predictions/" + file_name, index=False)

