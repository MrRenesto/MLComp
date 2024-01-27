import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load features, labels, and test features
features = pd.read_csv('Data/train_features.csv')
features_test = pd.read_csv('Data/test_features.csv')
label = pd.read_csv('Data/train_label.csv')

# Combine features and labels into a single DataFrame
train_data = pd.concat([features, label], axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data[['user', 'item']], train_data['rating'], test_size=0.2, random_state=42)

# Use Linear Regression as an example
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)


# Round predictions to the nearest integer and clip to the range [1, 5]
val_predictions = np.clip(np.round(val_predictions), 1, 5).astype(int)

print(val_predictions)

# Evaluate the model
val_mse = mean_squared_error(y_val, val_predictions)
print(f"Validation Mean Squared Error: {val_mse}")

# Now, you can use the trained model to make predictions on the test set features
test_predictions = model.predict(features_test[['user', 'item']])
