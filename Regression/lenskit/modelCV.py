import numpy as np
import pandas as pd
from lenskit.algorithms import als
from lenskit.metrics import predict
from sklearn.model_selection import train_test_split

# Load the data
features = pd.read_csv('../Data//train_features.csv')
label = pd.read_csv('../Data//train_label.csv')
predict_features = pd.read_csv('../Data//test_features.csv')

# Merge the data
merged_data = pd.merge(features, label, on='Id')

# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating', 'timestamp']]

# Split the data into training and test sets
train_data, test_data = train_test_split(transformed_data, test_size=0.2, random_state=42)

# Create a LensKit user-item frame for training
user_item_frame_train = train_data[['user', 'item', 'rating']]

# Create an ALS recommender
algo = als.BiasedMF(features=2)

# Fit the model on the training set
algo.fit(user_item_frame_train)

# Make predictions for the test set
predictions = []
for i in range(len(test_data)):
    user = test_data['user'].iloc[i]
    item = test_data['item'].iloc[i]

    # Get the predicted rating for the user-item pair
    predicted_rating_series = algo.predict_for_user(str(user), [str(item)])
    print(predicted_rating_series)
    predicted_rating = predicted_rating_series.iloc[0] if not predicted_rating_series.isnull().any() else 0.0  # Handle NaN values

    predictions.append((i, round(predicted_rating)))

# Convert the list of tuples to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Id', 'Predicted'])

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions_lenskit_test.csv', index=False)

# Calculate RMSE
print(predictions_df['Predicted'])
print(test_data['rating'])
rmse = predict.mae(predictions_df['Predicted'], test_data['rating'])
print('RMSE:', rmse)
