import numpy as np
import pandas as pd
from lenskit import batch, topn
from lenskit.algorithms import Recommender, als, item_knn
from lenskit.metrics import topn as tnmetrics
from lenskit import util

# Load the data
features = pd.read_csv('../Data//train_features.csv')
label = pd.read_csv('../Data//train_label.csv')
predict_features = pd.read_csv('../Data//test_features.csv')
# Merge the data
merged_data = pd.merge(features, label, on='Id')

# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating', 'timestamp']]

# Create a LensKit user-item frame
user_item_frame = transformed_data[['user', 'item', 'rating']]

# Create an ALS recommender
algo = als.BiasedMF(features=50)

# Fit the model
algo.fit(user_item_frame)

# Make predictions for the test set
predictions = []
for i in range(len(predict_features)):
    user = predict_features['user'][i]
    item = predict_features['item'][i]

    string_array_np = np.array([str(item)])
    predicted_rating = algo.predict_for_user(str(user), string_array_np)
    predictions.append((predict_features['Id'][i], round(predicted_rating)))

# Convert the list of tuples to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Id', 'Predicted'])

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions_lenskit.csv', index=False)
