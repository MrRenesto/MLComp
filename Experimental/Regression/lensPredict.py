from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
from lenskit.crossfold import sample_rows

import pandas as pd

from sklearn.metrics import mean_squared_error
import numpy as np

# Load your data
features = pd.read_csv('Data/train_features.csv')
features_test = pd.read_csv('Data/test_features.csv')
label = pd.read_csv('Data/train_label.csv')

# Combine features and labels into a single DataFrame
train_data = pd.concat([features, label], axis=1)

# Create the recommender model
algo = als.BiasedMF(50)

# Train the model on the entire training data
fittable = util.clone(algo)
fittable = Recommender.adapt(fittable)
fittable.fit(train_data[['user', 'item', 'rating']])

# Make predictions on the test data
test_preds = batch.predict(fittable, features_test[['user', 'item']])

# Merge predictions with the actual ratings
merged_test = pd.merge(features_test, test_preds, on=['user', 'item'])

# Calculate RMSE
rmse_test = np.sqrt(mean_squared_error(merged_test['rating'], merged_test['prediction']))

print(f"RMSE on test data: {rmse_test}")


