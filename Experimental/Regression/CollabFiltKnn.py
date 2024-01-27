# load_data.py

import pandas as pd
from surprise import SVD, NormalPredictor, KNNBasic, KNNWithMeans, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
import numpy as np

features = pd.read_csv('Data\\train_features.csv')
label = pd.read_csv('Data\\train_label.csv')

# Load the merged data
merged_data = pd.merge(features, label, on='Id')

# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating', 'timestamp']]
transformed_data.loc[:,'rating'] = transformed_data['rating'].astype(float).astype(str)

# Save Data in right format for Surprise Framework into CSV
# For some reason Dataset can only read timestamp from file, so i have to save a csv real quick.
transformed_data.to_csv('Data\\transformed_data.csv', index=False, header=False)

# Create a Surprise Reader object
reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp', sep=',')

# Load data into Surprise Dataset
data = Dataset.load_from_file('Data\\transformed_data.csv', reader)

results_dict = {}

# List of algorithms
algorithms = [SVD(), NormalPredictor(), SVDpp(), NMF(), SlopeOne(), CoClustering()]

# Cross-validate each algorithm and store the results in the dictionary
for algo in algorithms:
    algo_name = algo.__class__.__name__
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5)
    results_dict[algo_name] = results
    print(str(algo_name) + " Done")

# Set display options for Pandas and NumPy to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# Convert the dictionary to a Pandas DataFrame
results_df = pd.DataFrame(results_dict)

# Print the full results for each algorithm
for algo_name, result in results_dict.items():
    print(f"{algo_name}: {result}")

# Print the DataFrame
print(results_df)