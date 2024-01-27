import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

# Read the data
features = pd.read_csv('Data//train_features.csv')
label = pd.read_csv('Data//train_label.csv')
predict_features = pd.read_csv('Data//test_features.csv')

# Merge features and labels
merged_data = pd.merge(features, label, on='Id')

# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating', 'timestamp']]

# Remove duplicate rows based on 'user', 'item', 'rating', and 'timestamp'
transformed_data = transformed_data.drop_duplicates(subset=['user', 'item', 'rating', 'timestamp'])

# Reset index after removing duplicates
transformed_data = transformed_data.reset_index(drop=True)

# Save Data in the right format for the Surprise Framework into CSV
transformed_data.to_csv('Data/transformed_data.csv', index=False, header=False)

# Create a Surprise Reader object
reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp', sep=',')

# Load data into Surprise Dataset
data = Dataset.load_from_file('Data/transformed_data.csv', reader)

# Choose the timestamp to split the data
split_timestamp = 1528147188325 # Choose the timestamp to split the data

# Create two train datasets based on the timestamp
trainset_model1 = [tuple(rating[:3]) for rating in data.raw_ratings if int(rating[3]) < split_timestamp]
trainset_model2 = [tuple(rating[:3]) for rating in data.raw_ratings if int(rating[3]) >= split_timestamp]

print("Count of trainset_model1:", len(trainset_model1))
print("Count of trainset_model2:", len(trainset_model2))

# Create Surprise Datasets for training models
data_model1 = Dataset.load_from_df(pd.DataFrame(trainset_model1, columns=['user', 'item', 'rating']), reader)
data_model2 = Dataset.load_from_df(pd.DataFrame(trainset_model2, columns=['user', 'item', 'rating']), reader)

# Create the Surprise SVD algorithm
algo = SVD

# Define the parameter grid for grid search
param_grid = {'n_factors': [25, 40, 50, 100], 'reg_all': [0.05, 0.1, 0.15]}

# Create the GridSearchCV objects for model 1 and model 2
gs_model1 = GridSearchCV(algo, param_grid, measures=['rmse'], cv=5)
gs_model2 = GridSearchCV(algo, param_grid, measures=['rmse'], cv=5)

# Fit the grid search using the data for model 1
gs_model1.fit(data_model1)
best_params_model1 = gs_model1.best_params['rmse']

# Fit the grid search using the data for model 2
gs_model2.fit(data_model2)
best_params_model2 = gs_model2.best_params['rmse']

# Print the best parameters for both models
print("Best Parameters for Model 1:", best_params_model1)
print("Best Parameters for Model 2:", best_params_model2)

# Create new SVD algorithms with the best parameters for both models
best_algo_model1 = SVD(n_factors=best_params_model1['n_factors'], reg_all=best_params_model1['reg_all'])
best_algo_model2 = SVD(n_factors=best_params_model2['n_factors'], reg_all=best_params_model2['reg_all'])


result = cross_validate(best_algo_model1, data_model1, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print(result)
result2 = cross_validate(best_algo_model2, data_model2, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print(result2)


# Train models on the respective datasets
trainset_model1 = data_model1.build_full_trainset()
trainset_model2 = data_model2.build_full_trainset()

best_algo_model1.fit(trainset_model1)
best_algo_model2.fit(trainset_model2)



# Make predictions for predict_features using the trained models
# Use best_algo_model1 for predictions before split_timestamp and best_algo_model2 for predictions after split_timestamp
# ...

# Save the trained models for later use
# Save best_algo_model1 and best_algo_model2 using the Surprise model saving methods
# ...

# Save the predictions to CSV
# ...
