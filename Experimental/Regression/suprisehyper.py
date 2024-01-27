import pandas as pd
from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

# Load the data
features = pd.read_csv('Data//train_features.csv')
label = pd.read_csv('Data//train_label.csv')

# Merge data and remove duplicates
merged_data = pd.merge(features, label, on='Id')
merged_data.drop_duplicates(keep='first', inplace=True)

# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating', 'timestamp']]
transformed_data = transformed_data.drop_duplicates(subset=['user', 'item', 'rating', 'timestamp'])
transformed_data = transformed_data.reset_index(drop=True)


# Count the number of ratings per user
#user_rating_counts = transformed_data['user'].value_counts()

# Filter out users with only one rating
#filtered_data = transformed_data[transformed_data['user'].map(user_rating_counts) > 1]

# Reset index after filtering
#transformed_data = filtered_data.reset_index(drop=True)


# Save transformed data
transformed_data.to_csv('Data/transformed_data.csv', index=False, header=False)

# Create a Surprise Reader object
reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp', sep=',')

# Load data into Surprise Dataset
data = Dataset.load_from_file('Data/transformed_data.csv', reader)

# Define the parameter grid for hyperparameter tuning
param_grid = {'n_factors': [3, 4, 5],
              'reg_all': [0.05, 0.07, 0.08]}

# Initialize the SVD algorithm
algo = SVD

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(algo, param_grid, measures=['RMSE', 'MAE'], cv=5)
grid_search.fit(data)

# Get the best hyperparameters
best_params = grid_search.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the algorithm on the full training set with the best hyperparameters
best_algo = grid_search.best_estimator
trainset = data.build_full_trainset()
best_algo.fit(trainset)

# Make predictions on the test set
predict_features = pd.read_csv('Data//test_features.csv')
predictions = []

for i in range(len(predict_features)):
    result = best_algo.predict(str(predict_features['user'][i]), str(predict_features['item'][i]), clip=True)
    prediction = (predict_features['Id'][i], round(float(result.est)))
    predictions.append(prediction)

# Convert the list of tuples to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Id', 'Predicted'])

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions_SVD_tuned.csv', index=False)
