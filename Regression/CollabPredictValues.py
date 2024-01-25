import pandas as pd
from surprise import SVD, NormalPredictor, KNNBasic, KNNWithMeans, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

features = pd.read_csv('Data//train_features.csv')
label = pd.read_csv('Data//train_label.csv')
predict_features = pd.read_csv('Data//test_features.csv')

# Load the merged data
merged_data = pd.merge(features, label, on='Id')
# merged_data = merged_data[merged_data['timestamp']>= 1300000000000]

merged_data.drop_duplicates(keep='first', inplace=True)

# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating', 'timestamp']]

# Remove duplicate rows based on 'user', 'item', and 'rating'
transformed_data = transformed_data.drop_duplicates(subset=['user', 'item', 'rating', 'timestamp'])

# Reset index after removing duplicates
transformed_data = transformed_data.reset_index(drop=True)

# Save Data in right format for Surprise Framework into CSV
# For some reason Dataset can only read timestamp from file, so i have to save a csv real quick.
transformed_data.to_csv('Data/transformed_data.csv', index=False, header=False)

# Create a Surprise Reader object
reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp', sep=',')

# Load data into Surprise Dataset
data = Dataset.load_from_file('Data/transformed_data.csv', reader)


reader_test = Reader(rating_scale=(1, 5), line_format='user item rating timestamp', sep=',')
transformed_data_test = predict_features[['user', 'item', 'timestamp']]
# Add a new column 'rating' and set its value to 1 everywhere
transformed_data_test['rating'] = 1
# Reorder the columns
transformed_data_test = transformed_data_test[['user', 'item', 'rating', 'timestamp']]
transformed_data_test.to_csv('Data/transformed_data_test.csv', index=False, header=False)
data_test = Dataset.load_from_file('Data/transformed_data_test.csv', reader_test)

n_factors_value = 5
reg_all_value = 0.05

lr_all_value = 0.003

# Initialize the SVD algorithm with the specified parameters
algo = SVDpp(n_factors=n_factors_value, reg_all=reg_all_value, lr_all=lr_all_value)

#result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#print(result)

from surprise.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid = {'n_factors': [25, 40, 50, 100], 'reg_all': [0.05, 0.1, 0.15]}

# Create the Surprise SVD algorithm
algo = SVD

# Create the GridSearchCV object
gs = GridSearchCV(algo, param_grid, measures=['rmse'], cv=5)

# Fit the grid search using the data
gs.fit(data)

# Get the best parameters from the grid search
best_params = gs.best_params['rmse']

# Print the best parameters
print(best_params)

# Create a new SVD algorithm with the best parameters
best_algo = SVD(n_factors=best_params['n_factors'], reg_all=best_params['reg_all'])

# Perform cross-validation using the best algorithm
result = cross_validate(best_algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(result)


'''

# Train the algorithm on the trainset, and predict ratings for the testset
data = data.build_full_trainset()

algo.fit(data)
# predictions = algo.test(data_test)
predictions = []
for i in range(len(predict_features)):
    result = algo.predict(str(predict_features['user'][i]), str(predict_features['item'][i]), clip=True)
    prediction = (predict_features['Id'][i], round(float(result.est)))
    predictions.append(prediction)

# print(predictions)

# Convert the list of tuples to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Id', 'Predicted'])

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions_SVD.csv', index=False)

'''