
import pandas as pd
from surprise import SVD, NormalPredictor, KNNBasic, KNNWithMeans, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split


# This File creates a Model with Surprise and the SVD Model
# It first makes Cross Validation on the training set and prints the results.
# Then I Train the Model on the whole Dataset and predict the values for the Test Data.
# Result gets saved in predictions_SVD_new.csv

features = pd.read_csv('Data//train_features.csv')
label = pd.read_csv('Data//train_label.csv')
predict_features = pd.read_csv('Data//test_features.csv')

# Load the merged data
merged_data = pd.merge(features, label, on='Id')

# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating', 'timestamp']]
#transformed_data.loc[:,'rating'] = transformed_data['rating'].astype(float).astype(str)

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

# Initialize the SVD algorithm with the specified parameters
algo = SVD(n_factors=n_factors_value, reg_all=reg_all_value)

result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(result)


# Train the algorithm on the trainset, and predict ratings for the testset
data = data.build_full_trainset()
algo.fit(data)
# predictions = algo.test(data_test)
predictions = []
for i in range(len(predict_features)):
    result = algo.predict(str(predict_features['user'][i]), str(predict_features['item'][i]), clip=True)
    prediction = (predict_features['Id'][i], round(float(result.est)))
    predictions.append(prediction)

# Convert the list of tuples to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Id', 'Predicted'])

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions_SVD_new.csv', index=False)