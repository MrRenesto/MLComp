
import pandas as pd
from surprise import SVD, NormalPredictor, KNNBasic, KNNWithMeans, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

features = pd.read_csv('Data//train_features.csv')
label = pd.read_csv('Data//train_label.csv')
predict_features = pd.read_csv('Data//test_features.csv')

merged_data = pd.merge(features, label, on='Id')

# Load the merged data
# Transform the data into the required format
transformed_data = merged_data[['user', 'item', 'rating']]
# Remove duplicate rows based on 'user', 'item', and 'rating'
transformed_data = transformed_data.drop_duplicates(subset=['user', 'item', 'rating'])
# Reset index after removing duplicates
transformed_data = transformed_data.reset_index(drop=True)

# Save Data in right format for Surprise Framework into CSV
# For some reason Dataset can only read timestamp from file, so i have to save a csv real quick.
transformed_data.to_csv('Data/transformed_data.csv', index=False, header=False)

# Create a Surprise Reader object
reader = Reader(rating_scale=(1, 5), line_format='user item rating', sep=',')
# Load data into Surprise Dataset
data = Dataset.load_from_file('Data/transformed_data.csv', reader)


n_factors_value = 5
reg_all_value = 0.05

# Initialize the SVD algorithm with the specified parameters
algo = SVD(n_factors=n_factors_value, reg_all=reg_all_value, lr_all=lr_all_value)

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

