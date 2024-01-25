import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate, train_test_split

# Load the data
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

# Save transformed data in the right format for the Surprise Framework into CSV
transformed_data.to_csv('Data/transformed_data.csv', index=False, header=False)

# Create a Surprise Reader object
reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp', sep=',')

# Load data into Surprise Dataset
data = Dataset.load_from_file('Data/transformed_data.csv', reader)

# Split data by year
transformed_data['year'] = pd.to_datetime(transformed_data['timestamp']).dt.year

# Train a model for each year
models = {}
for year in transformed_data['year'].unique():
    train_data = transformed_data[transformed_data['year'] == year]
    trainset = Dataset.load_from_df(train_data[['user', 'item', 'rating']], reader).build_full_trainset()
    model = SVD()
    model.fit(trainset)
    models[year] = model

# Make predictions using the corresponding model for each year
predictions = []
for i in range(len(predict_features)):
    year = pd.to_datetime(predict_features['timestamp'][i]).year
    model = models.get(year)
    if model:
        result = model.predict(str(predict_features['user'][i]), str(predict_features['item'][i]), clip=True)
        prediction = (predict_features['Id'][i], round(float(result.est)))
        predictions.append(prediction)

# Convert the list of tuples to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Id', 'Predicted'])

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions_SVD_by_year.csv', index=False)
