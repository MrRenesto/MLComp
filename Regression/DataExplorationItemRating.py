import pandas as pd

# Read the features and label DataFrames
features = pd.read_csv('Data\\train_features.csv')
label = pd.read_csv('Data\\train_label.csv')

# Merge the features and label DataFrames on the 'Id' column
merged_data = pd.merge(features, label, on='Id')

# Calculate the average rating per item and count how many times it got rated
avg_ratings = merged_data.groupby('item')['rating'].agg(['mean', 'count'])

# Sort the DataFrame by average rating in descending order
sorted_avg_ratings = avg_ratings.sort_values(by='mean', ascending=False)

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print the sorted DataFrame
print(sorted_avg_ratings)

# Reset pandas display options to default (optional)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
