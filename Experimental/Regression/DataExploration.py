
import pandas as pd
import matplotlib.pyplot as plt

# feature = Id,item,user,timestamp      /timestamp in ms
features = pd.read_csv('Data/train_features.csv')
features_test = pd.read_csv('Data/test_features.csv')
# label = Id,rating        /rating = 1,2,3,4 or 5
label = pd.read_csv('Data/train_label.csv')

# Count the number of unique values in the 'user' column
unique_user_count = features['user'].nunique()

print(f'The number of unique values in the "user" column is: {unique_user_count}')

unique_item_count = features['item'].nunique()

print(f'The number of unique values in the "item" column is: {unique_item_count}')

# Check for users in features_test but not in features
new_users = features_test[~features_test['user'].isin(features['user'])]['user']

# Check for items in features_test but not in features
new_items = features_test[~features_test['item'].isin(features['item'])]['item']

# Print the results
if not new_users.empty:
    print(f'Users in features_test but not in features: {new_users.tolist()}')
else:
    print('All users in features_test are also in features')

if not new_items.empty:
    print(f'Items in features_test but not in features: {new_items.tolist()}')
else:
    print('All items in features_test are also in features')

columns_to_ignore = ['Id', 'timestamp']
# Check for duplicates in 'features' DataFrame
duplicates_features = features.duplicated(subset=[col for col in features.columns if col not in columns_to_ignore])
# Check for duplicates in 'features_test' DataFrame
duplicates_features_test = features_test.duplicated(subset=[col for col in features_test.columns if col not in columns_to_ignore])

# Print the results
print("Duplicates in 'features':", duplicates_features.any())
print("Duplicates in 'features_test':", duplicates_features_test.any())


# Extract relevant columns for comparison
columns_for_comparison = [col for col in features_test.columns if col not in columns_to_ignore]

# Check for common values in 'features_test' and 'features'
common_values = features_test[columns_for_comparison].isin(features[columns_for_comparison])

# Get the indices where common values exist
common_indices = common_values[common_values.all(axis=1)].index

# Print the combinations in 'features_test' that are also in 'features'
if len(common_indices) > 0:
    print("Combinations in 'features_test' that are also in 'features':")
    print(features_test.loc[common_indices, columns_for_comparison])
else:
    print("No combinations in 'features_test' are also in 'features'.")
'''

import pandas as pd
import matplotlib.pyplot as plt

features = pd.read_csv('Data/train_features.csv')  # Adjust the file path
label = pd.read_csv('Data/train_label.csv')  # Adjust the file path

# Get unique 'item' values from filtered_features
unique_items = features['item'].unique()

# Loop through each unique 'item' and create a separate plot for each
for item in unique_items:
    filtered_labels = label[label['Id'].isin(features[features['item'] == item]['Id'])]

    # Count the occurrences of each rating value (0 to 5)
    rating_counts = filtered_labels['rating'].value_counts().sort_index()

    # Plot the counts
    plt.figure(figsize=(8, 6))
    rating_counts.plot(kind='bar')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title(f'Counts of Ratings for Item {item}')
    plt.xticks(rotation=0)  # Keep the x-axis labels horizontal
    plt.show()
'''