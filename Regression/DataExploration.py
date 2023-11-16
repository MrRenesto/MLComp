
import pandas as pd
import matplotlib.pyplot as plt

# feature = Id,item,user,timestamp      /timestamp in ms
features = pd.read_csv('Data\\train_features.csv')
# label = Id,rating        /rating = 1,2,3,4 or 5
label = pd.read_csv('Data\\train_label.csv')

# Filter rows where 'item' is equal to 1000
filtered_features = features[features['item'] == 2980]

filtered_labels = label[label['Id'].isin(filtered_features['Id'])]
print(filtered_labels)
# Count the occurrences of each rating value (0 to 5)
rating_counts = label['rating'].value_counts().sort_index()

# Plot the counts
plt.figure(figsize=(8, 6))
rating_counts.plot(kind='bar')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Counts of Ratings in Filtered Labels')
plt.xticks(rotation=0)  # Keep the x-axis labels horizontal
plt.show()
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