import pandas as pd
import matplotlib.pyplot as plt
# feature = Id,item,user,timestamp
# /timestamp in ms
features = pd.read_csv('Data/train_features.csv')
features_test = pd.read_csv('Data/test_features.csv')
# label = Id,rating        /rating = 1,2,3,4 or 5
label = pd.read_csv('Data/train_label.csv')

# Convert the timestamp to datetime format
features_test['timestamp'] = pd.to_datetime(features_test['timestamp'], unit='ms')

# Group by months and years, and count the entries in each group
grouped_data = features_test.groupby(features_test['timestamp'].dt.to_period("M")).size().reset_index(name='count')

# Convert the 'timestamp' Period to string
grouped_data['timestamp'] = grouped_data['timestamp'].astype(str)

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(grouped_data['timestamp'], grouped_data['count'], color='skyblue')
plt.title('Entries Count Over Time (Grouped by Month/Year)')
plt.xlabel('Month/Year')
plt.ylabel('Entry Count')
plt.xticks(rotation=45)
plt.show()


# Alternatively, if you want the count of unique users and items:
num_unique_users = features_test['user'].nunique()
num_unique_items = features_test['item'].nunique()

# Or print the counts
print("Number of Unique Users:", num_unique_users)
print("Number of Unique Items:", num_unique_items)