import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have the 'features' and 'label' DataFrames loaded
features = pd.read_csv('Data\\train_features.csv')
label = pd.read_csv('Data\\train_label.csv')

# Merge with labels based on the common ID
merged_data = pd.merge(features, label, on='Id')

# Convert timestamp to datetime format (assuming it's in milliseconds)
merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], unit='ms')

# Group by month and year, and calculate the mean rating and count for each year
grouped_data_all_items = merged_data.groupby([merged_data['timestamp'].dt.to_period("Y")])['rating'].agg(['mean', 'count']).reset_index()

# Convert Period to a suitable representation for plotting
grouped_data_all_items['timestamp'] = grouped_data_all_items['timestamp'].astype(str)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting the main line with average ratings for all items
line1 = ax1.plot(grouped_data_all_items['timestamp'], grouped_data_all_items['mean'], marker=',', label='Average Rating (All Items)', color='b')

ax1.set_title('Yearly Average Rating and Count for All Items')
ax1.set_xlabel('Date')
ax1.set_ylabel('Average Rating', color='b')
ax1.tick_params('y', colors='b')
ax1.set_xticklabels(grouped_data_all_items['timestamp'], rotation=45)

# Creating a secondary y-axis on the right for count of ratings for all items
ax2 = ax1.twinx()

# Plotting the line for the count of ratings on the right y-axis for all items
line2 = ax2.plot(grouped_data_all_items['timestamp'], grouped_data_all_items['count'], marker='o', label='Rating Count (All Items)', color='r')

# Adding the count of ratings next to the line on the right y-axis for all items
for i, txt in enumerate(grouped_data_all_items['count']):
    ax2.annotate(txt, (grouped_data_all_items['timestamp'].iloc[i], grouped_data_all_items['count'].iloc[i]),
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='r')

ax2.set_ylabel('Rating Count', color='r')
ax2.tick_params('y', colors='r')

# Adding legends for both lines
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.show()
