import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have the 'features' and 'label' DataFrames loaded
features = pd.read_csv('Data\\train_features.csv')
label = pd.read_csv('Data\\train_label.csv')

# Find the top 20 most common items
top_items = features['item'].value_counts().nlargest(20).index

# Plotting for each of the top 20 items
for most_common_item in top_items:
    # Filter features for the current most common item
    filtered_features = features[features['item'] == most_common_item]

    # Merge with labels based on the common ID
    merged_data = pd.merge(filtered_features, label, on='Id')

    # Convert timestamp to datetime format (assuming it's in milliseconds)
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], unit='ms')

    # Convert the 'rating' column to numeric
    merged_data['rating'] = pd.to_numeric(merged_data['rating'], errors='coerce')

    # Group by month and year, and calculate the rating counts for each month
    grouped_data = merged_data.groupby([merged_data['timestamp'].dt.to_period("M"), 'rating']).size().unstack(fill_value=0).reset_index()

    # Convert Period to a suitable representation for plotting
    grouped_data['timestamp'] = grouped_data['timestamp'].astype(str)

    # Calculate the percentage of each rating count
    total_ratings = grouped_data.drop('timestamp', axis=1).sum(axis=1)
    grouped_data_percentages = grouped_data.drop('timestamp', axis=1).divide(total_ratings, axis=0) * 100
    grouped_data_percentages['timestamp'] = grouped_data['timestamp']

    # Manually set colors for each rating
    line_colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting lines for each rating count percentage with manual colors
    for rating, color in zip(range(1, 6), line_colors):
        ax1.plot(grouped_data_percentages['timestamp'], grouped_data_percentages[rating], marker=',', label=f'Rating {rating}', color=color)

    ax1.set_title(f'Monthly Rating Percentages for Item {most_common_item}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rating Percentage (%)')

    # Add a secondary y-axis for the total amount of ratings
    ax2 = ax1.twinx()
    ax2.plot(grouped_data_percentages['timestamp'], total_ratings, linestyle='--', color='black', label='Total Ratings')
    ax2.set_ylabel('Total Ratings')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()
