import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Assuming you already have the 'features' and 'label' DataFrames loaded
features = pd.read_csv('Data\\train_features.csv')
label = pd.read_csv('Data\\train_label.csv')

# Find the top 20 most common items
top_items = features['item'].value_counts().nlargest(20).index

# Function to calculate linear regression line
def calculate_linear_regression(x, y):
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    return model.predict(x.reshape(-1, 1))

# Plotting for each of the top 20 items
for most_common_item in top_items:
    # Filter features for the current most common item
    filtered_features = features[features['item'] == most_common_item]

    # Merge with labels based on the common ID
    merged_data = pd.merge(filtered_features, label, on='Id')

    # Convert timestamp to datetime format (assuming it's in milliseconds)
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], unit='ms')

    # Group by month and year, and calculate the mean rating and count for each month
    grouped_data = merged_data.groupby([merged_data['timestamp'].dt.to_period("Y")])['rating'].agg(['mean', 'count']).reset_index()

    # Convert Period to a suitable representation for plotting
    grouped_data['timestamp'] = grouped_data['timestamp'].astype(str)

    # Calculate linear regression line
    x_values = np.arange(len(grouped_data))
    regression_line = calculate_linear_regression(x_values, grouped_data['mean'])

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting the main line with average ratings
    line1 = ax1.plot(grouped_data['timestamp'], grouped_data['mean'], marker=',', label='Average Rating', color='b')

    # Plotting the linear regression line
    ax1.plot(grouped_data['timestamp'], regression_line, linestyle='--', color='orange', label='Linear Regression')

    ax1.set_title(f'Monthly Average Rating and Count for Item {most_common_item}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Rating', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xticklabels(grouped_data['timestamp'], rotation=45)

    # Creating a secondary y-axis on the right for count of ratings
    ax2 = ax1.twinx()

    # Plotting the line for the count of ratings on the right y-axis
    line2 = ax2.plot(grouped_data['timestamp'], grouped_data['count'], marker='o', label='Rating Count', color='r')

    # Adding the count of ratings next to the line on the right y-axis
    for i, txt in enumerate(grouped_data['count']):
        ax2.annotate(txt, (grouped_data['timestamp'].iloc[i], grouped_data['count'].iloc[i]),
                     textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='r')

    ax2.set_ylabel('Rating Count', color='r')
    ax2.tick_params('y', colors='r')

    # Adding legends for both lines
    lines = line1 + line2 + [ax1.lines[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.show()
