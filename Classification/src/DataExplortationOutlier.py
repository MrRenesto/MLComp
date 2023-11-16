import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load train data
dftrain = pd.read_csv('..\\Data\\train_features.csv')

# Choose the feature you want to visualize
feature_name = 'feature_13'  # Adjust the feature name as needed

# Get the data for the chosen feature
data = dftrain[feature_name]

# Set up the subplots vertically
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 16), sharex=True)

# Methods for outlier detection
methods = ['IQR', 'Standard Deviation', 'MAD', 'Z-Score']

for i, method in enumerate(methods):
    # Calculate outliers based on the chosen method
    if method == 'IQR':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        non_outliers_min = np.min(data[data >= lower_bound])
        non_outliers_max = np.max(data[data <= upper_bound])
        outliers = (data < lower_bound) | (data > upper_bound)
    elif method == 'Standard Deviation':
        mean_val = np.mean(data)
        std_dev = np.std(data)
        threshold = 2  # Adjust as needed
        lower_bound = mean_val - threshold * std_dev
        upper_bound = mean_val + threshold * std_dev
        non_outliers_min = np.min(data[data >= lower_bound])
        non_outliers_max = np.max(data[data <= upper_bound])
        outliers = (abs(data - mean_val) > threshold * std_dev)
    elif method == 'MAD':
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        threshold = 2  # Adjust as needed
        lower_bound = median_val - threshold * mad
        upper_bound = median_val + threshold * mad
        non_outliers_min = np.min(data[data >= lower_bound])
        non_outliers_max = np.max(data[data <= upper_bound])
        outliers = (abs(data - median_val) > threshold * mad)
    elif method == 'Z-Score':
        z_scores = (data - np.mean(data)) / np.std(data)
        threshold = 2  # Adjust as needed
        lower_bound = -threshold
        upper_bound = threshold
        non_outliers_min = np.min(data[data >= lower_bound])
        non_outliers_max = np.max(data[data <= upper_bound])
        outliers = (abs(z_scores) > threshold)

    # Plot min, max, and outliers
    axes[i].plot([non_outliers_min, non_outliers_max], [1, 1], color='black', linestyle='-', marker='o', markersize=8, label='Min/Max (Non-Outliers)')
    axes[i].scatter(data[outliers], np.ones(sum(outliers)), color='red', marker='o', label='Outliers')

    # Set y-axis to logarithmic scale for better visualization
    axes[i].set_xscale('log')
    axes[i].set_xlim(left=0.1)  # Adjust as needed

    axes[i].set_ylabel(f'{method} Min/Max/Outliers')
    axes[i].legend()

# Set x-axis label
axes[-1].set_xlabel('Data')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
