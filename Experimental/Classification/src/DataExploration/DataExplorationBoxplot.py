import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load train and test data
dftrain = pd.read_csv('../../Data/train_features.csv')
dftest = pd.read_csv('../../Data/test_features.csv')

for i in range(31):
    # Choose the feature you want to visualize as a boxplot
    feature_name = 'feature_' + str(i)

    # Create a boxplot for the feature
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size

    # Combine train and test data for the boxplot
    data_to_plot = [dftrain[feature_name], dftest[feature_name]]

    # Plot the boxplot with a logarithmic y-axis scale
    plt.boxplot(data_to_plot, labels=['Train Data', 'Test Data'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.yscale('log')  # Set y-axis to logarithmic scale

    plt.xlabel('Data')  # X-axis label
    plt.ylabel(feature_name)  # Y-axis label
    plt.title(f'Boxplot of {feature_name} for Train and Test Data')

    # Save the image
    plt.savefig('./ResultDataExploration/Boxplot/' + feature_name + '_logscale.png')

    # Close the figure to avoid overlapping plots
    plt.close()
