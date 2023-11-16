import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load train and test data
dftrain = pd.read_csv('..\\Data\\train_features.csv')
dftest = pd.read_csv('..\\Data\\test_features.csv')

for i in range(31):
    # Choose the feature you want to visualize as a violin plot
    feature_name = 'feature_' + str(i)

    # Create a violin plot for the feature
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size

    # Combine train and test data for the violin plot
    data_to_plot = [dftrain[feature_name], dftest[feature_name]]

    # Plot the violin plot
    parts = plt.violinplot(data_to_plot, showmedians=True, vert=False, widths=0.7, positions=[1, 2])

    # Customize the colors of the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('blue')

    plt.yticks([1, 2], ['Train Data', 'Test Data'])
    plt.xlabel(feature_name)  # X-axis label
    plt.title(f'Violin Plot of {feature_name} for Train and Test Data')

    # Save the image
    plt.savefig('./ResultDataExploration/ViolinPlot/' + feature_name + '_logscale.png')

    # Close the figure to avoid overlapping plots
    plt.close()
