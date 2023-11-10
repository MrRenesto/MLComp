import pandas as pd
import matplotlib.pyplot as plt

# Load train and test data
dftrain = pd.read_csv('..\\Data\\train_features.csv')
dftest = pd.read_csv('..\\Data\\test_features.csv')

for i in range(31):
    # Choose the feature you want to visualize as a histogram
    feature_name = 'feature_' + str(i)

    # Create a histogram for the feature
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size

    # Plot the histogram for the train data
    n_train, bins_train, patches_train = plt.hist(dftrain[feature_name], bins=20, alpha=0.5, label='Train Data', color='blue')

    # Plot the histogram for the test data
    n_test, bins_test, patches_test = plt.hist(dftest[feature_name], bins=20, alpha=0.5, label='Test Data', color='green')

    plt.xlabel(feature_name)  # X-axis label
    plt.ylabel('Frequency')  # Y-axis label
    plt.title(f'Histogram of {feature_name} for Train and Test Data')
    plt.legend()

    # Add count labels on top of the bars
    for bin_value, count, patches in zip(bins_train, n_train, patches_train):
        plt.text(bin_value + 0.5, count, str(int(count)), ha='center', va='bottom', fontsize=8, color='Blue')

    for bin_value, count, patches in zip(bins_test, n_test, patches_test):
        plt.text(bin_value + 0.5, count, str(int(count)), ha='center', va='bottom', fontsize=8, color='DarkGreen')

    # Save the image
    plt.savefig('./ResultDataExploration/Histogram/' + feature_name + '.png')

    # Close the figure to avoid overlapping plots
    plt.close()
