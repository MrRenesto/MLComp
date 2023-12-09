import pandas as pd
import matplotlib.pyplot as plt

# Load train data
dftrain = pd.read_csv('../../Data/train_features.csv')

dftrain = dftrain.drop('Id', axis=1)

# Display min and max for each feature
for column in dftrain.columns:
    min_val = dftrain[column].min()
    max_val = dftrain[column].max()
    print(f"{column}: Min = {min_val}, Max = {max_val}")

# Create a horizontal bar plot for train data with logarithmic x-axis
plt.figure(figsize=(15, 8))  # Optional: Set the figure size
plt.barh(y=dftrain.columns, width=dftrain.max() - dftrain.min(), color='lightblue')

plt.xscale('log')  # Set x-axis to logarithmic scale

plt.xlabel('Feature Range (log scale)')  # X-axis label
plt.title('Feature Ranges for Train Data')

# Save the image
plt.savefig('./ResultDataExploration/FeatureRanges_TrainData_LogScale.png')

# Show the plot
plt.show()
