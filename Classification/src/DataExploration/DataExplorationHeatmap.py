import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.DataProcessing import *


# Load data
df = get_raw_data('../../Data/train_features.csv')
df_label = get_raw_data('../../Data/train_label.csv')

df = pd.merge(df, df_label, on='Id')

if 'Id' in df:
    df = df.drop('Id', axis=1)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a mask to hide the upper triangular part
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create a heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, fmt=".2f")

# Show the plot
plt.show()
