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

# Replace any NaN or infinite values with a large finite number
correlation_matrix.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# Create a mask to hide the upper triangular part
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create a heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, fmt=".2f")

# Replace any NaN or infinite values with a large finite number for hierarchical clustering
correlation_matrix.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# Perform hierarchical clustering with annotated values
cluster_map = sns.clustermap(correlation_matrix, cmap='coolwarm', method='average', figsize=(12, 8), annot=True, fmt=".2f")
cluster_map.ax_row_dendrogram.set_visible(False)
cluster_map.ax_row_dendrogram.set_xlim([0, 0])
cluster_map.ax_col_dendrogram.set_visible(False)
cluster_map.ax_col_dendrogram.set_xlim([0, 0])

# Show the plot
plt.show()
