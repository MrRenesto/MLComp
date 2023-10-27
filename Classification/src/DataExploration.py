import matplotlib

from DataProcessing import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


df = get_raw_data('..\\Data\\train_features.csv')

import pandas as pd

import sweetviz as sv

my_report = sv.analyze(df)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"

""" Plot Histogram
# Set display options to prevent truncation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_colwidth', None)  # Show full content of columns without truncation

# Print the DataFrame
print(df.describe())

print(df.isnull().sum())

print(df.dtypes)

import matplotlib.pyplot as plt

charts_per_page = 8  # Set the number of charts per page (adjust as needed)
num_rows = 2  # Set the number of rows per page (adjust as needed)
num_cols = 4  # Set the number of columns per page (adjust as needed)

# Calculate the total number of subplots
total_subplots = len(df.columns)

# Calculate the total number of pages
total_pages = (total_subplots - 1) // charts_per_page + 1

# Set a larger figure size
fig = plt.figure(figsize=(12, 8))

# Iterate through the DataFrame columns and create charts for each page
for page in range(total_pages):
    plt.clf()  # Clear the current figure

    for i in range(charts_per_page):
        index = page * charts_per_page + i
        if index < total_subplots:
            row = i // num_cols
            col = i % num_cols

            ax = plt.subplot(num_rows, num_cols, i + 1)
            ax.hist(df[df.columns[index]], bins=20)  # Adjust the number of bins as needed
            ax.set_title(f'Histogram of {df.columns[index]}')
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')

    # Adjust layout for the current page
    plt.tight_layout()

    # Save or display the current page of charts
    plt.savefig(f'ResultDataExploration/page_{page + 1}.png', dpi=300)  # Save the current page as an image
    plt.show()

# Show a message indicating that images are saved
print(f"{total_subplots} charts saved across {total_pages} pages. You can now scroll through them using an image viewer.")
"""
