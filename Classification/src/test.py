import numpy as np
import pandas as pd

feature_csv_file = "..\\Data\\train_features.csv"
label_csv_file = "..\\Data\\train_label.csv"

feature_df = pd.read_csv(feature_csv_file)
label_df = pd.read_csv(label_csv_file)

merged_df = pd.merge(feature_df, label_df, on="Id")

data = []

# Iterate through rows and create nested arrays
for index, row in feature_df.iterrows():
    features = np.array(row[1:])  # Skip the "Id" column
    label = label_df.loc[label_df['Id'] == row['Id'], 'label'].values[0]
    data.append((features, label))

print(data)