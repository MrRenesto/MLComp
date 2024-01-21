import pandas as pd

features = pd.read_csv('Data//train_features.csv')


middle_timestamp = features['timestamp'].median()

print("Middle Timestamp:", middle_timestamp)