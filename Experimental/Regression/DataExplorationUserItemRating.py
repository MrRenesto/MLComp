import pandas as pd

# Load feature and label data
features_df = pd.read_csv("Data/train_features.csv")
labels_df = pd.read_csv("Data/train_label.csv")

# Merge the two dataframes based on the common "id" column
merged_df = pd.merge(features_df, labels_df, on="Id", how="inner")

# Check the resulting merged dataframe
print(merged_df.head())

# Group by "user" and calculate user-based metrics
user_metrics = merged_df.groupby('user').agg({
    'rating': ['count', 'mean', lambda x: (x == 1).sum(), lambda x: (x == 2).sum(),
               lambda x: (x == 3).sum(), lambda x: (x == 4).sum(), lambda x: (x == 5).sum()]
})

# Rename the columns for clarity
user_metrics.columns = ['num_ratings', 'avg_rating', 'num_rating_1', 'num_rating_2', 'num_rating_3', 'num_rating_4', 'num_rating_5']

# Reset the index to have 'user' as a regular column
user_metrics.reset_index(inplace=True)

# Display the resulting DataFrame
print(user_metrics)

# Save user_metrics to a CSV file
user_metrics.to_csv("result.csv", index=False)

