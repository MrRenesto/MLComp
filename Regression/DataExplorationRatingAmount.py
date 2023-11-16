import pandas as pd

user_metrics = pd.read_csv("result.csv")

# ['num_ratings', 'avg_rating', 'num_rating_1', 'num_rating_2', 'num_rating_3', 'num_rating_4', 'num_rating_5']
num_ratings_columns = ['num_ratings', 'num_rating_1', 'num_rating_2', 'num_rating_3', 'num_rating_4', 'num_rating_5']

# Sum the columns for each num_rating except 'avg_rating'
rating_sums = user_metrics[num_ratings_columns].sum()

# Print the results
print(rating_sums)