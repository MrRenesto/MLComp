import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load feature and label data
features_df = pd.read_csv("Data/train_features.csv")
labels_df = pd.read_csv("Data/train_label.csv")

# Merge the two dataframes based on the common "id" column
merged_df = pd.merge(features_df, labels_df, on="Id", how="inner")

# Group by "user" and calculate user-based metrics
user_metrics = merged_df.groupby('item').agg({
    'rating': ['count', 'mean', lambda x: (x == 1).sum(), lambda x: (x == 2).sum(),
               lambda x: (x == 3).sum(), lambda x: (x == 4).sum(), lambda x: (x == 5).sum()]
})

# Rename the columns for clarity
user_metrics.columns = ['num_ratings', 'avg_rating', 'num_rating_1', 'num_rating_2', 'num_rating_3', 'num_rating_4', 'num_rating_5']

# Reset the index to have 'user' as a regular column
user_metrics.reset_index(inplace=True)

rating_counts = user_metrics.groupby('num_ratings').size().reset_index(name='count')
# Update 'num_ratings' column
user_metrics['num_ratings'] = user_metrics['num_ratings'].apply(lambda x: min(x, 1000))

# Group by the updated 'num_ratings' and sum the counts
rating_counts = user_metrics.groupby('num_ratings').size().reset_index(name='count').groupby('num_ratings')['count'].sum().reset_index(name='count')

# Smooth the counts using a rolling average with window size 10 (you can adjust this)
rating_counts['smooth_count'] = rating_counts['count'].rolling(window=10).mean()

# Display the result
print(rating_counts)

# Plotting
plt.plot(rating_counts['num_ratings'], rating_counts['smooth_count'], marker=',', color='blue', linestyle='-')
plt.yscale('log')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Items (Smoothed)')
plt.title('Count of Items Grouped by Number of Ratings')
plt.grid(True)
plt.show()


# Define intervals for vote counts
intervals = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100), (101, float('inf'))]

# Create an empty dictionary to store the count of users in each interval
user_count_in_intervals = {}

# Loop through the intervals and count users in each interval
for interval in intervals:
    start, end = interval
    interval_label = f"{start}-{end} Ratings"
    users_in_interval = user_metrics[(user_metrics['num_ratings'] >= start) & (user_metrics['num_ratings'] <= end)]
    user_count_in_intervals[interval_label] = len(users_in_interval)

# Convert the dictionary to a DataFrame for better visualization
interval_counts_df = pd.DataFrame(list(user_count_in_intervals.items()), columns=['Vote Interval', 'Item Count'])

# Create a bar chart with a logarithmic Y-axis scale and value labels
plt.figure(figsize=(10, 6))
bars = plt.bar(interval_counts_df['Vote Interval'], interval_counts_df['Item Count'])
plt.yscale('log')  # Use a logarithmic scale for the Y-axis
plt.xlabel('Rating received Count Interval')
plt.ylabel('Number of Items (log scale)')
plt.title('Items by Ratings received Count Interval (log scale)')
plt.xticks(rotation=45, ha='right')

# Add value labels above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=9)

# Show the bar chart
plt.tight_layout()
plt.show()



