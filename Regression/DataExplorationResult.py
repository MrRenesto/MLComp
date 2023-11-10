import pandas as pd
import matplotlib.pyplot as plt

# Load the user metrics data from the saved "result.csv" file
user_metrics = pd.read_csv("result.csv")

# Define intervals for vote counts
intervals = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100), (101, float('inf'))]

# Create an empty dictionary to store the count of users in each interval
user_count_in_intervals = {}

# Loop through the intervals and count users in each interval
for interval in intervals:
    start, end = interval
    interval_label = f"{start}-{end} votes"
    users_in_interval = user_metrics[(user_metrics['num_ratings'] >= start) & (user_metrics['num_ratings'] <= end)]
    user_count_in_intervals[interval_label] = len(users_in_interval)

# Convert the dictionary to a DataFrame for better visualization
interval_counts_df = pd.DataFrame(list(user_count_in_intervals.items()), columns=['Vote Interval', 'User Count'])

# Create a bar chart with a logarithmic Y-axis scale and value labels
plt.figure(figsize=(10, 6))
bars = plt.bar(interval_counts_df['Vote Interval'], interval_counts_df['User Count'])
plt.yscale('log')  # Use a logarithmic scale for the Y-axis
plt.xlabel('Vote Count Interval')
plt.ylabel('Number of Users (log scale)')
plt.title('Users by Vote Count Interval (log scale)')
plt.xticks(rotation=45, ha='right')

# Add value labels above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=9)

# Show the bar chart
plt.tight_layout()
plt.show()
