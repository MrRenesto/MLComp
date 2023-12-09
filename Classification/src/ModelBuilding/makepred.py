import pandas as pd

# Read the input CSV file
df = pd.read_csv('/home/rsauermann/Desktop/labels.csv')

# Create a new column 'label' based on the condition
df['label'] = 0
df.loc[df['label_0_scores'] <= df['label_1_scores'], 'label'] = 1

# Create a new DataFrame with only 'Id' and 'label' columns
output_df = df[['Id', 'label']]

# Save the output to a new CSV file
output_df.to_csv('your_output_file.csv', index=False)
