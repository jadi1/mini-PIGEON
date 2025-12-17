import pandas as pd

# Load the two CSVs
df1 = pd.read_csv('data/coordinates10k.csv')
df2 = pd.read_csv('data/coordinates_data_added.csv')

# Merge them
merged_df = pd.concat([df1, df2], ignore_index=True)

# Sort by the "ADM1_Region" column alphabetically
merged_df = merged_df.sort_values(by='ADM1_Region', ascending=True)

# Save to a new CSV
merged_df.to_csv('data/coordinates12k.csv', index=False)
