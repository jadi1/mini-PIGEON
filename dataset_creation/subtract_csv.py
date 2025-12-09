import pandas as pd

# df1 = pd.read_csv("dataset_creation/adm1_sample_counts2_unusedsofar.csv")
# df2 = pd.read_csv("dataset_creation/adm1_sample_counts.csv")

# df1['Differences'] = df1['Samples'] - df2['Samples']

# # Save back if needed
# df1.to_csv("data_subtracted.csv", index=False)

# Load CSV
df = pd.read_csv("data_subtracted.csv")


df['Differences'] = df['Differences'].clip(lower=0) # clip all negative values at 0

positive_sum = df['Differences'].sum()

df.to_csv("data_subtracted_clipped.csv", index=False)

print("Sum of positive values:", positive_sum)