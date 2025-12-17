import pandas as pd

df1 = pd.read_csv("dataset_creation/adm1_sample_counts2.csv")
df2 = pd.read_csv("data_sample_counts.csv")
print(df1['Samples'].sum())
# 11655 total samples

# find difference
df1['Samples'] = df1['Samples'] + df2['Differences']

positive_sum = df1['Samples'].sum()

print(positive_sum)
# Save back if needed
df1.to_csv("adm1_sample_counts_final.csv", index=False)

# # # Load CSV
# # df = pd.read_csv("data_subtracted.csv")


# # df['Differences'] = df['Differences'].clip(lower=0) # clip all negative values at 0

# # positive_sum = df['Differences'].sum()

# # df.to_csv("data_subtracted_clipped.csv", index=False)

# # print("Sum of positive values:", positive_sum)