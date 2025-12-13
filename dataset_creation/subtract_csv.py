import pandas as pd

df1 = pd.read_csv("dataset_creation/adm1_sample_counts2.csv")

print(df1['Samples'].sum())
# 11655 total samples

# new df 
df1['Minimum'] = 26

# find difference
df1['Differences'] = df1['Minimum'] - df1['Samples']

df1['Differences'] = df1['Differences'].clip(lower=0) # clip all negative values at 0

positive_sum = df1['Differences'].sum()

print(positive_sum)
print(11655+positive_sum)
# Save back if needed
df1.to_csv("data_subtracted_clipped.csv", index=False)

# # # Load CSV
# # df = pd.read_csv("data_subtracted.csv")


# # df['Differences'] = df['Differences'].clip(lower=0) # clip all negative values at 0

# # positive_sum = df['Differences'].sum()

# # df.to_csv("data_subtracted_clipped.csv", index=False)

# # print("Sum of positive values:", positive_sum)