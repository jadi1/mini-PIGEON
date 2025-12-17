import pandas as pd
from sklearn.model_selection import train_test_split

input_csv = "data/img_dataset.csv"
output_csv = "data/metadata.csv"
region_col = "region"
RANDOM_SEED = 42

df = pd.read_csv(input_csv)

train_df, temp_df = train_test_split(
    df,
    test_size=(1 - 0.8),
    stratify=df[region_col],
    random_state=RANDOM_SEED,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df[region_col],
    random_state=RANDOM_SEED,
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["selection"] = "train"
val_df["selection"] = "val"
test_df["selection"] = "test"

# merge back to one df
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

full_df.to_csv(output_csv, index=False)