import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit

input_csv = "data/coordinates10k.csv"
output_csv = "data/coordinates10k_with_split.csv"
region_col = "ADM1_Region"
RANDOM_SEED = 42

df = pd.read_csv(input_csv)

train_df, temp_df = train_test_split(
    df,
    test_size=(1 - 0.8),
    stratify=df[region_col],
    random_state=RANDOM_SEED,
)

val_size = 0.5 # half of remaining 2k samples

ss = ShuffleSplit(
    n_splits=1,
    test_size=0.5,        # 1k val, 1k test
    random_state=RANDOM_SEED
)

val_idx, test_idx = next(ss.split(temp_df))

val_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["selection"] = "train"
val_df["selection"] = "val"
test_df["selection"] = "test"

# merge back to one df
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

full_df.to_csv(output_csv, index=False)