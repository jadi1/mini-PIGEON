import pandas as pd

# Load the two CSVs
df = pd.read_csv('data/metadata.csv')

# create "id" column
df["id"] = range(len(df))

df = df.rename(columns={"region": "admin_1_id"})

# drop cols containing images
df.drop(columns=["img1", "img2", "img3", "img4"], inplace=True)

# create country column
df["country_name"] = df["admin_1_id"].apply(lambda x: x.split(".")[0])

df.to_csv("data/metadata_for_geoson.csv")