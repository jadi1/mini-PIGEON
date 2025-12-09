import pandas as pd

input_csv = "data/coordinates10k_with_split.csv"
output_csv = "data/coordinates10k_for_geocells.csv"

df = pd.read_csv(input_csv)

# Rename latitude/longitude
df = df.rename(columns={"Latitude": "lat", "Longitude": "lng"})

# Add unique ID
df["id"] = range(len(df))

# Map ADM1_Region -> country_name
# If you don't have a mapping, just use a placeholder
df["country_name"] = df["ADM1_Region"].apply(lambda x: x.split(".")[0])  # crude guess: take prefix as country
# or simply: df["country_name"] = "unknown"

# Reorder columns to match NEEDED_COLS
df = df[["id", "lat", "lng", "selection", "country_name"]]

# Save
df.to_csv(output_csv, index=False)
print(f"Saved geocell-compatible CSV to {output_csv}")