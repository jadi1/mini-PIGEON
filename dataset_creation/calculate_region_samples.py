# map each country to number of samples proportional to area
import geopandas as gpd
import numpy as np
import pandas as pd

regions = gpd.read_file("data/gadm/countries_adm1_geometries.gpkg")
csv = pd.read_csv("dataset_creation/filtered_adm1_progress.csv")

csv_ok = csv[csv["Status"] == "OK"]

regions_filtered = regions[regions["GID_1"].isin(csv_ok["GID_1"])]
regions_filtered = regions_filtered.reset_index(drop=True)

N = 24386 # total number of samples
min_samples = 26 # min samples per region

# sum the area for each country where status is ok
region_areas = regions_filtered.geometry.area

# compute area fractions
fractions = region_areas / region_areas.sum()
print(fractions)

# give each region a minimum of 5 samples, so subtract that from N
N_remaining = N - (len(region_areas) * min_samples)

print(N_remaining)
# round down to nearest int
samples_float = fractions * N_remaining
samples_int = np.floor(samples_float).astype(int) + min_samples # ensure minimum samples

remaining = N - samples_int.sum() # remaining samples to be distributed
print("remaining:", remaining)
if remaining > 0:
    # assign remaining based on largest fractional remainders
    remainders = samples_float - samples_int
    indices = np.argsort(remainders)[-remaining:]
    samples_int[indices] += 1
elif remaining < 0:
    # remove extra samples from smallest fractional remainders
    remainders = samples_int - samples_float
    indices = np.argsort(remainders)[:abs(remaining)]
    samples_int[indices] -= 1

# finally, create a mapping from country name to number of samples
region_samples = dict(zip(regions_filtered.GID_1, samples_int))

# convert to pandas dataframe and save to a csv
df = pd.DataFrame(list(region_samples.items()), columns=["GID_1", "Samples"])
print("total:", df["Samples"].sum())
df.to_csv("dataset_creation/adm1_sample_counts2.csv", index=False)