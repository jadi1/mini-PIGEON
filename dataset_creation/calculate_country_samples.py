# map each country to number of samples proportional to area
import geopandas as gpd
import numpy as np
import pandas as pd

countries = gpd.read_file("data/gadm/countries_merged_geometries.gpkg")
print(countries[countries['COUNTRY'] == 'India'])
print(countries[countries['COUNTRY'] == 'China'])

N = 10000 # total number of samples

# sum the area for each country 
country_areas = countries.geometry.area

# print(country_areas)
# compute area fractions
fractions = country_areas / country_areas.sum()
# print(fractions)

# round down to nearest int
samples_float = fractions * N
samples_float = np.maximum(samples_float, np.eye(1)) # ensure at least 1 sample per country
samples_int = np.floor(samples_float).astype(int)
samples_int = samples_int[0]
print(samples_int)

remaining = N - samples_int.sum() # remaining samples to be distributed
print("remaining:", remaining)
if remaining > 0:
    # assign remaining samples to countries with largest fractional remainder
    remainders = samples_float - samples_int
    indices = np.argsort(remainders)[-remaining:]
    samples_int[indices] += 1

print(len(countries.COUNTRY), len(samples_int))

# problems??
# finally, create a mapping from country name to number of samples
country_samples = dict(zip(countries.COUNTRY, samples_int))

print(countries.COUNTRY)
# # check
print(country_samples)

# convert to pandas dataframe and save to a csv
df = pd.DataFrame(list(country_samples.items()), columns=["Country", "Samples"])
df.to_csv("dataset_creation/country_sample_counts.csv", index=False)