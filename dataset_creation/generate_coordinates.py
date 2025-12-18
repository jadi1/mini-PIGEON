# using the google streetview metadata api, generate VALID coordinates for each country
import geopandas as gpd
import pandas as pd
from utils import query_streetview, random_point_in_polygon

# read data
df = pd.read_csv("adm1_sample_counts_final.csv")
regions = gpd.read_file("data/gadm/countries_adm1_geometries.gpkg") # in equal area coords, not lat/lng
regions = regions.to_crs("EPSG:4326") # convert to lat/lng coords

coordinates = pd.read_csv("data/coordinates_data_final.csv") # load existing df

# for each adm1 region
for i in range(0, len(regions)):
    row = regions.iloc[i]
    region = row.GID_1
    exists = region in df["GID_1"].values

    samples = 0

    # if region exists, calculate corresponding number of coordinates!
    if exists:
        samples = df.loc[df["GID_1"] == region, "Samples"].iloc[0]
        print(f"Generating {samples} samples for {region}")
    else:
        continue # otherwise, move on to next region

    already_done = coordinates[coordinates["ADM1_Region"] == region].shape[0]
    remaining = samples - already_done

    if remaining <= 0:
        print(f"{region} already complete, skipping")
        continue

    geom = row.geometry # geometry for this region

    # k counts total number of tries
    k = 0
    for j in range(remaining):
        while True:
            k += 1

            # check if random point gets streetview data
            point = random_point_in_polygon(geom)
            
            # if point is invalid, just try again
            if point is None:
                continue
            lat, lng = point.y, point.x

            # otherwise, check if the point has streetview data
            print(lat, lng)
            has_streetview_data = query_streetview(lat,lng) # lat, lng
            
            # if so, add new coordinate to dataframe and break
            if has_streetview_data:
                coordinates.loc[len(coordinates)] = [region, point.y, point.x]          
                break
    coordinates.to_csv("data/coordinates_data_final.csv", index=False)
    print(f"Generating {samples} samples for {region} took {k} tries")
