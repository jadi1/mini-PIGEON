# using the google streetview metadata api, generate VALID coordinates for each country
import geopandas as gpd
import pandas as pd
from utils import query_streetview, random_point_in_polygon

df = pd.read_csv("data_subtracted_clipped.csv")
regions = gpd.read_file("data/gadm/countries_adm1_geometries.gpkg") # in equal area coords, not lat/lng
regions = regions.to_crs("EPSG:4326") # convert to lat/lng coords

coordinates = pd.DataFrame(columns=['ADM1_Region', 'Latitude', 'Longitude'])

# for each adm1 region
for i in range(0, len(regions)):
    row = regions.iloc[i]
    region = row.GID_1
    exists = region in df["GID_1"].values

    samples = 0

    # if region exists, calculate corresponding number of coordinates!
    if exists:
        samples = df.loc[df["GID_1"] == region, "Differences"].iloc[0]
        print(f"Generating {samples} samples for {region}")
    else:
        continue # otherwise, move on to next region

    geom = row.geometry # geometry for this region

    # k counts total number of tries
    k = 0
    for i in range(samples):
        while True:
            k += 1

            # check if random point gets streetview data
            point = random_point_in_polygon(geom)
            lat, lng = point.y, point.x

            # if point is invalid, just try again
            if point is None:
                continue

            # otherwise, check if the point has streetview data
            print(lat, lng)
            has_streetview_data = query_streetview(lat,lng) # lat, lng
            
            # if so, add new coordinate to dataframe and break
            if has_streetview_data:
                coordinates.loc[len(coordinates)] = [region, point.y, point.x]          
                break
    coordinates.to_csv("data/coordinates_data_added",index=False)
    print(f"Generating {samples} samples for {region} took {k} tries")
