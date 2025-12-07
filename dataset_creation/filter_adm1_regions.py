# filter out which regions aren't supported by google maps api
import geopandas as gpd
import pandas as pd
from utils import query_streetview, random_point_in_polygon

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("MAPS_API_KEY")


regions = gpd.read_file("data/gadm/countries_adm1_geometries.gpkg") # in equal area coords, not lat/lng
regions = regions.to_crs("EPSG:4326") # convert to lat/lng coords

# Resume logic: load already-processed CSV if it exists
csv_file = "dataset_creation/filtered_adm1_progress.csv"
if os.path.exists(csv_file):
    processed_df = pd.read_csv(csv_file)
    processed_regions = set(processed_df["GID_1"])
else:
    processed_df = pd.DataFrame(columns=['GID_1','Country'])
    processed_regions = set()

to_drop = []

# for each adm1 region
for row in regions.itertuples():
    adm1_name = row.GID_1
    if adm1_name in processed_regions:
        continue  # skip, already processed
    geom = row.geometry
    print(f"Checking {adm1_name} in {row.COUNTRY}")

    found_streetview = False # flag for if you found a supported pano
    for i in range(20): # try to generate 20 samples. if none are supported, eliminate this region
        # check if random point gets streetview data
        point = random_point_in_polygon(geom) # generate random point

        # if there is no random point, just break
        if point is None:
            print("Geometry invalid")
            break

        has_streetview_data = query_streetview(point.y,point.x) # lat, lng
        status = ''
        # if so, break
        if has_streetview_data:
            found_streetview = True # you've found a sample!
            processed_df = pd.concat([processed_df, pd.DataFrame([{'GID_1': row.GID_1, 'Country': row.COUNTRY, 'Status': "OK"}])], ignore_index=True)
            break
        

        # if not, try again.
    if not found_streetview: # if you were unable to find any samples
        # delete row
        processed_df = pd.concat([processed_df, pd.DataFrame([{'GID_1': row.GID_1, 'Country': row.COUNTRY, 'Status': "NO"}])], ignore_index=True)
        to_drop.append(adm1_name)

    # Save after each region to resume safely
    processed_df.to_csv(csv_file, index=False)
# at the end, delete all regions that are to be dropped
print(to_drop)
print("dropped ", len(to_drop))