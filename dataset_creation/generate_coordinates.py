# using the google streetview metadata api, generate VALID coordinates for each country
import geopandas as gpd
import pandas as pd
import random
import requests
from shapely.geometry import Point, Polygon, MultiPolygon

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("MAPS_API_KEY")

# returns a random point (lat,lng) in a polygon or multipolygon
def random_point_in_polygon(polygon):
    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    else:
        raise ValueError("Input must be a Polygon or MultiPolygon")
    
    # loops until it returns a real valid point
    while True:
        # pick a polygon if MultiPolygon
        poly = random.choice(polygons)
        minx, miny, maxx, maxy = poly.bounds
        # sample random point in bounding box
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p
        
def query_streetview(lat,lng):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    if data["status"] == "OK":
        print(f"Street View exists! Date: {data['date']}, Pano ID: {data['pano_id']}")
    else:
        print("No Street View imagery at this location")

# FILE STARTS HERE
df = pd.read_csv("dataset_creation/country_sample_counts.csv")
countries = gpd.read_file("data/gadm/countries_merged_geometries.gpkg") # in equal area coords, not lat/lng
countries = countries.to_crs("EPSG:4326") # convert to lat/lng coords

coordinates = pd.DataFrame(columns=['Country', 'Latitude', 'Longitude'])

# for each country
for index, row in df.iterrows():
    country = row["Country"]
    samples = row["Samples"]
    print(f"Generating samples for {country}")

    # geometry row for this country
    country_geom = countries.loc[countries["COUNTRY"] == country, "geometry"].values[0]

    # k counts total number of tries
    k = 0
    for i in range(samples):
        found = False # flag for if you've found a supported pano
        unsupported = False
        n_tries = 0 # counts number of tries for this sample
        while True:
            n_tries += 1
            k += 1

            # check if random point gets streetview data
            point = random_point_in_polygon(country_geom)
            print(point.y, point.x)
            has_streetview_data = query_streetview(point.y,point.x) # lat, lng
            
            # if so, add new row to dataframe and break
            if has_streetview_data:
                coordinates.loc[len(coordinates)] = [country, point.y, point.x]
                coordinates.to_csv("data/coordinates_data",index=False)
                found = True # you've found a sample!
                break

            # if not, try again.
            if n_tries > 50 and found == False: # if you've generated 50 samples and they're all unsupported, mark country unsupported
                coordinates.loc[len(coordinates)] = [country, -1, -1]
                coordinates.to_csv("data/coordinates_data",index=False)
                unsupported = True
                break
        if unsupported == True:
            break

    print(f"Generating {samples} samples for {country} took {k} tries")

# at the end, save to csv file
# coordinates.to_csv("data/coordinates_data",index=False)