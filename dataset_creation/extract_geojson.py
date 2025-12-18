import geopandas as gpd
import os

# load geojson
input_file = "data/geocells/geoBoundariesCGAZ_ADM2.geojson"
gdf = gpd.read_file(input_file)

# output dir
out_dir = "data/geocells"
os.makedirs(out_dir, exist_ok=True)

# create country-level geojson
countries = gdf[['geometry', 'shapeGroup']].drop_duplicates(subset='shapeGroup').copy()
countries = countries.rename(columns={'shapeGroup': 'country_id'})
countries.to_file(os.path.join(out_dir, "countries.geojson"), driver="GeoJSON")
print("... saved countries.geojson")

# create admin1-level geojson
admin1 = gdf[gdf['shapeType'] == 'ADM1'].copy()
admin1 = admin1.rename(columns={'shapeGroup': 'country_id', 'shapeID': 'admin_1_id'})
admin1 = admin1[['geometry', 'country_id', 'admin_1_id']]
admin1.to_file(os.path.join(out_dir, "admin_1.geojson"), driver="GeoJSON")
print("... saved admin_1.geojson")

# create admin2-level geojson
admin2 = gdf[gdf['shapeType'] == 'ADM2'].copy()
admin2 = admin2.rename(columns={'shapeGroup': 'country_id', 'shapeID': 'admin_2_id'})
admin2 = admin2[['geometry', 'country_id', 'admin_2_id']]
admin2.to_file(os.path.join(out_dir, "admin_2.geojson"), driver="GeoJSON")
print("... saved admin_2.geojson")
