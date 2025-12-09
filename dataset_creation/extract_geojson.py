# import geopandas as gpd

# # Path to your GADM GPKG file
# path = "data/geocells/geoBoundariesCGAZ_ADM2.geojson"

# gdf = gpd.read_file(path)

# print(gdf.columns)
# print(gdf.head())
# print(gdf.crs)
# # Load each layer
# # countries = gpd.read_file(gadm_file, layer="ADM_0")  # country-level
# # admin1 = gpd.read_file(gadm_file, layer="ADM_1")     # admin 1 level
# admin2 = gpd.read_file(gadm_file, layer="ADM_2")     # admin 2 level

# Save as GeoJSON for Pigeon
# countries.to_file("data/geocells/countries.geojson", driver="GeoJSON")
# admin1.to_file("data/geocells/admin_1.geojson", driver="GeoJSON")
# admin2.to_file("data/geocells/admin_2.geojson", driver="GeoJSON")
# import fiona
# from fiona.crs import CRS

# input_path = "data/geocells/geoBoundariesCGAZ_ADM2.geojson"

# # Output files we will create
# output_paths = {
#     "ADM0": "data/geocells/country.geojson",
#     "ADM1": "data/geocells/admin_1.geojson",
#     "ADM2": "data/geocells/admin_2.geojson",
# }

# # Prepare writers (empty dict for now)
# writers = {}

# with fiona.open(input_path, "r") as src:
#     src_crs = src.crs
#     src_schema = src.schema

#     # Create an empty output file for each ADM level
#     for adm_level, path in output_paths.items():
#         writers[adm_level] = fiona.open(
#             path,
#             "w",
#             driver="GeoJSON",
#             crs=src_crs,
#             schema=src_schema,
#         )

#     # Read and sort features into the right file
#     for feature in src:
#         shape_type = feature["properties"].get("shapeType")

#         if shape_type in writers:
#             writers[shape_type].write(feature)

# # Close writers
# for w in writers.values():
#     w.close()

# print("Finished splitting into ADM0, ADM1, ADM2.")
import geopandas as gpd
import os

# Input GeoJSON
input_file = "data/geocells/geoBoundariesCGAZ_ADM2.geojson"

# Output directory
out_dir = "data/geocells"
os.makedirs(out_dir, exist_ok=True)

# Load GeoJSON
gdf = gpd.read_file(input_file)

# --- Create Country-level GeoJSON ---
countries = gdf[['geometry', 'shapeGroup']].drop_duplicates(subset='shapeGroup').copy()
countries = countries.rename(columns={'shapeGroup': 'country_id'})
countries.to_file(os.path.join(out_dir, "countries.geojson"), driver="GeoJSON")
print("... saved countries.geojson")

# --- Create Admin1-level GeoJSON ---
admin1 = gdf[gdf['shapeType'] == 'ADM1'].copy()
admin1 = admin1.rename(columns={'shapeGroup': 'country_id', 'shapeID': 'admin_1_id'})
admin1 = admin1[['geometry', 'country_id', 'admin_1_id']]
admin1.to_file(os.path.join(out_dir, "admin_1.geojson"), driver="GeoJSON")
print("... saved admin_1.geojson")

# --- Create Admin2-level GeoJSON ---
admin2 = gdf[gdf['shapeType'] == 'ADM2'].copy()
admin2 = admin2.rename(columns={'shapeGroup': 'country_id', 'shapeID': 'admin_2_id'})
admin2 = admin2[['geometry', 'country_id', 'admin_2_id']]
admin2.to_file(os.path.join(out_dir, "admin_2.geojson"), driver="GeoJSON")
print("... saved admin_2.geojson")
