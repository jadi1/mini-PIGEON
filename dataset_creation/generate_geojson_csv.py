import geopandas as gpd
import pandas as pd

# Paths
geojson_path = "data/geocells/admin_1.geojson"        # your ADM1 GeoJSON
# coords_csv_path = "data/coordinates12k.csv"       # your coordinates CSV
output_csv_path = "data/geocells.csv"             # output for PIGEON

# Load ADM1 GeoJSON
gdf = gpd.read_file(geojson_path)

# Ensure we have the right ID column. Adjust if your GeoJSON uses a different property.
# Here we assume 'admin_1_id' is the unique ID
gdf = gdf[['admin_1_id', 'geometry']].copy()
gdf = gdf.rename(columns={'admin_1_id': 'cell_id', 'geometry': 'polygon'})

# Project to metric CRS
gdf_proj = gdf.set_geometry('polygon').to_crs(epsg=3857)
gdf_proj['centroid'] = gdf_proj['polygon'].centroid

# Convert centroids back to lat/lon
gdf['centroid'] = gdf_proj['centroid'].to_crs(epsg=4326)
gdf['lat'] = gdf['centroid'].y
gdf['lng'] = gdf['centroid'].x

# Load your coordinates CSV to know which ADM1 regions to keep
# coords_df = pd.read_csv(coords_csv_path)
# Extract unique regions from your dataset
# regions_to_keep = coords_df['region'].unique()

# Filter geocells to only those present in your dataset
# gdf = gdf[gdf['cell_id'].isin(regions_to_keep)]

# Convert geometry to WKT (text format)
gdf['polygon'] = gdf['polygon'].apply(lambda x: x.wkt)

# Drop temporary centroid column
gdf = gdf.drop(columns=['centroid'])

# Save to CSV
gdf.to_csv(output_csv_path, index=False)

print(f"Saved {len(gdf)} geocells to {output_csv_path}")
