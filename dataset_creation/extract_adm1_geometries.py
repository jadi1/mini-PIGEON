import geopandas as gpd

# read file
countries = gpd.read_file("data/gadm/gadm_410-levels.gpkg", layer="ADM_1")

# project to Mollweide (meters)
countries = countries.to_crs("ESRI:54009")

# save to GPKG
countries.to_file("data/gadm/countries_adm1_geometries.gpkg", driver="GPKG")
