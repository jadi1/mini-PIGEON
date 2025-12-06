import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# import fiona
# print(fiona.listlayers("data/gadm/gadm_410-levels.gpkg"))

# checks if valid, non-empty geometries
def is_good_geometry(geom):
    return geom is not None and not geom.is_empty and geom.is_valid and geom.geom_type in [Polygon, MultiPolygon]

# safe dissolve function
def safe_dissolve(gdf, group_col):
    merged_geoms = []
    group_names = []

    for name, group in gdf.groupby(group_col):
        # filter only good geometries inside the group
        geoms = [geom for geom in group.geometry if is_good_geometry(geom)]
        if not geoms:
            continue  # skip groups with no valid geometry
        merged_geom = unary_union(geoms)
        merged_geoms.append(merged_geom)
        group_names.append(name)

    merged_gdf = gpd.GeoDataFrame({group_col: group_names, 'geometry': merged_geoms}, crs=gdf.crs)
    return merged_gdf

# read file
countries = gpd.read_file("data/gadm/gadm_410-levels.gpkg", layer="ADM_1")

countries = countries[countries['geometry'].apply(is_good_geometry)]

# project to Mollweide (meters)
countries = countries.to_crs("ESRI:54009")


# # dissolve safely
# countries_merged = safe_dissolve(countries, 'COUNTRY')

# print(countries_merged)

# # save to GPKG
countries.to_file("data/gadm/countries_adm1_geometries.gpkg", driver="GPKG")
