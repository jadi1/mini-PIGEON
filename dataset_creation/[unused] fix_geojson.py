import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

from shapely.geometry import Point, LineString, Polygon, MultiPolygon

def remove_z(geom):
    """Remove Z coordinates from a geometry, recursively."""
    if geom is None:
        return None
    if geom.geom_type == "Point":
        x, y = geom.xy[0][0], geom.xy[1][0]
        return Point(x, y)
    if geom.geom_type == "LineString":
        return LineString([(x, y) for x, y, *rest in geom.coords])
    if geom.geom_type == "Polygon":
        exterior = [(x, y) for x, y, *rest in geom.exterior.coords]
        interiors = [[(x, y) for x, y, *rest in interior.coords] for interior in geom.interiors]
        return Polygon(exterior, interiors)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([remove_z(p) for p in geom.geoms])
    return geom  # fallback for other types


# Load your GeoJSON
gdf = gpd.read_file("data/geocells/countries.geojson")

# Apply Z-removal
gdf["geometry"] = gdf["geometry"].apply(remove_z)

# Save cleaned GeoJSON
gdf.to_file("data/geocells/countries_new.geojson", driver="GeoJSON")

print("Cleaned geometries saved to countries_new.geojson")
