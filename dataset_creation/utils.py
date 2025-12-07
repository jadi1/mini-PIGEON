import random
import requests
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("MAPS_API_KEY")


# try to fix geometry
def fix_or_none(g):
    if g is None or g.is_empty:
        return None

    # Try buffer(0)
    try:
        g = g.buffer(0)
    except Exception:
        return None

    if g.is_valid:
        return g

    # Try unary_union (handles self-crossing MultiPolygons)
    try:
        g = unary_union(g)
    except Exception:
        return None

    return g if g.is_valid else None

# returns a random point (lat,lng) in a polygon or multipolygon
def random_point_in_polygon(polygon):
    polygon = fix_or_none(polygon)
    if polygon is None:
        return None

    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    else:
        raise ValueError("Input must be a Polygon or MultiPolygon")
    
    # loop many times but do not hang
    for _ in range(1000):
        # pick a polygon if MultiPolygon
        poly = random.choice(polygons)
        if poly is None or poly.is_empty:
            continue

        minx, miny, maxx, maxy = poly.bounds
        if minx == maxx or miny == maxy:
            continue  # skip degenerate polygons

        # sample random point in bounding box
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        try:
            if poly.contains(p):
                return p
        except Exception:
            print("trying to generate another point")
            continue
    return None
        
def query_streetview(lat,lng):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    if data["status"] == "OK" and "Google" in data.get("copyright", ""):
        if data["date"] and data["pano_id"]:
            print(f"Street View exists! Date: {data['date']}, Pano ID: {data['pano_id']}")
        return True
    return False