import folium
import pickle
import random
import torch
import numpy as np

def plot_geocells_folium(geocell_centroids: torch.Tensor, 
                         location_clusters: dict,
                         map_center=(0,0), zoom_start=2):
    """
    Creates an interactive map with hierarchical geocells.
    """
    m = folium.Map(location=map_center, zoom_start=zoom_start)
    
    geocell_centroids_np = geocell_centroids.numpy()
    
    for i, center in enumerate(geocell_centroids_np):
        random_color ="#{:06x}".format(random.randint(0, 0xFFFFFF))
        # Geocell centroid
        folium.CircleMarker(location=(center[0], center[1]), radius=5, 
                            color= random_color, fill=True).add_to(m)
        
        # Subclusters
        if i in location_clusters:
            subcenters = location_clusters[i].numpy()
            for sub in subcenters:
                folium.CircleMarker(location=(sub[0], sub[1]), radius=2,
                                    color=random_color, fill=True, fill_opacity=0.6).add_to(m)
    return m

# load geocell data
with open("data/geocells.pt", "rb") as f:
    data = pickle.load(f)
    geocell_centroids = data["geocell_centroids"]
    location_clusters = data["location_clusters"]
    geocell_to_sublabels = data["geocell_to_sublabels"]

# print(len(geocell_centroids))
# print(geocell_centroids[0])
# print(len(location_clusters))
# print(location_clusters[0])
# print(len(geocell_to_sublabels))
# print(geocell_to_sublabels[0])
# max_points_in_subcluster = 0
# for g, sublabels in geocell_to_sublabels.items():
#     unique, counts = np.unique(sublabels, return_counts=True)
#     max_points_in_subcluster = max(max_points_in_subcluster, counts.max())

# print("Max number of points in a subcluster:", max_points_in_subcluster)

# plot geocells
m = plot_geocells_folium(geocell_centroids, location_clusters, map_center=(20,0))
m.save("geocells_map.html")
