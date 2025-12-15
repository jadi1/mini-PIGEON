import numpy as np
import torch
import pandas as pd
from sklearn.cluster import KMeans
import pickle

SEED = 42 # random seed

def create_hierarchical_geocells(csv_path, k1, k2, save_path = "data/geocells.pt"):
    """
    Instead of using optics, we use k-means clustering for both high level geocells and for location clusters within each geocell

    Returns:
    - geocell_centroids: Tensor (K1, 2) - center of each geocell
    - location_clusters: Dict {geocell_idx: Tensor(K2, 2)} - subclusters within each geocell
    - geocell_to_sublabels: Dict {geocell_idx: Array} - mapping of training points to subclusters
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter only training points
    df = df[df['selection'] == 'train'].reset_index(drop=True)
    
    latitudes = df['lat'].values
    longitudes = df['lng'].values

    print(f"Generating {k1} geocells with {k2} location clusters each using K-Means...")
    coords = np.stack([latitudes, longitudes], axis=1)

    # global geocells
    kmeans1 = KMeans(n_clusters=k1, random_state=SEED, n_init=10)
    geocell_labels = kmeans1.fit_predict(coords)
    geocell_centroids = torch.tensor(kmeans1.cluster_centers_, dtype=torch.float)

    # subclusters within each geocell
    location_clusters = {}
    geocell_to_sublabels = {}

    for g in range(k1):
        mask = geocell_labels == g
        points_in_g = coords[mask]

        if points_in_g.shape[0] == 0:
            continue

        # decide how many subclusters to make in this cell, can't exceed the number of points in this cell
        k_sub = min(k2, points_in_g.shape[0])
        if k_sub == 0:
            continue # Skip if no points for sub-clustering
        if k_sub == 1: # KMeans requires n_clusters > 1 unless n_samples = 1
             # If there's only one point, it forms its own cluster
            location_clusters[g] = torch.tensor([points_in_g[0]], dtype=torch.float)
            geocell_to_sublabels[g] = np.array([0]) # Assign to first (only) subcluster
            continue

        # Cluster within geocell
        kmeans2 = KMeans(n_clusters=k_sub, random_state=SEED, n_init=5)
        sublabels = kmeans2.fit_predict(points_in_g)

        # Store subcluster centroids
        location_clusters[g] = torch.tensor(kmeans2.cluster_centers_, dtype=torch.float)
        geocell_to_sublabels[g] = sublabels

    print(f"Created {len(location_clusters)} geocells with location clusters")
    
    # Save everything to file
    with open(save_path, "wb") as f:
        pickle.dump({
            "geocell_centroids": geocell_centroids,
            "location_clusters": location_clusters,
            "geocell_to_sublabels": geocell_to_sublabels
        }, f)
    print(f"Geocells saved to {save_path}")

create_hierarchical_geocells("data/metadata.csv", k1=900, k2=5)
# roughly 900 geocell clusters, and 5 subclusters within each, so about 4-5 points per subcluster