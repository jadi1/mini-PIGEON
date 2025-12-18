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
    df = pd.read_csv(csv_path)
    
    # filter only training points
    train_df = df[df['selection'] == 'train']
    
    latitudes = train_df['lat'].values
    longitudes = train_df['lng'].values

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
        geocell_mask = (geocell_labels == g)
        indices_in_g = np.where(geocell_mask)[0]
        points_in_g = coords[geocell_mask]

        if points_in_g.shape[0] == 0:
            continue

        # decide how many subclusters to make in this cell, can't exceed the number of points in this cell
        k_sub = min(k2, len(points_in_g))
        if k_sub == 0:
            continue # Skip if no points for sub-clustering
        if k_sub == 1:
             # if there's only one point, it forms its own cluster
            location_clusters[g] = torch.tensor([points_in_g[0]], dtype=torch.float)
            geocell_to_sublabels[g] =  {
                'sublabels': np.array([0]),
                'indices': indices_in_g # original dataset indices
            }
            continue

        # cluster within geocell
        kmeans2 = KMeans(n_clusters=k_sub, random_state=SEED, n_init=5)
        sublabels = kmeans2.fit_predict(points_in_g)

        # store subcluster centroids
        location_clusters[g] = torch.tensor(kmeans2.cluster_centers_, dtype=torch.float)
        geocell_to_sublabels[g] = {
            'sublabels': sublabels,
            'indices': indices_in_g  # original indices
        }
 
    # save everything to file
    with open(save_path, "wb") as f:
        pickle.dump({
            "geocell_centroids": geocell_centroids,
            "location_clusters": location_clusters,
            "geocell_to_sublabels": geocell_to_sublabels
        }, f)
    print(f"Geocells saved to {save_path}")

create_hierarchical_geocells("data/metadata.csv", k1=900, k2=5)
# roughly 900 geocell clusters, and 5 subclusters within each, so about 4-5 points per subcluster

df = pd.read_csv("data/metadata_with_geocells.csv", index_col=0)
df['numeric_idx'] = range(len(df))  # add numeric index to full df

df.to_csv("data/metadata_with_geocells2.csv")