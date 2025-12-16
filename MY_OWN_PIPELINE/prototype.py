import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from preprocessing.geo_utils import haversine


def generate_prototypes(df, geocell_to_sublabels):
    prototypes = []

    for geocell_idx, data in geocell_to_sublabels.items():
        sublabels = data['sublabels']  # Array of subcluster assignments
        indices_in_g = data['indices']  # Original df indices for this geocell
        
        # Get all points in this geocell
        points_in_g = df.iloc[indices_in_g]

        assert len(points_in_g) == len(sublabels)

        for sub_id in np.unique(sublabels):
            mask = (sublabels == sub_id)

            sub_indices = indices_in_g[mask]  # original indices
            points_in_sub = df.iloc[sub_indices]

            if len(points_in_sub) == 0:
                continue

            proto_lat = points_in_sub['lat'].mean()
            proto_lng = points_in_sub['lng'].mean()

            prototypes.append({
                "geocell_idx": geocell_idx,
                "subcluster": sub_id,
                "lat": proto_lat,
                "lng": proto_lng,
                "count": len(points_in_sub),
                "indices": sub_indices.tolist()
            })

    prototypes_df = pd.DataFrame(prototypes)
    prototypes_df.to_csv("data/prototypes_from_geocells.csv", index=False)
    return prototypes_df


if __name__ == '__main__':
    with open("data/geocells.pt", "rb") as f:
        data = pickle.load(f)

    geocell_to_sublabels = data["geocell_to_sublabels"]

    # # Load your dataset
    df = pd.read_csv("data/metadata_with_geocells2.csv", index_col=0)
    
    train_df = df[df['selection'] == 'train']
    
    prototypes_df = generate_prototypes(train_df, geocell_to_sublabels)
    print("\nPrototype distribution:")
    print(prototypes_df.groupby('geocell_idx').size().describe())
    print(f"\nTotal prototypes: {len(prototypes_df)}")
    print(prototypes_df.groupby('geocell_idx').size())