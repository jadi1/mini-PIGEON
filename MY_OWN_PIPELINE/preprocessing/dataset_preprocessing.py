import torch
import pickle
import numpy as np
from torch import Tensor
from typing import Dict
from datasets import DatasetDict
from functools import partial
from MY_OWN_PIPELINE.utils import haversine
from datasets import load_from_disk

def generate_label_cells_centroid(
    example: Dict,
    geocell_centroids: Tensor,
    location_clusters: Dict[int, Tensor] = None
) -> Dict:
    """
    Centroid-based replacement for PIGEON polygon labeling
    """

    lng, lat = example["lng"], example["lat"]
    point = torch.tensor([[lat, lng]], dtype=torch.float)

    # assign to nearest geocell
    dists = haversine(point, geocell_centroids)
    print(dists.shape)
    print(dists[:5])
    geocell_id = torch.argmin(dists).item()

    label_dict = {
        "labels": np.array([lat, lng]),
        "labels_clf": geocell_id
    }

    # assign to nearest subcluster within geocell
    if location_clusters is not None and geocell_id in location_clusters:
        sub_centroids = location_clusters[geocell_id]
        sub_dists = haversine(point, sub_centroids)
        sub_id = torch.argmin(sub_dists).item()
        label_dict["labels_sub"] = sub_id

    return label_dict

def add_embeddings(example: Dict, index: int, emb_array: np.ndarray) -> np.ndarray:
    """Adds embeddings for the given image."""
    return {
        'embedding': emb_array[index]
    }

def load_geocells_centroid(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    geocell_centroids = data["geocell_centroids"]
    location_clusters = data["location_clusters"]
    return geocell_centroids, location_clusters

"""
After running preprocess, each split is a hugging face dataset that contains
#     "embedding": Tensor(1024)       # averaged panorama embedding
#     "labels": Tensor([lat, lng])    # original coordinates
#     "labels_clf": int                # geocell ID
#     "labels_sub": int (optional)    # hierarchical subcluster ID
""" 
def preprocess(dataset_path: str, geocell_path: str) -> DatasetDict:
    """Preproccesses image dataset for vision input

    Args:
        dataset (DatasetDict): image dataset.
        geocell_path (str): path to geocells.

    Returns:
        DatasetDict: transformed dataset
    """
    # Load geocells
    geocell_centroids, location_clusters = load_geocells_centroid(geocell_path)

    splits = ["train", "val", "test"]
    processed_splits = {}

    for split in splits:
        ds = load_from_disk(f"{dataset_path}/{split}")
        embeddings = torch.from_numpy(np.load(f"data/embeddings_{split}.npy"))
        
        ds = ds.map(partial(add_embeddings, emb_array=embeddings), with_indices=True)
        ds = ds.map(lambda x: generate_label_cells_centroid(x, geocell_centroids, location_clusters))
        ds = ds.remove_columns(['lng', 'lat', 'image', 'image_2', 'image_3', 'image_4'])
        ds = ds.with_format('torch')
        processed_splits[split] = ds

    processed_dataset = DatasetDict(processed_splits)
    return processed_dataset

if __name__ == '__main__':
    dataset_path = "data/hf_geoguessr_finetune"
    geocell_path = "data/geocells.pt"  # or wherever your geocell centroids are stored

    processed_dataset = preprocess(dataset_path, geocell_path)

    # save processed splits back to disk
    for split, ds in processed_dataset.items():
        ds.save_to_disk(f"{dataset_path}_processed/{split}")
        print(f"Saved processed {split} split")