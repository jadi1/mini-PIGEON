import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor
from typing import Dict, List, Any
from datasets import DatasetDict
from transformers import AutoFeatureExtractor
from config import *
from functools import partial
from geo_utils import haversine
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
    geocell_id = torch.argmin(dists, dim=1).item()

    label_dict = {
        "labels": np.array([lng, lat]),
        "labels_clf": geocell_id
    }

    # assign to nearest subcluster within geocell
    if location_clusters is not None and geocell_id in location_clusters:
        sub_centroids = location_clusters[geocell_id]
        sub_dists = haversine(point, sub_centroids)
        sub_id = torch.argmin(sub_dists, dim=1).item()
        label_dict["labels_sub"] = sub_id

    return label_dict

def add_embeddings(example: Dict, index: int, emb_array: np.ndarray) -> np.ndarray:
    """Adds embeddings for the given image."""
    return {
        'embedding': emb_array[index]
    }

def load_geocells_centroid(path: str):
    data = torch.load(path, map_location="cpu")
    return data["geocell_centroids"], data["location_clusters"]

def preprocess(dataset: DatasetDict, geocell_path: str) -> DatasetDict:
    """Preproccesses image dataset for vision input

    Args:
        dataset (DatasetDict): image dataset.
        geocell_path (str): path to geocells.
        embedder (CLIPEmbedding): CLIP embedding model. Defaults to None.
        multi_task (bool, optional): if labels for multi-task setup should be generated.
            Defaults to False.

    Returns:
        DatasetDict: transformed dataset
    """
    # Load geocells
    geocell_centroids, location_clusters = load_geocells_centroid(geocell_path)

    splits = ["train", "val", "test"]

    for split in splits:
        ds = load_from_disk(f"data/hf_geoguessr_finetune/{split}")
        embeddings = torch.from_numpy(np.load(f"data/embeddings_{split}.npy"))
        
        ds = ds.map(partial(add_embeddings, emb_array=embeddings), with_indices=True)
        ds = ds.map(lambda x: generate_label_cells_centroid(x, geocell_centroids, location_clusters))
        ds = ds.remove_columns(['lng', 'lat', 'image', 'image_2', 'image_3', 'image_4'])
        ds = ds.with_format('torch')
