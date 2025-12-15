from os import path
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Image

def create_dataset_split(df: pd.DataFrame, shuffle: bool=False,
                         panorama: bool=False) -> Dataset:
    """Creates an image dataset from the given dataframe.

    Args:
        df (pd.DataFrame): metadata dataframe
        shuffle (bool): if dataset should be shuffled
        panorama (bool, optional): whether four images are passed in as a panorama.
            Defaults to False.

    Returns:
        Dataset: HuggingFace dataset
    """
    # rename columns to match existing implementation
    if panorama:
        df = df.rename(columns={
            'img1': 'image',
            'img2': 'image_2',
            'img3': 'image_3',
            'img4': 'image_4'
        })

    data_dict = {
        'image': df['image'].astype(str).tolist(),
        'lng': df['lng'].astype(float).tolist(),
        'lat': df['lat'].astype(float).tolist(),
        'index': df.index.astype(int).tolist()
    }

    if panorama:
        data_dict['image_2'] = df['image_2'].tolist()
        data_dict['image_3'] = df['image_3'].tolist()
        data_dict['image_4'] = df['image_4'].tolist()
        
    valid_indices = []
    for i in range(len(data_dict['image'])):
        has_nan = False
        for col in ['image', 'image_2', 'image_3', 'image_4']:
            if col in data_dict and not isinstance(data_dict[col][i], str):
                has_nan = True
                print("NAN")
                break
        if not has_nan:
            valid_indices.append(i)

    # Filter all columns to only valid rows
    data_dict = {k: [v[i] for i in valid_indices] for k, v in data_dict.items()}

    dataset = Dataset.from_dict(data_dict).cast_column('image', Image())
    if panorama:
        dataset = dataset.cast_column('image_2', Image())
        dataset = dataset.cast_column('image_3', Image())
        dataset = dataset.cast_column('image_4', Image())

    if shuffle:
        dataset = dataset.shuffle(seed=330)
    
    return dataset

def generate_finetune_dataset(metadata_path: str="data/metadata.csv") -> DatasetDict:
    """Generates the Geoguessr dataset

    Args:
        metadata_path (str, optional): metadata path. Defaults to METADATA_PATH.

    Returns:
        DatasetDict: HuggingFace DatasetDict
    """
    data_df = pd.read_csv(metadata_path)

    # Process image paths
    # image_cols = [x for x in data_df.columns if 'img' in x]

    panorama = True
    # for col in image_cols:
    #     data_df[col] = data_df[col].apply(lambda x: path.join(image_path, x))

    splits = []
    for split in ['train', 'val', 'test']:
        data_split = data_df.loc[data_df['selection'] == split]
        splits.append(create_dataset_split(data_split, panorama=panorama))

    dataset = DatasetDict(
        train=splits[0],
        val=splits[1],
        test=splits[2]
    )
    
    return dataset

if __name__ == '__main__':
    dataset = generate_finetune_dataset()
    print(dataset)

    save_path = "data/hf_geoguessr_finetune"
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")