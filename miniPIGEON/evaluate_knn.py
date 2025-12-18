# This file is only used to evaluate baselines (unfinetuned 
# CLIP ViT-B/32 and CLIP ViT-B/16) using k-nearest neighbors algorithm, with k=5.
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from transformers import TrainingArguments
from datasets import load_from_disk
from miniPIGEON.utils import haversine
from torch.utils.data import DataLoader

EVAL_ARGS = TrainingArguments(
    output_dir="saved_models",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    seed=330
)

def evaluate_knn(k_neighbors, metadata_df, test_dataset, test_embeddings_path, train_embeddings_path,
                   eval_args: TrainingArguments, save_path="results/eval_knn_results.csv") -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load test embeddings
    test_embeddings = torch.from_numpy(np.load(test_embeddings_path)).float().to(device)

    # create test dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=eval_args.per_device_eval_batch_size,
        shuffle=False, pin_memory=True
    )

    results_list = []
    all_errors = []
    
    # metrics at different distance thresholds
    distance_thresholds = {
        1: 0,      # 1 km
        25: 0,     # 25 km 
        200: 0,    # 200 km 
        750: 0,    # 750 km
        2500: 0    # 2500 km
    }
    embedding_idx = 0 # track position in embeddings

    # load train embeddings
    train_embeddings = torch.from_numpy(np.load(train_embeddings_path)).float().to(device)

    train_metadata =  metadata_df[metadata_df["selection"] == "train"].reset_index(drop=True)

    # extract train coordinates
    train_coords = torch.tensor(
        train_metadata[["lat", "lng"]].values,
        dtype=torch.float,
        device=device
    )

    # normalize train embeddings
    train_embeddings = torch.nn.functional.normalize(train_embeddings, dim=-1)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            batch_size = batch['embedding'].size(0)

            # get embeddings for this batch
            embeddings = test_embeddings[embedding_idx:embedding_idx + batch_size]
            embedding_idx += batch_size

            # normalize test embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            # compute cosine similarity between all train and test embeddings
            similarities = embeddings @ train_embeddings.T

            # get top-k neighbors
            topk_vals, topk_idxs = torch.topk(similarities, k=k_neighbors, dim=1)
            neighbor_coords = train_coords[topk_idxs] # get coords

            # compute softmax to get weights
            weights = torch.softmax(topk_vals, dim=1).unsqueeze(-1)

            # get weighted average of coordinates
            final_preds = (neighbor_coords * weights).sum(dim=1)

            true_coords = batch['labels'] 
            
            # compute errors
            errors = haversine(true_coords, final_preds) 
            all_errors.extend(errors.cpu().numpy())
            
            # save per-sample info
            for i in range(final_preds.size(0)):
                row = {
                    "true_lat": batch['labels'][i, 0].item(),
                    "true_lng": batch['labels'][i, 1].item(),
                    "pred_lat": final_preds[i, 0].item(),
                    "pred_lng": final_preds[i, 1].item(),
                    "true_geocell": batch['labels_clf'][i].item(),
                    "error_km": errors[i].item()
                }

                for t in distance_thresholds:
                    row[f"within_{t}km"] = int(errors[i].item() <= t)

                results_list.append(row)

            # count predictions within thresholds
            for threshold in distance_thresholds.keys():
                distance_thresholds[threshold] += (errors <= threshold).sum().item()

    # save results to csv
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(save_path, index=False)

    # compute final metrics
    num_samples = len(test_loader.dataset)
    all_errors = np.array(all_errors)
    
    results = {
        "accuracy_@1km": distance_thresholds[1] / num_samples * 100,
        "accuracy_@25km": distance_thresholds[25] / num_samples * 100,
        "accuracy_@200km": distance_thresholds[200] / num_samples * 100,
        "accuracy_@750km": distance_thresholds[750] / num_samples * 100,
        "accuracy_@2500km": distance_thresholds[2500] / num_samples * 100,
        "median_error_km": float(np.median(all_errors)),
        "mean_error_km": float(np.mean(all_errors)),
        "std_error_km": float(np.std(all_errors)),
    }
    
    return results

if __name__ == "__main__":
    # load prototypes and metadata dataset
    prototypes_csv = "data/prototypes_from_geocells.csv"
    metadata_df = pd.read_csv("data/metadata_with_geocells.csv", index_col=0)

    # load test_dataset
    test_dataset = load_from_disk("data/hf_geoguessr_finetune_processed_b16/test")

    # load embeddings
    train_embeddings_path = "data/embeddings_b16/embeddings_train.npy"
    test_embeddings_path = "data/embeddings_b16/embeddings_test.npy"

    # load geocells
    with open("data/geocells.pt", "rb") as f:
        data = pickle.load(f)
    geocell_centroids = data["geocell_centroids"]
    num_geocells = len(geocell_centroids)
    
    results = evaluate_knn(k_neighbors=5, metadata_df=metadata_df,
                             test_dataset=test_dataset,
                             test_embeddings_path=test_embeddings_path,
                             train_embeddings_path=train_embeddings_path,
                             eval_args=EVAL_ARGS,
                             save_path="results/eval_results_b16_knn.csv")
    
    print(results)