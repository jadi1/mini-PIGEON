import os
import torch
import pandas as pd
import numpy as np
import pickle
from model import PigeonModel
from tqdm import tqdm
from transformers import TrainingArguments
from datasets import load_from_disk
from MY_OWN_PIPELINE.pigeonrefiner import PigeonRefiner
from MY_OWN_PIPELINE.utils import haversine
from torch.utils.data import DataLoader

EVAL_ARGS = TrainingArguments(
    output_dir="saved_models",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # num_train_epochs=200, # set large, but train until convergence
    evaluation_strategy="epoch",
    eval_steps=5,
    save_strategy='epoch',
    save_steps=2,
    # learning_rate=1e-3, 
    # logging_steps=1,
    load_best_model_at_end=True,
    seed=330
)

def evaluate_model_with_embeddings(model, test_dataset, test_embeddings_path,
                   eval_args: TrainingArguments, refiner: PigeonRefiner=None, model_checkpoint_path="", save_path="results/eval_results.csv") -> dict:
    """Evaluates model on evaluation data

#     Returns:
#         Dict with evaluation metrics:
#       - loss: Test loss
#       - accuracy_@1km: % predictions within 1km
#       - accuracy_@25km: % predictions within 25km
#       - accuracy_@200km: % predictions within 200km
#       - accuracy_@750km: % predictions within 750km
#       - median_error_km: Median error in km
#       - mean_error_km: Mean error in km
#     """
    device = next(model.parameters()).device
    
    # Load checkpoint if provided
    if model_checkpoint_path and os.path.isfile(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"LOADED CHECKPOINT from {model_checkpoint_path}")

    # Load precomputed embeddings
    test_embeddings = torch.from_numpy(np.load(test_embeddings_path)).float().to(device)

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=eval_args.per_device_eval_batch_size,
        shuffle=False, pin_memory=True
    )

    results_list = []
    model.eval()
    all_errors = []
    
    # Metrics at different distance thresholds
    distance_thresholds = {
        1: 0,      # 1 km
        25: 0,     # 25 km (city level)
        200: 0,    # 200 km (region level)
        750: 0,    # 750 km (country level)
        2500: 0    # 2500 km (continent level)
    }
    embedding_idx = 0 # track position in embeddings

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            batch_size = batch['embedding'].size(0)

            # Get precomputed embeddings for this batch
            embeddings = test_embeddings[embedding_idx:embedding_idx + batch_size]
            embedding_idx += batch_size

            # Classify embeddings directly (skip forward through CLIP)
            logits = model.classifier(embeddings)
            
            geocell_probs = torch.softmax(logits, dim=-1)
            pred_coords = model.geocell_centroids[torch.argmax(geocell_probs, dim=-1)]
            top_probs, top_indices = torch.topk(geocell_probs, k=5, dim=-1)

            # Refine with PigeonRefiner
            if refiner is not None:
                print("Refining coordinates...")
                refined_lats, refined_lons, refined_geocells = refiner(
                    embedding=embeddings,
                    initial_preds=pred_coords,
                    candidate_cells=top_indices,  # indices from topk
                    candidate_probs=top_probs   # probs from topk
                )
                final_preds = torch.stack([refined_lats, refined_lons], dim=1)
                final_geocells = refined_geocells
            else:
                print("No refinement, using direct coordinates")
                final_preds = pred_coords
                final_geocells = top_indices[:,0]
            
            # True coordinates
            true_coords = batch['labels'] 
            
            # Compute errors
            errors = haversine(true_coords, final_preds) 
            
            all_errors.extend(errors.cpu().numpy())
            
            # Save per-sample info
            for i in range(final_preds.size(0)):
                row = {
                    "true_lat": batch['labels'][i, 0].item(),
                    "true_lng": batch['labels'][i, 1].item(),
                    "pred_lat": final_preds[i, 0].item(),
                    "pred_lng": final_preds[i, 1].item(),
                    "true_geocell": batch['labels_clf'][i].item(),
                    "pred_geocell": final_geocells[i].item(),
                    "error_km": errors[i].item()
                }
                # Distance threshold flags
                for t in distance_thresholds:
                    row[f"within_{t}km"] = int(errors[i].item() <= t)

                results_list.append(row)

            # Count predictions within thresholds
            for threshold in distance_thresholds.keys():
                distance_thresholds[threshold] += (errors <= threshold).sum().item()

    # save results to csv
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(save_path, index=False)

    # Compute final metrics
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

def print_eval_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Accuracy at different distance thresholds:")
    print(f"  @ 1 km (street):     {results['accuracy_@1km']:6.2f}%")
    print(f"  @ 25 km (city):      {results['accuracy_@25km']:6.2f}%")
    print(f"  @ 200 km (region):   {results['accuracy_@200km']:6.2f}%")
    print(f"  @ 750 km (country):  {results['accuracy_@750km']:6.2f}%")
    print(f"  @ 2500 km (cont.):   {results['accuracy_@2500km']:6.2f}%\n")
    print(f"Distance Errors:")
    print(f"  Median: {results['median_error_km']:8.2f} km")
    print(f"  Mean:   {results['mean_error_km']:8.2f} km")
    print(f"  Std:    {results['std_error_km']:8.2f} km")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Load prototypes and metadata dataset
    prototypes_csv = "data/prototypes_from_geocells.csv"
    metadata_df = pd.read_csv("data/metadata_with_geocells.csv", index_col=0)

    # Load test_dataset
    test_dataset = load_from_disk("data/hf_geoguessr_finetune_processed_b32/test")

    # Load embeddings
    train_embeddings_path = "data/embeddings_b32/embeddings_train.npy"
    test_embeddings_path = "data/embeddings_b32/embeddings_test.npy"

    # load geocells
    with open("data/geocells.pt", "rb") as f:
        data = pickle.load(f)
    geocell_centroids = data["geocell_centroids"]
    num_geocells = len(geocell_centroids)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize model
    model = PigeonModel(num_geocells=num_geocells,
                        geocell_centroids=geocell_centroids,
                        smooth_labels=True,
                        embedding_dim=768, # embedding dim for both b-32 and b-16 models
                        device=device)
    
    # Initialize refiner
    refiner = PigeonRefiner(
        proto_csv=prototypes_csv,
        dataset_df=metadata_df,
        embeddings_npy=train_embeddings_path, # must be path to train embeddings
        topk=5,
        temperature=1.6,
        max_refinement=1000,
        device=device
    )
    
    checkpoint_path = "saved_models/finetuned_b32_lr3e-4_185epochs/checkpoint_epoch180.pt"

    results = evaluate_model_with_embeddings(model=model,
                             test_dataset=test_dataset,
                             test_embeddings_path=test_embeddings_path,
                             eval_args=EVAL_ARGS,
                             refiner=None,
                             model_checkpoint_path=checkpoint_path,
                             save_path="results/eval_results_b32_lr_3e-4nowefaef.csv")
    
    print_eval_results(results)