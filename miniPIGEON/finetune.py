import pickle
import torch
from transformers import TrainingArguments, \
                         AutoModelForImageClassification
from model import PigeonModel
from datasets import DatasetDict, load_from_disk
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, AutoModel 
from typing import Any
from tqdm import tqdm


TRAIN_ARGS = TrainingArguments(
    output_dir="saved_models",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1000, # set large, but train until convergence
    evaluation_strategy="epoch",
    eval_steps=5,
    save_strategy='epoch',
    save_steps=2,
    learning_rate=3e-4, # a little higher, since only training on precomputed embeddings
    logging_steps=1,
    load_best_model_at_end=True,
    seed=330
)


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_model(model: Any, train_dataset: DatasetDict, val_dataset:DatasetDict,
                train_args: TrainingArguments, patience: int=None, save_checkpoint_path: str="", load_checkpoint_path: str=None) -> AutoModel:
    """Training and evaluation loop for the model with multi-GPU support.

    Args:
        loaded_model (Any):              Model used for training.
        dataset (DatasetDict):           Dataset containing train, val, and test splits.
        train_args (TrainingArguments):  Training arguments.
        patience (int, optional):        Patience for early stopping. Defaults to None.
        
    Returns:
        AutoModel: Trained automodel
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate)

    start_epoch = 0
    best_val_loss = None

    # optional: load from checkpoint to continue training!
    if load_checkpoint_path is not None and os.path.isfile(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', None)
        print(f"Loaded checkpoint from {load_checkpoint_path}, starting at epoch {start_epoch}")

    train_loader = DataLoader(train_dataset, batch_size=train_args.per_device_train_batch_size, shuffle=True,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_args.per_device_eval_batch_size,
                            shuffle=False, pin_memory=True)

    current_patience = 0
    best_model_state = None
    if best_val_loss is not None:
        # If resuming from checkpoint, start with the checkpoint’s model
        best_model_state = {k: v.to(device) for k, v in model.state_dict().items()}

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, train_args.num_train_epochs):
        model.train()
        running_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_args.num_train_epochs}"):
            # batch is a dict with 'embedding', 'labels_clf', 'labels'
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}

            # output returns pred_coords, (geocell_topk, top_indices), embedding
            output = model(batch['embedding'], batch.get('labels'), batch.get('labels_clf'))
            loss = output['loss']
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch['embedding'].size(0)

        # Average training loss
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if v is not None}
                output = model(batch['embedding'], batch.get('labels'), batch.get('labels_clf'))
                val_loss += output['loss'].item() * batch['embedding'].size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # if best_val_loss has improved, update it
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            current_patience = 0

            os.makedirs(save_checkpoint_path, exist_ok=True)  # creates dir if it doesn't exist

            # Save best model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }, f"{save_checkpoint_path}/checkpoint_epoch{epoch+1}.pt")
        else:
            current_patience += 1

        # Early stopping
        if patience is not None and current_patience >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    plot_losses(train_losses, val_losses)

    # load best model and return
    model.load_state_dict(best_model_state)
    return model

def finetune_on_embeddings(train_dataset: DatasetDict, val_dataset:DatasetDict,
                           early_stopping: int=None, 
                           train_args: TrainingArguments=TRAIN_ARGS) -> AutoModelForImageClassification:
    """Finetunes a model on embeddings.

    Args:
        dataset (DatasetDict): dataset containing embeddings
        early_stopping (int, optional): early stopping patience. Defaults to None.
        train_args (TrainingArguments, optional): training arguments. Defaults to DEFAULT_TRAIN_ARGS.

    Returns:
        AutoModel: finetuned model.
    """
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

    model.to(device)
    print(model)

    finetuned_model = train_model(model, train_dataset, val_dataset, train_args, early_stopping, save_checkpoint_path="saved_models/finetuned", load_checkpoint_path=None)
    return finetuned_model

if __name__ == "__main__":
    early_stopping = 5 # hardcode parameter - if no val improvement after 5 epochs, stop to prevent overfitting

    # Load dataset
    train_dataset = load_from_disk("data/hf_geoguessr_finetune_processed_b32/train")
    val_dataset = load_from_disk("data/hf_geoguessr_finetune_processed_b32/val")

    finetune_on_embeddings(train_dataset,val_dataset, early_stopping=early_stopping, train_args = TRAIN_ARGS)