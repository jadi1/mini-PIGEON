import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, AutoModel 
from typing import Any, Callable
from tqdm import tqdm


# def evaluate_model(model: nn.Module, dataset: Dataset, metrics: Callable,
#                    train_args: TrainingArguments, refiner: ProtoRefiner=None,
#                    yfcc: bool=False, writer: SummaryWriter=None, step: int=0) -> float:
#     """Evaluates model on evaluation data

#     Args:
#         model (nn.Module): model to use for evaluation
#         dataset (Dataset): validation dataset
#         metrics (Callable): function returning a dict of metrics given predictions
#                             and labels
#         train_args (TrainingArguments): training arguments
#         refiner (ProtoRefiner, optional): guess refinement model. Defaults to None.
#         yfcc (bool, optional): whether yfcc input data was used.
#         writer (SummaryWriter, optional): TensorBoard writer. Defaults to None.
#         step (int, optional): number of evaluation step. Defaults to 0.

#     Returns:
#         float: evaluation loss
#     """
    

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
                train_args: TrainingArguments, patience: int=None, checkpoint_path: str="", load_checkpoint_path: str=None) -> AutoModel:
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
        print(f"Loaded checkpoint from {checkpoint_path}, starting at epoch {start_epoch}")

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

            os.makedirs(checkpoint_path, exist_ok=True)  # creates dir if it doesn't exist

            # Save best model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }, f"{checkpoint_path}/checkpoint_epoch{epoch+1}.pt")
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