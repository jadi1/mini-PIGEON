import pickle
import torch
import numpy as np
from transformers import TrainingArguments, \
                         AutoModelForImageClassification
from model import PigeonModel
from datasets import DatasetDict, load_from_disk
from train_eval_loop import train_model

TRAIN_ARGS = TrainingArguments(
    output_dir="saved_models",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=200, # set large, but train until convergence
    evaluation_strategy="epoch",
    eval_steps=5,
    save_strategy='epoch',
    save_steps=2,
    learning_rate=1e-3, # a little higher, since only training on precomputed embeddings
    logging_steps=1,
    load_best_model_at_end=True,
    seed=330
)

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

    finetuned_model = train_model(model, train_dataset, val_dataset, train_args, early_stopping, "saved_models/finetuned")
    return finetuned_model

if __name__ == "__main__":
    early_stopping = 3 # hardcode parameter - if no val improvement after 3 epochs, stop to prevent overfitting

    # Load dataset
    train_dataset = load_from_disk("data/hf_geoguessr_finetune_processed/train")
    val_dataset = load_from_disk("data/hf_geoguessr_finetune_processed/train")

    finetune_on_embeddings(train_dataset,val_dataset, early_stopping=early_stopping, train_args = TRAIN_ARGS)