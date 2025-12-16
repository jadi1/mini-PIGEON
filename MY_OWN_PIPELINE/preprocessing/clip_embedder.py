import torch
from torch import nn, Tensor
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk

CLIP_MODEL = 'openai/clip-vit-base-patch16' 

class CLIPEmbedding(torch.nn.Module):
    def __init__(self, device: str='cuda', panorama: bool=False):
        """CLIP embedding model (not trainable)

        Args:
            model_name (str): CLIP model version
            device (str, optional): where to run the model. Defaults to 'cuda'.
            load_checkpoint (bool, optional): whether to load checkpoint from
                CLIP_SERVING path. Defaults to True.
            panorama (bool): if four images should be embedded.
                Defaults to False.
        """
        super().__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.clip_model = CLIPVisionModel.from_pretrained(CLIP_MODEL)
        self.panorama = panorama

        if type(device) == str:
            self.clip_model = self.clip_model.to(self.device)
        else:
            self.clip_model = self.clip_model.cuda(self.device)

        self.eval()

    def _get_embedding(self, image: Image) -> Tensor:
        """Computes embedding for a single image.

        Args:
            image (Image): jpg image

        Returns:
            Tensor: embedding
        """
        with torch.no_grad():
            if isinstance(image, Tensor) == False:
                inputs = self.processor(images=image, return_tensors='pt')
                pixel_values = inputs['pixel_values']
            else:
                pixel_values = image

            if type(self.device) == str:
                pixel_values = pixel_values.to(self.device)
            else:
                pixel_values = pixel_values.cuda(self.device)
        
            outputs = self.clip_model.base_model(pixel_values=pixel_values)
            cls_token_embedding = outputs.last_hidden_state
            cls_token_embedding = torch.mean(cls_token_embedding, dim=1)
            return cls_token_embedding
        

def embed_dataset(dataset, embedder, batch_size=1):
    embeddings = []
    indices = []

    for example in tqdm(dataset):
        if embedder.panorama:
            imgs = [
                example["image"],
                example["image_2"],
                example["image_3"],
                example["image_4"],
            ]

            img_embeds = []
            # first get all individual embeddings
            for img in imgs:
                emb = embedder._get_embedding(img)
                img_embeds.append(emb)

            # then average them all
            emb = torch.mean(torch.stack(img_embeds), dim=0)

        embeddings.append(emb.cpu())
        indices.append(example["index"])

    embeddings = torch.cat(embeddings, dim=0)
    indices = np.array(indices)

    return embeddings, indices

if __name__ == '__main__':
    splits = ["train", "val", "test"]

    for split in splits:
        # load in dataset
        dataset = load_from_disk("data/hf_geoguessr_finetune")[split]

        # instantiate embedder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = CLIPEmbedding(device=device, panorama=True) 

        embeddings, indices = embed_dataset(dataset, embedder)

        print(embeddings.shape)

        np.save(f"data/embeddings_{split}.npy", embeddings.numpy())
        np.save(f"data/indices_{split}.npy", indices)