import torch
from torch import nn
from utils import haversine_matrix, smooth_labels

class PigeonModel(nn.Module):
    def __init__(self, num_geocells, geocell_centroids, smooth_labels, embedding_dim, device):
        super().__init__()
        self.device = device
        self.geocell_centroids = geocell_centroids.to(device) # (900,2)
        self.embedding_dim = embedding_dim
        self.smooth_labels = smooth_labels

        # linear layer on top of CLIP's vision encoder to predict geocells
        self.classifier = nn.Linear(self.embedding_dim, num_geocells)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fnc = nn.CrossEntropyLoss()
        print(f'Initialized SuperGuessr classification model with {num_geocells} geocells.')
        
    # labels: coordinates, labels_clf: geocell classification
    def forward(self, embedding, labels, labels_clf):
        """
        embedding: precomputed embeddings, (B, hidden_dim)
        labels: (B,2) lat/lon
        labels_clf: (B,) geocell index
        """
        embedding = embedding.to(self.device)

        # Linear layer, run directly on embeddings
        logits = self.classifier(embedding)
        geocell_probs = self.softmax(logits)            

        # Compute initial coordinate prediction - this is before protorefiner
        geocell_preds = torch.argmax(geocell_probs, dim=-1) # geocell index prediction for each sample
        pred_coords = self.geocell_centroids[geocell_preds] # select along dim 0 

        # get topk geocell candidates
        geocell_topk, top_indices = torch.topk(geocell_probs, k=5, dim=-1) # top 5 candidates!

        # serving mode - return predictions and embeddings for refinement
        if labels is None or labels_clf is None:
            return pred_coords, (geocell_topk, top_indices), embedding
        
        # otherwise, TRAINING MODE
        labels = labels.to(self.device)
        labels_clf = labels_clf.to(self.device)

        # smooth labels based on distance
        if self.smooth_labels:
            distances = haversine_matrix(labels, self.geocell_centroids)
            label_probs = smooth_labels(distances)
        else:
            label_probs = labels_clf # just use classification directly
            
        # classification loss (cross entropy)
        loss_clf = self.loss_fnc(logits, label_probs)
 
        return {
            "loss": loss_clf,
            "loss_clf": loss_clf,   # same, but separate in PIGEON paper
            "geocell_preds": geocell_preds,
            "geocell_topk": geocell_topk,
            "probs": geocell_probs,
            "embedding": embedding,
        }
