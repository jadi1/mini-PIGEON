import json
import ast
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import Tuple
from utils import haversine

# Cluster refinement model

class PigeonRefiner(nn.Module):
    def __init__(self, proto_csv, dataset_df, embeddings_npy: str, topk: int=5, max_refinement: int=1000,
                 temperature: float=1.6, device: str = 'cuda'):
        super().__init__()

        # Variables
        self.topk = topk
        self.max_refinement = max_refinement
        self.device = device

        # Load prototypes
        self.proto_df = pd.read_csv(proto_csv)
        self.dataset_df=dataset_df

        # Load embeddings
        embeddings = np.load(embeddings_npy)
        self.embeddings = torch.from_numpy(embeddings).to(device)

        # Group prototypes by geocell for fast lookup
        self.geocell_prototypes = {}
        for geocell_idx, group in self.proto_df.groupby('geocell_idx'):
            group_records = []
            for _, row in group.iterrows():
                record = {
                    'geocell_idx': row['geocell_idx'],
                    'subcluster': row['subcluster'],
                    'lat': row['lat'],
                    'lng': row['lng'],
                    'count': row['count'],
                    'indices': json.loads(row['indices']) if isinstance(row['indices'], str) else row['indices']
                }
                group_records.append(record)
            self.geocell_prototypes[geocell_idx] = group.to_dict('records')
        
        # Temperature parameter
        self.temperature = Parameter(torch.tensor(temperature), requires_grad=False)


    def forward(self, embedding: Tensor=None, initial_preds: Tensor=None, candidate_cells: Tensor=None,
                candidate_probs: Tensor=None):
        """Forward function for proto refinement model.

        Args:
            embedding (Tensor): CLIP embeddings of images.
            initial_preds (Tensor): initial predictions.
            candidate_cells (Tensor): tensor of candidate geocell predictions.
            candidate_probs (Tensor): tensor of probabilities assigned to
                each candidate geocell. Defaults to None.
        """
        assert self.topk <= candidate_cells.size(1), \
            '"topk" parameter must be smaller or equal to the number of geocell candidates \
             passed into the forward function.'

        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)

        batch_size = embedding.shape[0]
        refined_lats = []
        refined_lons = []
        refined_geocells_list = []

        # Loop over every data sample
        for b in range(batch_size):
            emb = embedding[b]  # [D]
            candidates = candidate_cells[b]  # [K]
            c_probs = candidate_probs[b]  # [K]
            initial_pred = initial_preds[b]  #[2]

            top_preds = []
            top_scores = []

            # for every candidate cell in top k,
            for geocell_idx in candidates[:self.topk]:
                geocell_id  = geocell_idx.item()

                if geocell_id not in self.geocell_prototypes:
                    # skip if no prototypes for this geocell
                    top_scores.append(torch.tensor(-100000.0, device=self.device))
                    top_preds.append([0.0, 0.0])
                    continue

                # get prototypes for this geocell
                geocell_protos = self.geocell_prototypes[geocell_id]
                
                # Find best prototype in this geocell
                best_proto = None
                best_distance = float('inf')
                
                # for every prototype
                for proto in geocell_protos:
                    proto_indices = proto['indices']
                    if isinstance(proto_indices, str):
                        proto_indices = ast.literal_eval(proto_indices)  # converts string to list of ints
                    proto_indices = torch.tensor(proto_indices, dtype=torch.long, device=self.device)
                    proto_embeddings = self.embeddings[proto_indices]
                    
                    # Distance to prototype
                    proto_embedding = proto_embeddings.mean(dim=0)  # [D]
                    dist = self._euclidian_distance(proto_embedding.unsqueeze(0), emb)
                    dist_val = dist.item()

                    # smaller dist is better!
                    if dist_val < best_distance:
                        best_distance = dist_val
                        best_proto = proto
                # Within best prototype, find best individual sample
                if best_proto is not None:
                    lat, lng  = self._within_prototype_refinement(emb, best_proto)
                    top_preds.append([lat, lng])
                    top_scores.append(torch.tensor(-best_distance, device=self.device))
                else:
                    top_preds.append([0.0, 0.0])
                    top_scores.append(torch.tensor(-100000.0, device=self.device))

            # Temperature softmax over cluster candidates
            top_scores = torch.stack(top_scores)
            probs = self._temperature_softmax(top_scores)

            # Combine with initial geocell probabilities
            final_probs = c_probs[:self.topk] * probs
            best_idx = torch.argmax(final_probs).item()

            # SAFETY CHECK
            refined_coord = torch.tensor(top_preds[best_idx], device=self.device).unsqueeze(0)
            initial_coord = initial_pred.unsqueeze(0)
            distance = haversine(initial_coord, refined_coord)[0].item()
            
            if distance > self.max_refinement:
                # Refinement too far, use original
                best_idx = 0
            
            refined_lats.append(top_preds[best_idx][0])
            refined_lons.append(top_preds[best_idx][1])
            refined_geocells_list.append(candidates[best_idx].item())

        return (torch.tensor(refined_lats, device=self.device),
                torch.tensor(refined_lons, device=self.device),
                torch.tensor(refined_geocells_list, device=self.device))
        
    
    def _within_prototype_refinement(self, emb: Tensor, 
                                   proto: dict) -> Tuple[float, float]:
        """Refines the guess even further by picking the image in a cluster that matches the best.

        Args:
            emb (Tensor): embedding of query image.
            cluster (dict): dictionary

        Returns:
            Tuple[float, float]: (lat, lng)
        """
        proto_indices = proto['indices']
        if isinstance(proto_indices, str):
            proto_indices = ast.literal_eval(proto_indices)  # converts string to list of ints

        proto_indices = torch.tensor(proto_indices, dtype=torch.long, device=self.device)
        
        if len(proto_indices) == 1:
            return proto['lat'], proto['lng']
        
        # Get embeddings for all samples in prototype
        proto_embeddings = self.embeddings[proto_indices]
        
        # Find closest sample
        distances = self._euclidian_distance(proto_embeddings, emb)
        best_idx_local = torch.argmin(distances).item()
        best_global_idx = proto_indices[best_idx_local].item()
        
        # Get coordinates
        best_sample = self.dataset_df.iloc[best_global_idx]
        return best_sample['lat'], best_sample['lng']
    
    def _euclidian_distance(self, matrix: Tensor, vector: Tensor) -> Tensor:
        v = vector.unsqueeze(0)
        distances = torch.cdist(matrix, v)
        return distances.flatten()

    def _temperature_softmax(self, input: Tensor) -> Tensor:
        ex = torch.exp(input / self.temperature)
        sum = torch.sum(ex, axis=0)
        return ex / sum
