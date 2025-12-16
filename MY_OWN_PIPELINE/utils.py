# taken directly from [insert link here]

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

# Constants
rad_np = np.float64(6378137.0)        # Radius of the Earth (in meters)
f_np = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model

rad_torch = torch.tensor(6378137.0, dtype=torch.float64)
f_torch = torch.tensor(1.0/298.257223563, dtype=torch.float64)

PREC = 2.2204e-16
LABEL_SMOOTHING_CONSTANT = 75 # tau

# ECEF to LLA coordinate transformation
a_sq = rad_np ** 2
e = 8.181919084261345e-2
e_sq = 6.69437999014e-3
b = rad_np * (1- f_np)
ep_sq  = (rad_np**2 - b**2) / b**2
ee = (rad_np**2-b**2)


def haversine(x: Tensor, y: Tensor) -> Tensor:
    """Computes the haversine distance between two sets of points

    Args:
        x (Tensor): points 1 (lat, lng)
        y (Tensor): points 2 (lat, lng)

    Returns:
        Tensor: haversine distance in km
    """
    x = x[..., [1, 0]]
    y = y[..., [1, 0]]
    x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
    delta = y_rad - x_rad
    a = torch.sin(delta[:, 1] / 2)**2 + torch.cos(x_rad[:, 1]) * torch.cos(y_rad[:, 1]) * torch.sin(delta[:, 0] / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    km = (rad_torch * c) / 1000
    return km

# Implementation to calculate all possible combinations of distances in parallel
def haversine_matrix(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes Haversine distances between two sets of points.

    Args:
        x (Tensor): (N, 2) -> lat/lon in degrees
        y (Tensor): (M, 2) -> lat/lon in degrees

    Returns:
        Tensor: (N, M) Haversine distances in km
    """
    # Convert degrees to radians
    x_rad = torch.deg2rad(x)  # (N,2)
    y_rad = torch.deg2rad(y)  # (M,2)

    # Separate latitudes and longitudes
    lat1, lon1 = x_rad[:, 0].unsqueeze(1), x_rad[:, 1].unsqueeze(1)  # (N,1)
    lat2, lon2 = y_rad[:, 0].unsqueeze(0), y_rad[:, 1].unsqueeze(0)  # (1,M)

    # Differences
    dlat = lat2 - lat1  # (N,M)
    dlon = lon2 - lon1  # (N,M)

    # Haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Earth radius in km
    R = 6371.0
    km = R * c
    return km

def smooth_labels(distances: Tensor) -> Tensor:
    """Haversine smooths labels for shared representation learning across geocells.

    Args:
        distances (Tensor): distance (km) matrix of size (batch_size, num_geocells)

    Returns:
        Tensor: smoothed labels
    """
    adj_distances = distances - distances.min(dim=-1, keepdim=True)[0]
    smoothed_labels = torch.exp(-adj_distances / LABEL_SMOOTHING_CONSTANT)
    smoothed_labels = torch.nan_to_num(smoothed_labels, nan=0.0, posinf=0.0, neginf=0.0)
    return smoothed_labels