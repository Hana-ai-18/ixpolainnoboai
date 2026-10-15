"""
TC-PhysXAI Utilities
======================
Shared helper functions: CRPS, DTW, haversine, normalisation constants,
and small plotting helpers that every module reuses.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

# ─── ERA5 normalisation constants (copied from TC-XFlow loader) ─────────────
DATA3D_MEAN = np.array([
    12444.037, 5844.935, 1482.430, 752.466,   # GPH 200/500/850/925 hPa
       -1.052,   -0.188,   -0.407,  -0.888,   # U   200/500/850/925 hPa
       -0.055,    1.655,    1.343,   1.018,   # V   200/500/850/925 hPa
      301.133,                               # SST
], dtype=np.float32)

DATA3D_STD = np.array([
    113.583, 57.135, 37.852, 37.274,          # GPH
     13.315,  8.042,  7.911,  7.494,          # U
      8.377,  5.999,  6.203,  6.555,          # V
      3.023,                                  # SST
], dtype=np.float32)

# Channel names (for plotting)
CH_NAMES = [
    "GPH_200", "GPH_500", "GPH_850", "GPH_925",
    "U_200",   "U_500",   "U_850",   "U_925",
    "V_200",   "V_500",   "V_850",   "V_925",
    "SST",
]

# Pressure-level groupings
LEVEL_GROUPS = {
    "200hPa": [0, 4, 8],   # GPH, U, V at 200
    "500hPa": [1, 5, 9],   # GPH, U, V at 500
    "850hPa": [2, 6, 10],  # GPH, U, V at 850
    "925hPa": [3, 7, 11],  # GPH, U, V at 925
    "SST":    [12],
}

# Normalisation scales for trajectory coordinates
_MOVE_VEL_NORM = 150.0


def seed_crn(seed: int = 0) -> None:
    """Common random numbers: call right before every ensemble `model.sample`
    whose result is COMPARED with another run (masked vs unmasked, perturbed vs
    clean). Without it, the ratio/difference of two CRPS values also contains
    sampling noise, which can be as large as the effect being measured."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denorm_latlon(traj_norm: torch.Tensor) -> torch.Tensor:
    """Absolute normalised (lon_n, lat_n) -> (lat_deg, lon_deg).

    Dataset convention (Model/flow_matching_model.py::_norm_to_deg, see geo.py):
    index 0 = longitude, index 1 = latitude, lon = 5*x0 + 180, lat = 5*x1.
    NOTE the OUTPUT order is (lat, lon) because haversine_km expects it.
    """
    lon_deg = traj_norm[..., 0] * 5.0 + 180.0
    lat_deg = traj_norm[..., 1] * 5.0
    return torch.stack([lat_deg, lon_deg], dim=-1)


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Haversine distance in km between lat/lon pairs.

    Args:
        p1, p2: [..., 2] tensors with (lat_deg, lon_deg)
    Returns:
        [...] tensor of distances in km
    """
    R = 6371.0
    lat1 = torch.deg2rad(p1[..., 0])
    lat2 = torch.deg2rad(p2[..., 0])
    dlat = torch.deg2rad(p2[..., 0] - p1[..., 0])
    dlon = torch.deg2rad(p2[..., 1] - p1[..., 1])
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return R * 2 * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))


def crps_ensemble(ensemble: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Continuous Ranked Probability Score for ensemble forecasts.

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent draws from the ensemble.

    Args:
        ensemble: [K, T, B, 2] — K ensemble members, T horizons, B storms
        obs:      [T, B, 2]   — ground truth trajectories (normalised)
    Returns:
        [T, B] CRPS values in normalised units
    """
    K, T, B, _ = ensemble.shape
    # Mean absolute error to observation: [K, T, B]
    mae = (ensemble - obs.unsqueeze(0)).norm(dim=-1)  # [K, T, B]
    mean_mae = mae.mean(0)  # [T, B]
    # Intra-ensemble spread: E[|X-X'|] = 2/(K*(K-1)) * Σ_{i<j} |X_i - X_j|
    spread = torch.zeros(T, B, device=ensemble.device)
    for i in range(K):
        for j in range(i + 1, K):
            spread += (ensemble[i] - ensemble[j]).norm(dim=-1)
    if K > 1:
        spread = spread * 2.0 / (K * (K - 1))
    return mean_mae - 0.5 * spread


def crps_ensemble_fast(ensemble: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Vectorised CRPS (O(K²) in memory, faster on GPU).

    Args:
        ensemble: [K, T, B, 2]
        obs:      [T, B, 2]
    Returns:
        [T, B]
    """
    K, T, B, _ = ensemble.shape
    mae = (ensemble - obs.unsqueeze(0)).norm(dim=-1).mean(0)  # [T, B]
    # Pairwise spread: broadcast i vs j
    ei = ensemble.unsqueeze(1)   # [K, 1, T, B, 2]
    ej = ensemble.unsqueeze(0)   # [1, K, T, B, 2]
    spread = (ei - ej).norm(dim=-1).sum([0, 1]) / (K * K)  # [T, B]
    return mae - 0.5 * spread


def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Dynamic Time Warping distance between two trajectory arrays.

    Args:
        s1: [T1, 2] array
        s2: [T2, 2] array
    Returns:
        DTW distance (scalar)
    """
    T1, T2 = len(s1), len(s2)
    dtw = np.full((T1 + 1, T2 + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[T1, T2])


def dtw_distance_batch(trajs: np.ndarray) -> np.ndarray:
    """Compute pairwise DTW distance matrix for a set of trajectories.

    Args:
        trajs: [K, T, 2] array of K trajectories
    Returns:
        [K, K] symmetric distance matrix
    """
    K = len(trajs)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            d = dtw_distance(trajs[i], trajs[j])
            D[i, j] = D[j, i] = d
    return D


def era5_denorm(img_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalised ERA5 fields back to physical units.

    Args:
        img_norm: [..., 13, T, H, W] normalised ERA5 tensor
    Returns:
        [..., 13, T, H, W] in physical units
    """
    mean = torch.tensor(DATA3D_MEAN, device=img_norm.device, dtype=img_norm.dtype)
    std  = torch.tensor(DATA3D_STD,  device=img_norm.device, dtype=img_norm.dtype)
    # Broadcast: add dims for T, H, W
    mean = mean.view(*([1] * (img_norm.dim() - 4)), 13, 1, 1, 1)
    std  = std.view(*([1] * (img_norm.dim() - 4)),  13, 1, 1, 1)
    return img_norm * std + mean


def get_climatological_era5(img_obs: torch.Tensor) -> torch.Tensor:
    """Return the mean ERA5 field (climatological baseline) shaped like img_obs.

    The baseline is the channel-wise mean (zero in normalised space = mean in
    physical space). We add small Gaussian noise to avoid degenerate gradients.

    Args:
        img_obs: [B, 13, T, H, W]
    Returns:
        [B, 13, T, H, W] climatological baseline (normalised = ~0)
    """
    return torch.zeros_like(img_obs)


def get_climatological_traj(obs_traj: torch.Tensor) -> torch.Tensor:
    """Return mean trajectory (zero motion) as climatological baseline.

    Args:
        obs_traj: [T, B, 2] normalised trajectory
    Returns:
        [T, B, 2] zero-motion reference
    """
    # Keep last observed position, fill history with that position
    last = obs_traj[-1:].expand_as(obs_traj)
    return last


def soft_cluster_assignment(
    traj: torch.Tensor,
    centroids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft (differentiable) cluster assignment via softmax over distances.

    Args:
        traj:      [T, 2] trajectory
        centroids: [K, T, 2] cluster centroids
        temperature: sharpness of assignment
    Returns:
        [K] soft assignment probabilities
    """
    # Mean squared distance from traj to each centroid
    diff = traj.unsqueeze(0) - centroids  # [K, T, 2]
    dist = diff.pow(2).sum([-2, -1])       # [K]
    return F.softmax(-dist / temperature, dim=0)


def compute_dpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Displacement Prediction Error (mean haversine over all horizons).

    Args:
        pred: [T, B, 2] normalised predicted
        gt:   [T, B, 2] normalised ground truth
    Returns:
        scalar mean DPE in km
    """
    pred_deg = denorm_latlon(pred)
    gt_deg   = denorm_latlon(gt)
    return haversine_km(pred_deg, gt_deg).mean()
