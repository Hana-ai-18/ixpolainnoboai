"""
TC-PhysXAI — coordinate conventions of the TC-XFlow dataset (single source of truth)
====================================================================================
Trajectory tensors are [..., 2] = (lon_norm, lat_norm)  — INDEX 0 IS LONGITUDE.
Normalisation (see Model/flow_matching_model.py::_norm_to_deg and the dataset loader):

    lon[°E] = (x0 * 50 + 1800) / 10 = 5 * x0 + 180
    lat[°N] = (x1 * 50)        / 10 = 5 * x1

i.e. ONE NORMALISED UNIT = 5 DEGREES on both axes. Relative (displacement) tensors use
the same scale without the +180 offset.

Earlier XAI code assumed index 0 = latitude, index 1 = longitude with 12.8 / 22.68
degrees per unit. That is wrong for this dataset; use the helpers below instead.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

DEG_PER_UNIT = 5.0
KM_PER_DEG = 111.195


def norm_to_lonlat(x: np.ndarray) -> np.ndarray:
    """Absolute normalised (lon_n, lat_n) -> (lon°E, lat°N), shape preserved."""
    x = np.asarray(x, dtype=np.float64)
    return np.stack([x[..., 0] * DEG_PER_UNIT + 180.0, x[..., 1] * DEG_PER_UNIT], axis=-1)


def step_enu_km(traj_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-step eastward / northward displacement in km for [..., T, 2] absolute
    normalised tracks. Returns (east, north), each [..., T-1]."""
    ll = norm_to_lonlat(traj_norm)
    dlon = ll[..., 1:, 0] - ll[..., :-1, 0]
    dlat = ll[..., 1:, 1] - ll[..., :-1, 1]
    lat_mid = np.deg2rad(0.5 * (ll[..., 1:, 1] + ll[..., :-1, 1]))
    return dlon * KM_PER_DEG * np.cos(lat_mid), dlat * KM_PER_DEG


def net_displacement_deg(traj_norm: np.ndarray) -> Tuple[float, float]:
    """(dlon°, dlat°) from first to last point of a [T, 2] track."""
    ll = norm_to_lonlat(traj_norm)
    return float(ll[-1, 0] - ll[0, 0]), float(ll[-1, 1] - ll[0, 1])


def classify_recurvature(traj_norm: np.ndarray, threshold_dlat_deg: float = 3.0) -> bool:
    """Recurvature: >threshold poleward motion in the second half of the track while
    not moving westward by more than 2 degrees there."""
    traj_norm = np.asarray(traj_norm)
    if len(traj_norm) < 4:
        return False
    ll = norm_to_lonlat(traj_norm)
    mid = len(ll) // 2
    d_lat_late = ll[-1, 1] - ll[mid, 1]
    d_lon_late = ll[-1, 0] - ll[mid, 0]
    return bool(d_lat_late > threshold_dlat_deg and d_lon_late > -2.0)


# Regime thresholds in DEGREES over the forecast window (default: 12 steps = 72 h).
# These are heuristics used only to STRATIFY storms for reporting; confirm them with
# the thesis advisor before quoting regime-level numbers.
STALLED_MAX_DEG = 2.0
POLEWARD_MIN_DLAT_DEG = 3.0
WEST_MIN_DLON_DEG = -3.0


def classify_track_type(traj_norm: np.ndarray) -> str:
    """STRAIGHT_WEST | RECURVING_NE | POLEWARD_TRACK | STALLED | MIXED for a [T,2] track."""
    traj_norm = np.asarray(traj_norm)
    if len(traj_norm) < 2:
        return "MIXED"
    dlon, dlat = net_displacement_deg(traj_norm)
    if (dlon ** 2 + dlat ** 2) ** 0.5 < STALLED_MAX_DEG:
        return "STALLED"
    if classify_recurvature(traj_norm, POLEWARD_MIN_DLAT_DEG) and dlon > 0.0:
        return "RECURVING_NE"
    if dlat > POLEWARD_MIN_DLAT_DEG:
        return "POLEWARD_TRACK"
    if dlon < WEST_MIN_DLON_DEG and abs(dlat) < POLEWARD_MIN_DLAT_DEG:
        return "STRAIGHT_WEST"
    return "MIXED"
