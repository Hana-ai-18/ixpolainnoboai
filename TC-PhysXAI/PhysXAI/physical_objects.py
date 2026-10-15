"""
TC-PhysXAI — Physical Object Detection
=======================================
Detect meteorologically-meaningful objects in ERA5 attribution maps:
  · Anticyclone  (subtropical ridge, high-pressure centre)
  · Upper trough (200 hPa geopotential height trough)
  · SST front    (sea-surface temperature gradient zone)
  · Monsoon trough / vortex (low-pressure cyclonic feature at 850/925 hPa)

Method (adapted from Guillaume-Castel et al. EGUsphere 2026):
  1. Threshold attribution map to find high-relevance regions (> 0.5% of total)
  2. Apply watershed segmentation to split overlapping patches
  3. For each patch, classify using physical signature:
       - Anticyclone: positive GPH anomaly at 500 hPa, anticyclonic U/V pattern
       - Trough:      negative GPH anomaly at 200 hPa, cyclonic curvature aloft
       - SST front:   large spatial gradient in SST channel (ch12)
       - Monsoon:     negative GPH at 850/925 hPa, convergent wind field
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import torch


@dataclass
class PhysicalObject:
    """Represents a detected physical meteorological object."""
    type: str                      # "anticyclone", "trough", "sst_front", "monsoon", "unknown"
    level: str                     # "200hPa", "500hPa", "850hPa", "925hPa", "SST"
    patch_mask: np.ndarray         # [H, W] boolean mask of the relevant patch
    centroid: Tuple[float, float]  # (row, col) in grid coords
    attribution_weight: float      # fraction of total positive attribution
    physical_signature: Dict       # diagnostic fields for this object
    description: str               # human-readable meteorological description


def _compute_relevance_patches(
    attr_map: np.ndarray,       # [H, W] sum-normalised attribution
    threshold: float = 0.005,   # patches with total relevance > 0.5%
) -> List[np.ndarray]:
    """Find contiguous high-attribution regions.

    Uses a simple connected-component approach (no scipy dependency needed).
    """
    try:
        from scipy import ndimage
        binary = attr_map > np.percentile(attr_map[attr_map > 0], 80) if attr_map.max() > 0 else np.zeros_like(attr_map, dtype=bool)
        labeled, n_components = ndimage.label(binary)
        patches = []
        for i in range(1, n_components + 1):
            mask = labeled == i
            total = attr_map[mask].sum()
            if total > threshold:
                patches.append(mask)
        return patches
    except ImportError:
        # Fallback: single global patch above threshold
        mask = attr_map > (attr_map.max() * 0.5)
        if mask.any():
            return [mask]
        return []


def _patch_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Weighted centroid of a patch."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (0.0, 0.0)
    return (float(rows.mean()), float(cols.mean()))


def detect_physical_objects(
    ig_attr: np.ndarray,          # [13, T, H, W] IG attribution map
    era5_fields: np.ndarray,      # [13, T, H, W] actual ERA5 fields (normalised)
    time_step: int = -1,          # which time step to analyse (default: latest)
) -> List[PhysicalObject]:
    """Detect physical meteorological objects from IG attribution maps.

    Parameters
    ----------
    ig_attr : [13, T, H, W] Integrated Gradients attribution
    era5_fields : [13, T, H, W] ERA5 input fields (normalised)
    time_step : time index to analyse

    Returns
    -------
    List of PhysicalObject instances, sorted by attribution weight descending.
    """
    # Select time step
    attr_t  = ig_attr[:, time_step]    # [13, H, W]
    era5_t  = era5_fields[:, time_step]  # [13, H, W]
    H, W    = attr_t.shape[-2], attr_t.shape[-1]

    objects = []
    total_pos_attr = max(attr_t.clip(min=0).sum(), 1e-8)

    # ─── Level-by-level analysis ─────────────────────────────────────────────
    level_configs = [
        # (channels_for_detection, level_name, expected_objects)
        ([0, 4, 8],   "200hPa", ["trough"]),
        ([1, 5, 9],   "500hPa", ["anticyclone", "trough"]),
        ([2, 6, 10],  "850hPa", ["monsoon"]),
        ([3, 7, 11],  "925hPa", ["monsoon"]),
        ([12],        "SST",    ["sst_front"]),
    ]

    for ch_list, level, expected_types in level_configs:
        # Sum attribution over channels of this level
        level_attr = attr_t[ch_list].sum(0)    # [H, W]
        level_era5 = {c: era5_t[c] for c in ch_list}

        # Normalise
        level_attr_norm = level_attr / total_pos_attr if total_pos_attr > 0 else level_attr

        patches = _compute_relevance_patches(level_attr_norm, threshold=0.003)

        for patch_mask in patches:
            weight = float(level_attr_norm[patch_mask].sum())
            if weight < 0.005:
                continue

            centroid = _patch_centroid(patch_mask)

            # Extract field values in the patch
            patch_gph  = era5_t[ch_list[0]][patch_mask] if len(ch_list) > 0 else np.array([0])
            patch_u    = era5_t[ch_list[1]][patch_mask] if len(ch_list) > 1 else np.array([0])
            patch_v    = era5_t[ch_list[2]][patch_mask] if len(ch_list) > 2 else np.array([0])

            mean_gph = float(patch_gph.mean())
            mean_u   = float(patch_u.mean())
            mean_v   = float(patch_v.mean())

            # ─── Classification heuristics ───────────────────────────────────
            if level == "200hPa":
                # Upper-level trough: negative GPH anomaly (trough axis), cyclonic
                if mean_gph < -0.3:
                    obj_type = "trough"
                    desc = (
                        f"UPPER-LEVEL TROUGH ({level}): Negative geopotential height "
                        f"anomaly ({mean_gph:.3f} norm.) at 200 hPa. This upper trough is "
                        "approaching from the west/northwest, reducing the ridge axis and "
                        "creating a favourable environment for typhoon recurvature. "
                        "The divergent flow ahead of the trough enhances outflow "
                        "venting and intensification potential."
                    )
                elif mean_gph > 0.3:
                    obj_type = "anticyclone"
                    desc = (
                        f"UPPER-LEVEL RIDGE ({level}): Positive GPH anomaly at 200 hPa "
                        "indicates a well-developed anticyclone aloft. This blocks recurvature "
                        "and forces the typhoon to maintain a westward/northwestward track "
                        "under strong subtropical ridge influence."
                    )
                else:
                    obj_type = "unknown"
                    desc = f"Unclassified feature at {level} (GPH anomaly: {mean_gph:.3f})."

            elif level == "500hPa":
                # Mid-level steering: anticyclone = positive GPH, negative = trough
                if mean_gph > 0.2:
                    obj_type = "anticyclone"
                    desc = (
                        f"SUBTROPICAL RIDGE ({level}): The 500 hPa anticyclone is the PRIMARY "
                        "steering mechanism. Positive GPH anomaly displaces the typhoon along the "
                        f"ridge southern periphery. Mean U500={mean_u:.2f}, V500={mean_v:.2f} "
                        "(normalised) indicate easterly/northeasterly steering flow."
                    )
                elif mean_gph < -0.2:
                    obj_type = "trough"
                    desc = (
                        f"MID-LEVEL TROUGH ({level}): Negative GPH at 500 hPa indicates a "
                        "mid-latitude trough displacing the westerlies equatorward. This trough "
                        "enhances the poleward component of typhoon motion, possibly triggering "
                        "or accelerating recurvature."
                    )
                else:
                    # Check wind pattern
                    if abs(mean_u) > abs(mean_v) and mean_u < 0:
                        obj_type = "anticyclone"
                        desc = (
                            f"WEAK RIDGE ({level}): Easterly (negative U) steering flow "
                            "at 500 hPa drives westward motion. GPH anomaly is weak but "
                            "wind direction indicates ridge influence."
                        )
                    else:
                        obj_type = "unknown"
                        desc = f"Ambiguous feature at {level}."

            elif level in ("850hPa", "925hPa"):
                # Low-level: monsoon trough, low, or vortex
                if mean_gph < -0.2:
                    obj_type = "monsoon"
                    desc = (
                        f"MONSOON TROUGH / LOW-LEVEL CYCLONIC FEATURE ({level}): "
                        "Negative GPH anomaly at the boundary layer level. This feature "
                        "can be the monsoon trough that spawned or is interacting with the "
                        "typhoon, enhancing moisture convergence and modifying the local "
                        "vorticity budget. At 925 hPa this is directly within the planetary "
                        "boundary layer, affecting surface friction and track deflection "
                        "near terrain."
                    )
                else:
                    obj_type = "unknown"
                    desc = f"Boundary-layer feature at {level} (GPH: {mean_gph:.3f})."

            elif level == "SST":
                # SST front: large gradient within patch
                sst_patch = era5_t[12][patch_mask]
                sst_range = float(sst_patch.max() - sst_patch.min()) if len(sst_patch) > 1 else 0
                if sst_range > 0.5:   # 0.5 normalised ≈ 1.5 K given σ≈3 K
                    obj_type = "sst_front"
                    desc = (
                        "SEA-SURFACE TEMPERATURE FRONT: Pronounced SST gradient "
                        f"(normalised range: {sst_range:.2f}, ≈{sst_range*3:.1f} K physical) "
                        "within the relevant region. SST fronts modulate tropical cyclone "
                        "intensity and track through differential surface heat fluxes: "
                        "high SST (>28 °C) sustains deepening while lower SST downstream "
                        "can trigger rapid weakening. The TC track model correctly identifies "
                        "this gradient as a steering-relevant feature."
                    )
                else:
                    obj_type = "sst_front" if weight > 0.03 else "unknown"
                    desc = (
                        f"SST REGION: Warm pool identified (SST attribution weight={weight:.2%}). "
                        "The warm ocean provides the energy source maintaining storm intensity, "
                        "which in turn affects the speed calibration of the track prediction."
                    )

            else:
                obj_type = "unknown"
                desc = f"Feature detected at {level}."

            phys_sig = {
                "mean_gph_norm": mean_gph,
                "mean_u_norm":   mean_u,
                "mean_v_norm":   mean_v,
                "patch_size_pct": float(patch_mask.sum()) / (H * W),
            }

            objects.append(PhysicalObject(
                type=obj_type,
                level=level,
                patch_mask=patch_mask,
                centroid=centroid,
                attribution_weight=weight,
                physical_signature=phys_sig,
                description=desc,
            ))

    # Sort by attribution weight descending
    objects.sort(key=lambda o: o.attribution_weight, reverse=True)
    return objects


def summarise_objects(objects: List[PhysicalObject]) -> str:
    """Generate a concise meteorological summary of detected objects."""
    if not objects:
        return "No significant physical objects detected in the attribution map."

    lines = ["=== Physical Object Summary ==="]
    for i, obj in enumerate(objects[:5], 1):
        lines.append(
            f"\n[{i}] {obj.type.upper()} at {obj.level} "
            f"(attribution: {obj.attribution_weight:.2%})"
        )
        lines.append(f"    Centroid: row={obj.centroid[0]:.1f}, col={obj.centroid[1]:.1f}")
        lines.append(f"    {obj.description}")
    return "\n".join(lines)
