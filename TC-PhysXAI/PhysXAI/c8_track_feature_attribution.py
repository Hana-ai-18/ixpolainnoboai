"""
TC-PhysXAI — C8: Track Feature Attribution (Prediction Feature Analysis)
=========================================================================
Motivation
----------
Modules C1-C7 answer: "Which INPUT drives the prediction?"
C8 answers:           "Which PHYSICAL FEATURE of the predicted track
                       is caused by which input?"

This is a fundamentally different question — it decomposes the MODEL OUTPUT
into interpretable meteorological track features, then attributes each feature
back to the ERA5 inputs (channels + time windows).

Track Features Extracted
------------------------
Per-step features (from ensemble [K, T_pred, B, 2]):
  speed_t      : Translation speed at each forecast step (normalised units)
  heading_t    : Movement direction (degrees clockwise from north)
  curvature_t  : Rate of heading change (signed, degrees per 6h step)
  spread_t     : Ensemble positional uncertainty at each horizon

Event / scalar features (single value per case):
  recurvature_prob      : P(storm turns poleward > 45°) over ensemble
  deceleration_prob     : P(speed drops > 30% within 24h)
  bifurcation_index     : Bimodality of ensemble final-position distribution
  max_speed             : Peak translation speed in forecast window
  mean_speed            : Mean translation speed
  total_turning         : Accumulated heading change (positive = clockwise, i.e. NH recurvature W→N→E)
  poleward_displacement : Net northward movement (degrees latitude)
  zonal_displacement    : Net eastward/westward movement (degrees longitude)

Attribution Method
------------------
For each feature f, each ERA5 channel c, and each time window w:

  Δf_channel[f, c] = f_baseline - f_perturb(channel=c → climatological mean)
  Δf_window[f, w]  = f_baseline - f_perturb(window=w → temporal mean)

The SPATIOTEMPORAL MEAN PERTURBATION replaces a channel (or window) with its
spatiotemporal mean value (averaged over all spatial grid points and observed
time steps for this storm) rather than zero. This is more physically coherent:
  - Zeroing sets ERA5 fields to an unrealistic state (e.g. U-wind = 0 everywhere).
  - Replacing with the channel mean preserves the mean atmospheric state while
    removing spatial/temporal variation — equivalent to "if this channel gave
    no additional information beyond its storm-specific average state."
NOTE: This is the *storm-specific* spatiotemporal mean, not a multi-storm
climatological mean (which would require a pre-computed background dataset).
The perturbation asks: "what if all spatial/temporal structure in this channel
were removed, leaving only the mean value?" — a valid ablation baseline.

Positive Δf means: "removing variation in this input reduces the feature" →
    the input AMPLIFIES the feature.
Negative Δf means: "removing variation in this input increases the feature" →
    the input SUPPRESSES the feature.

The result is a [n_features × 13] channel attribution matrix and
a [n_features × 8] window attribution matrix, giving a complete
meteorological interpretation of why the model forecasts the specific
track behaviour it does.

Faithfulness Validation
-----------------------
C8 includes a built-in MoRF-AOPC faithfulness check (similar to C7):

  aopc[f] = (1/n_steps) Σ_k |Δf after removing k most-attributed channels|
             ─────────────────────────────────────────────────────────────────
                                    |f_baseline|

Higher AOPC = the attribution ranking is faithful: removing the top-ranked
channels causes the largest feature changes, confirming the attribution is
not arbitrary.

Feature Groups
--------------
Features are separated into two conceptually distinct groups:
  Physical   : speed, curvature, recurvature, deceleration, displacement
               (intrinsic TC track properties, independent of ensemble size)
  Uncertainty: spread, bifurcation_index
               (ensemble-derived measures of forecast uncertainty)

Physical Interpretation Examples
---------------------------------
  GPH_500 → high attribution to speed    : subtropical ridge strength
                                           directly determines translation speed
  U_200   → high attribution to curvature: upper jet position determines
                                           whether/when recurvature occurs
  Phase B → high attribution to recurvature: synoptic evolution at t−24h
                                             is the trigger for track turning
  SST     → attribution to deceleration : warm SST maintains intensity;
                                          cool water slows translation
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from .geo import norm_to_lonlat, step_enu_km, KM_PER_DEG
import torch


# ─── ERA5 channels ───────────────────────────────────────────────────────────

_CH_NAMES = [
    "GPH_200", "GPH_500", "GPH_850", "GPH_925",
    "U_200",   "U_500",   "U_850",   "U_925",
    "V_200",   "V_500",   "V_850",   "V_925",
    "SST",
]

_WINDOW_LABELS = [f"t-{(7 - w) * 6}h" for w in range(8)]   # t-42h … t-0h

# Feature names (order matters — referenced by index)
_FEATURE_NAMES = [
    "speed_mean",           # 0  mean translation speed over forecast
    "speed_peak",           # 1  peak translation speed
    "curvature_mean",       # 2  mean absolute curvature (turning rate)
    "curvature_total",      # 3  total heading change (signed, + = clockwise = NH recurvature)
    "spread_mean",          # 4  mean ensemble positional spread
    "spread_final",         # 5  ensemble spread at last forecast step
    "recurvature_prob",     # 6  probability of poleward recurvature
    "deceleration_prob",    # 7  probability of significant speed drop
    "bifurcation_index",    # 8  bimodality of ensemble end-positions
    "poleward_displacement",# 9  net northward displacement
    "zonal_displacement",   # 10 net zonal displacement (+ = east)
]

N_FEATURES = len(_FEATURE_NAMES)

# Feature groups — physical track properties vs ensemble uncertainty measures
_PHYSICAL_FEATURE_NAMES = [
    "speed_mean", "speed_peak", "curvature_mean", "curvature_total",
    "recurvature_prob", "deceleration_prob",
    "poleward_displacement", "zonal_displacement",
]
_UNCERTAINTY_FEATURE_NAMES = [
    "spread_mean", "spread_final", "bifurcation_index",
]


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class TrackFeatures:
    """Meteorological features extracted from an ensemble of forecast tracks.

    All values are in normalised lat/lon units unless stated otherwise.
    Use `.to_physical()` for km/h and km interpretations (approximate).
    """
    # Per-step arrays (averaged over ensemble)
    speed:      np.ndarray           # [T_pred-1]  per-step translation speed
    heading:    np.ndarray           # [T_pred-1]  degrees clockwise from North
    curvature:  np.ndarray           # [T_pred-2]  heading change per step
    spread:     np.ndarray           # [T_pred]    ensemble std in position

    # Scalar summary features
    speed_mean:            float
    speed_peak:            float
    curvature_mean:        float     # mean |curvature|
    curvature_total:       float     # signed total (+ = net clockwise turn, i.e. NH recurvature)
    spread_mean:           float
    spread_final:          float

    # Event features
    recurvature_prob:      float     # in [0, 1]
    deceleration_prob:     float     # in [0, 1]
    bifurcation_index:     float     # in [0, 1]  (0 = unimodal, 1 = bimodal)

    # Displacement
    poleward_displacement: float     # net Δlat (+ = northward)
    zonal_displacement:    float     # net Δlon (+ = eastward)

    def to_feature_vector(self) -> np.ndarray:
        """Return [N_FEATURES] vector aligned with _FEATURE_NAMES."""
        return np.array([
            self.speed_mean,
            self.speed_peak,
            self.curvature_mean,
            self.curvature_total,
            self.spread_mean,
            self.spread_final,
            self.recurvature_prob,
            self.deceleration_prob,
            self.bifurcation_index,
            self.poleward_displacement,
            self.zonal_displacement,
        ], dtype=np.float32)

    def summary(self) -> str:
        lines = [
            "Track Feature Summary:",
            f"  Translation speed   : mean={self.speed_mean:.4f}  peak={self.speed_peak:.4f}",
            f"  Curvature           : mean|κ|={self.curvature_mean:.4f}  "
            f"total turning={self.curvature_total:+.2f}°",
            f"  Ensemble spread     : mean={self.spread_mean:.4f}  final={self.spread_final:.4f}",
            f"  Recurvature prob    : {self.recurvature_prob:.1%}",
            f"  Deceleration prob   : {self.deceleration_prob:.1%}",
            f"  Bifurcation index   : {self.bifurcation_index:.3f}",
            f"  Poleward displ.     : {self.poleward_displacement:+.4f}  "
            f"({'northward' if self.poleward_displacement > 0 else 'southward'})",
            f"  Zonal displ.        : {self.zonal_displacement:+.4f}  "
            f"({'eastward' if self.zonal_displacement > 0 else 'westward'})",
        ]
        return "\n".join(lines)


@dataclass
class FeatureAttributionResult:
    """Full result of C8 Track Feature Attribution."""

    # Attribution matrices (point estimates)
    channel_attr_matrix: np.ndarray      # [N_FEATURES, 13]  Δf per channel
    window_attr_matrix:  np.ndarray      # [N_FEATURES,  8]  Δf per window

    # Normalised attribution (z-score across channels per feature)
    # Removes cross-feature scale differences; comparable across features/storms
    channel_attr_norm: np.ndarray        # [N_FEATURES, 13]  z-scored per row
    window_attr_norm:  np.ndarray        # [N_FEATURES,  8]  z-scored per row

    # Bootstrap confidence intervals on channel attributions
    # channel_attr_ci_lo/hi[f, c] = 95% CI bound from B bootstrap resamples
    channel_attr_ci_lo: np.ndarray       # [N_FEATURES, 13]
    channel_attr_ci_hi: np.ndarray       # [N_FEATURES, 13]
    channel_attr_sig:   np.ndarray       # [N_FEATURES, 13]  bool: CI excludes 0

    # Named axes
    feature_names: List[str]             # _FEATURE_NAMES
    channel_names: List[str]             # _CH_NAMES
    window_labels: List[str]             # _WINDOW_LABELS

    # Baseline track features
    baseline_features: TrackFeatures

    # Top driver per feature
    top_channel_per_feature: Dict[str, Tuple[str, float]]   # feature → (channel, Δ)
    top_window_per_feature:  Dict[str, Tuple[str, float]]   # feature → (window,  Δ)

    # Per-feature narrative
    feature_summaries: Dict[str, str]

    # Global physical interpretation
    physical_interpretation: str

    # Faithfulness (MoRF-AOPC): higher = attribution is validated
    faithfulness_scores: np.ndarray          # [N_FEATURES]

    # Feature group classification
    feature_groups: Dict[str, str]           # feature → "physical" | "uncertainty"


# ─── Feature Extractor ────────────────────────────────────────────────────────

class TrackFeatureExtractor:
    """Extract meteorological features from ensemble trajectory predictions.

    Parameters
    ----------
    recurvature_threshold_deg : heading change required to call recurvature
    deceleration_threshold    : fractional speed drop to call deceleration
    """

    def __init__(
        self,
        recurvature_threshold_deg: float = 45.0,
        deceleration_threshold:    float = 0.30,
    ):
        self.recurvature_threshold = recurvature_threshold_deg
        self.decel_threshold       = deceleration_threshold

    def extract(self, all_traj: torch.Tensor, storm_idx: int = 0) -> TrackFeatures:
        """Extract features from ensemble trajectories.

        Parameters
        ----------
        all_traj   : [K, T_pred, B, 2]  ensemble trajectories, ABSOLUTE normalised (lon_n, lat_n)
        storm_idx  : which storm in the batch to analyse

        Returns
        -------
        TrackFeatures
        """
        # Select storm → [K, T_pred, 2]  ABSOLUTE normalised (lon_n, lat_n), 1 unit = 5 deg
        traj = all_traj[:, :, storm_idx, :].cpu().float().numpy()
        K, T, _ = traj.shape
        ll = norm_to_lonlat(traj)                               # [K, T, 2] (lon°E, lat°N)

        # ── Per-step velocity components in km (east, north) ─────────────────
        east_km, north_km = step_enu_km(traj)                   # each [K, T-1]
        speed_raw = np.sqrt(east_km ** 2 + north_km ** 2)       # km per step (6 h)
        speed_mean_over_k = speed_raw.mean(0)                   # [T-1]

        # ── Heading (compass degrees, clockwise from North) ──────────────────
        heading_rad = np.arctan2(east_km, north_km)             # [K, T-1]
        heading_deg = np.degrees(heading_rad) % 360             # [K, T-1]
        heading_mean_over_k = self._circular_mean(heading_deg, axis=0)  # [T-1]

        # ── Curvature: rate of heading change ─────────────────────────────────
        # Signed difference of consecutive headings (handle wrap-around)
        h1 = heading_deg[:, :-1]   # [K, T-2]
        h2 = heading_deg[:, 1:]    # [K, T-2]
        curv = self._angle_diff(h2, h1)                    # [K, T-2] signed °
        curvature_mean_over_k = curv.mean(0)               # [T-2]

        # ── Ensemble spread ───────────────────────────────────────────────────
        # std of the member positions in km (east-west scaled by cos(mean lat))
        lat_km = ll[:, :, 1].std(0) * KM_PER_DEG                 # [T]
        lon_km = ll[:, :, 0].std(0) * KM_PER_DEG * np.cos(np.deg2rad(ll[:, :, 1].mean(0)))
        spread = np.sqrt(lat_km ** 2 + lon_km ** 2)              # [T] km

        # ── Scalar summaries ──────────────────────────────────────────────────
        speed_mean  = float(speed_mean_over_k.mean())
        speed_peak  = float(speed_mean_over_k.max())
        curv_mean   = float(np.abs(curvature_mean_over_k).mean()) if len(curvature_mean_over_k) > 0 else 0.0
        curv_total  = float(curvature_mean_over_k.sum()) if len(curvature_mean_over_k) > 0 else 0.0
        spread_mean = float(spread.mean())
        spread_fin  = float(spread[-1]) if len(spread) > 0 else 0.0

        # ── Event features ────────────────────────────────────────────────────
        recurvature_prob  = self._recurvature_probability(heading_deg)
        deceleration_prob = self._deceleration_probability(speed_raw)
        bifurcation_idx   = self._bifurcation_index(traj)

        # ── Displacement ──────────────────────────────────────────────────────
        poleward = float((ll[:, -1, 1] - ll[:, 0, 1]).mean())   # degrees latitude
        zonal    = float((ll[:, -1, 0] - ll[:, 0, 0]).mean())   # degrees longitude

        return TrackFeatures(
            speed=speed_mean_over_k,
            heading=heading_mean_over_k,
            curvature=curvature_mean_over_k,
            spread=spread,
            speed_mean=speed_mean,
            speed_peak=speed_peak,
            curvature_mean=curv_mean,
            curvature_total=curv_total,
            spread_mean=spread_mean,
            spread_final=spread_fin,
            recurvature_prob=float(recurvature_prob),
            deceleration_prob=float(deceleration_prob),
            bifurcation_index=float(bifurcation_idx),
            poleward_displacement=poleward,
            zonal_displacement=zonal,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _circular_mean(angles_deg: np.ndarray, axis: int = 0) -> np.ndarray:
        rad = np.radians(angles_deg)
        return np.degrees(np.arctan2(np.sin(rad).mean(axis), np.cos(rad).mean(axis))) % 360

    @staticmethod
    def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Signed shortest angular difference a − b in degrees ∈ (−180, 180]."""
        d = a - b
        return (d + 180) % 360 - 180

    def _recurvature_probability(self, heading_deg: np.ndarray) -> float:
        """Fraction of ensemble members that show a clockwise (recurving) turn.

        Recurvature = net heading change from first half to second half
        of the forecast is > threshold degrees poleward (northward = <180°).
        """
        K, T = heading_deg.shape
        if T < 4:
            return 0.0
        mid = T // 2
        h_early = self._circular_mean(heading_deg[:, :mid], axis=1)   # [K]
        h_late  = self._circular_mean(heading_deg[:, mid:], axis=1)   # [K]
        delta   = self._angle_diff(h_late, h_early)                   # [K] signed
        # Compass heading is clockwise from North. In the NH a westward-moving storm that
        # recurves turns W(270°)→N(360°)→E(90°), i.e. a CLOCKWISE (positive) heading change.
        poleward_turn = delta > self.recurvature_threshold
        return float(poleward_turn.mean())

    def _deceleration_probability(self, speed_raw: np.ndarray) -> float:
        """Fraction of ensemble members showing significant speed drop.

        Deceleration = max(speed[t] − speed[t+1]) / max(speed) > threshold.
        """
        K, T1 = speed_raw.shape
        if T1 < 2:
            return 0.0
        max_speed = speed_raw.max(1, keepdims=True).clip(1e-6)   # [K, 1]
        drops = (speed_raw[:, :-1] - speed_raw[:, 1:]) / max_speed  # [K, T1-1]
        sig_drop = (drops.max(1) > self.decel_threshold)             # [K]
        return float(sig_drop.mean())

    # Expected η² for a median-split of a perfectly unimodal Gaussian:
    #   E[x | x≥0] = σ√(2/π) ≈ 0.798σ for N(0,σ²)
    #   between_var / total_var = 2*(0.798)²/π / 1 ≈ 0.637
    # Any distribution with η²_raw ≤ this is consistent with unimodality.
    _ETA2_GAUSSIAN_NULL: float = 0.637

    def _bifurcation_index(self, traj: np.ndarray) -> float:
        """Track bifurcation index via principal-axis cluster separation (corrected η²).

        Method
        ------
        1. Project ensemble final positions onto their principal axis (PCA, 2D→1D).
        2. Split at the median to form two candidate clusters A and B.
        3. Compute raw η² = SS_between / SS_total  (ANOVA effect size).
        4. Correct for the unimodal Gaussian null: a perfectly unimodal Gaussian
           gives raw η² ≈ 0.637 with a median split, NOT 0.  The corrected index
           subtracts this null baseline and rescales to [0, 1]:

               η²_corrected = clip((η²_raw − 0.637) / (1.0 − 0.637), 0, 1)

           This ensures:
             η²_corrected = 0.0  →  distribution consistent with unimodal Gaussian
             η²_corrected = 1.0  →  two perfectly separated point clusters

        Scientific basis
        ----------------
        - η² (eta-squared) is the standard ANOVA effect size (Cohen 1988, §5.1):
            SS_between / SS_total, equivalent to R² from a 2-group regression.
        - The Gaussian null correction follows the same principle as Cohen's d
          corrected for reference distribution — attributing excess variance above
          baseline to bimodal structure.
        - For a unimodal N(0,σ²): E[x | x≥0] = σ√(2/π) ≈ 0.798σ (half-normal mean)
          → E[between_var] = (K/2*(0.798σ)² + K/2*(0.798σ)²)/K = 0.637σ²
          → raw η² ≈ 0.637 / 1.0 = 0.637 regardless of σ.
        - Natural threshold for corrected η²:
            > 0.30 : moderate bimodality (between-cluster distance > 1.1σ_within)
            > 0.50 : strong bimodality (two well-separated clusters)

        Advantages over bimodality coefficient (BC = (γ₁²+1)/γ₂):
          - BC requires kurtosis (γ₂), which is undefined/unstable for K < 20
          - BC is a 1D moment statistic; this method uses PCA to handle 2D structure
          - η²_corrected has a direct variance-decomposition interpretation
        """
        final_pos = traj[:, -1, :]    # [K, 2]
        K = final_pos.shape[0]
        if K < 4:
            return 0.0

        try:
            # PCA: find principal axis of dispersion in 2D
            mean     = final_pos.mean(0)            # [2]
            centered = final_pos - mean             # [K, 2]
            cov      = (centered.T @ centered) / max(K - 1, 1)   # [2, 2] sample cov
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, -1]              # [2] eigenvector of largest eigenvalue

            # Project onto principal axis → 1D scalar per ensemble member
            proj = centered @ principal             # [K]

            # total_var = SS_total / K  (population variance, ddof=0)
            total_var = float(proj.var())           # [scalar]
            if total_var < 1e-12:
                return 0.0

            # Median split → two groups A (≥ median) and B (< median)
            med    = float(np.median(proj))
            mask_a = proj >= med
            mask_b = ~mask_a
            if mask_a.sum() < 2 or mask_b.sum() < 2:
                return 0.0

            mean_a = float(proj[mask_a].mean())
            mean_b = float(proj[mask_b].mean())
            grand  = float(proj.mean())

            # SS_between / K  (between-cluster component of population variance)
            between_var = (mask_a.sum() * (mean_a - grand) ** 2 +
                           mask_b.sum() * (mean_b - grand) ** 2) / K

            # Raw η²  =  SS_between / SS_total  (both divided by K, K cancels)
            eta2_raw = float(np.clip(between_var / (total_var + 1e-10), 0.0, 1.0))

            # Correct for unimodal Gaussian null (η²_raw ≈ 0.637 for N(0,σ²))
            # Maps [0.637, 1.0] → [0.0, 1.0]; anything ≤ null → 0 (clipped)
            eta2_corrected = float(np.clip(
                (eta2_raw - self._ETA2_GAUSSIAN_NULL) / (1.0 - self._ETA2_GAUSSIAN_NULL),
                0.0, 1.0,
            ))
            return eta2_corrected

        except np.linalg.LinAlgError:
            return 0.0


# ─── Feature Attributor ───────────────────────────────────────────────────────

class TrackFeatureAttributor:
    """C8: Attribute each track feature to ERA5 channels and time windows.

    For each perturbation (zero-out one channel or one time window),
    re-run the model ensemble and measure the change in each track feature.

    Usage
    -----
    >>> attr = TrackFeatureAttributor(model, device, K=10)
    >>> result = attr.compute(batch_list, storm_idx=0)
    >>> print(result.physical_interpretation)

    Parameters
    ----------
    model   : trained TCFlowMatching
    device  : torch device
    K       : ensemble size for each perturbation run (default 10)
    obs_len : number of observed time steps (default 8)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        K: int = 10,
        obs_len: int = 8,
    ):
        self.model   = model
        self.device  = device
        self.K       = K
        self.obs_len = obs_len
        self.extractor = TrackFeatureExtractor()
        self.model.eval()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_device(self, batch_list: list) -> list:
        out = []
        for item in batch_list:
            if isinstance(item, torch.Tensor):
                out.append(item.to(self.device))
            elif isinstance(item, dict):
                out.append({k: v.to(self.device) if torch.is_tensor(v) else v
                             for k, v in item.items()})
            else:
                out.append(item)
        return out

    def _deep_copy(self, batch_list: list) -> list:
        out = []
        for item in batch_list:
            if isinstance(item, torch.Tensor):
                out.append(item.clone())
            elif isinstance(item, dict):
                out.append({k: v.clone() if torch.is_tensor(v) else copy.copy(v)
                             for k, v in item.items()})
            else:
                out.append(item)
        return out

    def _run_ensemble(self, batch_list: list) -> torch.Tensor:
        """Run model K times → [K, T_pred, B, 2]."""
        with torch.no_grad():
            _, _, all_traj = self.model.sample(
                batch_list,
                num_ensemble=self.K,
                use_speed_calibration=True,
                return_xai=False,
            )
        return all_traj  # [K, T_pred, B, 2]

    def _features_from_batch(self, batch_list: list, storm_idx: int) -> np.ndarray:
        """Run ensemble and return feature vector [N_FEATURES]."""
        all_traj = self._run_ensemble(batch_list)
        feats    = self.extractor.extract(all_traj, storm_idx=storm_idx)
        return feats.to_feature_vector(), feats

    # ── perturbation masks ────────────────────────────────────────────────────

    def _zero_channel(self, batch_list: list, ch_idx: int) -> list:
        """Replace ERA5 channel ch_idx with its climatological (spatiotemporal) mean.

        Instead of zeroing (which creates an unphysical state), we replace the
        channel values with their spatiotemporal mean — preserving the mean climate
        state while removing all spatial and temporal variation. This is equivalent
        to asking: "what if this channel gave no information beyond the climatological
        average?" — physically analogous to C1's climatological baseline perturbation.

        img_obs: [B, 13, T_obs, H, W]
        """
        bl = self._deep_copy(batch_list)
        if len(bl) > 11 and torch.is_tensor(bl[11]):
            img     = bl[11]                                   # [B, 13, T_obs, H, W]
            ch_mean = img[:, ch_idx, :, :, :].mean()          # scalar spatiotemporal mean
            bl[11][:, ch_idx, :, :, :] = ch_mean
        return bl

    def _zero_window(self, batch_list: list, window: int) -> list:
        """Replace observation window `window` with its temporal mean (climatological baseline).

        Instead of zeroing the window, we replace all fields at that time step
        with their temporal mean across all observation windows. This removes the
        synoptic signal at that moment while preserving the mean atmospheric state.

        Also replaces kinematics (obs_traj) and scalar obs (obs_Me) with their
        temporal mean to maintain internal consistency.
        """
        bl = self._deep_copy(batch_list)
        # ERA5 fields: [B, 13, T_obs, H, W] → replace window with temporal mean
        if len(bl) > 11 and torch.is_tensor(bl[11]):
            if window < bl[11].shape[2]:
                t_mean = bl[11].mean(dim=2)                    # [B, 13, H, W]
                bl[11][:, :, window, :, :] = t_mean
        # obs_traj: [T_obs, B, 2] → replace with temporal mean position
        if torch.is_tensor(bl[0]) and window < bl[0].shape[0]:
            t_mean_traj = bl[0].mean(dim=0)                    # [B, 2]
            bl[0][window] = t_mean_traj
        # obs_Me: [T_obs, B, 4] → replace with temporal mean
        if len(bl) > 7 and torch.is_tensor(bl[7]) and window < bl[7].shape[0]:
            t_mean_me = bl[7].mean(dim=0)                      # [B, 4]
            bl[7][window] = t_mean_me
        return bl

    # ── attribution normalisation ─────────────────────────────────────────────

    @staticmethod
    def _normalise_attr(matrix: np.ndarray) -> np.ndarray:
        """Z-score each row (feature) independently.

        Removes cross-feature scale differences so the matrix is comparable
        across features with very different magnitudes (e.g. speed_mean ≈ 0.1
        vs recurvature_prob ∈ [0,1]). Within each feature, the sign and ranking
        of channels is preserved.

        Returns the normalised matrix; rows with std ≈ 0 are left as-is.
        """
        normed = matrix.copy()
        for fi in range(matrix.shape[0]):
            row_std = matrix[fi].std()
            if row_std > 1e-8:
                normed[fi] = (matrix[fi] - matrix[fi].mean()) / row_std
        return normed

    # ── bootstrap confidence intervals ────────────────────────────────────────

    def compute_bootstrap_ci(
        self,
        batch_list:    list,
        storm_idx:     int   = 0,
        n_bootstrap:   int   = 30,
        alpha:         float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap 95% confidence intervals on channel attribution.

        Procedure
        ---------
        In each bootstrap replicate b:
          1. Draw a fresh baseline trajectory ensemble (re-run FM stochastic ODE)
             → base_vec_b   (different random noise seed each time)
          2. For each channel c, draw a fresh perturbed ensemble
             → vec_c_b      (also different seed)
          3. Attribution sample: Δf_b[f, c] = base_vec_b[f] − vec_c_b[f]

        The percentile CI over B resamples correctly captures the JOINT
        variability of (baseline − perturbed), including:
          - Variability from stochastic FM sampling of the baseline
          - Variability from stochastic FM sampling of the perturbed run

        This is more conservative than fixing the baseline (which underestimates
        variance and inflates statistical significance).  Formally this is a
        Monte Carlo CI under the FM stochastic model — a proper bootstrap
        in the sense of Efron (1979) applied to the attribution statistic.

        Parameters
        ----------
        batch_list   : original batch (on device)
        storm_idx    : which storm in the batch
        n_bootstrap  : number of bootstrap resamples (default 30, ~2min on GPU)
        alpha        : significance level (0.05 → 95% CI)

        Returns
        -------
        ci_lo  : [N_FEATURES, 13]  lower CI bound
        ci_hi  : [N_FEATURES, 13]  upper CI bound
        sig    : [N_FEATURES, 13]  bool — True if CI excludes 0 (significant)
        """
        bl = self._to_device(batch_list)

        # Collect Δf samples per channel — BOTH baseline and perturbed resampled
        # each iteration to correctly estimate the variance of the attribution.
        samples = np.zeros((n_bootstrap, N_FEATURES, 13), dtype=np.float32)

        for b in range(n_bootstrap):
            # Fresh baseline draw (different FM random seed)
            base_vec_b, _ = self._features_from_batch(bl, storm_idx)
            for c in range(13):
                bl_c     = self._zero_channel(bl, c)
                vec_c, _ = self._features_from_batch(bl_c, storm_idx)
                samples[b, :, c] = base_vec_b - vec_c

        lo  = np.percentile(samples, 100 * alpha / 2,       axis=0)  # [N_FEATURES, 13]
        hi  = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        sig = ~((lo <= 0) & (hi >= 0))   # True = CI does not straddle zero

        return lo.astype(np.float32), hi.astype(np.float32), sig.astype(bool)

    # ── faithfulness validation ───────────────────────────────────────────────

    def compute_faithfulness(
        self,
        batch_list: list,
        ch_attr_matrix: np.ndarray,
        baseline_fvec:  np.ndarray,
        storm_idx: int = 0,
        n_steps:   int = 6,
    ) -> np.ndarray:
        """MoRF-AOPC faithfulness check for C8 channel attributions.

        Removes ERA5 channels cumulatively in decreasing attribution order
        (Most Relevant First — MoRF). If the attribution ranking is correct,
        the most-attributed channel should cause the largest single drop, and
        cumulative removal should steadily increase the total feature change.

        This mirrors the AOPC faithfulness used in C7, adapted to features
        instead of trajectory errors.

        Parameters
        ----------
        batch_list      : original (unperturbed) batch on device
        ch_attr_matrix  : [N_FEATURES, 13] attribution matrix already computed
        baseline_fvec   : [N_FEATURES] baseline feature vector
        storm_idx       : which storm in the batch
        n_steps         : how many channels to remove cumulatively (max 13)

        Returns
        -------
        aopc : np.ndarray [N_FEATURES]
            Area-Over-the-Perturbation-Curve normalised by |baseline_feature|.
            Higher = attribution ranking is more faithful.
        """
        n_steps = min(n_steps, 13)
        aopc    = np.zeros(N_FEATURES, dtype=np.float32)

        for fi in range(N_FEATURES):
            attr_row   = ch_attr_matrix[fi]                    # [13]
            morf_order = np.argsort(-np.abs(attr_row))        # descending |Δf|

            cumulative_deltas = []
            bl_accum = self._deep_copy(batch_list)            # accumulate removals

            for k in range(n_steps):
                ch = int(morf_order[k])
                # Apply climatological mean to this channel (matching _zero_channel)
                if len(bl_accum) > 11 and torch.is_tensor(bl_accum[11]):
                    ch_mean = batch_list[11][:, ch, :, :, :].mean()
                    bl_accum[11][:, ch, :, :, :] = ch_mean

                vec_k, _ = self._features_from_batch(bl_accum, storm_idx)
                delta = float(abs(baseline_fvec[fi] - vec_k[fi]))
                cumulative_deltas.append(delta)

            if cumulative_deltas:
                # Normalise by max(|baseline|, 0.01) to avoid division-by-near-zero
                # when baseline features are close to 0 (e.g. recurvature_prob = 0.0
                # early in the season).  The floor of 0.01 bounds AOPC to ≤ 100×
                # the absolute change, which is still a meaningful faithfulness score.
                denom    = max(float(abs(baseline_fvec[fi])), 0.01)
                aopc[fi] = float(np.mean(cumulative_deltas) / denom)

        return aopc

    # ── main compute ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute(
        self,
        batch_list: list,
        storm_idx: int = 0,
        verbose: bool = False,
    ) -> FeatureAttributionResult:
        """Compute feature attribution over ERA5 channels and time windows.

        Returns
        -------
        FeatureAttributionResult
        """
        bl = self._to_device(batch_list)

        # ── Baseline ──────────────────────────────────────────────────────────
        if verbose:
            print("[C8] Computing baseline features ...")
        base_vec, base_feats = self._features_from_batch(bl, storm_idx)

        # ── Channel attribution: [N_FEATURES, 13] ─────────────────────────────
        ch_attr = np.zeros((N_FEATURES, 13), dtype=np.float32)
        for c in range(13):
            if verbose:
                print(f"[C8] Channel perturbation {c+1}/13: {_CH_NAMES[c]} ...", end="\r")
            bl_c  = self._zero_channel(bl, c)
            vec_c, _ = self._features_from_batch(bl_c, storm_idx)
            ch_attr[:, c] = base_vec - vec_c   # Δ = baseline − perturbed

        if verbose:
            print()

        # ── Window attribution: [N_FEATURES, 8] ───────────────────────────────
        win_attr = np.zeros((N_FEATURES, self.obs_len), dtype=np.float32)
        for w in range(self.obs_len):
            if verbose:
                print(f"[C8] Window perturbation {w+1}/{self.obs_len}: "
                      f"{_WINDOW_LABELS[w]} ...", end="\r")
            bl_w  = self._zero_window(bl, w)
            vec_w, _ = self._features_from_batch(bl_w, storm_idx)
            win_attr[:, w] = base_vec - vec_w

        if verbose:
            print()

        # ── Top drivers per feature ────────────────────────────────────────────
        top_ch  = {}
        top_win = {}
        for fi, fname in enumerate(_FEATURE_NAMES):
            best_c = int(np.abs(ch_attr[fi]).argmax())
            best_w = int(np.abs(win_attr[fi]).argmax())
            top_ch[fname]  = (_CH_NAMES[best_c], float(ch_attr[fi, best_c]))
            top_win[fname] = (_WINDOW_LABELS[best_w], float(win_attr[fi, best_w]))

        # ── Normalised attribution (z-score per feature) ──────────────────────
        ch_attr_norm  = self._normalise_attr(ch_attr)
        win_attr_norm = self._normalise_attr(win_attr)

        # ── Bootstrap CI on channel attribution ───────────────────────────────
        if verbose:
            print(f"[C8] Computing bootstrap CI (n_bootstrap=30) ...")
        ci_lo, ci_hi, sig = self.compute_bootstrap_ci(bl, storm_idx=storm_idx)
        n_sig = int(sig.sum())
        if verbose:
            print(f"[C8] {n_sig}/{N_FEATURES * 13} attributions significant (CI excludes 0)")

        # ── Faithfulness (MoRF-AOPC) ──────────────────────────────────────────
        if verbose:
            print("[C8] Computing MoRF-AOPC faithfulness scores ...")
        faithfulness = self.compute_faithfulness(
            bl, ch_attr, base_vec, storm_idx=storm_idx, n_steps=6
        )
        if verbose:
            print(f"[C8] Mean AOPC = {faithfulness.mean():.4f}")

        # ── Feature groups ────────────────────────────────────────────────────
        feature_groups = {
            f: ("physical" if f in _PHYSICAL_FEATURE_NAMES else "uncertainty")
            for f in _FEATURE_NAMES
        }

        # ── Per-feature narrative ─────────────────────────────────────────────
        feat_summaries = self._feature_summaries(
            base_feats, ch_attr, win_attr, top_ch, top_win, faithfulness, sig
        )

        # ── Global interpretation ─────────────────────────────────────────────
        phys = self._interpret(
            base_feats, ch_attr, win_attr, top_ch, top_win, faithfulness, sig
        )

        return FeatureAttributionResult(
            channel_attr_matrix=ch_attr,
            window_attr_matrix=win_attr,
            channel_attr_norm=ch_attr_norm,
            window_attr_norm=win_attr_norm,
            channel_attr_ci_lo=ci_lo,
            channel_attr_ci_hi=ci_hi,
            channel_attr_sig=sig,
            feature_names=list(_FEATURE_NAMES),
            channel_names=list(_CH_NAMES),
            window_labels=list(_WINDOW_LABELS[:self.obs_len]),
            baseline_features=base_feats,
            top_channel_per_feature=top_ch,
            top_window_per_feature=top_win,
            feature_summaries=feat_summaries,
            physical_interpretation=phys,
            faithfulness_scores=faithfulness,
            feature_groups=feature_groups,
        )

    # ── interpretation ────────────────────────────────────────────────────────

    def _feature_summaries(
        self,
        base: TrackFeatures,
        ch_attr:      np.ndarray,    # [N_FEATURES, 13]
        win_attr:     np.ndarray,    # [N_FEATURES, 8]
        top_ch:       Dict,
        top_win:      Dict,
        faithfulness: np.ndarray,    # [N_FEATURES]
        sig:          np.ndarray,    # [N_FEATURES, 13] bool significance mask
    ) -> Dict[str, str]:
        """Generate one-paragraph interpretation for each feature."""
        summaries = {}

        # ── speed_mean ────────────────────────────────────────────────────────
        fi_sm0 = _FEATURE_NAMES.index("speed_mean")
        ch_name, ch_val = top_ch["speed_mean"]
        wn_name, wn_val = top_win["speed_mean"]
        best_c_sm0      = int(np.abs(ch_attr[fi_sm0]).argmax())
        sig_mark        = "★" if sig[fi_sm0, best_c_sm0] else "○"
        level           = _ch_to_level(ch_name)
        summaries["speed_mean"] = (
            f"TRANSLATION SPEED (mean={base.speed_mean:.4f}): "
            f"The model's predicted translation speed is most sensitive to "
            f"{ch_name} ({level}), with attribution Δ={ch_val:+.4f} {sig_mark}. "
            f"{'Positive attribution indicates this channel amplifies storm speed, ' if ch_val > 0 else 'Negative attribution indicates this channel suppresses speed, '}"
            f"physically consistent with "
            f"{'the subtropical ridge steering the storm through strong advective flow.' if 'GPH' in ch_name else 'upper-level jet influence on translation speed.' if '200' in ch_name else 'low-level boundary layer steering current.'}"
            f" Temporally, {wn_name} is the most critical observation window (Δ={wn_val:+.4f})."
        )

        # ── speed_peak ────────────────────────────────────────────────────────
        ch_name, ch_val = top_ch["speed_peak"]
        summaries["speed_peak"] = (
            f"PEAK TRANSLATION SPEED (max={base.speed_peak:.4f}): "
            f"Peak speed is controlled primarily by {ch_name} (Δ={ch_val:+.4f}). "
            f"Peak speeds typically occur when the storm aligns with the axis of "
            f"maximum steering flow — the model captures this through the "
            f"{_ch_to_level(ch_name)} pattern."
        )

        # ── curvature_mean ────────────────────────────────────────────────────
        ch_name, ch_val = top_ch["curvature_mean"]
        wn_name, wn_val = top_win["curvature_mean"]
        summaries["curvature_mean"] = (
            f"TRACK CURVATURE (mean|κ|={base.curvature_mean:.4f}°/step): "
            f"How much the track curves at each step. Primary driver: {ch_name} "
            f"(Δ={ch_val:+.4f}). "
            f"{'Strong curvature driven by upper-level flow asymmetry — the jet stream creates differential steering across the vortex depth.' if '200' in ch_name or '500' in ch_name else 'Low-level flow channelling creates the curvature signal.'}"
            f" The {wn_name} window is most influential (Δ={wn_val:+.4f})."
        )

        # ── curvature_total ────────────────────────────────────────────────────
        ch_name, ch_val = top_ch["curvature_total"]
        wn_name, wn_val = top_win["curvature_total"]
        direction = "clockwise (recurving)" if base.curvature_total > 0 else "counter-clockwise (equatorward / straightening)"
        summaries["curvature_total"] = (
            f"TOTAL HEADING CHANGE (Σκ={base.curvature_total:+.2f}°, net {direction}): "
            f"The net turning of the track over the forecast window. "
            f"Primary driver: {ch_name} (Δ={ch_val:+.4f}). "
            f"{'A westerly trough approaching from the north deflects the track poleward — the model detects this signal through the upper geopotential pattern.' if 'GPH' in ch_name else 'Wind shear pattern encoded in this channel drives the net heading change.'}"
            f" Most critical window: {wn_name} (Δ={wn_val:+.4f})."
        )

        # ── spread_mean ──────────────────────────────────────────────────────
        fi_sm = _FEATURE_NAMES.index("spread_mean")
        ch_name, ch_val = top_ch["spread_mean"]
        summaries["spread_mean"] = (
            f"[UNCERTAINTY] ENSEMBLE SPREAD (mean={base.spread_mean:.4f}, "
            f"AOPC={faithfulness[fi_sm]:.3f}): "
            f"Epistemic uncertainty of the forecast — how much the ensemble members "
            f"disagree. NOTE: spread is an ensemble-level property, not an intrinsic "
            f"physical feature of a real typhoon track. Its attribution should be "
            f"interpreted as 'which input most constrains the model's uncertainty.' "
            f"Spread is most sensitive to {ch_name} (Δ={ch_val:+.4f}). "
            f"{'When geopotential height is well-defined, spread decreases — confirming GPH as the key uncertainty-reducing input in the steering flow.' if 'GPH' in ch_name else 'This channel reduces ensemble divergence by constraining the initial-condition noise in the flow-matching ODE.'}"
        )

        # ── spread_final ──────────────────────────────────────────────────────
        fi_sf = _FEATURE_NAMES.index("spread_final")
        ch_name, ch_val = top_ch["spread_final"]
        t_final = (base.spread.shape[0] - 1) * 6 if hasattr(base.spread, 'shape') else '?'
        summaries["spread_final"] = (
            f"[UNCERTAINTY] FINAL ENSEMBLE SPREAD (at t+{t_final}h, "
            f"spread={base.spread_final:.4f}, AOPC={faithfulness[fi_sf]:.3f}): "
            f"Long-range forecast uncertainty — how far apart ensemble members are "
            f"at the end of the forecast window. Large final spread often co-occurs "
            f"with track bifurcation. Controlled by {ch_name} (Δ={ch_val:+.4f}). "
            f"This uncertainty measure is meaningful for operational forecasters "
            f"but is not an attribute of the observed typhoon."
        )

        # ── recurvature_prob ──────────────────────────────────────────────────
        ch_name, ch_val = top_ch["recurvature_prob"]
        wn_name, wn_val = top_win["recurvature_prob"]
        summaries["recurvature_prob"] = (
            f"RECURVATURE PROBABILITY ({base.recurvature_prob:.1%}): "
            f"{'High recurvature probability — the model sees conditions favouring a poleward turn.' if base.recurvature_prob > 0.5 else 'Low recurvature probability — straight-track regime predicted.'} "
            f"The dominant atmospheric driver is {ch_name} (Δ={ch_val:+.4f}). "
            f"{'Geopotential height at this level encodes the position of the subtropical ridge. A weakened ridge allows poleward steering from the approaching trough.' if 'GPH' in ch_name else 'Upper-level wind pattern determines the outflow channel and recurvature timing.' if 'U_2' in ch_name or 'V_2' in ch_name else 'SST gradient zones can modulate recurvature by influencing convective asymmetry.'}"
            f" Temporally, the {wn_name} observation window is critical (Δ={wn_val:+.4f}), "
            f"{'consistent with synoptic precursors establishing 24-48h before recurvature.' if 't-3' in wn_name or 't-4' in wn_name else 'suggesting recurvature is driven by the near-current environment.'}"
        )

        # ── deceleration_prob ─────────────────────────────────────────────────
        ch_name, ch_val = top_ch["deceleration_prob"]
        summaries["deceleration_prob"] = (
            f"DECELERATION PROBABILITY ({base.deceleration_prob:.1%}): "
            f"Probability of the storm slowing down significantly during the forecast. "
            f"Primary driver: {ch_name} (Δ={ch_val:+.4f}). "
            f"{'SST modulates track speed by controlling the storm intensity (and thus the effective steering layer depth). Cool SST weakens the storm and reduces translation speed.' if 'SST' in ch_name else 'The steering flow weakening encoded in this channel causes deceleration.'}"
        )

        # ── bifurcation_index ─────────────────────────────────────────────────
        fi_bi = _FEATURE_NAMES.index("bifurcation_index")
        ch_name, ch_val = top_ch["bifurcation_index"]
        bi_val = base.bifurcation_index
        # η²_corrected > 0.30: moderate bimodality (raw η² > 0.746 after Gaussian null correction)
        regime = "BIMODAL — two distinct track scenarios" if bi_val > 0.30 else "unimodal — single dominant track"
        summaries["bifurcation_index"] = (
            f"[UNCERTAINTY] TRACK BIFURCATION INDEX η²={bi_val:.3f} "
            f"({regime}, AOPC={faithfulness[fi_bi]:.3f}): "
            f"Measures how much of the ensemble's endpoint variance is explained "
            f"by a binary (bifurcating) split along the principal axis of dispersion. "
            f"Method: PCA on final positions → median split → η² = between/total variance. "
            f"η² = 1 means two perfectly separated clusters; η² = 0 means unimodal. "
            f"Primary driver of bifurcation: {ch_name} (Δ={ch_val:+.4f}). "
            f"Bifurcation arises near the recurvature decision point when the steering "
            f"environment is ambiguous between ridge persistence and trough approach."
        )

        # ── poleward_displacement ──────────────────────────────────────────────
        ch_name, ch_val = top_ch["poleward_displacement"]
        summaries["poleward_displacement"] = (
            f"POLEWARD DISPLACEMENT (Δlat={base.poleward_displacement:+.4f}, "
            f"{'northward' if base.poleward_displacement > 0 else 'southward'}): "
            f"Net meridional movement. Controlled by {ch_name} (Δ={ch_val:+.4f}). "
            f"{'V-wind (meridional) directly drives poleward steering.' if 'V_' in ch_name else 'Geopotential gradient creates the meridional steering component.' if 'GPH' in ch_name else 'SST gradient zones create organised meridional flow.'}"
        )

        # ── zonal_displacement ────────────────────────────────────────────────
        ch_name, ch_val = top_ch["zonal_displacement"]
        summaries["zonal_displacement"] = (
            f"ZONAL DISPLACEMENT (Δlon={base.zonal_displacement:+.4f}, "
            f"{'eastward' if base.zonal_displacement > 0 else 'westward'}): "
            f"Net east-west movement. Controlled by {ch_name} (Δ={ch_val:+.4f}). "
            f"{'U-wind (zonal) directly drives east-west translation.' if 'U_' in ch_name else 'The subtropical ridge position encoded in GPH drives westward steering.' if 'GPH' in ch_name else 'Boundary layer flow determines zonal translation.'}"
        )

        return summaries

    def _interpret(
        self,
        base:         TrackFeatures,
        ch_attr:      np.ndarray,
        win_attr:     np.ndarray,
        top_ch:       Dict,
        top_win:      Dict,
        faithfulness: np.ndarray,
        sig:          np.ndarray,   # [N_FEATURES, 13] significance mask
    ) -> str:
        """Generate overall meteorological interpretation bulletin."""
        lines = ["\n[C8 — TRACK FEATURE ATTRIBUTION (PFA)]\n"]

        lines.append(base.summary())
        lines.append("")

        # Which channel is most universally important
        ch_importance = np.abs(ch_attr).mean(0)    # [13] mean across features
        top_global_ch = int(ch_importance.argmax())
        lines.append(
            f"GLOBAL DOMINANT ERA5 CHANNEL: {_CH_NAMES[top_global_ch]} "
            f"(mean |Δfeature| = {ch_importance[top_global_ch]:.4f})"
        )

        win_importance = np.abs(win_attr).mean(0)  # [8]
        top_global_win = int(win_importance.argmax())
        lines.append(
            f"GLOBAL DOMINANT TIME WINDOW:  {_WINDOW_LABELS[top_global_win]} "
            f"(mean |Δfeature| = {win_importance[top_global_win]:.4f})"
        )

        # Regime narrative
        lines.append("")
        lines.append("TRACK REGIME NARRATIVE:")

        if base.recurvature_prob > 0.6:
            lines.append(
                "→ HIGH RECURVATURE PROBABILITY. The model detects conditions for "
                "a poleward track shift. The key ERA5 driver ("
                f"{top_ch['recurvature_prob'][0]}) encodes the approaching trough / "
                "ridge weakness. Forecast confidence depends on how well the model "
                "can resolve the steering flow transition."
            )
        elif base.recurvature_prob < 0.2 and base.zonal_displacement < -0.5:
            lines.append(
                "→ STRAIGHT-WEST TRACK. The model predicts continued westward "
                "translation under a persistent subtropical ridge. "
                f"Speed controlled by {top_ch['speed_mean'][0]}. "
                "Deceleration risk is low unless the ridge weakens."
            )
        else:
            lines.append(
                "→ TRANSITIONAL / UNCERTAIN TRACK. The model sees mixed signals "
                "between continued straight motion and incipient recurvature. "
                f"Track bifurcation index η²_corr = {base.bifurcation_index:.3f} "
                f"({'BIMODAL' if base.bifurcation_index > 0.30 else 'unimodal'})."
            )

        if base.deceleration_prob > 0.4:
            lines.append(
                f"→ DECELERATION RISK ({base.deceleration_prob:.0%}). "
                f"Primary driver: {top_ch['deceleration_prob'][0]}. "
                "Deceleration scenarios are associated with weakening steering flow "
                "or approaching land interaction."
            )

        # Feature–channel attribution table with faithfulness + significance
        n_total_sig = int(sig.sum())
        lines.append("")
        lines.append(
            f"TOP DRIVER PER TRACK FEATURE  "
            f"(★ = CI excludes 0, {n_total_sig}/{N_FEATURES * 13} significant):"
        )
        lines.append(f"  {'Feature':30s}  {'Group':11s}  {'ERA5 Channel':12s}  "
                     f"{'Δ':>8s}  {'Sig':3s}  {'Window':8s}  {'Δ':>8s}  {'AOPC':>6s}")
        lines.append("  " + "-" * 102)
        for fi, fname in enumerate(_FEATURE_NAMES):
            ch_n, ch_v = top_ch[fname]
            wn_n, wn_v = top_win[fname]
            grp        = "physical   " if fname in _PHYSICAL_FEATURE_NAMES else "uncertainty"
            best_c     = int(np.abs(ch_attr[fi]).argmax())
            sig_mark   = " ★ " if sig[fi, best_c] else " ○ "
            lines.append(
                f"  {fname:30s}  {grp}  {ch_n:12s}  {ch_v:+8.4f}  "
                f"{sig_mark}  {wn_n:8s}  {wn_v:+8.4f}  {faithfulness[fi]:6.3f}"
            )

        # Faithfulness summary
        phys_idx  = [_FEATURE_NAMES.index(f) for f in _PHYSICAL_FEATURE_NAMES]
        uncert_idx = [_FEATURE_NAMES.index(f) for f in _UNCERTAINTY_FEATURE_NAMES]
        mean_aopc_phys   = faithfulness[phys_idx].mean()
        mean_aopc_uncert = faithfulness[uncert_idx].mean()
        lines.append("")
        lines.append(
            f"MoRF-AOPC FAITHFULNESS:  "
            f"physical features = {mean_aopc_phys:.3f}  |  "
            f"uncertainty features = {mean_aopc_uncert:.3f}  |  "
            f"overall = {faithfulness.mean():.3f}"
        )
        lines.append(
            "(AOPC > 0.1 indicates attribution is non-trivially faithful; "
            "higher = more confident the ranking is correct)"
        )

        return "\n".join(lines)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ch_to_level(ch_name: str) -> str:
    levels = {
        "200": "upper-tropospheric (200 hPa) outflow layer",
        "500": "mid-tropospheric (500 hPa) primary steering layer",
        "850": "lower-tropospheric (850 hPa) boundary layer",
        "925": "near-surface (925 hPa) boundary layer",
    }
    for code, desc in levels.items():
        if code in ch_name:
            return desc
    return "sea-surface (SST)" if "SST" in ch_name else ch_name


# ─── Convenience function ─────────────────────────────────────────────────────

def compute_track_feature_attribution(
    model,
    batch_list: list,
    device: torch.device,
    K: int = 10,
    storm_idx: int = 0,
    verbose: bool = False,
) -> FeatureAttributionResult:
    """One-line API for C8 Track Feature Attribution.

    Parameters
    ----------
    model      : trained TCFlowMatching
    batch_list : single batch from DataLoader
    device     : torch.device
    K          : ensemble size per perturbation run
    storm_idx  : which storm in batch to analyse

    Returns
    -------
    FeatureAttributionResult with full attribution matrices and interpretation

    Example
    -------
    >>> result = compute_track_feature_attribution(model, batch_list, device)
    >>> print(result.physical_interpretation)
    >>> print(result.feature_summaries["recurvature_prob"])
    """
    attributor = TrackFeatureAttributor(model, device, K=K)
    return attributor.compute(batch_list, storm_idx=storm_idx, verbose=verbose)
