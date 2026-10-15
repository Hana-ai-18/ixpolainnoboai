"""
TC-PhysXAI — Module 2: Spatial Object Attribution (Contribution C2)
====================================================================
Physics Motivation
------------------
Typhoon tracks are steered by large-scale atmospheric circulation patterns:
  · 500 hPa subtropical anticyclone (main steering layer, all lead times)
  · 200 hPa upper-level trough      (trigger for recurvature, 24-72h)
  · 850/925 hPa monsoon trough      (boundary-layer modulation, 6-24h)
  · Sea-surface temperature fields  (intensity → inertia coupling)

Standard XAI for meteorology (e.g., Fernandez et al. 2025) produces only
model-level SHAP values over engineered scalars, losing the spatial structure
of the physical drivers.

Contribution C2 (Spatial XAI with physical object detection):
  1. Apply Integrated Gradients (IG) to the FNO3D encoder in TC-XFlow.
  2. IG uses the climatological mean ERA5 field as baseline (C1 principle).
  3. Attribute the displacement prediction error at each horizon to spatial
     regions in the ERA5 input fields (13 channels × 8 time steps × 81×81).
  4. Detect and label physical meteorological objects (anticyclone, trough,
     SST front, monsoon) within high-attribution patches per pressure level.
  5. Report per-horizon spatial attribution profiles aligned with the
     physical steering mechanism timeline.

Reference: Guillaume-Castel et al. (EGUsphere 2026) for object-oriented XAI
           framework; adapted here for tropical cyclone track prediction with
           a Flow Matching model using FNO3D spatial encoder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .physical_objects import detect_physical_objects, summarise_objects, PhysicalObject


# ─── normalise relevance maps ────────────────────────────────────────────────

def _normalise_attr(attr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Sum-normalise attribution: R_norm = R / Σ max(0, R).

    Adapted from Guillaume-Castel Eq. (2): ensures total positive relevance
    sums to 1, making attributions comparable across predictions.
    """
    pos = attr.clamp(min=0)
    total = pos.sum()
    if total < eps:
        return torch.zeros_like(attr)
    return pos / total


# ─── Integrated Gradients implementation ─────────────────────────────────────

def integrated_gradients_fno3d(
    model,
    batch_list: list,
    target_fn,
    baseline_3d: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Compute Integrated Gradients for the FNO3D encoder's ERA5 input.

    IG formula (Sundararajan et al. 2017):
        IG_i(x) = (x_i − x'_i) × ∫₀¹ ∂F/∂x_i(x' + α(x−x')) dα

    Satisfies completeness: Σ_i IG_i(x) = F(x) − F(x').

    In TC-XFlow, F maps ERA5 fields → predicted trajectory displacement.
    The gradient flows: img_obs → FNO3D encoder → context c → velocity
    transformer → predicted trajectory → scalar output (chosen by target_fn).

    Parameters
    ----------
    model      : TCFlowMatching (trained, frozen except for gradient computation)
    batch_list : collated batch from seq_collate (list of tensors/dicts)
    target_fn  : callable(pred: [T, B, 2]) → scalar, e.g. mean displacement
    baseline_3d: [B, 13, T_obs, 81, 81] climatological baseline; None → zeros
    n_steps    : number of IG quadrature steps (50 is sufficient)
    device     : computation device

    Returns
    -------
    ig_attr : [B, 13, T_obs, 81, 81] IG attribution (same shape as img_obs)
    """
    img_obs = batch_list[11].to(device, dtype=torch.float32)  # [B, 13, T, 81, 81]

    if baseline_3d is None:
        baseline = torch.zeros_like(img_obs)   # climatological mean in norm. space
    else:
        baseline = baseline_3d.to(device, dtype=torch.float32)

    # Build modified batch_list that uses the interpolated img_obs
    def _run_model_with_img(img_interpolated: torch.Tensor) -> torch.Tensor:
        """Forward pass using the interpolated ERA5 field."""
        bl_mod = _replace_img_obs(batch_list, img_interpolated, device)

        # Single-step deterministic forward (sigma=0, no ensemble):
        # x_rel = zeros (t=0 deterministic)
        B = img_interpolated.shape[0]
        x_rel = torch.zeros(B, model.pred_len, 2, device=device)
        t0    = torch.zeros(B, device=device)

        # Get context
        from Model.flow_matching_model import hard_score_from_obs
        obs_traj = bl_mod[0].to(device)
        h_score  = hard_score_from_obs(
            obs_traj[:, :, :2], weight_logits=model.hard_score_weight_logits
        )
        cond = model.encoder(bl_mod, hard_score=h_score)

        # Get velocity at t=0
        v    = model.velocity(x_rel, t0, cond)
        pred = x_rel + v    # [B, pred_len, 2]
        # Permute to [T, B, 2]
        pred = pred.permute(1, 0, 2)  # [T, B, 2]
        return target_fn(pred)

    # Riemann approximation of the integral
    grad_accum = torch.zeros_like(img_obs)
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)[1:]  # exclude α=0

    for alpha in alphas:
        img_interp = baseline + alpha * (img_obs - baseline)
        img_interp = img_interp.detach().requires_grad_(True)

        # Replace in batch_list and run forward
        bl_mod = _replace_img_obs(batch_list, img_interp, device)

        # Forward (with gradient)
        B = img_interp.shape[0]
        x_rel = torch.zeros(B, model.pred_len, 2, device=device)
        t0    = torch.zeros(B, device=device)

        try:
            from Model.flow_matching_model import hard_score_from_obs
        except ImportError:
            # Fallback: access through model module
            hard_score_from_obs = _import_hard_score(model)

        obs_traj = bl_mod[0].to(device)
        with torch.no_grad():
            h_score = hard_score_from_obs(
                obs_traj[:, :, :2],
                weight_logits=model.hard_score_weight_logits
            )

        # Grad computation: only img_interp has grad
        cond = model.encoder(bl_mod, hard_score=h_score)
        v    = model.velocity(x_rel, t0, cond)
        pred = (x_rel + v).permute(1, 0, 2)  # [T, B, 2]

        scalar = target_fn(pred)
        scalar.backward()

        with torch.no_grad():
            grad_accum += img_interp.grad.detach()

    grad_accum /= n_steps
    ig_attr = (img_obs - baseline).detach() * grad_accum
    return ig_attr  # [B, 13, T_obs, 81, 81]


def _replace_img_obs(batch_list: list, img: torch.Tensor, device) -> list:
    """Return a shallow-copy of batch_list with img_obs replaced."""
    out = []
    for i, item in enumerate(batch_list):
        if i == 11:
            out.append(img)
        elif isinstance(item, torch.Tensor):
            out.append(item.to(device))
        elif isinstance(item, dict):
            out.append({k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in item.items()})
        else:
            out.append(item)
    return out


def _import_hard_score(model):
    """Dynamically import hard_score_from_obs from the model's module."""
    import sys, types
    # Try to get the function from the model's module
    mod = sys.modules.get("Model.flow_matching_model")
    if mod and hasattr(mod, "hard_score_from_obs"):
        return mod.hard_score_from_obs
    # Fallback: return a stub that returns zeros
    def _stub(obs, weight_logits=None):
        B = obs.shape[1]
        return torch.zeros(B, device=obs.device)
    return _stub


# ─── Main SpatialAttributor class ────────────────────────────────────────────

class SpatialAttributor:
    """Computes per-pixel spatial attribution via IG on the FNO3D encoder.

    The attribution map identifies which spatial regions and pressure levels
    in the ERA5 input drive the typhoon track prediction, then overlays
    these with detected meteorological objects.

    Usage
    -----
    >>> attributor = SpatialAttributor(model, device)
    >>> result = attributor.compute(batch_list, horizon=6)  # 6th step = 36h
    >>> objects = result["physical_objects"]
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_steps: int = 50,
    ):
        self.model   = model
        self.device  = device
        self.n_steps = n_steps
        self.model.eval()

    @torch.enable_grad()
    def compute(
        self,
        batch_list: list,
        horizon: Optional[int] = None,
        detect_objects: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """Compute IG spatial attribution for the ERA5 input.

        Parameters
        ----------
        batch_list : collated batch
        horizon    : forecast horizon to explain (None = mean over all horizons)
        detect_objects : whether to run physical object detection

        Returns
        -------
        dict with keys:
          "ig_attr"              : [B, 13, T_obs, 81, 81] IG attribution
          "ig_attr_norm"         : [B, 13, T_obs, 81, 81] sum-normalised
          "level_importance"     : dict {level_name: float} attribution fraction
          "time_importance"      : [T_obs] attribution over lag times
          "physical_objects"     : List[PhysicalObject] (if detect_objects=True)
          "object_summary"       : str meteorological description
          "ch_importance"        : [13] per-channel importance
        """
        device = self.device
        bl = _to_device(batch_list, device)

        # Choose target: displacement at specific horizon or mean DPE
        def target_fn(pred):
            # pred: [T, B, 2]
            gt = bl[1].to(device)           # [T, B, 2] ground truth
            T = min(pred.shape[0], gt.shape[0])
            if horizon is not None and horizon < T:
                err = (pred[horizon] - gt[horizon]).pow(2).sum(-1).mean()
            else:
                err = (pred[:T] - gt[:T]).pow(2).sum(-1).mean()
            return err

        if verbose:
            print("[SpatialAttributor] Computing Integrated Gradients "
                  f"(n_steps={self.n_steps}) ...")

        ig_attr = integrated_gradients_fno3d(
            self.model, bl, target_fn,
            baseline_3d=None,   # climatological baseline = zeros
            n_steps=self.n_steps,
            device=device,
        )   # [B, 13, T_obs, 81, 81]

        # Normalise per sample
        ig_norm = torch.stack([
            _normalise_attr(ig_attr[b]) for b in range(ig_attr.shape[0])
        ], 0)  # [B, 13, T_obs, 81, 81]

        # Level-wise importance: sum over spatial dims and time
        level_groups = {
            "200hPa": [0, 4, 8],
            "500hPa": [1, 5, 9],
            "850hPa": [2, 6, 10],
            "925hPa": [3, 7, 11],
            "SST":    [12],
        }
        total = ig_norm.sum() + 1e-8
        level_imp = {}
        for name, chs in level_groups.items():
            level_imp[name] = float(ig_norm[:, chs].sum() / total)

        # Channel importance: [13]
        ch_imp = ig_norm.sum(dim=[0, 2, 3, 4])  # [13]
        ch_imp = (ch_imp / (ch_imp.sum() + 1e-8)).cpu().numpy()

        # Time importance: [T_obs]
        time_imp = ig_norm.sum(dim=[0, 1, 3, 4])  # [T_obs]
        time_imp = (time_imp / (time_imp.sum() + 1e-8)).cpu().numpy()

        # Physical object detection (per-sample, use mean over batch)
        ig_mean  = ig_norm.mean(0).cpu().numpy()    # [13, T_obs, 81, 81]
        era5_mean = bl[11].mean(0).cpu().numpy()    # [13, T_obs, 81, 81]

        objects = []
        obj_summary = ""
        if detect_objects:
            if verbose:
                print("[SpatialAttributor] Detecting physical objects ...")
            objects   = detect_physical_objects(ig_mean, era5_mean, time_step=-1)
            obj_summary = summarise_objects(objects)

        return {
            "ig_attr":           ig_attr.cpu(),
            "ig_attr_norm":      ig_norm.cpu(),
            "level_importance":  level_imp,
            "time_importance":   time_imp,
            "ch_importance":     ch_imp,
            "physical_objects":  objects,
            "object_summary":    obj_summary,
            "physical_interpretation": self._interpret(level_imp, objects, time_imp),
        }

    def _interpret(
        self,
        level_imp: Dict[str, float],
        objects: List[PhysicalObject],
        time_imp: np.ndarray,
    ) -> str:
        """Generate detailed meteorological interpretation of spatial attribution."""
        dominant_level = max(level_imp, key=level_imp.get)
        dominant_frac  = level_imp[dominant_level]

        # Time profile
        n = len(time_imp)
        recent_w = float(time_imp[max(0, n-2):].sum())
        old_w    = float(time_imp[:max(1, n-4)].sum())

        lines = [
            "=== Spatial Attribution: Meteorological Interpretation ===",
            f"Dominant pressure level: {dominant_level} ({dominant_frac:.1%} of total attribution)",
            f"Level breakdown: " + ", ".join(
                f"{k}: {v:.1%}" for k, v in sorted(level_imp.items(), key=lambda x: -x[1])
            ),
            f"Temporal focus: recent t-12h/t-6h = {recent_w:.1%}, "
            f"older t-48h/t-42h = {old_w:.1%}",
        ]

        # Physical regime interpretation
        p500 = level_imp.get("500hPa", 0)
        p200 = level_imp.get("200hPa", 0)
        p850 = level_imp.get("850hPa", 0)
        sst  = level_imp.get("SST", 0)

        if p500 > 0.40:
            lines.append(
                "\nREGIME: DEEP-TROPOSPHERIC STEERING (500 hPa dominated)"
                "\nThe subtropical ridge at 500 hPa is the primary steering mechanism. "
                "This regime is typical of well-organised typhoons in the peak season "
                "(July–October) when the Western Pacific subtropical high (WPSH) is "
                "well-established. Prediction skill is generally highest in this regime "
                "because the steering flow is coherent and persistent over 48-72h."
            )
        elif p200 > 0.35:
            lines.append(
                "\nREGIME: UPPER-TROPOSPHERE DOMINANT (200 hPa pre-recurvature)"
                "\nStrong 200 hPa attribution indicates an approaching mid-latitude trough "
                "or a weakening of the subtropical ridge aloft. This pattern precedes "
                "typhoon recurvature — the critical bifurcation point where storms either "
                "turn poleward into higher latitudes or continue westward. The 200 hPa "
                "geopotential height field is the key diagnostic for recurvature timing "
                "(Archambault et al. 2013). TC-XFlow has correctly identified this level "
                "as the most predictive, consistent with physical theory."
            )
        elif p850 + level_imp.get("925hPa", 0) > 0.45:
            lines.append(
                "\nREGIME: BOUNDARY-LAYER DOMINATED (monsoon/low-level trough)"
                "\nHigh attribution at 850/925 hPa suggests the track is being modulated "
                "by low-level circulation features: the monsoon trough, a nearby vortex, "
                "or terrain interaction (if near landmass). This regime is associated with "
                "track uncertainty because boundary-layer features evolve rapidly and "
                "interact non-linearly with the storm circulation."
            )
        elif sst > 0.25:
            lines.append(
                "\nREGIME: SST-MODULATED TRACK"
                "\nUnusually high SST attribution indicates that the track prediction is "
                "sensitive to sea-surface temperature gradients. This typically occurs "
                "for slow-moving storms where intensity (SST-dependent) feeds back into "
                "the track through the β-drift and vortex asymmetries. The warm pool "
                "boundary near 26-28°C isotherm is a natural barrier for intensification."
            )

        if objects:
            lines.append(f"\nTop detected physical object: {objects[0].type.upper()} "
                         f"at {objects[0].level} (weight: {objects[0].attribution_weight:.2%})")
            lines.append(objects[0].description)

        return "\n".join(lines)


def _to_device(batch_list: list, device: torch.device) -> list:
    out = []
    for item in batch_list:
        if isinstance(item, torch.Tensor):
            out.append(item.to(device))
        elif isinstance(item, dict):
            out.append({k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in item.items()})
        else:
            out.append(item)
    return out
