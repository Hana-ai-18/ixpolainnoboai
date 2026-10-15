"""
TC-PhysXAI — Module 1: Temporal Attribution (Contribution C1)
==============================================================
Physics Motivation
------------------
A typhoon's track is steered by the synoptic flow at different lag times.
Early lags (t-48h to t-24h) capture large-scale pattern setup; recent lags
(t-12h to t-0h) capture the immediate steering environment and storm
kinematics.  Standard XAI methods use Gaussian noise for perturbation, which
violates physical constraints (e.g., it can create unphysical wind shear
patterns).

Contribution C1 (Typhoon-specific temporal perturbation):
  - Replace each 6-hourly observation window with the CLIMATOLOGICAL MEAN
    (zero in the normalised ERA5 space = mean field), not Gaussian noise.
  - The climatological mean is dynamically stable — it represents a
    "no information" state consistent with the seasonal-mean atmosphere.
  - Perturbation is applied simultaneously to:
      · ERA5 spatial fields (img_obs, channels 0-12)
      · Best-track kinematics (obs_traj lat/lon)
      · 1D storm scalars (obs_Me: pressure, max wind)
  - Attribution score = ΔCRPS = CRPS(perturbed) − CRPS(baseline)
    A large positive ΔCRPS means that window was critical.

Reference: analogous to LIME/SHAP but with physically meaningful baseline.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import crps_ensemble_fast, get_climatological_era5, get_climatological_traj


class TemporalAttributor:
    """Computes per-window temporal attribution scores via climatological
    baseline perturbation and CRPS difference measurement.

    Usage
    -----
    >>> attributor = TemporalAttributor(model, device)
    >>> result = attributor.compute(batch_list, n_ensemble=30)
    >>> print(result["attribution_scores"])   # [obs_len, B]

    Parameters
    ----------
    model : TCFlowMatching
        The trained flow-matching model (frozen).
    device : torch.device
        GPU/CPU device.
    obs_len : int
        Number of observed steps (default 8, i.e. 48 h at 6-hourly).
    n_ensemble : int
        Number of ensemble trajectories for CRPS estimation.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        obs_len: int = 8,
        n_ensemble: int = 30,
    ):
        self.model      = model
        self.device     = device
        self.obs_len    = obs_len
        self.n_ensemble = n_ensemble
        self.model.eval()

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _deep_copy_batch(self, batch_list: list) -> list:
        """Shallow-copy mutable tensors; leave non-tensor items as-is."""
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

    def _run_ensemble(self, batch_list: list, K: int) -> torch.Tensor:
        """Run the model K times and return stacked trajectories.

        Returns
        -------
        all_traj : [K, T_pred, B, 2] ensemble trajectories (normalised)
        """
        with torch.no_grad():
            _, _, all_traj = self.model.sample(
                batch_list,
                num_ensemble=K,
                use_speed_calibration=True,
                return_xai=False,
            )
        return all_traj  # [K, T, B, 2]

    def _baseline_crps(
        self, batch_list: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute baseline CRPS and ensemble for the unperturbed input.

        Returns
        -------
        crps_base : [T_pred, B] baseline CRPS
        all_traj  : [K, T_pred, B, 2] baseline ensemble
        """
        all_traj  = self._run_ensemble(batch_list, self.n_ensemble)
        pred_traj = batch_list[1]  # [T_pred, B, 2] ground truth
        T_pred = min(all_traj.shape[1], pred_traj.shape[0])
        crps_base = crps_ensemble_fast(
            all_traj[:, :T_pred],
            pred_traj[:T_pred],
        )
        return crps_base, all_traj

    # ─── index map ───────────────────────────────────────────────────────────
    # batch_list slots (from seq_collate):
    #   [0]  obs_traj          [T_obs, B, 2]  lat/lon normalised
    #   [1]  pred_traj         [T_pred, B, 2]
    #   [7]  obs_Me            [T_obs, B, 4]  Lat, Lon, MinPressure, MaxWind
    #   [11] img_obs           [B, 13, T_obs, 81, 81] ERA5
    # ─────────────────────────────────────────────────────────────────────────

    def _perturb_window(self, batch_list: list, window: int) -> list:
        """Replace observation step `window` with climatological baseline.

        Physical meaning:
          - Setting ERA5 to 0 in normalised space = setting to seasonal mean.
          - Setting kinematics (lat/lon displacements) to the last valid
            position = "no motion" state = climatological mean translation.
          - Setting obs_Me to climatological mean values.
        """
        bl = self._deep_copy_batch(batch_list)

        # 1. ERA5 fields: [B, 13, T_obs, 81, 81] — zero = climatological mean
        if len(bl) > 11 and torch.is_tensor(bl[11]):
            bl[11][:, :, window, :, :] = 0.0

        # 2. Best-track kinematics: [T_obs, B, 2]
        if torch.is_tensor(bl[0]) and window < bl[0].shape[0]:
            if window == 0:
                ref = bl[0][0].clone()   # keep first position
            else:
                ref = bl[0][window - 1].clone()  # fill from previous step
            bl[0][window] = ref          # no-motion = repeat last pos

        # 3. 1D storm scalars: [T_obs, B, 4] (Lat, Lon, MinPressure, MaxWind)
        if len(bl) > 7 and torch.is_tensor(bl[7]) and window < bl[7].shape[0]:
            # Climatological mean ≈ 0 in normalised space.
            # For window > 0, use previous step (smoother than zeroing).
            # For window == 0, no previous step exists — use zero (climatological mean).
            if window > 0:
                bl[7][window] = bl[7][window - 1]
            else:
                bl[7][window] = 0.0

        return bl

    # ─── main API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute(
        self,
        batch_list: list,
        n_ensemble: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        """Compute temporal attribution scores for all observation windows.

        Returns
        -------
        dict with keys:
          "attribution_scores"   : [obs_len, B] ΔCRPS per window
          "attribution_relative" : [obs_len, B] normalised to [0,1]
          "crps_baseline"        : [T_pred, B]
          "crps_perturbed"       : [obs_len, T_pred, B]
          "mean_attribution_per_window" : [obs_len] averaged over batch
          "most_important_window"       : int index (batch mean)
          "window_time_labels"          : list of str (e.g. "t-42h")
        """
        K = n_ensemble or self.n_ensemble
        device = self.device

        # Move batch to device
        bl = self._to_device(batch_list, device)

        # Baseline
        if verbose:
            print("[TemporalAttributor] Computing baseline CRPS ...")
        crps_base, _ = self._baseline_crps(bl)  # [T_pred, B]

        crps_perturbed = []
        for w in range(self.obs_len):
            if verbose:
                print(f"[TemporalAttributor] Perturbing window {w}/{self.obs_len-1} "
                      f"({self._window_label(w)}) ...")
            bl_pert = self._perturb_window(bl, w)
            all_t   = self._run_ensemble(bl_pert, K)
            pred_traj = bl[1]
            T_pred = min(all_t.shape[1], pred_traj.shape[0])
            crps_p = crps_ensemble_fast(all_t[:, :T_pred], pred_traj[:T_pred])  # [T, B]
            crps_perturbed.append(crps_p)

        crps_perturbed = torch.stack(crps_perturbed, dim=0)  # [obs_len, T, B]

        # Attribution = ΔCRPS summed over horizons
        delta_crps = crps_perturbed - crps_base.unsqueeze(0)  # [obs_len, T, B]
        attr_scores = delta_crps.sum(1)   # [obs_len, B]

        # Normalise per storm to [0, 1]
        attr_min = attr_scores.min(0, keepdim=True).values
        attr_max = attr_scores.max(0, keepdim=True).values
        attr_rel = (attr_scores - attr_min) / (attr_max - attr_min + 1e-8)

        mean_per_window = attr_scores.mean(1).cpu().numpy()  # [obs_len]
        most_important  = int(mean_per_window.argmax())

        labels = [self._window_label(w) for w in range(self.obs_len)]

        return {
            "attribution_scores":       attr_scores.cpu(),
            "attribution_relative":     attr_rel.cpu(),
            "crps_baseline":            crps_base.cpu(),
            "crps_perturbed":           crps_perturbed.cpu(),
            "mean_attribution_per_window": mean_per_window,
            "most_important_window":    most_important,
            "window_time_labels":       labels,
            "physical_interpretation":  self._interpret(mean_per_window, labels),
        }

    def _window_label(self, w: int) -> str:
        """Convert window index to lag-time label.
        Window 7 = t-6h (most recent), Window 0 = t-48h (oldest).
        """
        lag_h = (self.obs_len - 1 - w) * 6
        return f"t-{lag_h}h"

    def _interpret(
        self, mean_attr: np.ndarray, labels: List[str]
    ) -> str:
        """Generate physical interpretation of temporal attribution pattern."""
        # Normalise
        total = mean_attr.sum() + 1e-8
        weights = mean_attr / total

        # Categorise: recent (0-12h lag), medium (12-30h), remote (30-48h)
        n = len(weights)
        recent_idx  = [i for i in range(n) if (n - 1 - i) * 6 <= 12]
        medium_idx  = [i for i in range(n) if 12 < (n - 1 - i) * 6 <= 30]
        remote_idx  = [i for i in range(n) if (n - 1 - i) * 6 > 30]

        w_recent = weights[recent_idx].sum()  if recent_idx  else 0
        w_medium = weights[medium_idx].sum()  if medium_idx  else 0
        w_remote = weights[remote_idx].sum()  if remote_idx  else 0

        dom = max([(w_recent, "recent 0-12h"), (w_medium, "medium 12-30h"),
                   (w_remote, "remote 30-48h")], key=lambda x: x[0])
        best_w  = labels[int(mean_attr.argmax())]

        lines = [
            f"Most influential observation window: {best_w}",
            f"Attribution weight distribution — recent: {w_recent:.2%}, "
            f"medium: {w_medium:.2%}, remote: {w_remote:.2%}",
        ]
        if w_recent > 0.5:
            lines.append(
                "PHYSICAL INTERPRETATION: Track is dominated by the CURRENT "
                "local steering flow (925–500 hPa). The typhoon is in a well-"
                "defined steering channel. 6–12h lag is most predictive, "
                "consistent with a steady-state vortex embedded in a "
                "stable subtropical ridge."
            )
        elif w_remote > 0.4:
            lines.append(
                "PHYSICAL INTERPRETATION: Track depends heavily on REMOTE "
                "synoptic precursors (t-36h to t-48h). This pattern is "
                "typical of approaching recurvature — the 200 hPa trough "
                "approaching from the west amplifies at 24-36h lag before "
                "the storm interacts with the mid-latitude westerlies. "
                "The model needs early environmental information to resolve "
                "the downstream trajectory bifurcation."
            )
        else:
            lines.append(
                "PHYSICAL INTERPRETATION: Mixed attribution across temporal "
                "windows suggests a storm in complex multi-scale steering "
                "interaction. The subtropical ridge and approaching trough "
                "are competing influences; both the current flow state and "
                "the evolving synoptic pattern matter equally."
            )
        return "\n".join(lines)

    def _to_device(self, batch_list: list, device: torch.device) -> list:
        """Move tensors in batch_list to device."""
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
