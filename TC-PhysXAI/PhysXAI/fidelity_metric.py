"""
TC-PhysXAI — C3: Probabilistic Fidelity Metric
===============================================
Theory
------
Standard XAI fidelity metrics measure how well the explanation correlates with
model behaviour (completeness, faithfulness). For PROBABILISTIC ensemble models
like TC-XFlow, we need a metric that respects the distributional output.

The Probabilistic Fidelity metric H-mean-F is defined as:

    H-mean-F = 0.4 × Precision-F + 0.4 × Recall-F + 0.2 × Uncertainty-F

where:

  Precision-F (P_F):
    How much does MASKING a high-attribution region INCREASE error?
    P_F = CRPS(masked) / CRPS(unmasked)
    ↑ means the explanation identifies truly important features.

  Recall-F (R_F):
    How much does KEEPING ONLY high-attribution regions PRESERVE accuracy?
    R_F = 1 - CRPS(top_k_only) / CRPS(full_input)
    ↑ means top-k channels are sufficient to explain the prediction.

  Uncertainty-F (U_F):
    How well does the ensemble spread calibrate to actual track errors?
    U_F = 1 - |CRPS_spread - CRPS_error| / max(CRPS_spread, CRPS_error)
    ↑ means the uncertainty quantification is reliable.

  H-mean-F ranges in [0, 1]; higher is better.

Reference:
  Samek et al. (2017) "Evaluating the Visualization of What a DNN has Learned"
  → adapted for probabilistic ensemble tracks.
  Lemos et al. (2023) "Appropriate Use of Calibration Metrics" → U_F component.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .utils import crps_ensemble_fast


@dataclass
class FidelityResult:
    """Result of C3 probabilistic fidelity evaluation."""
    precision_f:     float   # P_F ∈ [0, ∞)
    recall_f:        float   # R_F ∈ [0, 1]
    uncertainty_f:   float   # U_F ∈ [0, 1]
    h_mean_f:        float   # H-mean-F = 0.4P + 0.4R + 0.2U ∈ [0, ∞)
    crps_baseline:   float   # unperturbed ensemble CRPS
    crps_masked:     float   # CRPS after masking top-k
    crps_top_k_only: float   # CRPS keeping only top-k
    top_k_channels:  List[int]  # which channels were selected
    interpretation:  str


class ProbabilisticFidelity:
    """C3: Probabilistic Fidelity Metric for TC Track XAI.

    Evaluates explanation quality via three components:
      Precision-F, Recall-F, Uncertainty-F → H-mean-F

    Usage
    -----
    >>> fidelity = ProbabilisticFidelity(model, device)
    >>> result = fidelity.compute(batch_list, ig_attr, K=30, top_k=3)
    >>> print(f"H-mean-F = {result.h_mean_f:.3f}")
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_ensemble: int = 30,
    ):
        self.model      = model
        self.device     = device
        self.n_ensemble = n_ensemble
        self.model.eval()

    @torch.no_grad()
    def _run_ensemble(self, batch_list: list, K: int) -> torch.Tensor:
        """Run K ensemble members. Returns [K, T_pred, B, 2]."""
        _, _, all_traj = self.model.sample(
            batch_list,
            num_ensemble=K,
            use_speed_calibration=True,
            return_xai=False,
        )
        return all_traj

    @torch.no_grad()
    def _compute_crps(
        self, batch_list: list, K: int
    ) -> Tuple[float, torch.Tensor]:
        """Compute mean CRPS for an input configuration.

        Returns
        -------
        scalar_crps : mean over T and B
        crps_tb     : [T_pred, B] CRPS
        """
        all_traj  = self._run_ensemble(batch_list, K)
        pred_traj = batch_list[1]  # [T_pred, B, 2]
        T_pred    = min(all_traj.shape[1], pred_traj.shape[0])
        crps_tb   = crps_ensemble_fast(all_traj[:, :T_pred], pred_traj[:T_pred])
        return float(crps_tb.mean()), crps_tb

    def _mask_channels(
        self,
        batch_list: list,
        channels: List[int],
        mask_value: float = 0.0,
    ) -> list:
        """Zero-out specified ERA5 channels in img_obs."""
        bl = list(batch_list)
        if len(bl) > 11 and torch.is_tensor(bl[11]):
            img = bl[11].clone()
            for ch in channels:
                img[:, ch, :, :, :] = mask_value
            bl[11] = img
        return bl

    def _keep_only_channels(
        self,
        batch_list: list,
        channels: List[int],
    ) -> list:
        """Zero ALL channels EXCEPT the specified ones."""
        all_channels = list(range(13))
        mask_channels = [c for c in all_channels if c not in channels]
        return self._mask_channels(batch_list, mask_channels)

    def _compute_spread_error(
        self,
        all_traj: torch.Tensor,    # [K, T_pred, B, 2]
        pred_traj: torch.Tensor,   # [T_pred, B, 2]
    ) -> Tuple[float, float]:
        """Compute ensemble spread and mean absolute error.

        Returns
        -------
        spread : mean |X_i - X_mean| over ensemble
        error  : mean |X_mean - y| (deterministic error)
        """
        K, T, B, _ = all_traj.shape
        mean_traj   = all_traj.mean(0)  # [T, B, 2]
        spread = float((all_traj - mean_traj.unsqueeze(0)).norm(dim=-1).mean())
        error  = float((mean_traj - pred_traj[:T]).norm(dim=-1).mean())
        return spread, error

    def compute(
        self,
        batch_list: list,
        ig_attr: torch.Tensor,        # [B, 13, T_obs, H, W] IG attribution
        K: Optional[int] = None,
        top_k: int = 3,               # number of channels to mask/keep
        verbose: bool = False,
    ) -> FidelityResult:
        """Compute H-mean-F fidelity score.

        Parameters
        ----------
        ig_attr : IG attribution from SpatialAttributor
        top_k   : number of most important channels to select

        Returns
        -------
        FidelityResult with P_F, R_F, U_F, H-mean-F
        """
        K = K or self.n_ensemble
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        attr = ig_attr.to(self.device) if torch.is_tensor(ig_attr) else ig_attr

        # Identify top-k channels by mean attribution magnitude
        if torch.is_tensor(attr):
            # Sum over B, T, H, W — keep channel dimension
            ch_importance = attr.abs().mean(dim=[0, 2, 3, 4]).cpu().numpy()  # [13]
        else:
            ch_importance = np.ones(13)

        top_k_channels = list(np.argsort(ch_importance)[::-1][:top_k])

        if verbose:
            print(f"[C3] Top-{top_k} channels: {[f'Ch{i}' for i in top_k_channels]}")

        # ─── Baseline CRPS ────────────────────────────────────────────────

        if verbose:
            print("[C3] Computing baseline CRPS ...")
        crps_base, crps_tb_base = self._compute_crps(bl, K)

        # ─── Precision-F: Mask top-k → CRPS should increase ──────────────

        if verbose:
            print("[C3] Computing Precision-F (mask top-k) ...")
        bl_masked   = self._mask_channels(bl, top_k_channels)
        crps_masked, _ = self._compute_crps(bl_masked, K)
        precision_f = crps_masked / (crps_base + 1e-8)

        # ─── Recall-F: Keep only top-k → CRPS should stay close to base ──

        if verbose:
            print("[C3] Computing Recall-F (keep top-k only) ...")
        bl_topk     = self._keep_only_channels(bl, top_k_channels)
        crps_topk, _ = self._compute_crps(bl_topk, K)
        recall_f    = 1.0 - crps_topk / (crps_base + 1e-8)
        recall_f    = float(np.clip(recall_f, 0, 1))

        # ─── Uncertainty-F: Spread-error reliability ──────────────────────

        if verbose:
            print("[C3] Computing Uncertainty-F (spread vs error) ...")
        all_traj_base = self._run_ensemble(bl, K)
        pred_traj     = bl[1]  # [T_pred, B, 2]
        spread, error = self._compute_spread_error(all_traj_base, pred_traj)
        uncertainty_f = 1.0 - abs(spread - error) / (max(spread, error) + 1e-8)
        uncertainty_f = float(np.clip(uncertainty_f, 0, 1))

        # ─── H-mean-F ─────────────────────────────────────────────────────

        h_mean_f = 0.4 * precision_f + 0.4 * recall_f + 0.2 * uncertainty_f

        interp = self._interpret(
            precision_f, recall_f, uncertainty_f, h_mean_f,
            top_k_channels, crps_base, crps_masked, crps_topk,
        )

        return FidelityResult(
            precision_f=float(precision_f),
            recall_f=float(recall_f),
            uncertainty_f=float(uncertainty_f),
            h_mean_f=float(h_mean_f),
            crps_baseline=float(crps_base),
            crps_masked=float(crps_masked),
            crps_top_k_only=float(crps_topk),
            top_k_channels=top_k_channels,
            interpretation=interp,
        )

    def evaluate_multiple_methods(
        self,
        batch_list: list,
        method_attrs: Dict[str, torch.Tensor],
        K: Optional[int] = None,
        top_k: int = 3,
    ) -> Dict[str, FidelityResult]:
        """Evaluate fidelity for multiple XAI methods (e.g. IG vs SHAP vs C4).

        Parameters
        ----------
        method_attrs : dict mapping method name → [B, 13, T, H, W] attribution

        Returns
        -------
        dict mapping method name → FidelityResult
        """
        results = {}
        for name, attr in method_attrs.items():
            print(f"[C3] Evaluating fidelity for {name} ...")
            results[name] = self.compute(batch_list, attr, K=K, top_k=top_k)
        return results

    def _interpret(
        self,
        pf: float, rf: float, uf: float, hmf: float,
        top_k: List[int],
        crps_base: float, crps_masked: float, crps_topk: float,
    ) -> str:
        """Qualitative interpretation of fidelity scores."""
        ch_names = [
            "GPH_200", "GPH_500", "GPH_850", "GPH_925",
            "U_200",   "U_500",   "U_850",   "U_925",
            "V_200",   "V_500",   "V_850",   "V_925",
            "SST",
        ]
        top_names = [ch_names[i] for i in top_k]

        lines = [
            "PROBABILISTIC FIDELITY REPORT (C3)",
            "=" * 40,
            f"  Precision-F  : {pf:.3f}  "
            f"(CRPS ratio masked/base = {crps_masked:.4f}/{crps_base:.4f})",
            f"  Recall-F     : {rf:.3f}  "
            f"(CRPS top-k = {crps_topk:.4f} vs base {crps_base:.4f})",
            f"  Uncertainty-F: {uf:.3f}  (ensemble spread vs mean error)",
            f"  H-mean-F     : {hmf:.3f}",
            f"  Top-{len(top_k)} channels identified: {top_names}",
            "",
        ]

        # Overall rating
        if hmf > 1.2:
            rating = "EXCELLENT"
        elif hmf > 0.9:
            rating = "GOOD"
        elif hmf > 0.6:
            rating = "MODERATE"
        else:
            rating = "POOR"

        lines.append(f"OVERALL FIDELITY: {rating} (H-mean-F = {hmf:.3f})")
        lines.append("")

        # Precision diagnosis
        if pf > 1.5:
            lines.append(
                "PRECISION: HIGH — Masking the top channels causes large CRPS increase. "
                "The explanation correctly identifies the meteorologically critical "
                "ERA5 channels. These channels are genuinely necessary for accurate "
                "track prediction."
            )
        elif pf > 1.1:
            lines.append(
                "PRECISION: MODERATE — Masking key channels causes moderate CRPS increase. "
                "The identified channels are relevant but the model can partially "
                "compensate using correlated channels (e.g. GPH500 correlates with U500)."
            )
        else:
            lines.append(
                "PRECISION: LOW — Masking identified channels has minimal impact on CRPS. "
                "This may indicate (a) the attribution map diffuses importance across "
                "many channels, (b) the model uses highly redundant features, or "
                "(c) the explanation method is not aligned with the model's decision logic."
            )

        # Recall diagnosis
        if rf > 0.8:
            lines.append(
                "RECALL: HIGH — Using only the top channels preserves most predictive skill. "
                "The explanation is SUFFICIENT: the identified channels capture nearly "
                "all the information used by the model. This is physically sensible for "
                "tropical cyclone steering: 2-3 key level pressure/wind channels "
                "typically dominate track guidance."
            )
        elif rf > 0.5:
            lines.append(
                "RECALL: MODERATE — Top channels are necessary but not fully sufficient. "
                "Additional channels (perhaps SST or lower-level winds) contribute "
                "complementary information not captured in the primary explanation."
            )
        else:
            lines.append(
                "RECALL: LOW — Information is distributed across many channels. "
                "The model uses a holistic multi-level representation; no small "
                "subset of channels is sufficient for accurate track prediction in "
                "this case. This can occur during rapid change regimes."
            )

        # Uncertainty diagnosis
        if uf > 0.8:
            lines.append(
                "CALIBRATION: WELL-CALIBRATED — Ensemble spread matches mean error. "
                "The model's probabilistic uncertainty estimates are reliable for "
                "this case. Track uncertainty cones are operationally trustworthy."
            )
        elif uf > 0.5:
            lines.append(
                "CALIBRATION: PARTIALLY CALIBRATED — Some mismatch between spread and error. "
                "This is common during recurvature transitions where the ensemble "
                "bifurcates but the actual track is outside the main cluster."
            )
        else:
            lines.append(
                "CALIBRATION: POORLY CALIBRATED — Large spread-error mismatch. "
                "The ensemble is either overconfident (spread < error: deterministic "
                "steering failure) or over-dispersed (spread > error: unnecessary "
                "uncertainty). Both degrade operational utility."
            )

        return "\n".join(lines)
