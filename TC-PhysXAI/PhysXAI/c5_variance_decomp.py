"""
TC-PhysXAI — C5: Noise-Context Variance Decomposition
=======================================================
Physics Motivation
------------------
TC-XFlow generates ensemble trajectories by:
  1. Sampling initial noise z0 ~ N(0, I)  — the stochastic component
  2. Conditioning on context c = ContextEncoder(ERA5, kinematics) — the deterministic component

The total forecast variance decomposes (law of total variance):

    Var_total[f(z0, c)] = Var_{z0}[ E_c[f] ] + E_{z0}[ Var_c[f] ]
                        = V_noise              + V_context

Where:
  V_noise   = variance due to initial noise z0 (irreducible stochastic uncertainty)
  V_context = expected variance due to contextual uncertainty (epistemic)

Interpretation:
  - If V_noise >> V_context:
      The model is confident given the current atmospheric state;
      uncertainty arises from chaotic dynamics (butterfly effect).
      → "Predictability limited by initial-condition sensitivity"

  - If V_context >> V_noise:
      The model's uncertainty is mostly driven by the context encoder's
      inability to resolve the steering pattern.
      → "Predictability limited by synoptic pattern ambiguity"

  - Both large:
      Bifurcation regime — stochastic and synoptic uncertainty compound.

Implementation:
  1. Fix z0 (multiple seeds), vary context → compute Var_c[f | z0]
     Approximation: perturb context by small additive noise (±0.1σ)
  2. Fix context, vary z0 → compute Var_{z0}[f | c]
  3. Compute ratio V_noise / V_total and V_context / V_total

Reference: Brier (1950) decomposition applied to ML-based ensemble TC tracks;
           Analogue to Sobol sensitivity indices in global sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .fm_sampling import sample_z0, fm_integrate


@dataclass
class VarianceDecompositionResult:
    """Result of C5 noise-context variance decomposition."""
    var_total:       float              # total ensemble variance
    var_noise:       float              # variance from z0 sampling
    var_context:     float              # variance from context perturbation
    frac_noise:      float              # V_noise / V_total
    frac_context:    float              # V_context / V_total
    per_horizon_noise:   np.ndarray    # [T_pred] — V_noise(t)
    per_horizon_context: np.ndarray    # [T_pred] — V_context(t)
    predictability_horizon: int        # first step where V_total > threshold
    physical_interpretation: str


class VarianceDecomposition:
    """C5: Noise-Context Variance Decomposition for TC track ensembles.

    Implements the law-of-total-variance decomposition:
      Var_total = Var_z0[E_c[f]] + E_z0[Var_c[f]]

    Usage
    -----
    >>> vd = VarianceDecomposition(model, device)
    >>> result = vd.compute(batch_list, K=30, n_context_perturb=10)
    >>> print(result.physical_interpretation)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_context_perturb: int = 10,
        context_noise_level: float = 0.1,
    ):
        self.model               = model
        self.device              = device
        self.n_context_perturb   = n_context_perturb
        self.context_noise_level = context_noise_level
        self.model.eval()

    @torch.no_grad()
    def _sample_fixed_z0(
        self,
        z0: torch.Tensor,        # [B, T_pred, 2]  (already ~ N(0, sigma_inference^2))
        cond: torch.Tensor,      # [B, d_cond]
        last_obs: torch.Tensor,  # [B, 2]
        ddim_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """ODE from a FIXED z0 with the given context, integrated exactly like
        TCFlowMatching.sample (see fm_sampling.fm_integrate). Returns ABSOLUTE
        normalised tracks [T_pred, B, 2]."""
        return fm_integrate(self.model, z0, cond, last_obs, ddim_steps)

    @torch.no_grad()
    def _get_context(self, batch_list: list) -> torch.Tensor:
        """Context vector from model.encoder (with hard_score). Raises on failure."""
        from Model.flow_matching_model import hard_score_from_obs
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        h_score = hard_score_from_obs(
            bl[0][:, :, :2], weight_logits=self.model.hard_score_weight_logits)
        return self.model.encoder(bl, hard_score=h_score)

    def compute(
        self,
        batch_list: list,
        K: int = 30,
        ddim_steps: Optional[int] = None,
        n_z0_for_context: int = 3,
        predictability_threshold: float = 0.5,
        storm_idx: int = 0,
        verbose: bool = False,
    ) -> VarianceDecompositionResult:
        """Run variance decomposition for a single storm.

        Parameters
        ----------
        K                        : number of ensemble members for V_noise
        ddim_steps               : Euler steps; None -> model.n_inference_steps (as in sample())
        n_z0_for_context         : V_context is averaged over this many FIXED z0 draws
        predictability_threshold : V_total normalised threshold for horizon
        storm_idx                : which storm in batch to analyse

        Returns
        -------
        VarianceDecompositionResult
        """
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]

        if torch.is_tensor(bl[0]):
            B      = bl[0].shape[1]
            T_obs  = bl[0].shape[0]
        else:
            B, T_obs = 1, 8

        if torch.is_tensor(bl[1]):
            T_pred = bl[1].shape[0]
        else:
            T_pred = 12

        # Both terms use the SAME raw generator (fm_integrate: z0 ~ N(0, sigma_inf^2),
        # model.n_inference_steps Euler steps) — i.e. the candidates of sample() before
        # scoring / top-3 weighting / speed calibration. Output: absolute normalised tracks.
        last_obs  = bl[0][-1]
        base_cond = self._get_context(bl)               # [B, d_cond]
        gen = torch.Generator(device=self.device).manual_seed(0)

        # ─── Step 1: V_noise = Var over z0 at FIXED context ────────────────
        if verbose:
            print(f"[C5] K={K} z0 draws at fixed context for V_noise ...")
        with torch.no_grad():
            traj_k = torch.stack([
                self._sample_fixed_z0(sample_z0(self.model, B, T_pred, self.device, gen),
                                      base_cond, last_obs, ddim_steps)[:, storm_idx, :]
                for _ in range(K)
            ], dim=0)                                   # [K, T_pred, 2]
        var_noise_per_horizon = traj_k.var(dim=0).mean(dim=-1)  # [T_pred]
        v_noise = float(var_noise_per_horizon.mean())

        # ─── Step 2: V_context = Var over context perturbations at FIXED z0 ──
        if verbose:
            print(f"[C5] {self.n_context_perturb} context perturbations x "
                  f"{n_z0_for_context} fixed z0 for V_context ...")
        eps = self.context_noise_level * base_cond.std()
        z0_list = [sample_z0(self.model, B, T_pred, self.device, gen)
                   for _ in range(n_z0_for_context)]    # drawn ONCE, reused for every perturbation
        # one set of context perturbations shared by all z0 (paired design)
        noises = [torch.randn(base_cond.shape, device=self.device, generator=gen) * eps
                  for _ in range(self.n_context_perturb)]
        var_ctx_list = []
        with torch.no_grad():
            for z0_fixed in z0_list:
                trajs = torch.stack([
                    self._sample_fixed_z0(z0_fixed, base_cond + nz, last_obs, ddim_steps
                                          )[:, storm_idx, :]
                    for nz in noises
                ], dim=0)                               # [n_perturb, T_pred, 2]
                var_ctx_list.append(trajs.var(dim=0).mean(dim=-1))      # [T_pred]
        var_context_per_horizon = torch.stack(var_ctx_list, 0).mean(0)  # [T_pred]
        v_context = float(var_context_per_horizon.mean())

        # ─── Step 3: Decomposition ────────────────────────────────────────

        v_total = v_noise + v_context + 1e-8
        frac_n  = v_noise / v_total
        frac_c  = v_context / v_total

        # Predictability horizon: first step where variance exceeds threshold
        var_norm = (var_noise_per_horizon + var_context_per_horizon).cpu().numpy()
        var_norm = var_norm / (var_norm.max() + 1e-8)
        pred_h = int(np.argmax(var_norm > predictability_threshold))
        if pred_h == 0 and var_norm[0] <= predictability_threshold:
            pred_h = T_pred  # never exceeds threshold (very predictable)

        phys = self._interpret(
            v_noise, v_context, v_total,
            var_noise_per_horizon.cpu().numpy(),
            var_context_per_horizon.cpu().numpy(),
            pred_h, T_pred,
        )

        return VarianceDecompositionResult(
            var_total=v_total,
            var_noise=v_noise,
            var_context=v_context,
            frac_noise=frac_n,
            frac_context=frac_c,
            per_horizon_noise=var_noise_per_horizon.cpu().numpy(),
            per_horizon_context=var_context_per_horizon.cpu().numpy(),
            predictability_horizon=pred_h,
            physical_interpretation=phys,
        )

    def _interpret(
        self,
        v_noise: float,
        v_context: float,
        v_total: float,
        var_n_h: np.ndarray,
        var_c_h: np.ndarray,
        pred_h: int,
        T_pred: int,
    ) -> str:
        """Generate physical interpretation of C5 variance decomposition."""
        frac_n = v_noise / v_total
        frac_c = v_context / v_total
        lines  = []

        lines.append(
            f"VARIANCE DECOMPOSITION (C5):\n"
            f"  V_noise = {v_noise:.4f} ({frac_n:.1%} of total)\n"
            f"  V_context = {v_context:.4f} ({frac_c:.1%} of total)\n"
            f"  Predictability horizon ≈ {pred_h * 6}h (step {pred_h}/{T_pred})"
        )

        if frac_n > 0.70:
            lines.append(
                "\nNOISE-DOMINATED UNCERTAINTY:\n"
                "  The majority of forecast spread arises from the stochastic "
                "initial condition z0 ~ N(0,I), not from ambiguity in the "
                "atmospheric context. The synoptic pattern is WELL-DEFINED — "
                "the subtropical ridge, trough position, and SST field are "
                "unambiguous. However, the storm track remains uncertain due to "
                "chaotic sensitivity to position and intensity perturbations. "
                "This is analogous to the 'predictability barrier' in NWP at "
                "3-5 day range. Confidence interval should be reported as "
                "ensemble spread, not attributed to model uncertainty."
            )
        elif frac_c > 0.60:
            lines.append(
                "\nCONTEXT-DOMINATED UNCERTAINTY:\n"
                "  The dominant source of track uncertainty is SYNOPTIC PATTERN "
                "AMBIGUITY — the ContextEncoder extracts features sensitive to "
                "the exact ridge position, trough timing, or SST gradient that "
                "are not perfectly constrained in the ERA5 analysis. This is "
                "epistemic (reducible) uncertainty. Better ERA5 assimilation "
                "quality, ensemble data assimilation, or higher-resolution "
                "ocean analysis would reduce V_context. The model flags a regime "
                "where the steering channel is weak or multiple synoptic patterns "
                "are plausible — a classic large-error situation in typhoon "
                "forecasting."
            )
        else:
            lines.append(
                "\nMIXED UNCERTAINTY SOURCES:\n"
                "  Noise and context contribute comparably to forecast spread. "
                "This mixed regime typically occurs at medium range (48-72h) "
                "when the immediately predictable steering flow is well-resolved "
                "(low V_noise at short range) but approaching synoptic changes "
                "introduce pattern uncertainty (rising V_context at longer range). "
                "The model correctly represents both predictable short-range "
                "motion and synoptically uncertain longer-range evolution."
            )

        # Horizon decomposition: identify where each source peaks
        peak_n = int(np.argmax(var_n_h))
        peak_c = int(np.argmax(var_c_h))

        lines.append(
            f"\nHORIZON ANALYSIS:\n"
            f"  V_noise peaks at t+{peak_n*6}h → chaotic uncertainty grows slowly with lead time.\n"
            f"  V_context peaks at t+{peak_c*6}h → synoptic ambiguity intensifies at longer range."
        )

        if pred_h < T_pred // 2:
            lines.append(
                "  SHORT PREDICTABILITY LIMIT: Variance exceeds threshold before mid-range. "
                "This case has below-average predictability; operational advisories "
                "should include wider uncertainty cones."
            )
        elif pred_h >= T_pred:
            lines.append(
                "  HIGH PREDICTABILITY: Variance remains below threshold throughout "
                "the forecast window — this typhoon is in a well-defined, predictable "
                "steering regime (strong, stable subtropical ridge). Track forecast "
                "confidence is high."
            )

        return "\n".join(lines)
