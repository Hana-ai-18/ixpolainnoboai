"""
TC-PhysXAI — C6: FM-Native Counterfactual via ODE Adjoint
==========================================================
Physics Motivation
------------------
"What would the typhoon track look like if the subtropical ridge were 5° further
north?" is a counterfactual question with direct operational relevance.

Standard counterfactual XAI approaches perturb the INPUT (ERA5) and rerun the
model — but this ignores the ODE structure of flow matching.

C6 takes a physics-native approach:
  1. Define a TARGET trajectory y* (the counterfactual outcome, e.g. recurved track)
  2. Use the ADJOINT ODE METHOD to find the minimum-perturbation to the context
     vector c that shifts the model's prediction from ŷ toward y*
  3. Project the context perturbation Δc back to ERA5 channels via the Jacobian
     ∂c/∂x, to produce a COUNTERFACTUAL PHYSICAL EXPLANATION:
     "To produce this alternative track, the 500 hPa ridge would need to be
     Δh500 = +3.2 dam higher over 130°E-150°E"

Mathematical formulation:
  Minimise:    ‖Δc‖²           (minimum context change)
  Subject to:  ‖F(c + Δc) - y*‖ ≤ ε     (match counterfactual outcome)

  Where F(c) = ODE_solve(c, z0) is the TC-XFlow forward model.

  Solved by projected gradient descent on c with fixed z0.

Reference:
  Wachter et al. (2017) "Counterfactual Explanations without Opening the Black Box"
  Adjoint method: Pontryagin maximum principle adapted to continuous-time ODE models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .geo import norm_to_lonlat
from .fm_sampling import sample_z0, fm_integrate


@dataclass
class CounterfactualResult:
    """Result of C6 FM counterfactual generation."""
    # Original and counterfactual tracks
    original_track:       np.ndarray       # [T_pred, 2] ABSOLUTE normalised (lon_n, lat_n); 1 unit = 5 deg
    counterfactual_track: np.ndarray       # [T_pred, 2] ABSOLUTE normalised (lon_n, lat_n)
    target_track:         np.ndarray       # [T_pred, 2] ABSOLUTE normalised (lon_n, lat_n)

    # Context perturbation
    delta_context:        np.ndarray       # [d_cond] — Δc
    context_change_norm:  float            # ‖Δc‖ / ‖c‖ — fractional change

    # ERA5 channel attribution of Δc
    channel_delta:        np.ndarray       # [13] — projected Δc onto ERA5 channels
    channel_delta_phys:   np.ndarray       # [13] — in physical units

    # Scenario label
    counterfactual_type:  str              # "RECURVATURE", "STRAIGHT", "INTENSIFIED"
    track_delta_km:       float            # max displacement between original and CF
    physical_explanation: str


_CH_NAMES = [
    "GPH_200", "GPH_500", "GPH_850", "GPH_925",
    "U_200",   "U_500",   "U_850",   "U_925",
    "V_200",   "V_500",   "V_850",   "V_925",
    "SST",
]

_CH_UNITS = [
    "m (GPH@200)",  "m (GPH@500)",  "m (GPH@850)",  "m (GPH@925)",
    "m/s (U@200)",  "m/s (U@500)",  "m/s (U@850)",  "m/s (U@925)",
    "m/s (V@200)",  "m/s (V@500)",  "m/s (V@850)",  "m/s (V@925)",
    "K (SST)",
]

# Physical std deviations (from DATA3D_STD in utils.py)
_CH_STD = np.array([
    113.583, 57.135, 37.852, 37.274,
     13.315,  8.042,  7.911,  7.494,
      8.377,  5.999,  6.203,  6.555,
      3.023,
], dtype=np.float32)


def _haversine_km_np(lat1, lon1, lat2, lon2):
    """Haversine distance in km (numpy)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _make_recurvature_target(
    original_traj: np.ndarray,   # [T, 2] ABSOLUTE normalised (lon_n, lat_n)
    deflect_north: float = 0.18, # normalised units of LATITUDE  (1 unit = 5 deg -> 0.9 deg)
    deflect_east:  float = 0.18, # normalised units of LONGITUDE (1 unit = 5 deg -> 0.9 deg)
    onset_step:    int   = 4,    # steps after which the deflection starts
) -> np.ndarray:
    """Counterfactual target: linearly growing north/east deflection after onset_step.

    Coordinate convention (see geo.py): index 0 = longitude, index 1 = latitude.
    """
    T   = original_traj.shape[0]
    cf  = original_traj.copy()
    for t in range(onset_step, T):
        frac = (t - onset_step) / max(T - onset_step - 1, 1)
        cf[t, 1] += deflect_north * frac   # latitude  (poleward)
        cf[t, 0] += deflect_east  * frac   # longitude (eastward)
    return cf


class FMCounterfactual:
    """C6: FM-Native Counterfactual via Projected Gradient on Context.

    Usage
    -----
    >>> cf = FMCounterfactual(model, device)
    >>> result = cf.compute(batch_list, cf_type="RECURVATURE")
    >>> print(result.physical_explanation)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        ddim_steps: Optional[int] = None,   # None -> model.n_inference_steps (same as sample())
        lr: float = 0.01,
        max_iter: int = 100,
        cf_epsilon: float = 0.1,     # stop when RMS position error to the target < cf_epsilon DEGREES
        reg_lambda: float = 0.1,     # L2 regularisation on Δc
    ):
        self.model      = model
        self.device     = device
        self.ddim_steps = ddim_steps
        self.lr         = lr
        self.max_iter   = max_iter
        self.cf_epsilon = cf_epsilon
        self.reg_lambda = reg_lambda
        self.model.eval()

    def _encode(self, bl: list) -> torch.Tensor:
        from Model.flow_matching_model import hard_score_from_obs
        h_score = hard_score_from_obs(
            bl[0][:, :, :2], weight_logits=self.model.hard_score_weight_logits)
        return self.model.encoder(bl, hard_score=h_score)

    @torch.no_grad()
    def _get_context_and_traj(
        self,
        batch_list: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Baseline context, fixed z0 (same law as sample(): N(0, sigma_inference^2))
        and the raw ODE track (no candidate scoring / speed calibration).

        Returns cond [B,d_cond], z0 [B,T_pred,2], base_traj [T_pred,B,2] (absolute)."""
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        self._last_obs = bl[0][-1]
        cond = self._encode(bl)
        gen = torch.Generator(device=self.device).manual_seed(0)
        z0 = sample_z0(self.model, cond.shape[0], bl[1].shape[0], self.device, gen)
        traj = self._ode_forward(z0, cond)
        return cond, z0, traj

    def _ode_forward(self, z0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Integrate exactly like TCFlowMatching.sample. Returns ABSOLUTE [T_pred, B, 2]."""
        return fm_integrate(self.model, z0, cond, self._last_obs, self.ddim_steps)

    def _project_delta_to_channels(
        self,
        delta_c: torch.Tensor,    # [d_cond]
        batch_list: list,
    ) -> np.ndarray:
        """Project Δc onto ERA5 input channels via Jacobian ∂c/∂x.

        Approximation: For each channel, compute the norm of
        ∂c/∂x_ch × delta_c projected contribution.

        Returns [13] array of channel importance for the counterfactual.
        """
        img_obs = batch_list[11] if len(batch_list) > 11 else None
        if img_obs is None or not torch.is_tensor(img_obs):
            raise ValueError("C6: batch has no ERA5 tensor at index 11")

        bl_g = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        img_g = bl_g[11].float().detach().requires_grad_(True)
        bl_g[11] = img_g
        with torch.enable_grad():
            cond_g = self._encode(bl_g)                       # [B, d_cond]
            # vector-Jacobian product  J^T delta_c
            g = torch.autograd.grad((cond_g * delta_c.to(self.device).unsqueeze(0)).sum(),
                                    img_g)[0]
        ch_proj = g.abs().mean(0).mean(dim=[1, 2, 3]).detach().cpu().numpy()   # [13]
        return ch_proj / (ch_proj.sum() + 1e-8)

    def compute(
        self,
        batch_list: list,
        cf_type: str = "RECURVATURE",
        storm_idx: int = 0,
        target_traj: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> CounterfactualResult:
        """Generate a counterfactual explanation.

        Parameters
        ----------
        cf_type     : "RECURVATURE" | "STRAIGHT" | "CUSTOM"
        target_traj : [T_pred, 2] normalised — required if cf_type="CUSTOM"

        Returns
        -------
        CounterfactualResult
        """
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]

        # Baseline
        with torch.no_grad():
            cond, z0, base_traj = self._get_context_and_traj(bl)

        base_np = base_traj[:, storm_idx, :].cpu().numpy()   # [T_pred, 2]
        T_pred  = base_np.shape[0]
        B       = cond.shape[0]

        # Build counterfactual target
        if target_traj is not None:
            cf_target_np = target_traj
        elif cf_type == "RECURVATURE":
            cf_target_np = _make_recurvature_target(
                base_np, deflect_north=0.18, deflect_east=0.18, onset_step=T_pred//3
            )
        elif cf_type == "STRAIGHT":
            cf_target_np = _make_recurvature_target(
                base_np, deflect_north=-0.076, deflect_east=-0.227, onset_step=T_pred//3
            )
        else:
            cf_target_np = base_np.copy()

        cf_target = torch.tensor(cf_target_np, dtype=torch.float32, device=self.device)
        cf_target_batch = cf_target.unsqueeze(1).expand(T_pred, B, 2)  # [T_pred, B, 2]

        # ─── Projected gradient search ─────────────────────────────────────

        delta_c = torch.zeros_like(cond, requires_grad=True)
        optimiser = torch.optim.Adam([delta_c], lr=self.lr)

        best_delta = None
        best_loss  = float("inf")

        if verbose:
            print(f"[C6] Running CF optimisation ({self.max_iter} iterations) ...")

        eps_norm = self.cf_epsilon / 5.0          # degrees -> normalised units
        for step in range(self.max_iter):
            optimiser.zero_grad()
            cand = delta_c.detach().clone()       # the delta that is evaluated below
            cond_cf = cond.detach() + delta_c

            traj_cf = self._ode_forward(z0.detach(), cond_cf)  # [T_pred, B, 2]

            track_mse  = (traj_cf - cf_target_batch).pow(2).sum(-1).mean()   # norm units^2
            track_loss = track_mse
            reg_loss   = self.reg_lambda * delta_c.pow(2).mean()
            loss       = track_loss + reg_loss

            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_delta = cand

            if verbose and (step + 1) % 20 == 0:
                print(f"  Step {step+1}: loss={loss.item():.5f}, "
                      f"track={track_loss.item():.5f}, reg={reg_loss.item():.5f}")

            if track_mse.item() < eps_norm ** 2:
                if verbose:
                    print(f"  Converged at step {step+1}.")
                break

            loss.backward()
            optimiser.step()

        # Final counterfactual trajectory
        with torch.no_grad():
            cond_best = cond + best_delta
            traj_best = self._ode_forward(z0, cond_best)

        cf_np  = traj_best[:, storm_idx, :].cpu().numpy()   # [T_pred, 2]
        delta_c_np = best_delta[storm_idx].cpu()             # [d_cond]
        c_norm_frac = float(
            best_delta.norm() / (cond.norm() + 1e-8)
        )

        # Project Δc to ERA5 channels
        ch_delta = self._project_delta_to_channels(delta_c_np, bl)  # [13]
        ch_delta_phys = ch_delta * _CH_STD  # scale to physical units

        # Max great-circle displacement between original and counterfactual track (km)
        ll_o = norm_to_lonlat(base_np)
        ll_c = norm_to_lonlat(cf_np)
        dist_km = float(_haversine_km_np(ll_o[:, 1], ll_o[:, 0], ll_c[:, 1], ll_c[:, 0]).max())

        phys = self._interpret(cf_type, ch_delta, ch_delta_phys, c_norm_frac, dist_km)

        return CounterfactualResult(
            original_track=base_np,
            counterfactual_track=cf_np,
            target_track=cf_target_np,
            delta_context=delta_c_np.numpy(),
            context_change_norm=c_norm_frac,
            channel_delta=ch_delta,
            channel_delta_phys=ch_delta_phys,
            counterfactual_type=cf_type,
            track_delta_km=float(dist_km),
            physical_explanation=phys,
        )

    def _interpret(
        self,
        cf_type: str,
        ch_delta: np.ndarray,       # [13] normalised
        ch_delta_phys: np.ndarray,  # [13] physical units
        c_norm_frac: float,
        dist_km: float,
    ) -> str:
        """Generate physical counterfactual explanation."""
        top_ch_idx   = int(ch_delta.argmax())
        top_ch_name  = _CH_NAMES[top_ch_idx]
        top_ch_phys  = float(ch_delta_phys[top_ch_idx])
        top_ch_unit  = _CH_UNITS[top_ch_idx]

        gph_delta = ch_delta[0:4].sum()
        u_delta   = ch_delta[4:8].sum()
        v_delta   = ch_delta[8:12].sum()
        sst_delta = ch_delta[12]

        scenario_map = {
            "RECURVATURE": "northeast recurvature",
            "STRAIGHT":    "continued westward/northwestward motion",
            "CUSTOM":      "specified alternative trajectory",
        }
        scenario_desc = scenario_map.get(cf_type, cf_type.lower())

        lines = [
            f"COUNTERFACTUAL EXPLANATION (C6): {cf_type}",
            f"",
            f"COUNTERFACTUAL QUESTION:",
            f"  'What atmospheric change would cause this typhoon to undergo {scenario_desc}?'",
            f"",
            f"MINIMUM CONTEXT PERTURBATION REQUIRED:",
            f"  Context norm change: {c_norm_frac:.1%} of original context vector",
            f"  Maximum implied track shift: {dist_km:.0f} km",
            f"",
            f"DOMINANT PHYSICAL PERTURBATION:",
            f"  {top_ch_name}: Δ = {top_ch_phys:+.2f} {top_ch_unit}",
            f"  (This is the single ERA5 channel requiring the largest change)",
        ]

        if cf_type == "RECURVATURE":
            if gph_delta > 0.4:
                lines += [
                    "",
                    "PHYSICAL INTERPRETATION (RECURVATURE):",
                    f"  WEAKENING of the subtropical ridge: A GPH reduction of "
                    f"~{abs(ch_delta_phys[1]):.0f} m at 500 hPa (equivalent to "
                    f"~{abs(ch_delta_phys[1])/10:.0f} dam) over the storm's "
                    "steering layer would open the western terminus of the ridge, "
                    "allowing the typhoon to turn northward and interact with "
                    "mid-latitude westerlies. This corresponds to either a stronger "
                    "approaching trough or a ridging retraction event.",
                ]
            elif v_delta > 0.3:
                lines += [
                    "",
                    "PHYSICAL INTERPRETATION (RECURVATURE):",
                    "  A stronger poleward V-wind component at 850-500 hPa would "
                    "be sufficient to initiate recurvature. This can arise from "
                    "an approaching 200 hPa trough deepening the northward "
                    "geostrophic flow ahead of its axis. A mean V500 increase "
                    f"of ~{abs(ch_delta_phys[9]):.1f} m/s would be required.",
                ]
            else:
                lines += [
                    "",
                    "PHYSICAL INTERPRETATION (RECURVATURE):",
                    "  The model requires only a SMALL context perturbation "
                    f"({c_norm_frac:.1%}) to produce recurvature — this suggests "
                    "the typhoon is already near the recurvature bifurcation point. "
                    "The atmosphere is in a marginally-recurving state; small "
                    "ridge perturbations (~mesoscale to sub-synoptic) can tip "
                    "the forecast toward recurvature.",
                ]

        elif cf_type == "STRAIGHT":
            lines += [
                "",
                "PHYSICAL INTERPRETATION (CONTINUED WESTWARD):",
                f"  A STRENGTHENING of the subtropical ridge — increasing 500 hPa "
                f"GPH by ~{abs(ch_delta_phys[1]):.0f} m over the Western Pacific — "
                "would maintain the westward steering channel and prevent recurvature. "
                "This is consistent with a stronger-than-forecast subtropical "
                "anticyclone, blocking the trough interaction. Such a scenario "
                "has been observed when the monsoon gyre intensifies and reinforces "
                "the southern arm of the ridge.",
            ]

        lines += [
            "",
            f"CHANNEL BREAKDOWN OF REQUIRED CHANGE:",
            f"  GPH: {gph_delta:.2%}  |  U: {u_delta:.2%}  |  V: {v_delta:.2%}  |  SST: {sst_delta:.2%}",
        ]

        return "\n".join(lines)
