"""
TC-PhysXAI — C7: Segment-Level Temporal XAI + AOPC Faithfulness
================================================================
Extended with:
  ① Adaptive Phase Discovery     — learns optimal phase boundaries per case
  ② Global Attribution Mining    — cross-TC pattern analysis per track regime

Physics Motivation & XAI4Traj Foundation
-----------------------------------------
Standard temporal attribution (C1) perturbs ONE 6-hourly observation window at
a time.  This misses the fact that TC steering evolves in physically meaningful
PHASES that span multiple consecutive windows.

Foundation — XAI4Traj (Neurocomputing 658, 2025):
  "Towards Explainable Trajectory Classification: A Segment-based Perturbation
   Approach" — Le Xuan Tung, Bui Dang Phuc, Vo Nguyen Le Duy, UIT-VNU HCM.
   DOI: 10.1016/j.neucom.2025.131691

  Key idea: segment trajectory → binary-mask perturb each segment →
  surrogate model (Ridge Regression) → importance map θ.
  Perturbation: Gaussian noise, Scaling, Rotation.
  Segmentation: Douglas-Peucker (RDP), MDL, Sliding Window (all spatial/data-driven).
  Fidelity: H-mean = harmonic(Precision, Recall).

TC-PhysXAI extends XAI4Traj in four dimensions:
  ┌─────────────────────┬──────────────────────────┬────────────────────────────────┐
  │ Dimension           │ XAI4Traj                 │ TC-PhysXAI C7                  │
  ├─────────────────────┼──────────────────────────┼────────────────────────────────┤
  │ Task                │ Classification (label)   │ Ensemble probabilistic (CRPS)  │
  │ Segmentation        │ Data-driven spatial      │ Physics-driven temporal A/B/C  │
  │                     │ (RDP / MDL / sliding)    │ + Adaptive boundary search     │
  │ Perturbation        │ Gaussian/Scaling/Rotation│ Climatological baseline (zero) │
  │ Attribution signal  │ Surrogate coefficient θ  │ ΔCRPS direct (no surrogate)    │
  │ Fidelity metric     │ H-mean(P, R)             │ H-mean-F(P-F, R-F, U-F)        │
  │ Scope               │ Local (per-instance)     │ Local + Global (cross-TC)      │
  └─────────────────────┴──────────────────────────┴────────────────────────────────┘

Three Physical Phases (Fixed Mode)
------------------------------------
  Phase A — REMOTE PRECURSOR  (t₋₄₈ to t₋₂₄h, steps 0–3):
    Large-scale pattern setup: 500 hPa ridge axis, MJO phase, upper trough 2d lag.
    Determines the STEERING CHANNEL the typhoon will travel through.

  Phase B — SYNOPTIC EVOLUTION (t₋₂₄ to t₋₁₂h, steps 4–5):
    Mid-range pattern change: ridge weakening or trough approach.
    The RECURVATURE BIFURCATION window — operationally highest uncertainty.

  Phase C — IMMEDIATE STEERING (t₋₁₂ to t₀, steps 6–7):
    Current synoptic state. Short-range mode — model responds to contemporaneous flow.

① Adaptive Phase Discovery (NEW — addresses XAI4Traj limitation #5: segmentation sensitivity)
------------------------------------------------------------------------------------------------
  Instead of fixed A/B/C boundaries, search ALL valid 3-phase partitions
  of 8 obs steps and select the one that maximises AOPC faithfulness.

  Search space: τ = (τ₁, τ₂) with 1 ≤ τ₁ < τ₂ ≤ 7  →  C(7,2) = 21 configs.
    Phase α: steps [0, …, τ₁−1]
    Phase β: steps [τ₁, …, τ₂−1]
    Phase γ: steps [τ₂, …, n_obs−1]

  Criterion: τ* = argmax_τ Faithfulness(τ) = argmax [MoRF_area(τ) − LeRF_area(τ)]

  This makes the segmentation DATA-ADAPTIVE while the fixed-mode retains
  PHYSICAL INTERPRETABILITY.  Both modes are available simultaneously.

② Global Attribution Mining (NEW — extends XAI4Traj from local to global)
---------------------------------------------------------------------------
  Aggregate C7 results across many TC cases, grouped by track regime
  (STRAIGHT_WEST, RECURVING_NE, STALLED, POLEWARD_TRACK).

  Outputs:
    • Mean phase attribution per regime  →  which regime depends on which phase
    • Attribution consistency (std)      →  how stable is the pattern?
    • Faithfulness distribution          →  when does C7 explanation succeed?
    • Cross-regime contrast matrix       →  differential importance analysis

AOPC Faithfulness (Samek et al. 2017)
--------------------------------------
  MoRF: mask phases from MOST-to-LEAST relevant  →  fast CRPS degradation = good
  LeRF: mask phases from LEAST-to-MOST relevant  →  slow degradation = good
  Faithfulness = MoRF_area − LeRF_area  (> 0 means attribution is faithful)

References
----------
  - XAI4Traj: Le Xuan Tung et al., Neurocomputing 658 (2025) 131691
  - AOPC: Samek et al., IEEE TNNLS 28(11) (2017) 2660-2673
  - Shapley interaction: Grabisch & Roubens (1999) IJUFKS 7(5) 505-539
  - FreeFM: Choi et al. (2024) for C4 attribution context
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from dataclasses import dataclass, field


# ─── Physical phase definitions (fixed mode) ─────────────────────────────────

PHASES: Dict[str, Dict] = {
    "A_REMOTE_PRECURSOR": {
        "obs_steps":   [0, 1, 2, 3],
        "time_label":  "t−48h to t−24h",
        "short":       "A",
        "description": (
            "REMOTE PRECURSOR PHASE (t−48h to t−24h): "
            "Large-scale pattern setup — subtropical ridge position, MJO phase, "
            "upper trough at 2-day lag.  Determines the steering channel."
        ),
    },
    "B_SYNOPTIC_EVOLUTION": {
        "obs_steps":   [4, 5],
        "time_label":  "t−24h to t−12h",
        "short":       "B",
        "description": (
            "SYNOPTIC EVOLUTION PHASE (t−24h to t−12h): "
            "Mid-range pattern change — ridge weakening or trough approach. "
            "The recurvature bifurcation window.  Highest-uncertainty phase."
        ),
    },
    "C_IMMEDIATE_STEERING": {
        "obs_steps":   [6, 7],
        "time_label":  "t−12h to t0",
        "short":       "C",
        "description": (
            "IMMEDIATE STEERING PHASE (t−12h to t0): "
            "Current synoptic state — near-term steering flow. "
            "High attribution here = model responds to contemporaneous conditions."
        ),
    },
}

# Phase interpretation templates keyed by rank (0=dominant, 1=secondary, 2=background)
_PHASE_INTERP: Dict[str, Dict[str, str]] = {
    "A_REMOTE_PRECURSOR": {
        "0": (
            "PRECURSOR PATTERN CRITICAL: The large-scale atmospheric setup 2 days "
            "before analysis strongly controls this forecast.  High Phase A attribution "
            "indicates the model relies on the 500 hPa ridge configuration and upper "
            "trough position established 48h earlier — consistent with medium-range "
            "predictability theory where the steering channel is set before the TC "
            "enters it.  Cross-check 500 hPa analyses from D−2."
        ),
        "1": (
            "PRECURSOR PATTERN SECONDARY: The remote (48-24h) atmospheric pattern "
            "provides meaningful but non-dominant attribution.  Combined with the "
            "primary phase, it suggests the TC is transitioning between steering regimes."
        ),
        "2": (
            "PRECURSOR PHASE BACKGROUND: The remote pattern contributes minimally. "
            "The model relies on more recent observations — typical of a rapidly evolving "
            "synoptic setup or a storm in a well-defined, fast-moving steering environment."
        ),
    },
    "B_SYNOPTIC_EVOLUTION": {
        "0": (
            "BIFURCATION WINDOW CRITICAL: Synoptic evolution in the 24-12h period "
            "before analysis is the dominant attribution source.  The model is sensitive "
            "to the EXACT TIMING of ridge weakening or trough approach — the classic "
            "recurvature bifurcation window.  Operationally: small analysis errors here "
            "produce large track divergence.  High Phase B = HIGH forecast uncertainty."
        ),
        "1": (
            "BIFURCATION WINDOW SECONDARY: Mid-range synoptic change contributes "
            "significantly but is not the sole driver.  Monitor ridge/trough timing "
            "as a secondary factor alongside the dominant phase."
        ),
        "2": (
            "SYNOPTIC EVOLUTION NOT CRITICAL: The 24-12h evolution does not strongly "
            "drive this forecast.  Either the pattern is stable (steady ridge, no "
            "approaching trough) or the TC is in a post-recurvature phase."
        ),
    },
    "C_IMMEDIATE_STEERING": {
        "0": (
            "CURRENT STATE DOMINANT: The most recent observations (t−12h to t0) are "
            "the primary attribution source — short-range forecast mode.  The TC is in "
            "an actively changing environment; the model responds to the current steering "
            "flow.  A 500 hPa trough currently influencing the TC or monsoon trough "
            "interaction in the lower troposphere produce strong Phase C attribution."
        ),
        "1": (
            "RECENT STATE SECONDARY: The immediate steering contributes significantly "
            "as a secondary source.  The current synoptic state reinforces the pattern "
            "established by the dominant phase."
        ),
        "2": (
            "RECENT STATE LEAST INFORMATIVE: The immediate steering (t−12h to t0) "
            "contributes least.  The model relies on memory of earlier pattern evolution "
            "rather than the current state."
        ),
    },
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PhaseAttributionResult:
    """Attribution result for one temporal phase."""
    phase_name:             str
    obs_steps:              List[int]
    time_label:             str
    delta_crps:             float   # CRPS_masked − CRPS_full  (> 0 = informative)
    delta_crps_pct:         float   # % increase in CRPS when phase is masked
    relative_attribution:   float   # fraction of total positive attribution
    description:            str
    physical_interpretation: str


@dataclass
class SegmentXAIResult:
    """Full C7 result (fixed physics-driven phases)."""
    # Per-phase attribution
    phase_results:          List[PhaseAttributionResult]
    ranked_phases:          List[str]   # sorted by delta_crps DESC

    # AOPC curves  [n_phases+1]
    aopc_morf_curve:        np.ndarray
    aopc_lerf_curve:        np.ndarray
    aopc_morf_score:        float       # normalised area under MoRF curve
    aopc_lerf_score:        float       # normalised area under LeRF curve
    aopc_faithfulness:      float       # morf_score − lerf_score  (>0 = faithful)

    # Pairwise interaction: Δ(A+B) − ΔA − ΔB
    joint_attribution:      Dict[str, float]
    interaction_effects:    Dict[str, float]

    # CRPS anchors
    crps_full:              float
    crps_all_masked:        float

    # Interpretation
    physical_interpretation: str
    dominant_phase:         str
    uncertainty_source:     str   # "A", "B", "C", or "MIXED"


@dataclass
class AdaptivePhaseConfig:
    """One candidate 3-phase partition discovered by AdaptivePhaseDiscovery."""
    tau1:           int             # boundary between phase α and β (first step of β)
    tau2:           int             # boundary between phase β and γ (first step of γ)
    steps_alpha:    List[int]
    steps_beta:     List[int]
    steps_gamma:    List[int]
    delta_alpha:    float           # ΔCRPS when α is masked
    delta_beta:     float
    delta_gamma:    float
    faithfulness:   float           # MoRF − LeRF for this partition
    morf_area:      float
    lerf_area:      float
    # Physical labels inferred from step ranges
    label_alpha:    str
    label_beta:     str
    label_gamma:    str


@dataclass
class AdaptiveDiscoveryResult:
    """Full result of the adaptive phase boundary search."""
    # Optimal configuration
    optimal:            AdaptivePhaseConfig
    # All 21 candidate configs sorted by faithfulness DESC
    all_configs:        List[AdaptivePhaseConfig]
    # Comparison with fixed A/B/C phases
    fixed_faithfulness: float
    adaptive_gain:      float       # optimal.faithfulness − fixed_faithfulness
    # Whether adaptive boundaries match the physics-driven ones
    matches_physics:    bool
    # Interpretation
    interpretation:     str


@dataclass
class TrackTypeStats:
    """Per-track-type aggregated attribution statistics."""
    track_type:             str
    n_cases:                int
    mean_delta:             Dict[str, float]   # phase → mean ΔCRPS
    std_delta:              Dict[str, float]   # phase → std  ΔCRPS
    mean_relative:          Dict[str, float]   # phase → mean relative attribution
    mean_faithfulness:      float
    std_faithfulness:       float
    dominant_phase_counts:  Dict[str, int]     # phase → how often it's dominant
    dominant_phase:         str                # most common dominant phase
    adaptive_tau1_mean:     Optional[float]    # mean optimal τ₁ (if adaptive was run)
    adaptive_tau2_mean:     Optional[float]
    interpretation:         str


@dataclass
class GlobalAttributionResult:
    """Cross-TC global attribution pattern analysis."""
    n_total_cases:      int
    track_types:        List[str]
    per_type:           Dict[str, TrackTypeStats]
    # Phase importance ranking per track type
    regime_phase_rank:  Dict[str, List[str]]
    # Cross-regime contrast: which phase differentiates regimes most?
    contrast_matrix:    Dict[str, Dict[str, float]]   # [regime_a][phase] vs [regime_b][phase]
    most_discriminative_phase: str
    interpretation:     str


# ─── Low-level helpers ────────────────────────────────────────────────────────

def _crps_ensemble(
    ensemble:     torch.Tensor,     # [K, T_pred, B, 2]  or  [K, T, 2]
    ground_truth: torch.Tensor,     # [T_pred, B, 2]     or  [T, 2]
    storm_idx:    int = 0,
) -> float:
    """Compute CRPS averaged over time steps and spatial coordinates.

    CRPS(X, y) = E[|X − y|] − ½ E[|X − X'|]
    """
    if ensemble.dim() == 4:
        ens = ensemble[:, :, storm_idx, :]   # [K, T, 2]
        gt  = ground_truth[:, storm_idx, :]  # [T, 2]
    else:
        ens = ensemble   # [K, T, 2]
        gt  = ground_truth

    K = ens.shape[0]

    # E[|X − y|]
    term1 = (ens - gt.unsqueeze(0)).norm(dim=-1).mean(0)   # [T]

    # E[|X − X'|]  via pairwise O(K²) — exact for small K
    diffs = torch.stack(
        [(ens - ens[i].unsqueeze(0)).norm(dim=-1).mean(0) for i in range(K)],
        dim=0,
    ).mean(0)   # [T]

    return float((term1 - 0.5 * diffs).mean().item())


def _mask_obs_steps(
    batch_list:    list,
    steps_to_mask: List[int],
    device:        torch.device,
) -> list:
    """Replace selected observation time steps with climatological baseline (zeros).

    Slots modified:
      batch_list[0]  = obs_traj  [T_obs, B, 2]     → zero selected rows
      batch_list[7]  = obs_Me    [T_obs, B, 4]     → zero selected rows
      batch_list[11] = img_obs   [B, 13, T_obs, H, W] → zero selected slices
    """
    out = []
    for i, item in enumerate(batch_list):
        if not torch.is_tensor(item):
            out.append(item)
            continue

        t = item.to(device)

        if i == 0:                  # obs_traj  [T_obs, B, 2]
            t_mod = t.clone()
            for s in steps_to_mask:
                if s < t_mod.shape[0]:
                    t_mod[s] = 0.0
            out.append(t_mod)

        elif i == 7:                # obs_Me  [T_obs, B, 4]
            t_mod = t.clone()
            for s in steps_to_mask:
                if s < t_mod.shape[0]:
                    t_mod[s] = 0.0
            out.append(t_mod)

        elif i == 11:               # img_obs  [B, 13, T_obs, H, W]
            t_mod = t.clone()
            for s in steps_to_mask:
                if s < t_mod.shape[2]:
                    t_mod[:, :, s, :, :] = 0.0
            out.append(t_mod)

        else:
            out.append(t)

    return out


def _infer_phase_label(steps: List[int]) -> str:
    """Infer a meteorological label from step indices (0=t-48h, 7=t0)."""
    if not steps:
        return "EMPTY"
    s_min, s_max = min(steps), max(steps)
    # Map step range to approximate lead time labels (6-hourly, 8 steps → 48h)
    t_start = 48 - s_min * 6   # hours before t0
    t_end   = 48 - s_max * 6
    return f"t−{t_start}h to t−{t_end}h" if t_end > 0 else f"t−{t_start}h to t0"


def _aopc_areas(
    morf_curve: np.ndarray,   # [n+1]
    lerf_curve: np.ndarray,   # [n+1]
    crps_full:  float,
) -> Tuple[float, float, float]:
    """Compute normalised AOPC areas and faithfulness from degradation curves."""
    crps_range = max(morf_curve[-1] - crps_full, 1e-8)
    morf_area  = float((morf_curve[1:] - crps_full).clip(0).mean()) / crps_range
    lerf_area  = float((lerf_curve[1:] - crps_full).clip(0).mean()) / crps_range
    return morf_area, lerf_area, morf_area - lerf_area


# ─── Core C7 class ────────────────────────────────────────────────────────────

class SegmentPerturbationXAI:
    """C7: Physics-Driven Segment Temporal XAI with AOPC Faithfulness.

    Direct extension of XAI4Traj (Neurocomputing 2025) to probabilistic
    typhoon track prediction with Conditional Flow Matching.

    Key differences from XAI4Traj:
      • Segments defined by physical meteorological phases (not data-driven RDP)
      • Attribution signal is ΔCRPS from ensemble sampling (not surrogate θ)
      • Perturbation baseline is climatological mean (not Gaussian/scaling)
      • Fidelity uses H-mean-F with Uncertainty-F (not plain H-mean)

    Usage
    -----
    >>> sp = SegmentPerturbationXAI(model, device, K=20)
    >>> result = sp.compute(batch_list, verbose=True)
    >>> print(result.physical_interpretation)
    >>> print(f"AOPC faithfulness: {result.aopc_faithfulness:.4f}")
    """

    def __init__(
        self,
        model,
        device:  torch.device,
        K:       int = 20,
    ):
        self.model  = model
        self.device = device
        self.K      = K
        self.model.eval()

    # ─── Ensemble runner ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_ensemble(
        self,
        batch_list: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (ensemble_traj [K,T,B,2], pred_gt [T,B,2])."""
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        _, _, all_traj = self.model.sample(
            bl,
            num_ensemble=self.K,
            use_speed_calibration=True,
            return_xai=False,
        )
        return all_traj, bl[1]

    # ─── Phase attribution ────────────────────────────────────────────────────

    def _phase_attribution(
        self,
        batch_list:   list,
        phase_dict:   Dict[str, List[int]],   # {name: [steps]}
        crps_full:    float,
        storm_idx:    int,
        verbose:      bool,
    ) -> Dict[str, float]:
        """Mask each phase independently; return {phase_name: ΔCRPS}."""
        deltas: Dict[str, float] = {}
        for pname, steps in phase_dict.items():
            bl_m = _mask_obs_steps(batch_list, steps, self.device)
            try:
                at, pg = self._run_ensemble(bl_m)
                delta  = _crps_ensemble(at, pg, storm_idx) - crps_full
            except Exception as exc:
                if verbose:
                    print(f"  [C7] Phase {pname} masking failed: {exc}")
                delta = 0.0
            deltas[pname] = delta
        return deltas

    # ─── AOPC curves ─────────────────────────────────────────────────────────

    def _aopc_curves(
        self,
        batch_list:     list,
        ranked_phases:  List[str],          # most → least important
        phase_steps:    Dict[str, List[int]],
        crps_full:      float,
        storm_idx:      int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MoRF and LeRF CRPS degradation curves [n_phases+1]."""
        n = len(ranked_phases)
        morf = np.zeros(n + 1);  morf[0] = crps_full
        lerf = np.zeros(n + 1);  lerf[0] = crps_full

        cum: List[int] = []
        for k, pn in enumerate(ranked_phases):
            cum.extend(phase_steps[pn])
            bl_m = _mask_obs_steps(batch_list, cum, self.device)
            try:
                at, pg = self._run_ensemble(bl_m)
                morf[k + 1] = _crps_ensemble(at, pg, storm_idx)
            except Exception:
                morf[k + 1] = morf[k]

        cum_r: List[int] = []
        for k, pn in enumerate(reversed(ranked_phases)):
            cum_r.extend(phase_steps[pn])
            bl_m = _mask_obs_steps(batch_list, cum_r, self.device)
            try:
                at, pg = self._run_ensemble(bl_m)
                lerf[k + 1] = _crps_ensemble(at, pg, storm_idx)
            except Exception:
                lerf[k + 1] = lerf[k]

        return morf, lerf

    # ─── Interaction effects ─────────────────────────────────────────────────

    def _interaction_effects(
        self,
        batch_list:    list,
        phase_names:   List[str],
        phase_steps:   Dict[str, List[int]],
        phase_deltas:  Dict[str, float],
        crps_full:     float,
        storm_idx:     int,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute Δ(A+B) and interaction = Δ(A+B) − ΔA − ΔB.

        Interaction > 0: synergistic (phases cooperate; combined > sum of parts)
        Interaction < 0: redundant (phases share information)
        Interaction ≈ 0: additive (phases are independent information sources)
        """
        joint:   Dict[str, float] = {}
        interact: Dict[str, float] = {}

        for i in range(len(phase_names)):
            for j in range(i + 1, len(phase_names)):
                pa, pb   = phase_names[i], phase_names[j]
                steps_ab = phase_steps[pa] + phase_steps[pb]
                bl_m = _mask_obs_steps(batch_list, steps_ab, self.device)
                try:
                    at, pg = self._run_ensemble(bl_m)
                    delta_ab = _crps_ensemble(at, pg, storm_idx) - crps_full
                except Exception:
                    delta_ab = phase_deltas.get(pa, 0.0) + phase_deltas.get(pb, 0.0)

                key           = f"{pa}+{pb}"
                joint[key]    = delta_ab
                interact[key] = delta_ab - (phase_deltas.get(pa, 0.0) + phase_deltas.get(pb, 0.0))

        return joint, interact

    # ─── Physical interpretation ──────────────────────────────────────────────

    def _fill_phase_results(
        self,
        deltas:         Dict[str, float],
        crps_full:      float,
        ranked_phases:  List[str],
    ) -> List[PhaseAttributionResult]:
        """Assemble PhaseAttributionResult objects with interpretations."""
        total_pos = sum(max(v, 0.0) for v in deltas.values()) + 1e-8
        results   = []

        for pname, pinfo in PHASES.items():
            delta = deltas.get(pname, 0.0)
            rank  = ranked_phases.index(pname)

            interp = _PHASE_INTERP.get(pname, {}).get(
                str(rank),
                f"Phase {pname}: ΔCRPS = {delta:.4f}"
            )

            results.append(PhaseAttributionResult(
                phase_name=pname,
                obs_steps=pinfo["obs_steps"],
                time_label=pinfo["time_label"],
                delta_crps=delta,
                delta_crps_pct=delta / (abs(crps_full) + 1e-8) * 100.0,
                relative_attribution=max(delta, 0.0) / total_pos,
                description=pinfo["description"],
                physical_interpretation=interp,
            ))

        return results

    @staticmethod
    def _classify_uncertainty_source(deltas: Dict[str, float]) -> str:
        total = sum(max(v, 0) for v in deltas.values()) + 1e-8
        best  = max(deltas, key=deltas.get)
        if deltas[best] / total > 0.60:
            return best.split("_")[0]   # "A", "B", or "C"
        return "MIXED"

    # ─── Textual interpretation ───────────────────────────────────────────────

    def _interpret(
        self,
        phase_results:   List[PhaseAttributionResult],
        ranked_phases:   List[str],
        faithfulness:    float,
        morf_area:       float,
        lerf_area:       float,
        crps_full:       float,
        crps_all_masked: float,
        interact:        Dict[str, float],
        adaptive_note:   str = "",
    ) -> str:
        lines = [
            "=" * 68,
            "TC-PhysXAI C7 — Segment-Level Temporal XAI  (extends XAI4Traj)",
            "=" * 68,
        ]

        # ── Phase attribution table ───────────────────────────────────────────
        lines.append("\nPHASE ATTRIBUTION SUMMARY:")
        rank_symbols = {0: "★ MOST CRITICAL ", 1: "☆ SECONDARY     ", 2: "○ BACKGROUND    "}
        for r in sorted(phase_results, key=lambda x: x.delta_crps, reverse=True):
            rank  = ranked_phases.index(r.phase_name)
            label = rank_symbols.get(rank, "  ")
            bar   = "█" * int(r.relative_attribution * 30)
            lines.append(
                f"  {label}{r.phase_name:28s} [{bar:<30s}] "
                f"ΔCRPS={r.delta_crps:+.4f} ({r.delta_crps_pct:+.1f}%)"
            )

        # ── AOPC faithfulness ─────────────────────────────────────────────────
        lines.append("\nAOPC FAITHFULNESS EVALUATION:")
        lines.append(f"  MoRF score (mask most-relevant first):  {morf_area:.4f}")
        lines.append(f"  LeRF score (mask least-relevant first): {lerf_area:.4f}")
        lines.append(f"  Faithfulness = MoRF − LeRF:             {faithfulness:+.4f}")

        if faithfulness > 0.15:
            verdict = "✓ FAITHFUL"
            detail  = (
                "Masking high-attribution phases hurts prediction significantly more "
                "than masking low-attribution phases.  The C7 attribution ranking "
                "correctly identifies the most informative temporal windows."
            )
        elif faithfulness > -0.05:
            verdict = "~ MARGINAL"
            detail  = (
                "Attribution has limited ordering power over phases.  The three phases "
                "carry similar information amounts — the storm is in a phase-balanced "
                "steering regime, or all observation windows are equally informative."
            )
        else:
            verdict = "✗ COUNTER-FAITHFUL"
            detail  = (
                "Unusual result.  Masking disrupts temporal coherence in an unexpected "
                "way.  Consider soft-masking (linear interpolation to baseline) instead "
                "of hard zeroing, or verify that the ensemble size K is sufficient."
            )
        lines.append(f"  {verdict}: {detail}")

        # ── Information completeness ──────────────────────────────────────────
        if not np.isnan(crps_all_masked):
            loss_pct = (crps_all_masked - crps_full) / (abs(crps_full) + 1e-8) * 100
            lines.append(
                f"\nINFORMATION COMPLETENESS:"
                f"\n  CRPS (full observations):    {crps_full:.4f}"
                f"\n  CRPS (all phases masked):    {crps_all_masked:.4f}"
                f"\n  Total information loss:      {loss_pct:+.1f}%"
                f"\n  → Removing ALL observations degrades CRPS by {loss_pct:.0f}%.  "
                f"This is the XAI attribution budget for this case."
            )

        # ── Interaction effects ───────────────────────────────────────────────
        sig_interact = {k: v for k, v in interact.items() if abs(v) > 0.001}
        if sig_interact:
            lines.append("\nPHASE INTERACTION EFFECTS (non-additivity):")
            for key, val in sorted(sig_interact.items(), key=lambda x: abs(x[1]), reverse=True):
                nature = "SYNERGISTIC (phases cooperate)" if val > 0 else "REDUNDANT (phases share info)"
                lines.append(f"  {key}: {val:+.4f}  →  {nature}")

        # ── Dominant phase narrative ──────────────────────────────────────────
        dom = next((r for r in phase_results if r.phase_name == ranked_phases[0]), None)
        if dom:
            lines.append(f"\nDOMINANT PHASE INTERPRETATION:")
            for line in dom.physical_interpretation.split(". "):
                lines.append(f"  {line}.")

        # ── Operational recommendation ────────────────────────────────────────
        lines.append("\nOPERATIONAL RECOMMENDATION:")
        dom_key = ranked_phases[0] if ranked_phases else ""
        if "A_REMOTE" in dom_key:
            lines.append(
                "  → Prioritise the D−2 synoptic analysis (500 hPa height pattern, "
                "upper trough position, ridge axis).  Medium-range ensemble guidance "
                "(ECMWF EPS, GEFS) from 2 days before is the primary uncertainty driver."
            )
        elif "B_SYNOPTIC" in dom_key:
            lines.append(
                "  → Monitor ridge-weakening/trough-approach timing in the 24-12h "
                "window.  Ensemble spread in 500 hPa is maximally informative here.  "
                "Use satellite upper-troposphere water vapour for real-time tracking."
            )
        elif "C_IMMEDIATE" in dom_key:
            lines.append(
                "  → Rely on the latest surface and upper-air analyses.  Short-range "
                "deterministic guidance (HWRF, CReSS) is most relevant.  Watch for "
                "rapid evolution in the 12h window before forecast issuance."
            )
        else:
            lines.append(
                "  → Mixed attribution: use multi-model consensus across short-, "
                "medium- and extended-range guidance.  Track uncertainty is high — "
                "issue wider-than-normal forecast cones."
            )

        if adaptive_note:
            lines.append(f"\nADAPTIVE PHASE NOTE:\n  {adaptive_note}")

        lines.append("=" * 68)
        return "\n".join(lines)

    # ─── Main compute API ─────────────────────────────────────────────────────

    def compute(
        self,
        batch_list: list,
        storm_idx:  int  = 0,
        verbose:    bool = False,
    ) -> SegmentXAIResult:
        """Run C7 with fixed physics-driven phase segmentation (A / B / C).

        Parameters
        ----------
        batch_list  : model input list (obs_traj[0], pred_traj[1], ..., img_obs[11])
        storm_idx   : which sample in the batch to evaluate (default 0)
        verbose     : print progress

        Returns
        -------
        SegmentXAIResult
        """
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]

        # Step 1 — baseline CRPS
        if verbose:
            print("[C7] Step 1/6: baseline CRPS …")
        all_traj, pred_gt = self._run_ensemble(bl)
        crps_full = _crps_ensemble(all_traj, pred_gt, storm_idx)
        if verbose:
            print(f"         CRPS_full = {crps_full:.4f}")

        # Step 2 — per-phase attribution
        if verbose:
            print("[C7] Step 2/6: per-phase attribution (A, B, C) …")
        phase_steps = {pn: pi["obs_steps"] for pn, pi in PHASES.items()}
        deltas      = self._phase_attribution(bl, phase_steps, crps_full, storm_idx, verbose)

        ranked_phases = sorted(deltas, key=deltas.get, reverse=True)
        if verbose:
            for rk, pn in enumerate(ranked_phases):
                print(f"         [{rk}] {pn}: ΔCRPS={deltas[pn]:+.4f}")

        # Step 3 — AOPC curves
        if verbose:
            print("[C7] Step 3/6: AOPC MoRF / LeRF curves …")
        morf_curve, lerf_curve = self._aopc_curves(
            bl, ranked_phases, phase_steps, crps_full, storm_idx
        )
        morf_area, lerf_area, faithfulness = _aopc_areas(morf_curve, lerf_curve, crps_full)

        # Step 4 — all-phases-masked CRPS (information upper bound)
        if verbose:
            print("[C7] Step 4/6: all-phases-masked CRPS …")
        all_steps = [s for pi in PHASES.values() for s in pi["obs_steps"]]
        bl_all_m  = _mask_obs_steps(bl, list(set(all_steps)), self.device)
        try:
            at_all, pg_all = self._run_ensemble(bl_all_m)
            crps_all_masked = _crps_ensemble(at_all, pg_all, storm_idx)
        except Exception:
            crps_all_masked = float("nan")

        # Step 5 — interaction effects
        if verbose:
            print("[C7] Step 5/6: pairwise phase interaction effects …")
        joint_attr, interact = self._interaction_effects(
            bl, list(PHASES.keys()), phase_steps, deltas, crps_full, storm_idx
        )

        # Step 6 — assemble & interpret
        if verbose:
            print("[C7] Step 6/6: assembling results …")
        phase_results = self._fill_phase_results(deltas, crps_full, ranked_phases)
        dom_phase     = ranked_phases[0] if ranked_phases else "UNKNOWN"
        unc_source    = self._classify_uncertainty_source(deltas)
        phys_interp   = self._interpret(
            phase_results, ranked_phases, faithfulness, morf_area, lerf_area,
            crps_full, crps_all_masked, interact,
        )

        return SegmentXAIResult(
            phase_results=phase_results,
            ranked_phases=ranked_phases,
            aopc_morf_curve=morf_curve,
            aopc_lerf_curve=lerf_curve,
            aopc_morf_score=morf_area,
            aopc_lerf_score=lerf_area,
            aopc_faithfulness=faithfulness,
            joint_attribution=joint_attr,
            interaction_effects=interact,
            crps_full=crps_full,
            crps_all_masked=crps_all_masked,
            physical_interpretation=phys_interp,
            dominant_phase=dom_phase,
            uncertainty_source=unc_source,
        )


# ─── ① Adaptive Phase Discovery ──────────────────────────────────────────────

class AdaptivePhaseDiscovery:
    """Discover the optimal temporal phase partition for a TC case.

    Extends XAI4Traj's fixed segmentation by making the phase BOUNDARIES
    themselves data-adaptive — maximising AOPC faithfulness over all
    valid 3-phase splits of the 8 observation steps.

    This directly addresses the "segmentation sensitivity" limitation
    noted in XAI4Traj's Conclusion.

    Algorithm
    ---------
    1. Enumerate all C(n_obs-1, 2) = 21 valid (τ₁, τ₂) splits.
    2. For each split:
       a. Define α=[0..τ₁-1], β=[τ₁..τ₂-1], γ=[τ₂..n_obs-1].
       b. Compute ΔCRPS for each of the 3 phases.
       c. Rank phases → compute MoRF / LeRF curves.
       d. Faithfulness = MoRF_area − LeRF_area.
    3. Return split with highest faithfulness as τ*.

    Comparison with fixed A/B/C is reported automatically.

    Usage
    -----
    >>> apd = AdaptivePhaseDiscovery(model, device, K=10)  # use smaller K for speed
    >>> result = apd.discover(batch_list, storm_idx=0, verbose=True)
    >>> print(f"Optimal τ* = ({result.optimal.tau1}, {result.optimal.tau2})")
    >>> print(f"Adaptive gain over fixed: {result.adaptive_gain:+.4f}")
    """

    def __init__(
        self,
        model,
        device:       torch.device,
        K:            int = 10,     # smaller K for fast search
        n_obs_steps:  int = 8,
    ):
        self.model        = model
        self.device       = device
        self.K            = K
        self.n_obs_steps  = n_obs_steps
        self._sp          = SegmentPerturbationXAI(model, device, K=K)
        self.model.eval()

    @staticmethod
    def _enumerate_splits(n: int) -> List[Tuple[int, int]]:
        """All valid (τ₁, τ₂) with 1 ≤ τ₁ < τ₂ ≤ n−1, each phase ≥ 1 step."""
        return [(t1, t2) for t1 in range(1, n - 1) for t2 in range(t1 + 1, n)]

    def _one_config_faithfulness(
        self,
        batch_list:  list,
        tau1:        int,
        tau2:        int,
        crps_full:   float,
        storm_idx:   int,
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Evaluate one (τ₁, τ₂) split. Returns (morf, lerf, faith, deltas)."""
        n = self.n_obs_steps
        alpha = list(range(0, tau1))
        beta  = list(range(tau1, tau2))
        gamma = list(range(tau2, n))

        phase_steps = {"alpha": alpha, "beta": beta, "gamma": gamma}
        deltas = self._sp._phase_attribution(
            batch_list, phase_steps, crps_full, storm_idx, verbose=False
        )

        ranked = sorted(deltas, key=deltas.get, reverse=True)
        morf_c, lerf_c = self._sp._aopc_curves(
            batch_list, ranked, phase_steps, crps_full, storm_idx
        )
        morf_a, lerf_a, faith = _aopc_areas(morf_c, lerf_c, crps_full)
        return morf_a, lerf_a, faith, deltas

    @torch.no_grad()
    def discover(
        self,
        batch_list: list,
        storm_idx:  int  = 0,
        verbose:    bool = False,
    ) -> AdaptiveDiscoveryResult:
        """Run exhaustive search over all valid phase splits.

        Parameters
        ----------
        batch_list : model input list
        storm_idx  : sample in batch to evaluate
        verbose    : print search progress

        Returns
        -------
        AdaptiveDiscoveryResult with optimal split and comparison to fixed phases
        """
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        n  = self.n_obs_steps

        # Baseline CRPS (shared across all splits)
        if verbose:
            print(f"[AdaptivePhaseDiscovery] Baseline CRPS …")
        all_traj, pred_gt = self._sp._run_ensemble(bl)
        crps_full = _crps_ensemble(all_traj, pred_gt, storm_idx)
        if verbose:
            print(f"  CRPS_full = {crps_full:.4f}")

        # Fixed A/B/C faithfulness (for comparison)
        fixed_phase_steps = {pn: pi["obs_steps"] for pn, pi in PHASES.items()}
        fixed_deltas = self._sp._phase_attribution(bl, fixed_phase_steps, crps_full, storm_idx, False)
        fixed_ranked = sorted(fixed_deltas, key=fixed_deltas.get, reverse=True)
        fixed_morf, fixed_lerf = self._sp._aopc_curves(bl, fixed_ranked, fixed_phase_steps, crps_full, storm_idx)
        _, _, fixed_faithfulness = _aopc_areas(fixed_morf, fixed_lerf, crps_full)

        # Exhaustive search
        splits = self._enumerate_splits(n)
        if verbose:
            print(f"[AdaptivePhaseDiscovery] Searching {len(splits)} splits …")

        configs: List[AdaptivePhaseConfig] = []
        for k, (tau1, tau2) in enumerate(splits):
            if verbose and k % 5 == 0:
                print(f"  [{k+1}/{len(splits)}] τ=({tau1},{tau2}) …", end=" ", flush=True)

            morf_a, lerf_a, faith, deltas = self._one_config_faithfulness(
                bl, tau1, tau2, crps_full, storm_idx
            )

            alpha = list(range(0, tau1))
            beta  = list(range(tau1, tau2))
            gamma = list(range(tau2, n))

            cfg = AdaptivePhaseConfig(
                tau1=tau1, tau2=tau2,
                steps_alpha=alpha, steps_beta=beta, steps_gamma=gamma,
                delta_alpha=deltas.get("alpha", 0.0),
                delta_beta=deltas.get("beta",  0.0),
                delta_gamma=deltas.get("gamma", 0.0),
                faithfulness=faith,
                morf_area=morf_a, lerf_area=lerf_a,
                label_alpha=_infer_phase_label(alpha),
                label_beta=_infer_phase_label(beta),
                label_gamma=_infer_phase_label(gamma),
            )
            configs.append(cfg)

            if verbose and k % 5 == 0:
                print(f"faith={faith:+.4f}")

        # Sort by faithfulness descending
        configs.sort(key=lambda c: c.faithfulness, reverse=True)
        optimal = configs[0]

        # Does the optimal split match the fixed A/B/C partition?
        fixed_tau1 = min(PHASES["B_SYNOPTIC_EVOLUTION"]["obs_steps"])
        fixed_tau2 = min(PHASES["C_IMMEDIATE_STEERING"]["obs_steps"])
        matches_physics = (optimal.tau1 == fixed_tau1 and optimal.tau2 == fixed_tau2)

        adaptive_gain = optimal.faithfulness - fixed_faithfulness

        # Build interpretation
        interp = self._interpret(optimal, configs, fixed_faithfulness, adaptive_gain, matches_physics)

        if verbose:
            print(f"\n[AdaptivePhaseDiscovery] RESULT:")
            print(f"  τ* = ({optimal.tau1}, {optimal.tau2})")
            print(f"  Phase α: steps {optimal.steps_alpha}  →  {optimal.label_alpha}")
            print(f"  Phase β: steps {optimal.steps_beta }  →  {optimal.label_beta}")
            print(f"  Phase γ: steps {optimal.steps_gamma}  →  {optimal.label_gamma}")
            print(f"  Faithfulness = {optimal.faithfulness:+.4f}  (fixed = {fixed_faithfulness:+.4f}, gain = {adaptive_gain:+.4f})")
            if matches_physics:
                print("  ✓ Optimal split MATCHES the physics-driven A/B/C partition!")
            else:
                print(f"  ✗ Optimal split DIFFERS from physics (fixed τ=({fixed_tau1},{fixed_tau2}))")

        return AdaptiveDiscoveryResult(
            optimal=optimal,
            all_configs=configs,
            fixed_faithfulness=fixed_faithfulness,
            adaptive_gain=adaptive_gain,
            matches_physics=matches_physics,
            interpretation=interp,
        )

    @staticmethod
    def _interpret(
        optimal:            AdaptivePhaseConfig,
        all_configs:        List[AdaptivePhaseConfig],
        fixed_faithfulness: float,
        adaptive_gain:      float,
        matches_physics:    bool,
    ) -> str:
        lines = [
            "=" * 68,
            "ADAPTIVE PHASE DISCOVERY — τ* Boundary Search",
            "=" * 68,
            "",
            f"Optimal partition:  τ* = (τ₁={optimal.tau1}, τ₂={optimal.tau2})",
            f"  Phase α: steps {optimal.steps_alpha}  ({optimal.label_alpha})",
            f"           ΔCRPS = {optimal.delta_alpha:+.4f}",
            f"  Phase β: steps {optimal.steps_beta }  ({optimal.label_beta})",
            f"           ΔCRPS = {optimal.delta_beta:+.4f}",
            f"  Phase γ: steps {optimal.steps_gamma}  ({optimal.label_gamma})",
            f"           ΔCRPS = {optimal.delta_gamma:+.4f}",
            "",
            f"Faithfulness (optimal):   {optimal.faithfulness:+.4f}",
            f"Faithfulness (fixed A/B/C): {fixed_faithfulness:+.4f}",
            f"Adaptive gain:            {adaptive_gain:+.4f}",
        ]

        if matches_physics:
            lines.append(
                "\n✓ PHYSICS CONFIRMED: The data-optimal phase boundaries (τ₁, τ₂) "
                "coincide with the physics-driven A/B/C partition.  This validates "
                "the meteorological phase definitions empirically — the observational "
                "data agrees with the physical-phase hypothesis."
            )
        elif adaptive_gain > 0.05:
            lines.append(
                f"\n✗ PHYSICS REFINED: The optimal split τ*=({optimal.tau1},{optimal.tau2}) "
                f"differs from fixed A/B/C (τ=({min(PHASES['B_SYNOPTIC_EVOLUTION']['obs_steps'])},"
                f"{min(PHASES['C_IMMEDIATE_STEERING']['obs_steps'])})).  "
                f"Adaptive gain = {adaptive_gain:+.4f}.  For this TC case, the physically "
                f"meaningful boundary between precursor and synoptic phases shifts relative "
                f"to the canonical 24h / 12h thresholds.  This may indicate an anomalous "
                f"synoptic situation (e.g. fast-moving trough, early recurvature trigger)."
            )
        else:
            lines.append(
                f"\n~ MARGINAL DIFFERENCE: τ*=({optimal.tau1},{optimal.tau2}) gives "
                f"a small gain ({adaptive_gain:+.4f}) over fixed A/B/C.  Both segmentation "
                f"schemes are approximately equivalent for this case."
            )

        # Show top-5 configs
        lines.append("\nTop-5 candidate partitions by faithfulness:")
        for rank, cfg in enumerate(all_configs[:5]):
            mark = " ← OPTIMAL" if rank == 0 else ""
            lines.append(
                f"  [{rank+1}] τ=({cfg.tau1},{cfg.tau2})  "
                f"faith={cfg.faithfulness:+.4f}  "
                f"α-steps={cfg.steps_alpha}  β-steps={cfg.steps_beta}  γ-steps={cfg.steps_gamma}"
                f"{mark}"
            )

        return "\n".join(lines)


# ─── ② Global Attribution Miner ──────────────────────────────────────────────

class GlobalAttributionMiner:
    """Aggregate C7 phase attributions across multiple TC cases.

    Extends XAI4Traj from local (per-instance) to global (cross-TC) analysis,
    revealing systematic relationships between temporal observational phases
    and track regime predictability.

    Key questions answered:
      • Which phase is most important for RECURVING vs STRAIGHT-WEST TCs?
      • Which phase attribution is most consistent (low std) across cases?
      • Which phase best DIFFERENTIATES track regimes?
      • When does C7 produce faithful explanations (faithfulness distribution)?

    Usage
    -----
    >>> miner = GlobalAttributionMiner()
    >>> for typhoon_case in dataset:
    ...     result = sp.compute(typhoon_case.batch_list)
    ...     adaptive = apd.discover(typhoon_case.batch_list)   # optional
    ...     miner.add(
    ...         result,
    ...         track_type = typhoon_case.track_label,
    ...         typhoon_id = typhoon_case.name,
    ...         adaptive   = adaptive,
    ...     )
    >>> global_result = miner.mine()
    >>> print(global_result.interpretation)
    """

    # Canonical track type labels
    TRACK_TYPES = [
        "STRAIGHT_WEST",
        "RECURVING_NE",
        "STALLED",
        "POLEWARD_TRACK",
        "MIXED",
    ]

    def __init__(self):
        # Accumulated records: {track_type: [{phase: delta, 'faithfulness': f, ...}]}
        self._records: Dict[str, List[Dict]] = defaultdict(list)

    def add(
        self,
        result:      SegmentXAIResult,
        track_type:  str,
        typhoon_id:  str = "",
        adaptive:    Optional[AdaptiveDiscoveryResult] = None,
    ) -> None:
        """Register one TC case's C7 result.

        Parameters
        ----------
        result      : output of SegmentPerturbationXAI.compute()
        track_type  : one of TRACK_TYPES (or any string)
        typhoon_id  : identifier for logging (e.g. "HAGIBIS_2019")
        adaptive    : optional AdaptiveDiscoveryResult for τ* statistics
        """
        rec: Dict = {
            "typhoon_id":   typhoon_id,
            "track_type":   track_type,
            "faithfulness": result.aopc_faithfulness,
            "crps_full":    result.crps_full,
            "dominant":     result.dominant_phase,
            "uncertainty":  result.uncertainty_source,
        }
        for r in result.phase_results:
            rec[f"delta_{r.phase_name}"]    = r.delta_crps
            rec[f"rel_{r.phase_name}"]      = r.relative_attribution
        for key, val in result.interaction_effects.items():
            rec[f"interact_{key}"] = val

        if adaptive is not None:
            rec["adaptive_tau1"]         = adaptive.optimal.tau1
            rec["adaptive_tau2"]         = adaptive.optimal.tau2
            rec["adaptive_faithfulness"] = adaptive.optimal.faithfulness
            rec["adaptive_gain"]         = adaptive.adaptive_gain
            rec["matches_physics"]       = adaptive.matches_physics

        self._records[track_type].append(rec)

    def mine(self) -> GlobalAttributionResult:
        """Compute global attribution statistics across all accumulated cases.

        Returns
        -------
        GlobalAttributionResult
        """
        if not self._records:
            raise RuntimeError("GlobalAttributionMiner: no cases registered.  Call add() first.")

        track_types  = sorted(self._records.keys())
        phase_names  = list(PHASES.keys())
        per_type:    Dict[str, TrackTypeStats] = {}

        # ── Per-track-type statistics ─────────────────────────────────────────
        for tt in track_types:
            recs = self._records[tt]
            n    = len(recs)

            mean_delta   = {}
            std_delta    = {}
            mean_rel     = {}
            dom_counts   = defaultdict(int)

            for pn in phase_names:
                vals_d = [r.get(f"delta_{pn}", 0.0) for r in recs]
                vals_r = [r.get(f"rel_{pn}",   0.0) for r in recs]
                mean_delta[pn] = float(np.mean(vals_d))
                std_delta[pn]  = float(np.std(vals_d, ddof=1)) if n > 1 else 0.0
                mean_rel[pn]   = float(np.mean(vals_r))

            for r in recs:
                dom_counts[r.get("dominant", "UNKNOWN")] += 1

            dominant_phase = max(dom_counts, key=dom_counts.get)
            faithfulness_vals = [r.get("faithfulness", float("nan")) for r in recs]
            faithfulness_vals = [v for v in faithfulness_vals if not np.isnan(v)]

            # Adaptive τ* statistics (if available)
            tau1_vals = [r["adaptive_tau1"] for r in recs if "adaptive_tau1" in r]
            tau2_vals = [r["adaptive_tau2"] for r in recs if "adaptive_tau2" in r]

            per_type[tt] = TrackTypeStats(
                track_type=tt,
                n_cases=n,
                mean_delta=mean_delta,
                std_delta=std_delta,
                mean_relative=mean_rel,
                mean_faithfulness=float(np.mean(faithfulness_vals)) if faithfulness_vals else float("nan"),
                std_faithfulness=float(np.std(faithfulness_vals, ddof=1)) if len(faithfulness_vals) > 1 else 0.0,
                dominant_phase_counts=dict(dom_counts),
                dominant_phase=dominant_phase,
                adaptive_tau1_mean=float(np.mean(tau1_vals)) if tau1_vals else None,
                adaptive_tau2_mean=float(np.mean(tau2_vals)) if tau2_vals else None,
                interpretation="",  # filled below
            )

        # ── Phase importance ranking per track type ────────────────────────────
        regime_phase_rank: Dict[str, List[str]] = {}
        for tt, stats in per_type.items():
            regime_phase_rank[tt] = sorted(
                phase_names, key=lambda pn: stats.mean_delta[pn], reverse=True
            )

        # ── Cross-regime contrast matrix ──────────────────────────────────────
        # For each pair of track types, compute |mean_delta[A] − mean_delta[B]| per phase
        contrast: Dict[str, Dict[str, float]] = {}
        for tt_a, tt_b in itertools.combinations(track_types, 2):
            key = f"{tt_a} vs {tt_b}"
            contrast[key] = {
                pn: abs(per_type[tt_a].mean_delta[pn] - per_type[tt_b].mean_delta[pn])
                for pn in phase_names
            }

        # Most discriminative phase: highest mean contrast across all pairs
        phase_contrast_mean: Dict[str, float] = {}
        for pn in phase_names:
            vals = [d[pn] for d in contrast.values()]
            phase_contrast_mean[pn] = float(np.mean(vals)) if vals else 0.0
        most_disc_phase = max(phase_contrast_mean, key=phase_contrast_mean.get)

        # ── Per-type interpretations ──────────────────────────────────────────
        for tt, stats in per_type.items():
            stats.interpretation = self._interpret_type(stats, most_disc_phase)

        # ── Global interpretation ─────────────────────────────────────────────
        total_n = sum(len(v) for v in self._records.values())
        global_interp = self._global_narrative(
            per_type, regime_phase_rank, contrast, most_disc_phase,
            phase_contrast_mean, total_n,
        )

        return GlobalAttributionResult(
            n_total_cases=total_n,
            track_types=track_types,
            per_type=per_type,
            regime_phase_rank=regime_phase_rank,
            contrast_matrix=contrast,
            most_discriminative_phase=most_disc_phase,
            interpretation=global_interp,
        )

    @staticmethod
    def _interpret_type(stats: TrackTypeStats, most_disc_phase: str) -> str:
        pn_names = {"A_REMOTE_PRECURSOR": "Phase A (remote)", "B_SYNOPTIC_EVOLUTION": "Phase B (synoptic)", "C_IMMEDIATE_STEERING": "Phase C (immediate)"}
        lines = [f"[{stats.track_type}]  n={stats.n_cases}  "
                 f"dominant_phase={stats.dominant_phase}  "
                 f"mean_faithfulness={stats.mean_faithfulness:.3f}±{stats.std_faithfulness:.3f}"]
        for pn in sorted(stats.mean_delta, key=stats.mean_delta.get, reverse=True):
            lines.append(
                f"  {pn_names.get(pn, pn)}: "
                f"ΔCRPS={stats.mean_delta[pn]:+.4f}±{stats.std_delta[pn]:.4f}  "
                f"rel={stats.mean_relative[pn]:.1%}"
            )
        if stats.adaptive_tau1_mean is not None:
            lines.append(
                f"  Adaptive τ* mean: ({stats.adaptive_tau1_mean:.1f}, {stats.adaptive_tau2_mean:.1f})"
            )
        return "\n".join(lines)

    @staticmethod
    def _global_narrative(
        per_type:           Dict[str, TrackTypeStats],
        regime_phase_rank:  Dict[str, List[str]],
        contrast:           Dict[str, Dict[str, float]],
        most_disc_phase:    str,
        phase_contrast_mean: Dict[str, float],
        total_n:            int,
    ) -> str:
        phase_labels = {
            "A_REMOTE_PRECURSOR":   "Phase A (t−48h to t−24h, remote precursor)",
            "B_SYNOPTIC_EVOLUTION": "Phase B (t−24h to t−12h, synoptic evolution)",
            "C_IMMEDIATE_STEERING": "Phase C (t−12h to t0, immediate steering)",
        }
        lines = [
            "=" * 68,
            "GLOBAL ATTRIBUTION MINING — Cross-TC Pattern Analysis",
            f"Total cases: {total_n}  |  Track regimes: {len(per_type)}",
            "=" * 68,
        ]

        lines.append("\nPER-REGIME DOMINANT PHASE:")
        for tt, stats in per_type.items():
            prank = regime_phase_rank.get(tt, [])
            phase_str = " > ".join(p.split("_")[0] for p in prank)
            lines.append(f"  {tt:20s}  {phase_str}  (n={stats.n_cases})")

        lines.append(
            f"\nMOST DISCRIMINATIVE PHASE: {phase_labels.get(most_disc_phase, most_disc_phase)}"
        )
        lines.append(
            "  This phase best separates track regimes in attribution space — "
            "the strongest signal for automated track classification via XAI."
        )
        lines.append("  Cross-regime contrast by phase:")
        for pn, cm in sorted(phase_contrast_mean.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(cm / max(phase_contrast_mean.values()) * 20)
            lines.append(f"    {pn:30s} [{bar:<20s}] mean |Δ| = {cm:.4f}")

        lines.append("\nFAITHFULNESS DISTRIBUTION BY REGIME:")
        for tt, stats in per_type.items():
            faith_label = (
                "FAITHFUL" if stats.mean_faithfulness > 0.1 else
                "MARGINAL" if stats.mean_faithfulness > -0.05 else
                "COUNTER-FAITHFUL"
            )
            lines.append(
                f"  {tt:20s}  {stats.mean_faithfulness:+.4f}±{stats.std_faithfulness:.4f}  [{faith_label}]"
            )

        lines.append("\nSYNTHESIS:")
        lines.append(
            f"  The global analysis across {total_n} TC cases reveals that temporal "
            f"attribution patterns are track-regime-dependent.  "
            f"{phase_labels.get(most_disc_phase, most_disc_phase)} is the most "
            f"discriminative temporal window for distinguishing TC track regimes, "
            f"which has direct operational implications for understanding forecast "
            f"uncertainty sources across different synoptic scenarios."
        )

        return "\n".join(lines)

    def summary_table(self) -> str:
        """Return a compact ASCII table for quick inspection before mine()."""
        if not self._records:
            return "(no cases registered)"
        lines = [f"  {'Track Type':22s} {'N':>5} {'Dom Phase':30s}"]
        lines.append("  " + "-" * 57)
        for tt, recs in sorted(self._records.items()):
            doms = [r.get("dominant", "?") for r in recs]
            dom  = max(set(doms), key=doms.count)
            lines.append(f"  {tt:22s} {len(recs):>5}  {dom}")
        return "\n".join(lines)

    def n_cases(self) -> int:
        return sum(len(v) for v in self._records.values())


# ─── Convenience wrappers ─────────────────────────────────────────────────────

def compute_segment_xai(
    model:      object,
    batch_list: list,
    device:     torch.device,
    K:          int = 20,
    storm_idx:  int = 0,
    verbose:    bool = False,
) -> SegmentXAIResult:
    """One-call convenience: instantiate SegmentPerturbationXAI and run compute().

    Example
    -------
    >>> result = compute_segment_xai(model, batch_list, device, K=20, verbose=True)
    """
    sp = SegmentPerturbationXAI(model, device, K=K)
    return sp.compute(batch_list, storm_idx=storm_idx, verbose=verbose)


def compute_adaptive_and_fixed(
    model:      object,
    batch_list: list,
    device:     torch.device,
    K_fixed:    int = 20,
    K_adaptive: int = 10,
    storm_idx:  int = 0,
    verbose:    bool = False,
) -> Tuple[SegmentXAIResult, AdaptiveDiscoveryResult]:
    """Run both fixed C7 and adaptive phase discovery in one call.

    Parameters
    ----------
    K_fixed    : ensemble size for fixed-phase analysis (higher = more accurate)
    K_adaptive : ensemble size for boundary search (lower = faster)

    Returns
    -------
    (SegmentXAIResult, AdaptiveDiscoveryResult)

    Example
    -------
    >>> fixed, adaptive = compute_adaptive_and_fixed(model, batch_list, device)
    >>> print(f"Fixed faithfulness:    {fixed.aopc_faithfulness:.4f}")
    >>> print(f"Adaptive faithfulness: {adaptive.optimal.faithfulness:.4f}")
    >>> print(f"Adaptive gain:         {adaptive.adaptive_gain:+.4f}")
    """
    sp  = SegmentPerturbationXAI(model, device, K=K_fixed)
    apd = AdaptivePhaseDiscovery(model, device, K=K_adaptive)

    fixed_result    = sp.compute(batch_list, storm_idx=storm_idx, verbose=verbose)
    adaptive_result = apd.discover(batch_list, storm_idx=storm_idx, verbose=verbose)

    if verbose:
        print(f"\n[C7] Fixed faithfulness:    {fixed_result.aopc_faithfulness:+.4f}")
        print(f"[C7] Adaptive faithfulness: {adaptive_result.optimal.faithfulness:+.4f}")
        print(f"[C7] Gain from adaptive:    {adaptive_result.adaptive_gain:+.4f}")
        if adaptive_result.matches_physics:
            print("[C7] ✓ Adaptive boundaries CONFIRM the physics-driven A/B/C partition.")
        else:
            tau1 = adaptive_result.optimal.tau1
            tau2 = adaptive_result.optimal.tau2
            print(f"[C7] ✗ Adaptive boundaries DIFFER → τ*=({tau1},{tau2})")

    return fixed_result, adaptive_result
