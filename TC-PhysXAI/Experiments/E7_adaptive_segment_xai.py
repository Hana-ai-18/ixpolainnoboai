"""
TC-PhysXAI — Experiment E7: Adaptive Segment-Level XAI (C7 Extended)
======================================================================
Demonstrates the three novel contributions of C7:

  1. Fixed A/B/C Phase Attribution (XAI4Traj-inspired baseline)
     - Physics-driven 3-phase segmentation: Remote (t-48h to t-24h),
       Synoptic (t-24h to t-12h), Immediate (t-12h to t0)
     - ΔCRPS attribution + AOPC faithfulness (MoRF – LeRF)
     - Shapley phase-interaction effects (synergy vs. redundancy)

  2. Adaptive Phase Discovery  [NEW — C7 contribution]
     - Exhaustive search over C(7,2) = 21 valid 3-phase boundary configs
     - Selects τ* = (τ₁*, τ₂*) maximising AOPC faithfulness
     - Compares optimal τ* vs. physics-fixed A/B/C boundaries
     - Reports adaptive_gain = faithfulness(τ*) − faithfulness(fixed)

  3. Global Attribution Mining  [NEW — C7 contribution]
     - Aggregates results across multiple TC cases
     - Groups by track regime (STRAIGHT_WEST, RECURVING_NE, STALLED, ...)
     - Contrast matrix: which phase most discriminates track types
     - Identifies regime-level dominant steering phase

Output CSV columns:
  typhoon_id, track_type,
  delta_A, delta_B, delta_C,        # fixed phase ΔCRPS (C7 fixed)
  rel_A, rel_B, rel_C,              # normalised attribution
  dominant_phase_fixed,             # A/B/C
  faithfulness_fixed,               # MoRF − LeRF (fixed)
  interaction_AB, interaction_AC, interaction_BC,
  tau1_opt, tau2_opt,               # adaptive τ* boundaries
  faithfulness_opt,                 # AOPC at τ*
  adaptive_gain,                    # faithfulness_opt − faithfulness_fixed
  matches_physics,                  # True if τ* == fixed A/B/C
  most_discriminative_phase,        # from GlobalAttributionMiner
  global_n_cases                    # total cases in miner at time of row
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PhysXAI import (
    SegmentPerturbationXAI,
    AdaptivePhaseDiscovery,
    GlobalAttributionMiner,
    SegmentXAIResult,
    AdaptiveDiscoveryResult,
    compute_adaptive_and_fixed,
)


# ─── Regime inference ────────────────────────────────────────────────────────

def _infer_track_type(batch_list: list) -> str:
    """Regime of the best-track future (storm 0): STRAIGHT_WEST | RECURVING_NE |
    POLEWARD_TRACK | STALLED | MIXED. Degree thresholds are heuristics (PhysXAI/geo.py);
    coordinates are (lon_n, lat_n), 1 unit = 5 degrees."""
    from PhysXAI.geo import classify_track_type
    pred = batch_list[1]  # [T_pred, B, 2]
    if not torch.is_tensor(pred) or pred.ndim < 3:
        return "MIXED"
    return classify_track_type(pred[:, 0, :].detach().cpu().numpy())


# ─── Single-storm runner ──────────────────────────────────────────────────────

def _run_one_storm(
    model,
    batch_list: list,
    device: torch.device,
    K_fixed: int,
    K_adaptive: int,
    storm_idx: int,
    typhoon_id: str,
    verbose: bool = False,
) -> Optional[dict]:
    """Run C7 fixed + adaptive for one storm, return row dict."""
    try:
        fixed_result, adaptive_result = compute_adaptive_and_fixed(
            model=model,
            batch_list=batch_list,
            device=device,
            K_fixed=K_fixed,
            K_adaptive=K_adaptive,
            storm_idx=storm_idx,
            verbose=verbose,
        )
    except Exception as e:
        warnings.warn(f"[E7] {typhoon_id}: compute failed — {e}")
        return None

    # Extract fixed-phase attribution
    delta_A = float(fixed_result.delta_crps.get("A_REMOTE_PRECURSOR",   0.0))
    delta_B = float(fixed_result.delta_crps.get("B_SYNOPTIC_EVOLUTION", 0.0))
    delta_C = float(fixed_result.delta_crps.get("C_IMMEDIATE_STEERING", 0.0))

    rel_A   = float(fixed_result.relative_importance.get("A_REMOTE_PRECURSOR",   0.0))
    rel_B   = float(fixed_result.relative_importance.get("B_SYNOPTIC_EVOLUTION", 0.0))
    rel_C   = float(fixed_result.relative_importance.get("C_IMMEDIATE_STEERING", 0.0))

    dominant_fixed = fixed_result.dominant_phase

    faith_fixed = float(fixed_result.faithfulness)

    # Shapley interactions
    inter = fixed_result.interaction_effects
    inter_AB = float(inter.get(("A_REMOTE_PRECURSOR",   "B_SYNOPTIC_EVOLUTION"), 0.0))
    inter_AC = float(inter.get(("A_REMOTE_PRECURSOR",   "C_IMMEDIATE_STEERING"), 0.0))
    inter_BC = float(inter.get(("B_SYNOPTIC_EVOLUTION", "C_IMMEDIATE_STEERING"), 0.0))

    # Adaptive
    opt = adaptive_result.optimal
    tau1_opt      = int(opt.tau1)
    tau2_opt      = int(opt.tau2)
    faith_opt     = float(opt.faithfulness)
    adaptive_gain = float(adaptive_result.adaptive_gain)
    matches_phys  = bool(adaptive_result.matches_physics)

    return {
        "typhoon_id":          typhoon_id,
        "delta_A":             delta_A,
        "delta_B":             delta_B,
        "delta_C":             delta_C,
        "rel_A":               rel_A,
        "rel_B":               rel_B,
        "rel_C":               rel_C,
        "dominant_phase_fixed":  dominant_fixed,
        "faithfulness_fixed":    faith_fixed,
        "interaction_AB":      inter_AB,
        "interaction_AC":      inter_AC,
        "interaction_BC":      inter_BC,
        "tau1_opt":            tau1_opt,
        "tau2_opt":            tau2_opt,
        "faithfulness_opt":    faith_opt,
        "adaptive_gain":       adaptive_gain,
        "matches_physics":     matches_phys,
        # attached after mining
        "_fixed_result":       fixed_result,
        "_adaptive_result":    adaptive_result,
    }


# ─── Main experiment runner ───────────────────────────────────────────────────

def run_e7(
    model,
    batches: list,
    device: torch.device,
    out_dir: str,
    K_fixed: int = 20,
    K_adaptive: int = 10,
    n_storms: int = 20,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run E7 on up to n_storms storms.

    Parameters
    ----------
    model       : trained TCFlowMatching (eval mode)
    batches     : list of batch_list items from DataLoader
    device      : CUDA or CPU device
    out_dir     : output directory for CSV + text report
    K_fixed     : ensemble size for fixed A/B/C phase attribution
    K_adaptive  : ensemble size for each of the 21 adaptive configs
    n_storms    : max number of storms to process
    verbose     : print per-storm progress

    Returns
    -------
    pd.DataFrame with one row per storm
    """
    os.makedirs(out_dir, exist_ok=True)
    miner = GlobalAttributionMiner()
    rows  = []

    batches_to_run = batches[:n_storms]
    print(f"[E7] Running Adaptive Segment XAI on {len(batches_to_run)} storms …")
    print(f"     K_fixed={K_fixed}, K_adaptive={K_adaptive} × 21 configs")

    t_start = time.time()
    for i, bl in enumerate(batches_to_run):
        typhoon_id = f"storm_{i:03d}"

        # Try to extract typhoon ID from batch metadata
        try:
            if isinstance(bl, (list, tuple)) and len(bl) > 12:
                meta = bl[12]
                if isinstance(meta, (list, tuple)) and len(meta) > 0:
                    typhoon_id = str(meta[0])
        except Exception:
            pass

        track_type = _infer_track_type(bl)

        if verbose:
            print(f"\n[E7] Storm {i+1}/{len(batches_to_run)}: {typhoon_id} "
                  f"(regime={track_type})")

        row = _run_one_storm(
            model=model,
            batch_list=bl,
            device=device,
            K_fixed=K_fixed,
            K_adaptive=K_adaptive,
            storm_idx=0,
            typhoon_id=typhoon_id,
            verbose=verbose,
        )

        if row is None:
            continue

        # Add to global miner
        miner.add(
            result=row.pop("_fixed_result"),
            track_type=track_type,
            typhoon_id=typhoon_id,
            adaptive=row.pop("_adaptive_result"),
        )

        row["track_type"] = track_type
        rows.append(row)

        elapsed = time.time() - t_start
        rate    = elapsed / (i + 1)
        eta     = rate * (len(batches_to_run) - i - 1)
        print(f"  [{i+1:3d}/{len(batches_to_run)}] {typhoon_id:20s} "
              f"regime={track_type:15s}  "
              f"faith_fixed={row['faithfulness_fixed']:.3f}  "
              f"faith_opt={row['faithfulness_opt']:.3f}  "
              f"gain={row['adaptive_gain']:+.3f}  "
              f"τ*=({row['tau1_opt']},{row['tau2_opt']})  "
              f"ETA {eta/60:.1f}min")

    if not rows:
        print("[E7] No valid results.")
        return pd.DataFrame()

    # ── Mine global patterns ──────────────────────────────────────────────────
    print("\n[E7] Mining global attribution patterns …")
    global_result = miner.mine()
    mdp = global_result.most_discriminative_phase

    # Attach global stats to every row
    df = pd.DataFrame(rows)
    df["most_discriminative_phase"] = mdp
    df["global_n_cases"]            = global_result.n_total_cases

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "E7_adaptive_segment_xai.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[E7] Results saved → {csv_path}")

    # Text report
    report_path = os.path.join(out_dir, "E7_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TC-PhysXAI — E7: Adaptive Segment-Level XAI Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total storms processed : {len(df)}\n")
        f.write(f"Track regimes          : {df['track_type'].value_counts().to_dict()}\n\n")

        # Fixed phase summary
        f.write("[FIXED A/B/C PHASE ATTRIBUTION]\n")
        phase_dominance = df["dominant_phase_fixed"].value_counts()
        for ph, cnt in phase_dominance.items():
            f.write(f"  Dominant = {ph}: {cnt} storms ({cnt/len(df):.0%})\n")
        f.write(f"  Mean faithfulness (fixed) : {df['faithfulness_fixed'].mean():.4f}\n\n")

        # Adaptive summary
        f.write("[ADAPTIVE PHASE DISCOVERY τ*]\n")
        f.write(f"  Mean τ₁* : {df['tau1_opt'].mean():.2f}  "
                f"(physics fixed τ₁=4)\n")
        f.write(f"  Mean τ₂* : {df['tau2_opt'].mean():.2f}  "
                f"(physics fixed τ₂=6)\n")
        f.write(f"  Mean faithfulness (adaptive) : {df['faithfulness_opt'].mean():.4f}\n")
        f.write(f"  Mean adaptive gain           : {df['adaptive_gain'].mean():+.4f}\n")
        match_rate = df["matches_physics"].mean()
        f.write(f"  τ* matches physics A/B/C     : {match_rate:.0%} of storms\n\n")

        # Global mining report
        f.write("[GLOBAL ATTRIBUTION MINING — REGIME PATTERNS]\n")
        f.write(global_result.interpretation + "\n\n")

        # Global miner summary table
        f.write(miner.summary_table() + "\n")

        f.write("=" * 70 + "\n")
        f.write("END OF E7 REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[E7] Report saved → {report_path}")

    # Print summary table inline
    print("\n" + miner.summary_table())
    print(f"\n[E7] Most discriminative phase across regimes: {mdp}")
    print(f"[E7] Adaptive gain (mean): {df['adaptive_gain'].mean():+.4f}")
    print(f"[E7] Physics-confirmed τ*: {df['matches_physics'].mean():.0%} of storms")

    return df


# ─── Plot helper (optional) ───────────────────────────────────────────────────

def plot_e7(
    df: pd.DataFrame,
    out_dir: str,
) -> None:
    """Generate E7 diagnostic plots (requires matplotlib).

    Produces:
      - phase_attribution_by_regime.png : grouped bar chart Δ(A/B/C) per track type
      - adaptive_gain_histogram.png     : distribution of faithfulness gain
      - tau_star_heatmap.png            : heatmap of τ* frequency in (τ₁, τ₂) space
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import numpy as np
    except ImportError:
        print("[E7] matplotlib not available — skipping plots.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1. Phase attribution by regime
    track_types = df["track_type"].unique()
    phases = ["A", "B", "C"]
    x = np.arange(len(track_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, ph in enumerate(phases):
        means = [df[df["track_type"] == tt][f"rel_{ph}"].mean() for tt in track_types]
        ax.bar(x + j * width, means, width=width, label=f"Phase {ph}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([tt.replace("_", "\n") for tt in track_types], fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Relative Attribution")
    ax.set_title("C7 — Phase Attribution by Track Regime")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "E7_phase_attribution_by_regime.png"), dpi=150)
    plt.close(fig)

    # 2. Adaptive gain histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["adaptive_gain"], bins=20, edgecolor="white", color="#4C9BE8")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No gain")
    ax.set_xlabel("Adaptive Gain (faithfulness_opt − faithfulness_fixed)")
    ax.set_ylabel("Count")
    ax.set_title("C7 — Distribution of Adaptive Phase Discovery Gain")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "E7_adaptive_gain_histogram.png"), dpi=150)
    plt.close(fig)

    # 3. τ* heatmap  (τ₁ in [1,6], τ₂ in [2,7])
    grid = np.zeros((8, 8))
    for _, row in df.iterrows():
        t1, t2 = int(row["tau1_opt"]), int(row["tau2_opt"])
        if 0 <= t1 < 8 and 0 <= t2 < 8:
            grid[t1, t2] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="Blues", origin="lower")
    plt.colorbar(im, ax=ax, label="Count of τ*")
    ax.set_xlabel("τ₂ (start of γ phase)")
    ax.set_ylabel("τ₁ (start of β phase)")
    ax.set_title("C7 — Optimal Partition Frequency (τ₁, τ₂)")
    # Mark physics-fixed boundary
    ax.plot(6, 4, "r*", markersize=14, label="Physics (4,6)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "E7_tau_star_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"[E7] Plots saved to {out_dir}/")
