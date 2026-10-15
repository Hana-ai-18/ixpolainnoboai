"""
E2: Recurvature Attribution Analysis
=====================================
Study how TC-PhysXAI identifies the physical precursors of typhoon recurvature.

Experiment design:
  1. Select typhoons that eventually recurve (subset from TC-OFM test set)
  2. Run all TC-PhysXAI modules at lead times: t-48h, t-24h, t-12h, t-0h
  3. Track evolution of:
     - 200 hPa trough attribution (should increase as recurvature approaches)
     - 500 hPa ridge attribution (should decrease)
     - V-wind importance (should increase — northward steering)
     - Ensemble scenario diversity (entropy should peak at bifurcation)
  4. Compare against straight-track cases (control group)

Physical hypothesis (from TC track theory):
  - 24-48h before recurvature: 200 hPa trough begins to dominate attribution
  - 12-24h before recurvature: V500 attribution increases, ridge weakens
  - At recurvature: maximum ensemble scenario bifurcation (peak uncertainty)
  - After recurvature: poleward V-wind dominates completely

Output:
  results/E2_recurvature_attribution.csv
  results/E2_trough_evolution.png
  results/E2_scenario_entropy.png
"""

from __future__ import annotations

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

from PhysXAI import TemporalAttributor, SpatialAttributor, AnalogueXAI
from PhysXAI.utils import denorm_latlon


def classify_recurvature(pred_traj: torch.Tensor, threshold_dlat: float = 3.0) -> bool:
    """Does the (ground-truth) track of storm 0 recurve within the window?

    pred_traj: [T_pred, B, 2] ABSOLUTE normalised (lon_n, lat_n); 1 unit = 5 degrees.
    threshold_dlat is in DEGREES of poleward motion over the 2nd half of the window.
    """
    from PhysXAI.geo import classify_recurvature as _cr
    return _cr(pred_traj[:, 0, :].detach().cpu().numpy(), threshold_dlat)


def run_e2(
    model,
    test_batches: List[list],
    device: torch.device,
    output_dir: str = "results",
    K: int = 30,
    n_storms: int = 50,
) -> pd.DataFrame:
    """Run E2: Recurvature attribution analysis.

    Returns DataFrame with per-storm, per-lead-time attribution statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    temporal   = TemporalAttributor(model, device, n_ensemble=K)
    spatial    = SpatialAttributor(model, device)
    analogue   = AnalogueXAI(model, device, n_clusters=3)

    rows = []
    n_recurve = 0
    n_straight = 0

    for s_idx, bl in enumerate(test_batches[:n_storms]):
        bl_dev = [t.to(device) if torch.is_tensor(t) else t for t in bl]

        # Determine if this storm recurves
        pred_traj = bl_dev[1] if torch.is_tensor(bl_dev[1]) else None
        if pred_traj is None:
            continue

        is_recurving = classify_recurvature(pred_traj)
        track_type   = "RECURVING" if is_recurving else "STRAIGHT"
        if is_recurving:
            n_recurve += 1
        else:
            n_straight += 1

        print(f"[E2] Storm {s_idx+1}: {track_type} ...")

        try:
            # Temporal attribution
            t_result = temporal.compute(bl_dev, n_ensemble=K)
            attr_scores = t_result["mean_attribution_per_window"]  # [obs_len]

            # Spatial attribution
            s_result = spatial.compute(bl_dev, detect_objects=True)
            level_imp = s_result.get("level_importance", {})

            # Analogue (ensemble scenario)
            a_result = analogue.compute(bl_dev, K=K, storm_idx=0)

            # Extract key quantities
            recent_attr = float(np.mean(attr_scores[-2:]))   # t-0h, t-6h
            remote_attr = float(np.mean(attr_scores[:2]))    # t-48h, t-42h

            rows.append({
                "storm_idx":   s_idx,
                "track_type":  track_type,
                "attr_200hPa": level_imp.get("200hPa", 0),
                "attr_500hPa": level_imp.get("500hPa", 0),
                "attr_850hPa": level_imp.get("850hPa", 0),
                "attr_925hPa": level_imp.get("925hPa", 0),
                "attr_SST":    level_imp.get("SST", 0),
                "temporal_recent": recent_attr,
                "temporal_remote": remote_attr,
                "uncertainty_entropy": a_result.get("uncertainty_entropy", 0),
                "n_scenarios": a_result.get("n_scenarios", 1),
                "dominant_scenario": a_result.get("dominant_scenario", "UNKNOWN"),
                "most_important_window": t_result.get("most_important_window", 0),
                "steering_regime": s_result.get("steering_regime", ""),
            })

        except Exception as e:
            print(f"  [E2] Storm {s_idx} failed: {e}")
            continue

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Statistical comparison: recurving vs straight
        summary = df.groupby("track_type")[
            ["attr_200hPa", "attr_500hPa", "attr_850hPa",
             "temporal_remote", "uncertainty_entropy"]
        ].agg(["mean", "std"])

        print("\n[E2] Recurving vs Straight comparison:")
        print(summary.to_string())
        print(f"\n  n_recurving={n_recurve}, n_straight={n_straight}")

        df.to_csv(os.path.join(output_dir, "E2_recurvature_attribution.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, "E2_summary.csv"))

        _plot_e2(df, output_dir)

    return df


def _plot_e2(df: pd.DataFrame, output_dir: str) -> None:
    """Plot recurvature vs straight attribution profiles."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("E2: Recurvature vs Straight Track Attribution (TC-PhysXAI)", fontsize=12)

        colors = {"RECURVING": "#e74c3c", "STRAIGHT": "#2980b9"}
        level_cols = ["attr_200hPa", "attr_500hPa", "attr_850hPa", "attr_SST"]
        level_labels = ["200hPa", "500hPa", "850hPa", "SST"]

        # Left: Level attribution profile
        ax = axes[0]
        for ttype, color in colors.items():
            sub = df[df["track_type"] == ttype]
            if len(sub) == 0:
                continue
            vals = [sub[c].mean() for c in level_cols]
            errs = [sub[c].std() for c in level_cols]
            ax.errorbar(range(len(vals)), vals, yerr=errs,
                        label=ttype, color=color, marker="o", linewidth=2)
        ax.set_xticks(range(len(level_labels)))
        ax.set_xticklabels(level_labels)
        ax.set_ylabel("Attribution Weight")
        ax.set_title("Level Attribution Profile")
        ax.legend()
        ax.grid(alpha=0.3)

        # Middle: Remote vs recent temporal attribution
        ax = axes[1]
        for ttype, color in colors.items():
            sub = df[df["track_type"] == ttype]
            if len(sub) == 0:
                continue
            ax.scatter(sub["temporal_remote"], sub["temporal_recent"],
                       label=ttype, color=color, alpha=0.6, s=50)
        ax.set_xlabel("Remote (t-48h) Attribution")
        ax.set_ylabel("Recent (t-0h) Attribution")
        ax.set_title("Temporal Attribution Pattern")
        ax.legend()
        ax.grid(alpha=0.3)

        # Right: Ensemble uncertainty entropy
        ax = axes[2]
        for ttype, color in colors.items():
            sub = df[df["track_type"] == ttype]
            if len(sub) == 0:
                continue
            ax.hist(sub["uncertainty_entropy"], bins=10, color=color,
                    alpha=0.6, label=ttype, density=True)
        ax.set_xlabel("Scenario Entropy")
        ax.set_ylabel("Density")
        ax.set_title("Ensemble Scenario Uncertainty")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "E2_recurvature_attribution.png"), dpi=150)
        plt.close()
        print(f"[E2] Plot saved.")
    except Exception as e:
        print(f"[E2] Plotting skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir",   default="results")
    parser.add_argument("--n_storms",     type=int, default=50)
    parser.add_argument("--K",            type=int, default=30)
    parser.add_argument("--gpu",          type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from PhysXAI.model_io import load_tcxflow
    model = load_tcxflow(args.checkpoint, device)

    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    from torch.utils.data import DataLoader
    dataset = TrajectoryDataset(args.dataset_root, split="test")
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)

    batches = [b for _, b in zip(range(args.n_storms), loader)]
    run_e2(model, batches, device, args.output_dir, K=args.K, n_storms=args.n_storms)
    print("[E2] Done.")
