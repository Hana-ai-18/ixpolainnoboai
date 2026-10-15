"""
E6: Counterfactual Scenario Analysis
======================================
Generate and validate FM-native counterfactual explanations (C6) for:
  1. RECURVATURE scenarios: what ridge weakening triggers recurvature?
  2. STRAIGHT scenarios: what ridge strengthening prevents recurvature?

Evaluation metrics:
  - Context change norm ‖Δc‖/‖c‖ (smaller = more likely physically)
  - Channel specificity: fraction of Δc concentrated in ≤3 channels
  - Physical plausibility: does the dominant Δc channel match the
    expected synoptic change? (GPH_500 for ridge, U_200 for trough)
  - Track delta: km shift between original and counterfactual

Output:
  results/E6_counterfactual_cases.csv
  results/E6_cf_tracks.png
  results/E6_channel_breakdown.png
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

from PhysXAI import FMCounterfactual
from PhysXAI.meteorological_interp import MeteorologyInterpreter


_CH_NAMES = [
    "GPH_200", "GPH_500", "GPH_850", "GPH_925",
    "U_200",   "U_500",   "U_850",   "U_925",
    "V_200",   "V_500",   "V_850",   "V_925",
    "SST",
]


def run_e6(
    model,
    test_batches: List[list],
    device: torch.device,
    output_dir: str = "results",
    n_storms: int = 20,
    cf_types: List[str] = None,
    ddim_steps: int = 20,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Run E6: Counterfactual scenario analysis."""
    os.makedirs(output_dir, exist_ok=True)

    if cf_types is None:
        cf_types = ["RECURVATURE", "STRAIGHT"]

    cf_model = FMCounterfactual(
        model, device,
        ddim_steps=ddim_steps,
        lr=0.01,
        max_iter=max_iter,
        cf_epsilon=0.25,
        reg_lambda=0.05,
    )
    interp = MeteorologyInterpreter()

    rows = []
    cf_track_data: List[Dict] = []

    for s_idx, bl in enumerate(test_batches[:n_storms]):
        print(f"[E6] Storm {s_idx+1}/{min(n_storms, len(test_batches))} ...")
        bl_dev = [t.to(device) if torch.is_tensor(t) else t for t in bl]

        for cf_type in cf_types:
            try:
                result = cf_model.compute(
                    bl_dev,
                    cf_type=cf_type,
                    storm_idx=0,
                    verbose=False,
                )

                # Channel concentration: top-3 fraction
                ch_delta_sorted = np.sort(result.channel_delta)[::-1]
                top3_frac = ch_delta_sorted[:3].sum() if len(ch_delta_sorted) >= 3 else 1.0
                dominant_ch = _CH_NAMES[int(np.argmax(result.channel_delta))]

                # Physical plausibility check
                phys_plausible = False
                if cf_type == "RECURVATURE":
                    # Should require change in GPH_500 (ridge) or V_200 (trough)
                    phys_plausible = bool(
                        dominant_ch in ["GPH_200", "GPH_500", "V_200", "V_500"]
                    )
                elif cf_type == "STRAIGHT":
                    # Should require strengthening the ridge
                    phys_plausible = bool(
                        dominant_ch in ["GPH_500", "U_500", "GPH_200"]
                    )

                row = {
                    "storm_idx":     s_idx,
                    "cf_type":       cf_type,
                    "context_norm":  result.context_change_norm,
                    "track_delta_km":result.track_delta_km,
                    "dominant_ch":   dominant_ch,
                    "top3_frac":     float(top3_frac),
                    "phys_plausible": phys_plausible,
                }
                # Add per-channel delta
                for ch_i, ch_name in enumerate(_CH_NAMES):
                    row[f"delta_{ch_name}"] = float(result.channel_delta[ch_i])

                rows.append(row)

                # Save track data for plotting
                cf_track_data.append({
                    "storm_idx":    s_idx,
                    "cf_type":      cf_type,
                    "original":     result.original_track,
                    "counterfactual": result.counterfactual_track,
                    "target":       result.target_track,
                })

            except Exception as e:
                print(f"  [E6] Storm {s_idx}, {cf_type} failed: {e}")
                continue

    df = pd.DataFrame(rows)

    if len(df) > 0:
        summary = df.groupby("cf_type")[
            ["context_norm", "track_delta_km", "top3_frac", "phys_plausible"]
        ].agg(["mean", "std"])

        print("\n[E6] Counterfactual summary:")
        print(summary.to_string())

        df.to_csv(os.path.join(output_dir, "E6_counterfactual_cases.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, "E6_summary.csv"))

        _plot_e6(df, cf_track_data, output_dir)

    return df


def _plot_e6(df: pd.DataFrame, track_data: List[Dict], output_dir: str) -> None:
    """Plot counterfactual tracks and channel breakdown."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("E6: FM Counterfactual Scenarios (TC-PhysXAI)", fontsize=12)

        cf_colors = {"RECURVATURE": "#e74c3c", "STRAIGHT": "#2980b9"}

        # Left: Sample counterfactual tracks
        ax = axes[0]
        plotted = 0
        for td in track_data[:6]:
            orig = td["original"]
            cf   = td["counterfactual"]
            cf_type = td["cf_type"]
            color   = cf_colors.get(cf_type, "grey")

            # Denorm to degrees
            orig_lat = orig[:, 0] * 12.8
            orig_lon = orig[:, 1] * 22.68 + 130  # rough Pacific offset
            cf_lat   = cf[:, 0] * 12.8
            cf_lon   = cf[:, 1] * 22.68 + 130

            if plotted == 0:
                ax.plot(orig_lon, orig_lat, "k-", linewidth=2, label="Original", zorder=5)
            ax.plot(cf_lon, cf_lat, "--", color=color, linewidth=1.5,
                    label=cf_type if plotted < 2 else "_", alpha=0.7)
            plotted += 1

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title("Counterfactual Track Examples")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Middle: Context norm distribution
        ax = axes[1]
        for cf_type, color in cf_colors.items():
            sub = df[df["cf_type"] == cf_type]
            if len(sub) == 0:
                continue
            ax.hist(sub["context_norm"], bins=10, color=color, alpha=0.6,
                    label=cf_type, density=True)
        ax.set_xlabel("Context Change Norm ‖Δc‖/‖c‖")
        ax.set_ylabel("Density")
        ax.set_title("Minimum Required Context Change")
        ax.legend()
        ax.grid(alpha=0.3)

        # Right: Mean channel breakdown
        ax = axes[2]
        ch_cols  = [f"delta_{c}" for c in _CH_NAMES]
        x = np.arange(len(_CH_NAMES))
        width = 0.35
        for i, (cf_type, color) in enumerate(cf_colors.items()):
            sub = df[df["cf_type"] == cf_type]
            if len(sub) == 0 or not all(c in sub.columns for c in ch_cols):
                continue
            vals = [sub[c].mean() for c in ch_cols]
            ax.bar(x + i*width, vals, width, color=color, alpha=0.75, label=cf_type)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(_CH_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean Channel Attribution for Δc")
        ax.set_title("ERA5 Channel Breakdown of Counterfactual")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "E6_counterfactual.png"), dpi=150)
        plt.close()
        print("[E6] Plot saved.")
    except Exception as e:
        print(f"[E6] Plotting skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir",   default="results")
    parser.add_argument("--n_storms",     type=int, default=20)
    parser.add_argument("--max_iter",     type=int, default=100)
    parser.add_argument("--gpu",          type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from Model.flow_matching_model import TCFlowMatching
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = TCFlowMatching(**ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    from torch.utils.data import DataLoader
    dataset = TrajectoryDataset(args.dataset_root, split="test")
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)

    batches = [b for _, b in zip(range(args.n_storms), loader)]
    run_e6(model, batches, device, args.output_dir, n_storms=args.n_storms, max_iter=args.max_iter)
    print("[E6] Done.")
