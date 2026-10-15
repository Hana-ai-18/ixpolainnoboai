"""
E3: Rapid Intensification and SST Attribution
==============================================
Study how TC-PhysXAI attributes track prediction to SST during
rapid intensification (RI) events.

Physical hypothesis:
  During RI (ΔVmax ≥ 30 kt / 24h), the TC deepens rapidly, modifying
  the effective steering layer depth and increasing intensity-track coupling.
  SST attribution should be ELEVATED during RI because:
  1. Warm SST sustains convective heating → deeper vortex → deeper steering
  2. SST gradients cause convective asymmetry → track deflection toward warm side
  3. TC-XFlow speed calibration uses storm intensity (obs_Me) which depends on SST

Experiment design:
  1. Identify RI periods in test set (ΔMinPressure ≤ -15 hPa / 24h or ΔMaxWind ≥ 30 kt)
  2. Compare SST attribution vs non-RI cases
  3. Correlate SST attribution with subsequent intensification rate
  4. Show physical object detection: SST front detection frequency

Output:
  results/E3_RI_sst_attribution.csv
  results/E3_sst_intensification_correlation.png
"""

from __future__ import annotations

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

from PhysXAI import SpatialAttributor, TemporalAttributor


def detect_rapid_intensification(
    obs_Me: torch.Tensor,   # [T_obs, B, 4] (Lat, Lon, MinPressure, MaxWind) normalised
    threshold_pressure: float = -0.4,   # normalised units ~ -15 hPa physical
) -> torch.Tensor:
    """Detect RI: pressure drop > threshold over 4 steps (24h).

    Returns [B] bool tensor — True if RI is detected.
    """
    if obs_Me is None or not torch.is_tensor(obs_Me):
        return torch.zeros(1, dtype=torch.bool)

    T_obs, B, _ = obs_Me.shape
    pressure = obs_Me[:, :, 2]  # [T_obs, B]

    if T_obs < 4:
        return torch.zeros(B, dtype=torch.bool)

    # Minimum pressure drop over any 4-step window
    min_change = torch.zeros(B)
    for t in range(T_obs - 4):
        dp = pressure[t + 4] - pressure[t]   # negative = intensification
        min_change = torch.minimum(min_change, dp.cpu())

    return min_change < threshold_pressure


def run_e3(
    model,
    test_batches: List[list],
    device: torch.device,
    output_dir: str = "results",
    K: int = 30,
    n_storms: int = 80,
) -> pd.DataFrame:
    """Run E3: RI and SST attribution study."""
    os.makedirs(output_dir, exist_ok=True)

    spatial  = SpatialAttributor(model, device)
    temporal = TemporalAttributor(model, device, n_ensemble=K)

    rows = []

    for s_idx, bl in enumerate(test_batches[:n_storms]):
        bl_dev = [t.to(device) if torch.is_tensor(t) else t for t in bl]

        # Check RI
        obs_Me = bl[7] if len(bl) > 7 else None
        ri_flags = detect_rapid_intensification(obs_Me)
        is_ri    = bool(ri_flags[0]) if len(ri_flags) > 0 else False

        print(f"[E3] Storm {s_idx+1}: {'RI' if is_ri else 'NON-RI'} ...")

        try:
            s_result = spatial.compute(bl_dev, detect_objects=True)
            level_imp = s_result.get("level_importance", {})

            # Check if SST front was detected
            objs = s_result.get("physical_objects", [])
            has_sst_front = any(o.type == "sst_front" for o in objs)
            sst_front_weight = sum(o.attribution_weight for o in objs if o.type == "sst_front")

            # Intensity variables (raw, before normalisation estimate)
            mean_pressure_norm = 0.0
            if obs_Me is not None and torch.is_tensor(obs_Me):
                mean_pressure_norm = float(obs_Me[:, 0, 2].mean().item())
            mean_wind_norm = 0.0
            if obs_Me is not None and torch.is_tensor(obs_Me):
                mean_wind_norm = float(obs_Me[:, 0, 3].mean().item())

            rows.append({
                "storm_idx":      s_idx,
                "is_ri":          is_ri,
                "attr_200hPa":    level_imp.get("200hPa", 0),
                "attr_500hPa":    level_imp.get("500hPa", 0),
                "attr_850hPa":    level_imp.get("850hPa", 0),
                "attr_SST":       level_imp.get("SST", 0),
                "has_sst_front":  has_sst_front,
                "sst_front_w":    sst_front_weight,
                "n_objects":      len(objs),
                "mean_pressure":  mean_pressure_norm,
                "mean_wind":      mean_wind_norm,
                "steering_regime": s_result.get("steering_regime", ""),
            })

        except Exception as e:
            print(f"  [E3] Storm {s_idx} failed: {e}")
            continue

    df = pd.DataFrame(rows)

    if len(df) > 0:
        summary = df.groupby("is_ri")[
            ["attr_SST", "attr_200hPa", "attr_500hPa", "has_sst_front", "sst_front_w"]
        ].agg(["mean", "std", "count"])

        print("\n[E3] RI vs Non-RI SST Attribution:")
        print(summary.to_string())

        df.to_csv(os.path.join(output_dir, "E3_RI_sst_attribution.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, "E3_summary.csv"))

        _plot_e3(df, output_dir)

    return df


def _plot_e3(df: pd.DataFrame, output_dir: str) -> None:
    """Plot RI vs non-RI SST attribution and correlation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("E3: Rapid Intensification & SST Attribution (TC-PhysXAI)", fontsize=12)

        ri_colors = {True: "#e74c3c", False: "#2980b9"}
        ri_labels = {True: "Rapid Intensification", False: "Non-RI"}

        # SST attribution distribution
        ax = axes[0]
        for ri_flag in [True, False]:
            sub = df[df["is_ri"] == ri_flag]
            if len(sub) == 0:
                continue
            ax.hist(sub["attr_SST"], bins=15, color=ri_colors[ri_flag],
                    label=ri_labels[ri_flag], alpha=0.6, density=True)
        ax.set_xlabel("SST Attribution Weight")
        ax.set_ylabel("Density")
        ax.set_title("SST Attribution: RI vs Non-RI")
        ax.legend()
        ax.grid(alpha=0.3)

        # Level attribution comparison
        ax = axes[1]
        levels = ["attr_200hPa", "attr_500hPa", "attr_850hPa", "attr_SST"]
        level_labels = ["200hPa", "500hPa", "850hPa", "SST"]
        x = np.arange(len(levels))
        width = 0.35
        for i, (ri_flag, color) in enumerate(ri_colors.items()):
            sub = df[df["is_ri"] == ri_flag]
            if len(sub) == 0:
                continue
            vals = [sub[c].mean() for c in levels]
            errs = [sub[c].std() for c in levels]
            ax.bar(x + i*width, vals, width, color=color, alpha=0.8,
                   label=ri_labels[ri_flag], yerr=errs, capsize=4)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(level_labels)
        ax.set_ylabel("Attribution Weight")
        ax.set_title("Attribution by Level")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # SST attribution vs wind intensity correlation
        ax = axes[2]
        for ri_flag, color in ri_colors.items():
            sub = df[df["is_ri"] == ri_flag]
            if len(sub) == 0:
                continue
            ax.scatter(sub["mean_wind"], sub["attr_SST"],
                       color=color, alpha=0.5, s=40, label=ri_labels[ri_flag])
        ax.set_xlabel("Mean Wind Speed (norm.)")
        ax.set_ylabel("SST Attribution Weight")
        ax.set_title("Intensity vs SST Attribution")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "E3_sst_intensification.png"), dpi=150)
        plt.close()
        print(f"[E3] Plot saved.")
    except Exception as e:
        print(f"[E3] Plotting skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir",   default="results")
    parser.add_argument("--n_storms",     type=int, default=80)
    parser.add_argument("--K",            type=int, default=20)
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
    run_e3(model, batches, device, args.output_dir, K=args.K, n_storms=args.n_storms)
    print("[E3] Done.")
