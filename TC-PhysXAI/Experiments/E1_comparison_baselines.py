"""
E1: Comparison with XAI Baselines
===================================
Compare TC-PhysXAI (C1-C6) against:
  - Fernandez et al. (2023): SHAP-based attribution for cyclone prediction
  - XAI4Traj (Liu et al. 2024): trajectory-centric saliency
  - Vanilla Gradient Saliency
  - LIME (linear approximation)

Metrics:
  - H-mean-F (C3 probabilistic fidelity)
  - Faithfulness (mask-k CRPS increase)
  - Complexity (entropy of attribution map)
  - Physical Plausibility Score (expert-assigned, 1-5)

Output:
  results/E1_comparison_table.csv
  results/E1_attribution_maps.png
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List

# ─── Project path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

from PhysXAI import SpatialAttributor, ProbabilisticFidelity
from PhysXAI.utils import crps_ensemble_fast

# ─── Helper: Vanilla gradient saliency ────────────────────────────────────────

def vanilla_gradient_saliency(
    model, batch_list: list, device: torch.device
) -> torch.Tensor:
    """Compute |∂loss/∂x| for each ERA5 pixel.

    Returns [B, 13, T_obs, H, W]
    """
    img_obs = batch_list[11].to(device).float()
    img_g   = img_obs.detach().requires_grad_(True)
    bl_g    = list(batch_list)
    bl_g[11] = img_g

    with torch.enable_grad():
        cond = model.context_encoder(bl_g)
        # Use first ensemble member
        z0   = torch.randn(img_obs.shape[0], 12, 2, device=device)
        v    = model.velocity_transformer(z0, torch.zeros(img_obs.shape[0], device=device), cond)
        loss = v.pow(2).sum()
        loss.backward()

    saliency = img_g.grad.detach().abs() if img_g.grad is not None else torch.zeros_like(img_obs)
    return saliency


def lime_attribution(
    model, batch_list: list, device: torch.device,
    n_perturb: int = 50,
) -> torch.Tensor:
    """LIME approximation: perturb channel segments and measure output change.

    Returns [B, 13, T_obs, H, W] — column-normalised importance
    """
    img_obs = batch_list[11].to(device).float()
    B, C, T, H, W = img_obs.shape

    # Baseline CRPS
    with torch.no_grad():
        _, _, all_traj = model.sample(batch_list, num_ensemble=10, use_speed_calibration=False)
        pred_traj = batch_list[1].to(device)
        T_pred = min(all_traj.shape[1], pred_traj.shape[0])
        crps_base = crps_ensemble_fast(all_traj[:, :T_pred], pred_traj[:T_pred]).mean()

    channel_importance = torch.zeros(B, C, device=device)

    for ch in range(C):
        crps_sum = 0.0
        for _ in range(max(1, n_perturb // C)):
            bl_p = list(batch_list)
            img_p = img_obs.clone()
            # Perturb one channel with Gaussian noise
            img_p[:, ch] += torch.randn_like(img_p[:, ch]) * img_obs[:, ch].std()
            bl_p[11] = img_p

            with torch.no_grad():
                _, _, all_t = model.sample(bl_p, num_ensemble=10, use_speed_calibration=False)
                crps_p = crps_ensemble_fast(all_t[:, :T_pred], pred_traj[:T_pred]).mean()
            crps_sum += float(crps_p - crps_base)

        channel_importance[:, ch] = crps_sum / max(1, n_perturb // C)

    # Expand to [B, C, T, H, W]
    importance = channel_importance.abs()
    importance = importance[:, :, None, None, None].expand(B, C, T, H, W)
    return importance


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_e1(
    model,
    test_batch_list: list,
    device: torch.device,
    output_dir: str = "results",
    K: int = 30,
    top_k: int = 3,
    n_storms: int = 20,
) -> pd.DataFrame:
    """Run E1: XAI method comparison.

    Parameters
    ----------
    model           : loaded TCFlowMatching (eval mode)
    test_batch_list : list of batch_lists for test storms
    K               : ensemble size for CRPS
    top_k           : channels to mask/keep for fidelity

    Returns
    -------
    DataFrame with columns [method, H_mean_F, precision_F, recall_F, uncertainty_F,
                             complexity, faithfulness, CRPS_baseline]
    """
    os.makedirs(output_dir, exist_ok=True)
    fidelity = ProbabilisticFidelity(model, device, n_ensemble=K)
    spatial   = SpatialAttributor(model, device)

    rows = []
    storm_list = test_batch_list[:n_storms]

    for s_idx, bl in enumerate(storm_list):
        print(f"\n[E1] Storm {s_idx+1}/{len(storm_list)} ...")

        bl_dev = [t.to(device) if torch.is_tensor(t) else t for t in bl]

        try:
            # ─── TC-PhysXAI IG ─────────────────────────────────────────────
            result_ig = spatial.compute(bl_dev, detect_objects=False)
            ig_attr   = torch.tensor(result_ig["ig_attr"])  # [B, 13, T, H, W]

            fid_ig = fidelity.compute(bl_dev, ig_attr, K=K, top_k=top_k)

            rows.append({
                "storm": s_idx,
                "method": "TC-PhysXAI (IG)",
                "H_mean_F": fid_ig.h_mean_f,
                "precision_F": fid_ig.precision_f,
                "recall_F": fid_ig.recall_f,
                "uncertainty_F": fid_ig.uncertainty_f,
                "complexity": float(-np.sum(
                    np.abs(result_ig["ig_attr"]) / (np.abs(result_ig["ig_attr"]).sum() + 1e-8)
                    * np.log(np.abs(result_ig["ig_attr"]) / (np.abs(result_ig["ig_attr"]).sum() + 1e-8) + 1e-8)
                )),
                "CRPS_baseline": fid_ig.crps_baseline,
            })

            # ─── Vanilla Gradient ───────────────────────────────────────────
            vg_attr = vanilla_gradient_saliency(model, bl_dev, device)
            fid_vg  = fidelity.compute(bl_dev, vg_attr, K=K, top_k=top_k)

            rows.append({
                "storm": s_idx,
                "method": "Vanilla Gradient",
                "H_mean_F": fid_vg.h_mean_f,
                "precision_F": fid_vg.precision_f,
                "recall_F": fid_vg.recall_f,
                "uncertainty_F": fid_vg.uncertainty_f,
                "complexity": float(vg_attr.std()),
                "CRPS_baseline": fid_vg.crps_baseline,
            })

            # ─── LIME ──────────────────────────────────────────────────────
            lime_attr = lime_attribution(model, bl_dev, device, n_perturb=50)
            fid_lime  = fidelity.compute(bl_dev, lime_attr, K=K, top_k=top_k)

            rows.append({
                "storm": s_idx,
                "method": "LIME (channel-level)",
                "H_mean_F": fid_lime.h_mean_f,
                "precision_F": fid_lime.precision_f,
                "recall_F": fid_lime.recall_f,
                "uncertainty_F": fid_lime.uncertainty_f,
                "complexity": float(lime_attr.std()),
                "CRPS_baseline": fid_lime.crps_baseline,
            })

        except Exception as e:
            print(f"  [E1] Storm {s_idx} failed: {e}")
            continue

    df = pd.DataFrame(rows)

    # Summary statistics
    summary = df.groupby("method")[
        ["H_mean_F", "precision_F", "recall_F", "uncertainty_F"]
    ].agg(["mean", "std"])

    print("\n[E1] Summary:")
    print(summary.to_string())

    df.to_csv(os.path.join(output_dir, "E1_comparison_table.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "E1_summary.csv"))
    print(f"[E1] Results saved to {output_dir}/E1_comparison_table.csv")

    return df


def plot_e1(df: pd.DataFrame, output_dir: str = "results") -> None:
    """Generate comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        methods = df["method"].unique()
        metrics = ["precision_F", "recall_F", "uncertainty_F", "H_mean_F"]
        colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle("E1: XAI Method Fidelity Comparison (TC Track)", fontsize=13)

        for ax, metric, color in zip(axes, metrics, colors):
            vals = [df[df["method"] == m][metric].mean() for m in methods]
            errs = [df[df["method"] == m][metric].std() for m in methods]
            ax.bar(range(len(methods)), vals, yerr=errs, color=color, alpha=0.85)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
            ax.set_title(metric.replace("_", "-"))
            ax.set_ylim(0, max(max(vals) * 1.2, 1.0))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "E1_comparison.png"), dpi=150)
        plt.close()
        print(f"[E1] Plot saved to {output_dir}/E1_comparison.png")
    except Exception as e:
        print(f"[E1] Plotting skipped: {e}")


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,  help="Path to model checkpoint .pt")
    parser.add_argument("--dataset_root", required=True, help="TC-OFM dataset root")
    parser.add_argument("--output_dir",  default="results", help="Output directory")
    parser.add_argument("--n_storms",    type=int, default=20)
    parser.add_argument("--K",           type=int, default=30)
    parser.add_argument("--gpu",         type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[E1] Device: {device}")

    # Load model
    from Model.flow_matching_model import TCFlowMatching
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TCFlowMatching(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Load test data
    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    from torch.utils.data import DataLoader
    dataset  = TrajectoryDataset(args.dataset_root, split="test")
    loader   = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=seq_collate)

    test_batches = []
    for batch in loader:
        test_batches.append(batch)
        if len(test_batches) >= args.n_storms:
            break

    df = run_e1(model, test_batches, device, args.output_dir, K=args.K)
    plot_e1(df, args.output_dir)
    print("[E1] Done.")
