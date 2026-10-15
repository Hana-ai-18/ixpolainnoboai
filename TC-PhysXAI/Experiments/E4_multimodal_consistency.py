"""
E4: Multi-Modal XAI Consistency Check
=======================================
Verify that different TC-PhysXAI modules (C1 temporal, C2 spatial, C4 FM-path,
C5 variance) produce CONSISTENT physical attributions.

Physical consistency principle:
  If C1 identifies t-36h as the most important temporal window and C2 identifies
  the 200 hPa trough as the dominant spatial feature, then:
  - C4 should show high channel attribution to GPH_200 / U_200 / V_200
  - C5 should show V_context dominant (synoptic ambiguity) at the same time
  - The counterfactual (C6) should show minimum change required in GPH_200

  These four attribution dimensions should tell the SAME physical story.

Metrics:
  - Cross-module attribution correlation (Spearman ρ between C1/C2/C4)
  - Physical label agreement rate (% cases where C2 regime matches C5 regime)
  - Temporal-spatial consistency index (TCS):
    TCS = correlation between temporal attribution variance and spatial entropy

Output:
  results/E4_consistency_matrix.csv
  results/E4_consistency_heatmap.png
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
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

from PhysXAI import (
    TemporalAttributor, SpatialAttributor,
    FMIntegrationAttribution, VarianceDecomposition, AnalogueXAI,
)


def run_e4(
    model,
    test_batches: List[list],
    device: torch.device,
    output_dir: str = "results",
    K: int = 20,
    n_storms: int = 30,
) -> pd.DataFrame:
    """Run E4: Multi-modal consistency analysis."""
    os.makedirs(output_dir, exist_ok=True)

    temporal = TemporalAttributor(model, device, n_ensemble=K)
    spatial  = SpatialAttributor(model, device)
    fm_attr  = FMIntegrationAttribution(model, device, ddim_steps=15)
    var_dec  = VarianceDecomposition(model, device, n_context_perturb=8)
    analogue = AnalogueXAI(model, device, n_clusters=3)

    rows = []

    for s_idx, bl in enumerate(test_batches[:n_storms]):
        print(f"[E4] Storm {s_idx+1}/{min(n_storms, len(test_batches))} ...")
        bl_dev = [t.to(device) if torch.is_tensor(t) else t for t in bl]

        try:
            # C1 Temporal
            t_res = temporal.compute(bl_dev, n_ensemble=K)
            attr_scores = t_res["mean_attribution_per_window"]  # [T_obs]
            temporal_entropy = float(
                -np.sum(attr_scores / (attr_scores.sum() + 1e-8) *
                        np.log(attr_scores / (attr_scores.sum() + 1e-8) + 1e-8))
            )
            most_imp_w = t_res["most_important_window"]

            # C2 Spatial
            s_res = spatial.compute(bl_dev, detect_objects=False)
            level_imp = s_res.get("level_importance", {})
            spatial_upper = level_imp.get("200hPa", 0) + level_imp.get("500hPa", 0)
            spatial_lower = level_imp.get("850hPa", 0) + level_imp.get("925hPa", 0)
            ig_attr = s_res.get("ig_attr", None)
            # Spatial entropy: distribution over channels
            if ig_attr is not None:
                ch_w = np.abs(ig_attr).mean(axis=(0, 2, 3, 4))  # [13]
                ch_w = ch_w / (ch_w.sum() + 1e-8)
                spatial_entropy = float(-np.sum(ch_w * np.log(ch_w + 1e-8)))
            else:
                spatial_entropy = 0.0

            # C4 FM-path attribution
            c4_res = fm_attr.compute(bl_dev, ddim_steps=15)
            c4_ch  = c4_res.channel_attribution  # [13]
            c4_upper = float(c4_ch[0] + c4_ch[1] + c4_ch[4] + c4_ch[5] + c4_ch[8] + c4_ch[9])
            c4_lower = float(c4_ch[2] + c4_ch[3] + c4_ch[6] + c4_ch[7] + c4_ch[10] + c4_ch[11])

            # C5 Variance decomposition
            vd_res = var_dec.compute(bl_dev, K=K, ddim_steps=10)
            frac_context = vd_res.frac_context
            frac_noise   = vd_res.frac_noise

            # Analogue
            an_res = analogue.compute(bl_dev, K=K)
            unc_entropy = an_res.get("uncertainty_entropy", 0)

            rows.append({
                "storm_idx": s_idx,
                # C1
                "c1_temporal_entropy":  temporal_entropy,
                "c1_most_imp_window":   most_imp_w,
                # C2
                "c2_spatial_upper":     spatial_upper,
                "c2_spatial_lower":     spatial_lower,
                "c2_spatial_entropy":   spatial_entropy,
                "c2_steering_regime":   s_res.get("steering_regime", ""),
                # C4
                "c4_upper":             c4_upper,
                "c4_lower":             c4_lower,
                # C5
                "c5_frac_context":      frac_context,
                "c5_frac_noise":        frac_noise,
                # Module 3
                "m3_unc_entropy":       unc_entropy,
                # Consistency checks
                "spatial_c2_c4_corr":  float(
                    np.corrcoef([spatial_upper, spatial_lower],
                                [c4_upper, c4_lower])[0, 1]
                    if c4_upper + c4_lower > 0 else 0.0
                ),
                "c1_c5_consistent":    (
                    (frac_context > 0.5 and temporal_entropy > 1.0) or
                    (frac_noise   > 0.5 and temporal_entropy < 0.8)
                ),
            })

        except Exception as e:
            print(f"  [E4] Storm {s_idx} failed: {e}")
            continue

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Consistency statistics
        c2_c4_corr = float(df["spatial_c2_c4_corr"].mean())
        c1_c5_agree = float(df["c1_c5_consistent"].mean())

        # Spearman correlation between C2 spatial entropy and C1 temporal entropy
        if len(df) >= 3:
            rho, pval = sp_stats.spearmanr(df["c2_spatial_entropy"], df["c1_temporal_entropy"])
            print(f"\n[E4] C2↔C1 Spearman ρ = {rho:.3f} (p={pval:.3f})")
            print(f"[E4] C2↔C4 mean corr = {c2_c4_corr:.3f}")
            print(f"[E4] C1↔C5 consistency rate = {c1_c5_agree:.1%}")

        df.to_csv(os.path.join(output_dir, "E4_consistency.csv"), index=False)
        _plot_e4(df, output_dir)

    return df


def _plot_e4(df: pd.DataFrame, output_dir: str) -> None:
    """Plot consistency heatmap and scatter plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("E4: Multi-Modal XAI Consistency (TC-PhysXAI)", fontsize=12)

        # C2 spatial entropy vs C1 temporal entropy
        ax = axes[0]
        ax.scatter(df["c2_spatial_entropy"], df["c1_temporal_entropy"],
                   alpha=0.6, s=50, color="#8e44ad")
        if len(df) >= 3:
            try:
                rho, _ = __import__("scipy").stats.spearmanr(
                    df["c2_spatial_entropy"], df["c1_temporal_entropy"]
                )
                ax.set_title(f"C2 vs C1 Entropy (ρ={rho:.2f})")
            except Exception:
                ax.set_title("C2 vs C1 Entropy")
        ax.set_xlabel("C2 Spatial Entropy")
        ax.set_ylabel("C1 Temporal Entropy")
        ax.grid(alpha=0.3)

        # C2 upper attribution vs C4 upper attribution
        ax = axes[1]
        ax.scatter(df["c2_spatial_upper"], df["c4_upper"],
                   alpha=0.6, s=50, color="#27ae60")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="1:1")
        ax.set_xlabel("C2 Upper-Level Attr.")
        ax.set_ylabel("C4 Upper-Level Attr.")
        ax.set_title("C2 vs C4 Upper Troposphere")
        ax.legend()
        ax.grid(alpha=0.3)

        # C5 context fraction vs M3 ensemble entropy
        ax = axes[2]
        ax.scatter(df["c5_frac_context"], df["m3_unc_entropy"],
                   alpha=0.6, s=50, color="#e67e22")
        ax.set_xlabel("C5 Context Variance Fraction")
        ax.set_ylabel("M3 Scenario Entropy")
        ax.set_title("C5 vs M3 Uncertainty Consistency")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "E4_consistency.png"), dpi=150)
        plt.close()
        print("[E4] Plot saved.")
    except Exception as e:
        print(f"[E4] Plotting skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir",   default="results")
    parser.add_argument("--n_storms",     type=int, default=30)
    parser.add_argument("--K",            type=int, default=20)
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
    run_e4(model, batches, device, args.output_dir, K=args.K, n_storms=args.n_storms)
    print("[E4] Done.")
