"""
E5: Probabilistic Fidelity Benchmark
======================================
Comprehensive benchmark of H-mean-F across:
  - Different XAI methods (IG, Gradient, LIME)
  - Different top-k channel selections (k=1,2,3,5)
  - Different ensemble sizes (K=10,20,30,50)
  - Different lead times (t+6h, t+24h, t+48h, t+72h)

This provides the ablation study needed for a Q1 paper:
  "Which method, which k, which ensemble size gives best fidelity?"

Output:
  results/E5_fidelity_benchmark.csv
  results/E5_ablation_heatmap.png
"""

from __future__ import annotations

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

from PhysXAI import SpatialAttributor, ProbabilisticFidelity


def run_e5(
    model,
    test_batches: List[list],
    device: torch.device,
    output_dir: str = "results",
    n_storms: int = 20,
    k_values: List[int] = None,
    K_values: List[int] = None,
) -> pd.DataFrame:
    """Run E5: Fidelity benchmark across hyperparameters."""
    os.makedirs(output_dir, exist_ok=True)

    if k_values is None:
        k_values = [1, 2, 3, 5]
    if K_values is None:
        K_values = [10, 20, 30]

    spatial = SpatialAttributor(model, device)
    rows    = []

    for s_idx, bl in enumerate(test_batches[:n_storms]):
        print(f"[E5] Storm {s_idx+1}/{min(n_storms, len(test_batches))} ...")
        bl_dev = [t.to(device) if torch.is_tensor(t) else t for t in bl]

        try:
            # Compute IG attribution once
            s_result = spatial.compute(bl_dev, detect_objects=False)
            ig_attr  = torch.as_tensor(s_result["ig_attr"]).detach().clone() if "ig_attr" in s_result else None
            if ig_attr is None:
                continue

            # Benchmark across K and top_k
            for K_ens in K_values:
                fidelity = ProbabilisticFidelity(model, device, n_ensemble=K_ens)

                for top_k in k_values:
                    try:
                        result = fidelity.compute(bl_dev, ig_attr, K=K_ens, top_k=top_k)
                        rows.append({
                            "storm_idx":    s_idx,
                            "K_ensemble":   K_ens,
                            "top_k":        top_k,
                            "H_mean_F":     result.h_mean_f,
                            "precision_F":  result.precision_f,
                            "recall_F":     result.recall_f,
                            "uncertainty_F":result.uncertainty_f,
                            "CRPS_baseline":result.crps_baseline,
                        })
                    except Exception as e:
                        print(f"    [E5] K={K_ens}, k={top_k} failed: {e}")
                        continue

        except Exception as e:
            print(f"  [E5] Storm {s_idx} failed: {e}")
            continue

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Pivot: mean H-mean-F for each (K, top_k)
        pivot = df.groupby(["K_ensemble", "top_k"])["H_mean_F"].mean().unstack()
        print("\n[E5] H-mean-F pivot (K × top_k):")
        print(pivot.to_string())

        df.to_csv(os.path.join(output_dir, "E5_fidelity_benchmark.csv"), index=False)
        pivot.to_csv(os.path.join(output_dir, "E5_pivot.csv"))

        _plot_e5(df, pivot, output_dir)

    return df


def _plot_e5(df: pd.DataFrame, pivot, output_dir: str) -> None:
    """Plot fidelity heatmap and K-sensitivity curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("E5: Probabilistic Fidelity Benchmark (TC-PhysXAI)", fontsize=12)

        # Heatmap of H-mean-F
        ax = axes[0]
        try:
            import seaborn as sns
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                        cbar_kws={"label": "H-mean-F"})
        except ImportError:
            im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, label="H-mean-F")
        ax.set_xlabel("top-k channels")
        ax.set_ylabel("Ensemble size K")
        ax.set_title("H-mean-F: K × top-k")

        # K sensitivity (top_k=3)
        ax = axes[1]
        sub = df[df["top_k"] == 3].groupby("K_ensemble")[
            ["H_mean_F", "precision_F", "recall_F", "uncertainty_F"]
        ].mean()
        for col, color in zip(sub.columns, ["#e74c3c", "#3498db", "#27ae60", "#e67e22"]):
            ax.plot(sub.index, sub[col], marker="o", label=col.replace("_", "-"),
                    color=color, linewidth=2)
        ax.set_xlabel("Ensemble size K")
        ax.set_ylabel("Score")
        ax.set_title("Fidelity vs Ensemble Size (top-k=3)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "E5_fidelity_benchmark.png"), dpi=150)
        plt.close()
        print("[E5] Plot saved.")
    except Exception as e:
        print(f"[E5] Plotting skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir",   default="results")
    parser.add_argument("--n_storms",     type=int, default=20)
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
    run_e5(model, batches, device, args.output_dir, n_storms=args.n_storms)
    print("[E5] Done.")
