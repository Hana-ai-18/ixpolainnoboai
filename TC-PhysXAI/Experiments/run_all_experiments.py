"""
TC-PhysXAI — Run All Experiments
==================================
Master script that sequentially runs E1-E8 and produces a combined report.

Usage:
  python Experiments/run_all_experiments.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results \
    --gpu 0 \
    --n_storms 20 \
    --K 30 \
    --experiments E1 E2 E3 E4 E5 E6 E7 E8

Experiment descriptions:
  E1 — XAI baseline comparison (Gradient / LIME / SHAP vs TC-PhysXAI)
  E2 — Recurvature attribution analysis (temporal + spatial XAI)
  E3 — Rapid intensification & SST attribution (C4/C5)
  E4 — Multi-modal XAI consistency across modules
  E5 — Probabilistic fidelity benchmark (H-mean-F, C3)
  E6 — Counterfactual scenario analysis (C6 adjoint)
  E7 — Adaptive Segment-Level XAI (C7): fixed A/B/C + τ* search + global mining
  E8 — Track Feature Attribution (C8): physical feature × ERA5 channel/window attribution
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

import torch
from torch.utils.data import DataLoader


def load_model_and_data(args, device):
    """Load TC-XFlow model and test dataset."""
    print(f"[Master] Loading model from {args.checkpoint} ...")

    from PhysXAI.model_io import load_tcxflow
    model = load_tcxflow(args.checkpoint, device)
    print(f"[Master] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    dataset = TrajectoryDataset(args.dataset_root, split="test")
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=seq_collate, num_workers=2,
    )
    print(f"[Master] Dataset loaded: {len(dataset)} test storms.")

    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= args.n_storms:
            break

    print(f"[Master] Using {len(batches)} storms for experiments.")
    return model, batches


def run_experiments(args, model, batches, device) -> dict:
    """Run selected experiments and return timing info."""
    results   = {}
    timing    = {}
    exps      = set(args.experiments)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if "E1" in exps:
        print("\n" + "="*60)
        print("E1: XAI Baseline Comparison")
        print("="*60)
        from Experiments.E1_comparison_baselines import run_e1, plot_e1
        t0 = time.time()
        df = run_e1(model, batches, device, out_dir, K=args.K, n_storms=args.n_storms)
        plot_e1(df, out_dir)
        timing["E1"] = time.time() - t0
        results["E1"] = {"rows": len(df)}

    if "E2" in exps:
        print("\n" + "="*60)
        print("E2: Recurvature Attribution Analysis")
        print("="*60)
        from Experiments.E2_recurvature_attribution import run_e2
        t0 = time.time()
        df = run_e2(model, batches, device, out_dir, K=args.K, n_storms=args.n_storms)
        timing["E2"] = time.time() - t0
        results["E2"] = {"rows": len(df)}

    if "E3" in exps:
        print("\n" + "="*60)
        print("E3: Rapid Intensification & SST Attribution")
        print("="*60)
        from Experiments.E3_rapid_intensification import run_e3
        t0 = time.time()
        df = run_e3(model, batches, device, out_dir, K=max(args.K//2, 10), n_storms=args.n_storms)
        timing["E3"] = time.time() - t0
        results["E3"] = {"rows": len(df)}

    if "E4" in exps:
        print("\n" + "="*60)
        print("E4: Multi-Modal XAI Consistency")
        print("="*60)
        from Experiments.E4_multimodal_consistency import run_e4
        t0 = time.time()
        df = run_e4(model, batches, device, out_dir, K=max(args.K//2, 10), n_storms=min(args.n_storms, 20))
        timing["E4"] = time.time() - t0
        results["E4"] = {"rows": len(df)}

    if "E5" in exps:
        print("\n" + "="*60)
        print("E5: Probabilistic Fidelity Benchmark")
        print("="*60)
        from Experiments.E5_fidelity_benchmark import run_e5
        t0 = time.time()
        df = run_e5(model, batches, device, out_dir, n_storms=min(args.n_storms, 15))
        timing["E5"] = time.time() - t0
        results["E5"] = {"rows": len(df)}

    if "E6" in exps:
        print("\n" + "="*60)
        print("E6: Counterfactual Scenario Analysis")
        print("="*60)
        from Experiments.E6_counterfactual_scenarios import run_e6
        t0 = time.time()
        df = run_e6(model, batches, device, out_dir, n_storms=min(args.n_storms, 15))
        timing["E6"] = time.time() - t0
        results["E6"] = {"rows": len(df)}

    if "E7" in exps:
        print("\n" + "="*60)
        print("E7: Adaptive Segment-Level XAI (C7 Extended)")
        print("="*60)
        from Experiments.E7_adaptive_segment_xai import run_e7, plot_e7
        t0 = time.time()
        df = run_e7(
            model, batches, device, out_dir,
            K_fixed=max(args.K // 2, 10),
            K_adaptive=max(args.K // 3, 5),
            n_storms=args.n_storms,
        )
        if not df.empty:
            plot_e7(df, out_dir)
        timing["E7"] = time.time() - t0
        results["E7"] = {"rows": len(df)}

    if "E8" in exps:
        print("\n" + "="*60)
        print("E8: Track Feature Attribution (C8)")
        print("="*60)
        from Experiments.E8_track_feature_attribution import (
            run_e8, plot_e8, run_e8_k_sensitivity,
        )
        t0 = time.time()
        df = run_e8(
            model, batches, device, out_dir,
            K=max(args.K // 3, 5),
            n_storms=args.n_storms,
        )
        if not df.empty:
            plot_e8(df, out_dir)
        # K-sensitivity ablation (lightweight — uses 5 storms × 3 K values)
        run_e8_k_sensitivity(
            model, batches, device, out_dir,
            K_values=(5, 10, 20),
            n_storms=min(5, args.n_storms),
        )
        timing["E8"] = time.time() - t0
        results["E8"] = {"rows": len(df)}

    return results, timing


def main():
    parser = argparse.ArgumentParser(
        description="TC-PhysXAI: Run all experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to TC-XFlow model checkpoint (.pt)"
    )
    parser.add_argument(
        "--dataset_root", required=True,
        help="Root directory of TC-OFM dataset"
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (0 = first GPU)"
    )
    parser.add_argument(
        "--n_storms", type=int, default=20,
        help="Number of test storms per experiment"
    )
    parser.add_argument(
        "--K", type=int, default=30,
        help="Ensemble size for CRPS and scenario analysis"
    )
    parser.add_argument(
        "--experiments", nargs="+",
        default=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
        help="Which experiments to run"
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Master] Device: {device}")
    print(f"[Master] Experiments: {args.experiments}")
    print(f"[Master] n_storms={args.n_storms}, K={args.K}")

    model, batches = load_model_and_data(args, device)
    results, timing = run_experiments(args, model, batches, device)

    # Save timing report
    report = {
        "experiments_run": list(results.keys()),
        "timing_seconds": {k: round(v, 1) for k, v in timing.items()},
        "total_seconds":  round(sum(timing.values()), 1),
        "n_storms": args.n_storms,
        "K": args.K,
    }
    report_path = os.path.join(args.output_dir, "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for exp, t in timing.items():
        print(f"  {exp}: {t:.1f}s")
    print(f"  TOTAL: {sum(timing.values()):.1f}s")
    print(f"\nAll results saved to: {args.output_dir}/")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
