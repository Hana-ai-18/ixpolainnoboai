"""
TC-PhysXAI — Multi-Seed Experiment Runner
==========================================
Chạy E1–E8 với nhiều checkpoint (seed) theo quy tắc:
  • E1 (XAI baseline comparison) : chạy trên TẤT CẢ seeds → báo mean ± std
  • E2–E8 (XAI analysis)         : chạy trên seed0 (checkpoint đầu tiên)

Kết quả lưu vào:
  <output_dir>/seed0/  ...  <output_dir>/seed2/   (per-seed)
  <output_dir>/multiseed_E1_report.json            (aggregate)
  <output_dir>/multiseed_summary.json              (timing + all metrics)

Usage:
  python Experiments/run_multiseed.py \\
    --checkpoints \\
        /kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_fm_seed0_new_ver.pth \\
        /kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_fm_seed1_new_ver.pth \\
        /kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_fm_seed2_new_ver.pth \\
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \\
    --output_dir /kaggle/working/TC-PhysXAI/results \\
    --gpu 0 \\
    --n_storms 20 \\
    --K 30 \\
    --experiments E1 E2 E3 E4 E5 E6 E7 E8
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

import torch
from torch.utils.data import DataLoader


# ── Robust checkpoint loader ──────────────────────────────────────────────────

_MODEL_STATE_KEYS = [
    "model_state_dict",   # TC-XFlow default
    "state_dict",         # common alternative
    "model",              # some trainers
    "net",
]

_MODEL_CFG_KEYS = [
    "model_kwargs",       # TC-XFlow default
    "model_config",
    "model_cfg",
    "config",
    "cfg",
]

_DEFAULT_MODEL_KWARGS = {
    "obs_len": 8, "pred_len": 12,
    "d_cond": 256, "d_model": 256, "nhead": 8,
    "num_dec_layers": 4, "unet_in_ch": 13,
}


def _load_model(ckpt_path: str, device: torch.device):
    """Load TC-XFlow via the shared loader (full config + EMA weights)."""
    from PhysXAI.model_io import load_tcxflow
    return load_tcxflow(ckpt_path, device)


# ── Dataset loader (shared across all seeds) ─────────────────────────────────

def _load_batches(dataset_root: str, n_storms: int):
    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    dataset = TrajectoryDataset(dataset_root, split="test")
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=seq_collate, num_workers=2,
    )
    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= n_storms:
            break
    print(f"[Data] {len(dataset)} test storms → using first {len(batches)}")
    return batches


# ── Single-seed experiment runner ─────────────────────────────────────────────

def _run_one_seed(
    model, batches, device,
    out_dir: str, exps: set,
    K: int, n_storms: int,
) -> tuple[dict, dict]:
    """Run a subset of experiments on one model. Returns (results, timing)."""
    results = {}
    timing  = {}
    os.makedirs(out_dir, exist_ok=True)

    if "E1" in exps:
        from Experiments.E1_comparison_baselines import run_e1, plot_e1
        t0 = time.time()
        df = run_e1(model, batches, device, out_dir, K=K, n_storms=n_storms)
        plot_e1(df, out_dir)
        timing["E1"] = time.time() - t0
        results["E1"] = {"rows": len(df), "df": df}

    if "E2" in exps:
        from Experiments.E2_recurvature_attribution import run_e2
        t0 = time.time()
        df = run_e2(model, batches, device, out_dir, K=K, n_storms=n_storms)
        timing["E2"] = time.time() - t0
        results["E2"] = {"rows": len(df)}

    if "E3" in exps:
        from Experiments.E3_rapid_intensification import run_e3
        t0 = time.time()
        df = run_e3(model, batches, device, out_dir, K=max(K//2, 10), n_storms=n_storms)
        timing["E3"] = time.time() - t0
        results["E3"] = {"rows": len(df)}

    if "E4" in exps:
        from Experiments.E4_multimodal_consistency import run_e4
        t0 = time.time()
        df = run_e4(model, batches, device, out_dir, K=max(K//2, 10), n_storms=min(n_storms, 20))
        timing["E4"] = time.time() - t0
        results["E4"] = {"rows": len(df)}

    if "E5" in exps:
        from Experiments.E5_fidelity_benchmark import run_e5
        t0 = time.time()
        df = run_e5(model, batches, device, out_dir, n_storms=min(n_storms, 15))
        timing["E5"] = time.time() - t0
        results["E5"] = {"rows": len(df)}

    if "E6" in exps:
        from Experiments.E6_counterfactual_scenarios import run_e6
        t0 = time.time()
        df = run_e6(model, batches, device, out_dir, n_storms=min(n_storms, 15))
        timing["E6"] = time.time() - t0
        results["E6"] = {"rows": len(df)}

    if "E7" in exps:
        from Experiments.E7_adaptive_segment_xai import run_e7, plot_e7
        t0 = time.time()
        df = run_e7(
            model, batches, device, out_dir,
            K_fixed=max(K // 2, 10),
            K_adaptive=max(K // 3, 5),
            n_storms=n_storms,
        )
        if not df.empty:
            plot_e7(df, out_dir)
        timing["E7"] = time.time() - t0
        results["E7"] = {"rows": len(df)}

    if "E8" in exps:
        from Experiments.E8_track_feature_attribution import run_e8, plot_e8, run_e8_k_sensitivity
        t0 = time.time()
        df = run_e8(model, batches, device, out_dir, K=max(K // 3, 5), n_storms=n_storms)
        if not df.empty:
            plot_e8(df, out_dir)
        run_e8_k_sensitivity(
            model, batches, device, out_dir,
            K_values=(5, 10, 20),
            n_storms=min(5, n_storms),
        )
        timing["E8"] = time.time() - t0
        results["E8"] = {"rows": len(df)}

    return results, timing


# ── Aggregate E1 results across seeds ─────────────────────────────────────────

def _aggregate_e1(per_seed_dfs: list) -> dict:
    """Aggregate E1 across seeds.

    Per seed, each method's metrics are first averaged over that seed's storms;
    mean and sample std (ddof=1) are then taken ACROSS SEEDS. Seeds may have a
    different number of successful storms (failed storms are dropped per seed).
    Also reports each method's paired difference vs the random-channel floor.
    """
    if not per_seed_dfs:
        return {}

    skip = {"storm"}
    cols = [c for c in per_seed_dfs[0].select_dtypes(include="number").columns if c not in skip]
    S = len(per_seed_dfs)
    ddof = 1 if S > 1 else 0

    per_method: dict = {}
    for m in per_seed_dfs[0]["method"].unique():
        rows = []
        for df in per_seed_dfs:
            sub = df[df["method"] == m]
            if len(sub):
                rows.append(sub[cols].mean(axis=0, skipna=True).values)
        if not rows:
            continue
        stk = np.stack(rows, axis=0)                      # [S_ok, C]
        per_method[str(m)] = {
            "n_seeds": int(stk.shape[0]),
            "n_storms_per_seed": [int((df["method"] == m).sum()) for df in per_seed_dfs],
            **{col: {"mean": float(np.nanmean(stk[:, ci])),
                     "std": float(np.nanstd(stk[:, ci], ddof=ddof)) if stk.shape[0] > 1 else 0.0}
               for ci, col in enumerate(cols)},
        }

    rnd = next((m for m in per_method if m.startswith("Random")), None)
    paired: dict = {}
    if rnd is not None:
        for m in per_method:
            if m == rnd:
                continue
            paired[m] = {}
            for metric in ("h_score_no_u", "precision_F", "recall_F"):
                if metric not in cols:
                    continue
                diffs = []
                for df in per_seed_dfs:
                    a_ = df[df["method"] == m].set_index("storm")[metric]
                    b_ = df[df["method"] == rnd].set_index("storm")[metric]
                    d_ = (a_ - b_).dropna()
                    if len(d_):
                        diffs.append(float(d_.mean()))
                if diffs:
                    paired[m][metric] = {
                        "mean_diff_vs_random": float(np.mean(diffs)),
                        "std_across_seeds": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                        "per_seed": diffs,
                    }

    return {"n_seeds": S, "per_method": per_method, "paired_vs_random": paired}


def main():
    parser = argparse.ArgumentParser(
        description="TC-PhysXAI: Multi-seed experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Paths to TC-XFlow checkpoints (.pt / .pth), one per seed"
    )
    parser.add_argument(
        "--dataset_root", required=True,
        help="Root directory of TC-OFM dataset"
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Output directory; per-seed results go into <output_dir>/seed{i}/"
    )
    parser.add_argument("--gpu",      type=int, default=0)
    parser.add_argument("--n_storms", type=int, default=20)
    parser.add_argument("--K",        type=int, default=30)
    parser.add_argument(
        "--experiments", nargs="+",
        default=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    exps   = set(args.experiments)
    n_seed = len(args.checkpoints)

    print("=" * 60)
    print("TC-PhysXAI Multi-Seed Runner")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Seeds:       {n_seed}")
    print(f"  Experiments: {sorted(exps)}")
    print(f"  n_storms:    {args.n_storms}  |  K: {args.K}")
    print(f"  Strategy:    E1 → all {n_seed} seeds (mean ± std)")
    print(f"               E2–E8 → seed0 only")
    print()

    # ── Load dataset once ────────────────────────────────────────────────────
    print("[Data] Loading test batches ...")
    batches = _load_batches(args.dataset_root, args.n_storms)

    # ── Run E1 on all seeds ─────────────────────────────────────────────────
    e1_dfs   = []
    e1_timing = []
    all_timing: dict[str, list] = {}

    if "E1" in exps:
        print("\n" + "=" * 60)
        print(f"E1: XAI Baseline Comparison (running on all {n_seed} seeds)")
        print("=" * 60)

        for si, ckpt_path in enumerate(args.checkpoints):
            seed_dir = os.path.join(args.output_dir, f"seed{si}")
            os.makedirs(seed_dir, exist_ok=True)
            print(f"\n  ── Seed {si} ──")
            model = _load_model(ckpt_path, device)
            res, tim = _run_one_seed(
                model, batches, device, seed_dir, {"E1"}, args.K, args.n_storms
            )
            if "E1" in res and "df" in res["E1"]:
                e1_dfs.append(res["E1"]["df"])
            e1_timing.append(tim.get("E1", 0.0))
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

        all_timing["E1"] = e1_timing
        agg = _aggregate_e1(e1_dfs)
        agg_path = os.path.join(args.output_dir, "multiseed_E1_report.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n[E1] Aggregated report → {agg_path}")
        for m, v in agg.get("per_method", {}).items():
            print(f"     {m:<34} H_mean_F {v['H_mean_F']['mean']:.3f} ± {v['H_mean_F']['std']:.3f}"
                  f"   h_score_no_u {v['h_score_no_u']['mean']:.3f} ± {v['h_score_no_u']['std']:.3f}"
                  f"   (seeds={v['n_seeds']})")

    # ── Run E2–E8 on seed0 only ─────────────────────────────────────────────
    xai_exps = exps - {"E1"}
    if xai_exps:
        print("\n" + "=" * 60)
        print(f"E2–E8: XAI Analysis on seed0 ({Path(args.checkpoints[0]).name})")
        print("=" * 60)

        seed0_dir = os.path.join(args.output_dir, "seed0")
        os.makedirs(seed0_dir, exist_ok=True)
        model0 = _load_model(args.checkpoints[0], device)
        res0, tim0 = _run_one_seed(
            model0, batches, device, seed0_dir, xai_exps, args.K, args.n_storms
        )
        for exp, t in tim0.items():
            all_timing.setdefault(exp, []).append(t)
        del model0
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Summary report ───────────────────────────────────────────────────────
    flat_timing = {
        exp: round(float(np.mean(ts)), 1)
        for exp, ts in all_timing.items()
    }
    summary = {
        "seeds_used": n_seed,
        "checkpoints": [str(p) for p in args.checkpoints],
        "strategy": {
            "E1": f"all {n_seed} seeds (mean ± std)",
            "E2-E8": "seed0 only",
        },
        "experiments_run": sorted(exps),
        "timing_per_exp_seconds": flat_timing,
        "total_seconds": round(sum(flat_timing.values()), 1),
        "n_storms": args.n_storms,
        "K": args.K,
    }
    summary_path = os.path.join(args.output_dir, "multiseed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)
    for exp, t in flat_timing.items():
        seeds_note = f"({n_seed} seeds)" if exp == "E1" else "(seed0)"
        print(f"  {exp} {seeds_note}: {t:.1f}s")
    print(f"  TOTAL: {summary['total_seconds']:.1f}s")
    print(f"\nPer-seed results : {args.output_dir}/seed{{0..{n_seed-1}}}/")
    print(f"E1 aggregate     : {agg_path if 'E1' in exps else 'N/A'}")
    print(f"Summary          : {summary_path}")


if __name__ == "__main__":
    main()
