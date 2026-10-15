"""
TC-PhysXAI — Main Kaggle Analysis Entry Point
===============================================
Runs the complete TC-PhysXAI analysis pipeline on a pre-trained TC-XFlow model.

This is the primary script for producing XAI results in Kaggle notebooks.

Usage (Kaggle):
--------------
# Step 1: Train TC-XFlow (existing command — unchanged)
!python /kaggle/working/Tropical_Typhoon_Architecture/Scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v3_ot_fixed/seed0 \
    --seed 2 --gpu_num 0 --use_ot --ot_epsilon 0.05

# Step 2: Run TC-PhysXAI analysis
!python /kaggle/working/TC-PhysXAI/Scripts/run_physxai.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results \
    --storm_ids all \
    --modules all \
    --K 30 \
    --gpu 0 \
    --save_bulletins

Output structure:
  results/
    bulletins/          ← per-storm meteorological bulletins (.txt)
    attribution_maps/   ← IG attribution map arrays (.npy)
    temporal_attr/      ← per-storm temporal attribution (.json)
    scenario_clusters/  ← ensemble cluster analysis (.json)
    experiments/        ← E1-E6 benchmark results
    summary_report.json ← aggregated statistics
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict

# ─── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TC_XFLOW_ROOT))

# ─── TC-PhysXAI imports ──────────────────────────────────────────────────────
from PhysXAI import (
    TemporalAttributor,
    SpatialAttributor,
    AnalogueXAI,
    FMIntegrationAttribution,
    VarianceDecomposition,
    FMCounterfactual,
    ProbabilisticFidelity,
    MeteorologyInterpreter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TC-PhysXAI: Physics-Guided XAI for Typhoon Track Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",    required=True,  help="TC-XFlow model checkpoint (.pt)")
    parser.add_argument("--dataset_root",  required=True,  help="TC-OFM dataset root directory")
    parser.add_argument("--output_dir",    default="/kaggle/working/TC-PhysXAI/results",
                        help="Output directory for results")
    parser.add_argument("--storm_ids",     nargs="+", default=["all"],
                        help="Storm IDs to analyse, or 'all'")
    parser.add_argument("--modules",       nargs="+",
                        default=["temporal", "spatial", "analogue", "c4", "c5", "c6", "fidelity"],
                        help="Which modules to run")
    parser.add_argument("--K",             type=int, default=30,
                        help="Ensemble size")
    parser.add_argument("--n_storms",      type=int, default=50,
                        help="Max storms to analyse")
    parser.add_argument("--gpu",           type=int, default=0)
    parser.add_argument("--save_bulletins",action="store_true",
                        help="Save per-storm meteorological bulletins to text files")
    parser.add_argument("--run_experiments",action="store_true",
                        help="Also run E1-E6 experiments after main analysis")
    parser.add_argument("--ddim_steps",    type=int, default=20,
                        help="ODE integration steps for C4/C6")
    return parser.parse_args()


def setup_output_dirs(output_dir: str) -> Dict[str, str]:
    """Create output directory structure."""
    dirs = {
        "root":        output_dir,
        "bulletins":   os.path.join(output_dir, "bulletins"),
        "attr_maps":   os.path.join(output_dir, "attribution_maps"),
        "temporal":    os.path.join(output_dir, "temporal_attr"),
        "scenarios":   os.path.join(output_dir, "scenario_clusters"),
        "experiments": os.path.join(output_dir, "experiments"),
        "plots":       os.path.join(output_dir, "plots"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def load_model(checkpoint_path: str, device: torch.device):
    """Load TC-XFlow model from checkpoint."""
    print(f"[PhysXAI] Loading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    from Model.flow_matching_model import TCFlowMatching
    model_kwargs = ckpt.get("model_kwargs", {
        "obs_len": 8, "pred_len": 12,
        "d_cond": 256, "d_model": 256,
        "nhead": 8, "num_dec_layers": 4, "unet_in_ch": 13,
    })
    model = TCFlowMatching(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PhysXAI] Model loaded. Parameters: {n_params:,}")
    print(f"[PhysXAI] Epoch: {ckpt.get('epoch', 'N/A')}, "
          f"Best CRPS: {ckpt.get('best_crps', 'N/A'):.4f}" if 'best_crps' in ckpt else "")
    return model


def load_dataset(dataset_root: str, n_storms: int):
    """Load test dataset."""
    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    from torch.utils.data import DataLoader

    dataset = TrajectoryDataset(dataset_root, split="test")
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=seq_collate, num_workers=2, pin_memory=True,
    )
    print(f"[PhysXAI] Test dataset: {len(dataset)} storms total.")

    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= n_storms:
            break

    print(f"[PhysXAI] Loaded {len(batches)} storms for analysis.")
    return batches


def get_storm_id(batch_list: list, idx: int) -> str:
    """Extract storm ID from batch_list[15] if available."""
    try:
        if len(batch_list) > 15 and batch_list[15] is not None:
            ids = batch_list[15]
            if hasattr(ids, '__iter__') and not isinstance(ids, str):
                return str(ids[0]) if len(ids) > 0 else f"storm_{idx:04d}"
    except Exception:
        pass
    return f"storm_{idx:04d}"


def analyse_storm(
    s_idx: int,
    batch_list: list,
    storm_id: str,
    device: torch.device,
    modules_to_run: List[str],
    K: int,
    ddim_steps: int,
    dirs: Dict[str, str],
    save_bulletins: bool,
    # Pre-built module instances (passed in for efficiency)
    temporal_mod: Optional[TemporalAttributor],
    spatial_mod:  Optional[SpatialAttributor],
    analogue_mod: Optional[AnalogueXAI],
    fidelity_mod: Optional[ProbabilisticFidelity],
    c4_mod:       Optional[FMIntegrationAttribution],
    c5_mod:       Optional[VarianceDecomposition],
    c6_mod:       Optional[FMCounterfactual],
    interp:       MeteorologyInterpreter,
) -> Dict:
    """Run all modules for one storm."""
    bl = [t.to(device) if torch.is_tensor(t) else t for t in batch_list]
    result_store = {"storm_id": storm_id}

    t_result = s_result = a_result = fid_result = c4_result = c5_result = c6_result = None

    # ─── Module 1: Temporal ──────────────────────────────────────────────────
    if "temporal" in modules_to_run and temporal_mod is not None:
        try:
            t_result = temporal_mod.compute(bl, n_ensemble=K)
            result_store["temporal"] = {
                "most_important_window": t_result["most_important_window"],
                "window_labels":         t_result["window_time_labels"],
                "attribution_scores":    t_result["mean_attribution_per_window"].tolist(),
                "physical_interpretation": t_result["physical_interpretation"],
            }
            # Save JSON
            jpath = os.path.join(dirs["temporal"], f"{storm_id}_temporal.json")
            with open(jpath, "w") as f:
                json.dump(result_store["temporal"], f, indent=2)
        except Exception as e:
            print(f"  [M1] Failed: {e}")

    # ─── Module 2: Spatial ───────────────────────────────────────────────────
    if "spatial" in modules_to_run and spatial_mod is not None:
        try:
            s_result = spatial_mod.compute(bl, detect_objects=True)
            result_store["spatial"] = {
                "level_importance":     s_result.get("level_importance", {}),
                "steering_regime":      s_result.get("steering_regime", ""),
                "n_objects":            len(s_result.get("physical_objects", [])),
                "object_types":         [o.type for o in s_result.get("physical_objects", [])],
                "object_levels":        [o.level for o in s_result.get("physical_objects", [])],
            }
            # Save attribution map as numpy array
            if "ig_attr" in s_result:
                np.save(
                    os.path.join(dirs["attr_maps"], f"{storm_id}_ig_attr.npy"),
                    s_result["ig_attr"],
                )
        except Exception as e:
            print(f"  [M2] Failed: {e}")

    # ─── Module 3: Analogue ──────────────────────────────────────────────────
    if "analogue" in modules_to_run and analogue_mod is not None:
        try:
            a_result = analogue_mod.compute(bl, K=K, storm_idx=0)
            result_store["analogue"] = {
                "dominant_scenario":    a_result.get("dominant_scenario", ""),
                "uncertainty_entropy":  a_result.get("uncertainty_entropy", 0),
                "n_scenarios":          a_result.get("n_scenarios", 1),
                "physical_interpretation": a_result.get("physical_interpretation", ""),
            }
            # Save cluster summary
            jpath = os.path.join(dirs["scenarios"], f"{storm_id}_clusters.json")
            with open(jpath, "w") as f:
                json.dump(result_store["analogue"], f, indent=2)
        except Exception as e:
            print(f"  [M3] Failed: {e}")

    # ─── C4: FM integration path ─────────────────────────────────────────────
    if "c4" in modules_to_run and c4_mod is not None:
        try:
            c4_result = c4_mod.compute(bl, ddim_steps=ddim_steps)
            result_store["c4"] = {
                "channel_attribution":  c4_result.channel_attribution.tolist(),
                "dominant_channels":    [(n, float(s)) for n, s in c4_result.dominant_channels],
                "physical_interpretation": c4_result.physical_interpretation,
            }
        except Exception as e:
            print(f"  [C4] Failed: {e}")

    # ─── C5: Variance decomposition ──────────────────────────────────────────
    if "c5" in modules_to_run and c5_mod is not None:
        try:
            c5_result = c5_mod.compute(bl, K=max(K//3, 5), ddim_steps=ddim_steps//2)
            result_store["c5"] = {
                "frac_noise":         c5_result.frac_noise,
                "frac_context":       c5_result.frac_context,
                "predictability_h6h": c5_result.predictability_horizon * 6,
                "physical_interpretation": c5_result.physical_interpretation,
            }
        except Exception as e:
            print(f"  [C5] Failed: {e}")

    # ─── C6: Counterfactual ──────────────────────────────────────────────────
    if "c6" in modules_to_run and c6_mod is not None:
        try:
            c6_result = c6_mod.compute(bl, cf_type="RECURVATURE", storm_idx=0)
            result_store["c6"] = {
                "cf_type":           c6_result.counterfactual_type,
                "context_norm":      float(c6_result.context_change_norm),
                "track_delta_km":    float(c6_result.track_delta_km),
                "dominant_channel":  str(c6_result.channel_delta.argmax()),
                "physical_explanation": c6_result.physical_explanation,
            }
        except Exception as e:
            print(f"  [C6] Failed: {e}")

    # ─── C3: Fidelity ────────────────────────────────────────────────────────
    if "fidelity" in modules_to_run and fidelity_mod is not None and s_result is not None:
        try:
            ig_attr = s_result.get("ig_attr", None)
            if ig_attr is not None:
                ig_t = torch.tensor(ig_attr, dtype=torch.float32)
                fid_result = fidelity_mod.compute(bl, ig_t, K=K, top_k=3)
                result_store["fidelity"] = {
                    "H_mean_F":     fid_result.h_mean_f,
                    "precision_F":  fid_result.precision_f,
                    "recall_F":     fid_result.recall_f,
                    "uncertainty_F": fid_result.uncertainty_f,
                }
        except Exception as e:
            print(f"  [C3] Failed: {e}")

    # ─── Meteorological Bulletin ─────────────────────────────────────────────
    if save_bulletins:
        try:
            import datetime
            bulletin = interp.synthesise(
                typhoon_id=storm_id,
                analysis_time=datetime.datetime.utcnow().strftime("%Y-%m-%d %HZ"),
                temporal_result=t_result,
                spatial_result=s_result,
                analogue_result=a_result,
                fidelity_result=fid_result,
                cf_result=c6_result,
                variance_result=c5_result,
                fm_attr_result=c4_result,
            )
            bpath = os.path.join(dirs["bulletins"], f"{storm_id}_bulletin.txt")
            with open(bpath, "w") as f:
                f.write(bulletin.full_text)
            result_store["bulletin_path"] = bpath
        except Exception as e:
            print(f"  [Bulletin] Failed: {e}")

    return result_store


def save_summary(all_results: List[Dict], dirs: Dict[str, str]) -> None:
    """Compute and save aggregate statistics across all storms."""
    summary = {
        "n_storms_analysed": len(all_results),
        "modules_run": [],
    }

    # Aggregate fidelity scores
    fid_scores = [r["fidelity"]["H_mean_F"]
                  for r in all_results if "fidelity" in r]
    if fid_scores:
        summary["fidelity_H_mean_F"] = {
            "mean": float(np.mean(fid_scores)),
            "std":  float(np.std(fid_scores)),
            "min":  float(np.min(fid_scores)),
            "max":  float(np.max(fid_scores)),
        }

    # Aggregate steering regimes
    regimes = [r.get("spatial", {}).get("steering_regime", "")
               for r in all_results if "spatial" in r]
    if regimes:
        regime_counts: Dict[str, int] = {}
        for r in regimes:
            if r:
                regime_counts[r] = regime_counts.get(r, 0) + 1
        summary["steering_regime_distribution"] = regime_counts

    # Aggregate variance decomposition
    noise_fracs = [r["c5"]["frac_noise"]
                   for r in all_results if "c5" in r]
    ctx_fracs   = [r["c5"]["frac_context"]
                   for r in all_results if "c5" in r]
    if noise_fracs:
        summary["variance_decomposition"] = {
            "mean_frac_noise":   float(np.mean(noise_fracs)),
            "mean_frac_context": float(np.mean(ctx_fracs)),
        }

    jpath = os.path.join(dirs["root"], "summary_report.json")
    with open(jpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[PhysXAI] Summary saved to {jpath}")
    print(json.dumps(summary, indent=2))


def main():
    args = parser = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[PhysXAI] TC-PhysXAI Analysis Pipeline")
    print(f"[PhysXAI] Device: {device}")
    print(f"[PhysXAI] Modules: {args.modules}")
    print(f"[PhysXAI] K={args.K}, n_storms={args.n_storms}")

    dirs    = setup_output_dirs(args.output_dir)
    model   = load_model(args.checkpoint, device)
    batches = load_dataset(args.dataset_root, args.n_storms)
    interp  = MeteorologyInterpreter()

    # ─── Initialise modules ──────────────────────────────────────────────────
    mods = args.modules
    temporal_mod = TemporalAttributor(model, device, n_ensemble=args.K) if "temporal" in mods else None
    spatial_mod  = SpatialAttributor(model, device)                      if "spatial"  in mods else None
    analogue_mod = AnalogueXAI(model, device, n_clusters=3)              if "analogue" in mods else None
    fidelity_mod = ProbabilisticFidelity(model, device, n_ensemble=args.K) if "fidelity" in mods else None
    c4_mod       = FMIntegrationAttribution(model, device, ddim_steps=args.ddim_steps) if "c4" in mods else None
    c5_mod       = VarianceDecomposition(model, device)                  if "c5" in mods else None
    c6_mod       = FMCounterfactual(model, device, ddim_steps=args.ddim_steps) if "c6" in mods else None

    # ─── Main loop ───────────────────────────────────────────────────────────
    all_results = []
    t0_total    = time.time()

    for s_idx, bl in enumerate(batches):
        storm_id = get_storm_id(bl, s_idx)

        # Filter by storm_ids if specified
        if args.storm_ids != ["all"] and storm_id not in args.storm_ids:
            continue

        print(f"\n[PhysXAI] ── Storm {s_idx+1}/{len(batches)}: {storm_id} ──")
        t0 = time.time()

        result = analyse_storm(
            s_idx=s_idx,
            batch_list=bl,
            storm_id=storm_id,
            device=device,
            modules_to_run=mods,
            K=args.K,
            ddim_steps=args.ddim_steps,
            dirs=dirs,
            save_bulletins=args.save_bulletins,
            temporal_mod=temporal_mod,
            spatial_mod=spatial_mod,
            analogue_mod=analogue_mod,
            fidelity_mod=fidelity_mod,
            c4_mod=c4_mod,
            c5_mod=c5_mod,
            c6_mod=c6_mod,
            interp=interp,
        )

        elapsed = time.time() - t0
        print(f"  [PhysXAI] Done in {elapsed:.1f}s")

        # Print quick summary
        if "fidelity" in result:
            print(f"  H-mean-F: {result['fidelity']['H_mean_F']:.3f}")
        if "spatial" in result:
            print(f"  Steering: {result['spatial'].get('steering_regime', 'N/A')}")
        if "analogue" in result:
            print(f"  Scenario: {result['analogue'].get('dominant_scenario', 'N/A')} "
                  f"(entropy={result['analogue'].get('uncertainty_entropy', 0):.2f})")
        if args.save_bulletins and "bulletin_path" in result:
            print(f"  Bulletin: {result['bulletin_path']}")

        all_results.append(result)

    total_elapsed = time.time() - t0_total
    print(f"\n[PhysXAI] Analysed {len(all_results)} storms in {total_elapsed:.1f}s "
          f"({total_elapsed/max(len(all_results),1):.1f}s/storm avg)")

    save_summary(all_results, dirs)

    # ─── Optionally run experiments ──────────────────────────────────────────
    if args.run_experiments:
        print("\n[PhysXAI] Running E1-E6 experiments ...")
        from Experiments.run_all_experiments import run_experiments

        class FakeArgs:
            experiments = ["E1", "E2", "E3", "E4", "E5", "E6"]
            output_dir  = dirs["experiments"]
            K           = args.K
            n_storms    = args.n_storms

        run_experiments(FakeArgs(), model, batches, device)

    print("\n[PhysXAI] Analysis complete. Results in:", args.output_dir)


if __name__ == "__main__":
    main()
