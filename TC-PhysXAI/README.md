# TC-PhysXAI

**Physics-Grounded Explainable AI for Probabilistic Tropical Cyclone Prediction via Conditional Flow Matching**

> Extends TC-XFlow (ESWA 2025) with 3 XAI modules and 6 novel contributions (C1–C6).

---

## Framework Overview

TC-PhysXAI provides post-hoc and intrinsic explanations for the TC-XFlow conditional flow matching (CFM) model at four levels:

| Module | Contributions | Purpose |
|--------|--------------|---------|
| **Module 1 — Temporal** | C1 | Climatological-baseline temporal perturbation → ΔCRPS per 6h window |
| **Module 2 — Spatial** | C2, C3 | IG on FNO3D + physical object detection + H-mean-F fidelity metric |
| **Module 3 — Scenario** | C4, C5, C6 | FM-path attribution, variance decomposition, counterfactual generation |

### Contributions

- **C1** Temporal attribution via climatological perturbation (ΔCRPS)
- **C2** Integrated Gradients on FNO3D + physical object detection (anticyclone, trough, monsoon, SST front)
- **C3** H-mean-F fidelity metric = 0.4·Precision-F + 0.4·Recall-F + 0.2·Uncertainty-F
- **C4** FreeFM integration-path attribution A_j = ∫₀¹ αⱼ dt
- **C5** Variance decomposition: V_total = V_noise + V_context (law of total variance)
- **C6** FM counterfactual via projected gradient on context space

---

## Directory Structure

```
TC-PhysXAI/
├── PhysXAI/                  # Core library
│   ├── __init__.py
│   ├── utils.py              # Shared utilities, ERA5 constants, CRPS, DTW
│   ├── physical_objects.py   # Synoptic object detection
│   ├── module1_temporal.py   # C1: Temporal attributor
│   ├── module2_spatial.py    # C2: Spatial IG attributor
│   ├── module3_analogue.py   # M3: DTW K-medoids + latent analogue retrieval
│   ├── c4_fm_attribution.py  # C4: FM integration-path attribution
│   ├── c5_variance_decomp.py # C5: Variance decomposition
│   ├── c6_counterfactual.py  # C6: Counterfactual generation
│   ├── fidelity_metric.py    # C3: H-mean-F fidelity benchmark
│   └── meteorological_interp.py  # Physical bulletin synthesis
├── Experiments/              # Reproducible experiments (E1–E6)
│   ├── E1_comparison_baselines.py
│   ├── E2_recurvature_attribution.py
│   ├── E3_rapid_intensification.py
│   ├── E4_multimodal_consistency.py
│   ├── E5_fidelity_benchmark.py
│   ├── E6_counterfactual_scenarios.py
│   └── run_all_experiments.py
├── Scripts/
│   └── run_physxai.py        # Main Kaggle entry point
├── requirements.txt
├── setup.py
└── README.md
```

---

## Kaggle Usage

### Step 1 — Train TC-XFlow (existing command, unchanged)

```bash
!python /kaggle/working/Tropical_Typhoon_Architecture/Scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v3_ot_fixed/seed0 \
    --seed 2 --gpu_num 0 --use_ot --ot_epsilon 0.05
```

### Step 2 — Run TC-PhysXAI Analysis

```bash
# Setup
!cp -r /kaggle/input/datasets/kaggle1234uitvn/tc-physxai/TC-PhysXAI /kaggle/working/TC-PhysXAI

# Full analysis on 20 test storms
!python /kaggle/working/TC-PhysXAI/Scripts/run_physxai.py \
    --checkpoint    /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root  /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir    /kaggle/working/TC-PhysXAI/results \
    --gpu           0 \
    --n_storms      20 \
    --K             30 \
    --run_experiments

# Single storm quick test (storm index 0)
!python /kaggle/working/TC-PhysXAI/Scripts/run_physxai.py \
    --checkpoint    /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root  /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir    /kaggle/working/TC-PhysXAI/results \
    --gpu           0 \
    --n_storms      1 \
    --K             10
```

### Step 3 — Run Individual Experiments

```bash
# E1: XAI Baseline Comparison
!python /kaggle/working/TC-PhysXAI/Experiments/E1_comparison_baselines.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results --n_storms 20 --K 30 --gpu 0

# E2: Recurvature Attribution
!python /kaggle/working/TC-PhysXAI/Experiments/E2_recurvature_attribution.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results --n_storms 20 --K 30 --gpu 0

# E3: Rapid Intensification & SST
!python /kaggle/working/TC-PhysXAI/Experiments/E3_rapid_intensification.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results --n_storms 20 --K 15 --gpu 0

# E4: Multi-Modal Consistency
!python /kaggle/working/TC-PhysXAI/Experiments/E4_multimodal_consistency.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results --n_storms 20 --K 20 --gpu 0

# E5: Fidelity Benchmark (ablation study)
!python /kaggle/working/TC-PhysXAI/Experiments/E5_fidelity_benchmark.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results --n_storms 15 --gpu 0

# E6: Counterfactual Scenarios
!python /kaggle/working/TC-PhysXAI/Experiments/E6_counterfactual_scenarios.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results --n_storms 15 --gpu 0

# ALL experiments at once
!python /kaggle/working/TC-PhysXAI/Experiments/run_all_experiments.py \
    --checkpoint /kaggle/working/runs/fm_v3_ot_fixed/seed0/best_model.pt \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir /kaggle/working/TC-PhysXAI/results \
    --gpu 0 --n_storms 20 --K 30 \
    --experiments E1 E2 E3 E4 E5 E6
```

---

## ERA5 Channels

| Index | Variable | Level |
|-------|----------|-------|
| 0 | Geopotential Height (GPH) | 200 hPa |
| 1 | Geopotential Height (GPH) | 500 hPa |
| 2 | Geopotential Height (GPH) | 850 hPa |
| 3 | Geopotential Height (GPH) | 925 hPa |
| 4 | U-wind | 200 hPa |
| 5 | U-wind | 500 hPa |
| 6 | U-wind | 850 hPa |
| 7 | U-wind | 925 hPa |
| 8 | V-wind | 200 hPa |
| 9 | V-wind | 500 hPa |
| 10 | V-wind | 850 hPa |
| 11 | V-wind | 925 hPa |
| 12 | SST | Surface |

---

## Citation

If you use TC-PhysXAI in your research, please cite:

```bibtex
@article{nguyen2025tcxflow,
  title   = {TC-XFlow: Tropical Cyclone Track Forecasting via Conditional Flow Matching with Physics-Constrained ERA5 Encoder},
  author  = {Nguyen, Hong Gam and others},
  journal = {Expert Systems with Applications},
  year    = {2025}
}
```

---

## Physical Interpretation Guide

TC-PhysXAI outputs are structured around four physical questions:

1. **WHEN?** (C1): Which 6h observation window most influences the forecast? High ΔCRPS at t−36h indicates the synoptic pattern 36h before analysis time is critical.

2. **WHERE?** (C2): Which ERA5 channel/level drives the model? `UPPER-TROPOSPHERE` regime → anticyclone ridge at 200/500 hPa steering. `SST-MODULATED` → sea surface thermal forcing dominant.

3. **HOW CERTAIN?** (C5, M3): What fraction of ensemble spread is predictability-limited (noise) vs. scenario-dependent (context)? High V_context → different synoptic scenarios possible; high V_noise → chaotic trajectory around a single attractor.

4. **WHAT IF?** (C6): What minimal synoptic change would alter the typhoon's track? Counterfactual shows the physically closest alternative reality.
