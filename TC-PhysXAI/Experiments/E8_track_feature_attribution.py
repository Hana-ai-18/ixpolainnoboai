"""
TC-PhysXAI — Experiment E8: Track Feature Attribution (C8)
===========================================================
Runs C8 Track Feature Attribution on test storms and produces:

  1. Attribution matrix CSV:
     Rows = meteorological track features (11 features)
     Cols = ERA5 channels (13) + time windows (8)
     Values = Δfeature (positive = that input amplifies the feature)

  2. Per-feature physical interpretation report

  3. Diagnostic plots:
     - Heatmap: feature × ERA5 channel attribution matrix
     - Heatmap: feature × time window attribution matrix
     - Radar chart: dominant channels per track regime
     - Bar chart: which feature is most sensitive to each channel

Output CSV columns per storm:
  typhoon_id, track_type,
  speed_mean, speed_peak, curvature_mean, curvature_total,
  spread_mean, spread_final, recurvature_prob, deceleration_prob,
  bifurcation_index, poleward_displacement, zonal_displacement,
  [feat]_top_channel, [feat]_top_channel_delta,
  [feat]_top_window, [feat]_top_window_delta,
  ...  (one set per feature)
"""

from __future__ import annotations

import sys
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    def _spearmanr(a: np.ndarray, b: np.ndarray):
        """Spearman ρ with scipy; falls back to numpy rank-correlation."""
        return _scipy_spearmanr(a, b)
except ImportError:
    def _spearmanr(a: np.ndarray, b: np.ndarray):  # type: ignore[misc]
        """Numpy-only Spearman ρ via rank inversion (Pearson on ranks)."""
        n = len(a)
        if n < 3:
            return 0.0, 1.0
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        d2 = float(((ra - rb) ** 2).sum())
        rho = 1.0 - 6.0 * d2 / max(n * (n ** 2 - 1), 1)
        return float(rho), None

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PhysXAI import (
    TrackFeatureAttributor,
    TrackFeatureExtractor,
    FeatureAttributionResult,
    TrackFeatures,
)
from PhysXAI.c8_track_feature_attribution import (
    _FEATURE_NAMES,
    _CH_NAMES,
    _WINDOW_LABELS,
    _PHYSICAL_FEATURE_NAMES,
    _UNCERTAINTY_FEATURE_NAMES,
    N_FEATURES,
)


# ─── Track type inference (same as E7) ───────────────────────────────────────

def _infer_track_type(batch_list: list) -> str:
    """Regime of the best-track future (storm 0): STRAIGHT_WEST | RECURVING_NE |
    POLEWARD_TRACK | STALLED | MIXED. Degree thresholds are heuristics (PhysXAI/geo.py);
    coordinates are (lon_n, lat_n), 1 unit = 5 degrees."""
    from PhysXAI.geo import classify_track_type
    pred = batch_list[1]  # [T_pred, B, 2]
    if not torch.is_tensor(pred) or pred.ndim < 3:
        return "MIXED"
    return classify_track_type(pred[:, 0, :].detach().cpu().numpy())


# ─── Single-storm runner ──────────────────────────────────────────────────────

def _run_one_storm(
    attributor: TrackFeatureAttributor,
    batch_list: list,
    storm_idx: int,
    typhoon_id: str,
    verbose: bool,
) -> Optional[dict]:
    try:
        result: FeatureAttributionResult = attributor.compute(
            batch_list, storm_idx=storm_idx, verbose=verbose
        )
    except Exception as e:
        warnings.warn(f"[E8] {typhoon_id}: failed — {e}")
        return None

    base = result.baseline_features
    row  = {"typhoon_id": typhoon_id}

    # Baseline feature values
    fvec = base.to_feature_vector()
    for fi, fname in enumerate(_FEATURE_NAMES):
        row[fname] = float(fvec[fi])

    # Top driver per feature
    for fname in _FEATURE_NAMES:
        ch_name, ch_val = result.top_channel_per_feature[fname]
        wn_name, wn_val = result.top_window_per_feature[fname]
        row[f"{fname}_top_ch"]    = ch_name
        row[f"{fname}_top_ch_d"]  = ch_val
        row[f"{fname}_top_win"]   = wn_name
        row[f"{fname}_top_win_d"] = wn_val

    # Faithfulness scores
    for fi, fname in enumerate(_FEATURE_NAMES):
        row[f"{fname}_aopc"] = float(result.faithfulness_scores[fi])

    row["_result"] = result   # kept temporarily for aggregation
    return row


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_e8(
    model,
    batches: list,
    device: torch.device,
    out_dir: str,
    K: int = 10,
    n_storms: int = 20,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run E8 on up to n_storms storms.

    Parameters
    ----------
    model     : trained TCFlowMatching
    batches   : list of batch_list items from DataLoader
    device    : torch device
    out_dir   : directory for outputs
    K         : ensemble size per perturbation (21 perturbations per storm)
    n_storms  : max storms to process

    Returns
    -------
    pd.DataFrame with per-storm rows
    """
    os.makedirs(out_dir, exist_ok=True)
    attributor = TrackFeatureAttributor(model, device, K=K)

    # Accumulate attribution matrices across storms
    ch_attr_accum    = np.zeros((N_FEATURES, 13),              dtype=np.float32)
    win_attr_accum   = np.zeros((N_FEATURES, len(_WINDOW_LABELS)), dtype=np.float32)
    faith_accum      = np.zeros(N_FEATURES,                   dtype=np.float32)
    n_valid          = 0

    rows = []
    batches_to_run = batches[:n_storms]

    print(f"[E8] Track Feature Attribution on {len(batches_to_run)} storms "
          f"(K={K}, {13 + len(_WINDOW_LABELS)} perturbations each)")
    t_start = time.time()

    for i, bl in enumerate(batches_to_run):
        typhoon_id = f"storm_{i:03d}"
        try:
            if isinstance(bl, (list, tuple)) and len(bl) > 12:
                meta = bl[12]
                if isinstance(meta, (list, tuple)) and meta:
                    typhoon_id = str(meta[0])
        except Exception:
            pass

        track_type = _infer_track_type(bl)

        row = _run_one_storm(attributor, bl, 0, typhoon_id, verbose)
        if row is None:
            continue

        result: FeatureAttributionResult = row.pop("_result")
        ch_attr_accum  += result.channel_attr_matrix
        win_attr_accum += result.window_attr_matrix
        faith_accum    += result.faithfulness_scores
        n_valid        += 1

        row["track_type"] = track_type
        rows.append(row)

        elapsed = time.time() - t_start
        eta     = elapsed / (i + 1) * (len(batches_to_run) - i - 1)
        rec_p   = row.get("recurvature_prob", 0.0)
        spd     = row.get("speed_mean", 0.0)
        print(f"  [{i+1:3d}/{len(batches_to_run)}] {typhoon_id:20s} "
              f"regime={track_type:15s}  "
              f"speed={spd:.4f}  recur={rec_p:.0%}  "
              f"ETA {eta/60:.1f}min")

    if not rows:
        print("[E8] No valid results.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Mean attribution matrices ─────────────────────────────────────────────
    ch_attr_mean   = ch_attr_accum  / max(n_valid, 1)  # [N_FEATURES, 13]
    win_attr_mean  = win_attr_accum / max(n_valid, 1)  # [N_FEATURES, 8]
    faith_mean     = faith_accum    / max(n_valid, 1)  # [N_FEATURES]

    # ── Save matrices ─────────────────────────────────────────────────────────
    ch_df  = pd.DataFrame(ch_attr_mean,  index=_FEATURE_NAMES, columns=_CH_NAMES)
    win_df = pd.DataFrame(win_attr_mean, index=_FEATURE_NAMES,
                          columns=_WINDOW_LABELS[:win_attr_mean.shape[1]])

    ch_df.to_csv(os.path.join(out_dir,  "E8_channel_attribution_matrix.csv"))
    win_df.to_csv(os.path.join(out_dir, "E8_window_attribution_matrix.csv"))
    df.to_csv(os.path.join(out_dir,     "E8_per_storm.csv"), index=False)

    # ── Text report ───────────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "E8_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TC-PhysXAI — E8: Track Feature Attribution Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Storms processed : {n_valid}\n")
        f.write(f"Track regimes    : {df['track_type'].value_counts().to_dict()}\n\n")

        f.write("[MEAN FEATURE VALUES ACROSS CASES]\n")
        for fname in _FEATURE_NAMES:
            if fname in df.columns:
                f.write(f"  {fname:30s}: {df[fname].mean():.4f} ± {df[fname].std():.4f}\n")

        f.write("\n[MEAN ERA5 CHANNEL ATTRIBUTION MATRIX]\n")
        f.write("(Δ = feature change when channel is removed; "
                "+ = channel amplifies feature)\n\n")
        f.write(ch_df.to_string(float_format=lambda x: f"{x:+.4f}"))
        f.write("\n\n")

        f.write("[MEAN TIME WINDOW ATTRIBUTION MATRIX]\n\n")
        f.write(win_df.to_string(float_format=lambda x: f"{x:+.4f}"))
        f.write("\n\n")

        # Top driver per feature with faithfulness (from mean matrix)
        f.write("[DOMINANT DRIVER PER FEATURE  (baseline = climatological mean, "
                "faithfulness = MoRF-AOPC)]\n")
        f.write(f"  {'Feature':30s}  {'Group':11s}  {'Top Channel':12s}  {'Δ':>8s}  "
                f"{'Top Window':10s}  {'Δ':>8s}  {'AOPC':>6s}\n")
        f.write("  " + "-" * 96 + "\n")
        for fi, fname in enumerate(_FEATURE_NAMES):
            best_c  = int(np.abs(ch_attr_mean[fi]).argmax())
            best_w  = int(np.abs(win_attr_mean[fi]).argmax())
            grp     = "physical   " if fname in _PHYSICAL_FEATURE_NAMES else "uncertainty"
            f.write(
                f"  {fname:30s}  {grp}  {_CH_NAMES[best_c]:12s}  "
                f"{ch_attr_mean[fi, best_c]:+8.4f}  "
                f"{_WINDOW_LABELS[best_w]:10s}  "
                f"{win_attr_mean[fi, best_w]:+8.4f}  "
                f"{faith_mean[fi]:6.3f}\n"
            )

        phys_idx    = [_FEATURE_NAMES.index(fn) for fn in _PHYSICAL_FEATURE_NAMES]
        uncert_idx  = [_FEATURE_NAMES.index(fn) for fn in _UNCERTAINTY_FEATURE_NAMES]
        f.write("\n[FAITHFULNESS SUMMARY]\n")
        f.write(f"  Physical features    mean AOPC = {faith_mean[phys_idx].mean():.4f}\n")
        f.write(f"  Uncertainty features mean AOPC = {faith_mean[uncert_idx].mean():.4f}\n")
        f.write(f"  Overall mean AOPC              = {faith_mean.mean():.4f}\n")
        f.write("  (AOPC > 0.10 = attribution is faithfully ranked; "
                "higher is better)\n")

        f.write("\n" + "=" * 70 + "\nEND OF E8 REPORT\n" + "=" * 70 + "\n")

    print(f"\n[E8] Results saved:")
    print(f"  {os.path.join(out_dir, 'E8_per_storm.csv')}")
    print(f"  {os.path.join(out_dir, 'E8_channel_attribution_matrix.csv')}")
    print(f"  {os.path.join(out_dir, 'E8_window_attribution_matrix.csv')}")
    print(f"  {report_path}")

    # Print dominant drivers inline
    phys_idx   = [_FEATURE_NAMES.index(fn) for fn in _PHYSICAL_FEATURE_NAMES]
    uncert_idx = [_FEATURE_NAMES.index(fn) for fn in _UNCERTAINTY_FEATURE_NAMES]
    print(f"\n[E8] MoRF-AOPC faithfulness: "
          f"physical={faith_mean[phys_idx].mean():.3f}  "
          f"uncertainty={faith_mean[uncert_idx].mean():.3f}  "
          f"overall={faith_mean.mean():.3f}")
    print("\n[E8] Dominant ERA5 driver per track feature:")
    print(f"  {'Feature':30s}  {'Group':11s}  {'Channel':12s}  {'Window':8s}  {'AOPC':>6s}")
    print("  " + "-" * 76)
    for fi, fname in enumerate(_FEATURE_NAMES):
        best_c = int(np.abs(ch_attr_mean[fi]).argmax())
        best_w = int(np.abs(win_attr_mean[fi]).argmax())
        grp    = "physical   " if fname in _PHYSICAL_FEATURE_NAMES else "uncertainty"
        print(f"  {fname:30s}  {grp}  {_CH_NAMES[best_c]:12s}  "
              f"{_WINDOW_LABELS[best_w]:8s}  {faith_mean[fi]:.3f}")

    return df


# ─── K-sensitivity analysis ───────────────────────────────────────────────────

def run_e8_k_sensitivity(
    model,
    batches: list,
    device: torch.device,
    out_dir: str,
    K_values: Tuple[int, ...] = (5, 10, 20),
    n_storms: int = 5,
) -> pd.DataFrame:
    """Ablation on ensemble size K.

    For each K in K_values, runs C8 channel attribution on n_storms storms,
    then computes Spearman ρ of per-feature attribution rankings relative to
    the largest K (used as reference).  High ρ (> 0.85) means smaller K is
    sufficient; low ρ reveals instability.

    Saves:
      E8_k_sensitivity.csv   — per-feature Spearman ρ table
      E8_k_sensitivity.png   — bar chart of ρ vs K
    """
    os.makedirs(out_dir, exist_ok=True)
    storm_subset = batches[:n_storms]
    ref_K        = max(K_values)

    # Maps  K → [N_FEATURES, 13] mean attribution matrix
    attr_by_K: dict[int, np.ndarray] = {}

    for K in sorted(set(K_values)):
        attributor = TrackFeatureAttributor(model, device, K=K)
        ch_accum   = np.zeros((N_FEATURES, 13), dtype=np.float32)
        n_ok       = 0

        print(f"[E8-Ksens] K={K}: running {len(storm_subset)} storms …")
        for i, bl in enumerate(storm_subset):
            tid = f"storm_{i:03d}"
            try:
                result = attributor.compute(bl, storm_idx=0, verbose=False)
                ch_accum += result.channel_attr_matrix
                n_ok     += 1
            except Exception as exc:
                warnings.warn(f"[E8-Ksens] {tid} K={K}: {exc}")

        if n_ok == 0:
            warnings.warn(f"[E8-Ksens] K={K}: no valid storms — skipped.")
            continue
        attr_by_K[K] = ch_accum / n_ok
        print(f"[E8-Ksens] K={K}: done ({n_ok}/{len(storm_subset)} valid)")

    if ref_K not in attr_by_K or len(attr_by_K) < 2:
        print("[E8-Ksens] Insufficient results — skipping K-sensitivity report.")
        return pd.DataFrame()

    ref_mat  = attr_by_K[ref_K]   # [N_FEATURES, 13]
    other_Ks = sorted(k for k in attr_by_K if k != ref_K)

    # Per-feature Spearman ρ: compare |attr| ranking vectors
    rho_table: dict[str, dict] = {}
    for fi, fname in enumerate(_FEATURE_NAMES):
        ref_rank = np.abs(ref_mat[fi])           # [13]
        entry    = {"feature": fname}
        for K in other_Ks:
            rho, pval = _spearmanr(ref_rank, np.abs(attr_by_K[K][fi]))
            entry[f"rho_K{K}"]  = float(round(rho,  4))
            entry[f"pval_K{K}"] = float(round(pval, 4))
        rho_table[fname] = entry

    rho_df = pd.DataFrame(list(rho_table.values()))
    rho_df.to_csv(os.path.join(out_dir, "E8_k_sensitivity.csv"), index=False)

    # Summary
    print("\n[E8-Ksens] Spearman ρ (channel attribution ranking vs reference K="
          f"{ref_K})")
    header = f"  {'Feature':30s}" + "".join(
        f"  K={k:>2d}  pval" for k in other_Ks
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fname, entry in rho_table.items():
        row_str = f"  {fname:30s}"
        for K in other_Ks:
            r = entry[f"rho_K{K}"]
            p = entry[f"pval_K{K}"]
            row_str += f"  {r:+.3f}  {p:.3f}"
        print(row_str)

    # Overall mean ρ per K
    overall_rho = {}
    for K in other_Ks:
        overall_rho[K] = float(np.mean([rho_table[fn][f"rho_K{K}"]
                                        for fn in _FEATURE_NAMES
                                        if fn in rho_table]))
    print("\n[E8-Ksens] Mean Spearman ρ across all features:")
    for K, r in overall_rho.items():
        stable = "✔ STABLE" if r >= 0.85 else ("△ MARGINAL" if r >= 0.70 else "✘ UNSTABLE")
        print(f"  K={K:2d} vs K={ref_K}: ρ={r:.3f}  {stable}")

    # Append summary to E8 report
    report_path = os.path.join(out_dir, "E8_report.txt")
    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n[K-SENSITIVITY ANALYSIS]\n")
            f.write(f"  Reference K = {ref_K}\n")
            for K, r in overall_rho.items():
                stable = "STABLE" if r >= 0.85 else ("MARGINAL" if r >= 0.70 else "UNSTABLE")
                f.write(f"  K={K:2d}: mean Spearman ρ={r:.4f}  [{stable}]\n")
            f.write("  (ρ ≥ 0.85 = attribution stable at this ensemble size)\n\n")
            f.write("  Per-feature ρ table:\n")
            f.write(f"  {'Feature':30s}")
            for K in other_Ks:
                f.write(f"  rho_K{K:02d}  pval")
            f.write("\n  " + "-" * 70 + "\n")
            for fname, entry in rho_table.items():
                f.write(f"  {fname:30s}")
                for K in other_Ks:
                    r = entry[f"rho_K{K}"]
                    p = entry[f"pval_K{K}"]
                    f.write(f"  {r:+.4f}  {p:.3f}")
                f.write("\n")
    except Exception:
        pass

    # Plot
    _plot_k_sensitivity(rho_df, other_Ks, ref_K, out_dir)

    return rho_df


def _plot_k_sensitivity(
    rho_df: pd.DataFrame,
    other_Ks: List[int],
    ref_K:    int,
    out_dir:  str,
) -> None:
    """Bar chart of Spearman ρ per feature and K value."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_feats = len(_FEATURE_NAMES)
    x       = np.arange(n_feats)
    width   = 0.35
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, K in enumerate(other_Ks):
        col  = f"rho_K{K}"
        rhos = [float(rho_df.loc[rho_df["feature"] == fn, col].values[0])
                if col in rho_df.columns and fn in rho_df["feature"].values
                else 0.0
                for fn in _FEATURE_NAMES]
        offset = (i - len(other_Ks) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, rhos, width * 0.9,
                        label=f"K={K}", color=colors[i % len(colors)], alpha=0.85)

    ax.axhline(0.85, color="red",    linewidth=1.2, linestyle="--",
               label="Stable threshold (ρ=0.85)")
    ax.axhline(0.70, color="orange", linewidth=1.0, linestyle=":",
               label="Marginal threshold (ρ=0.70)")
    ax.set_xticks(x)
    ax.set_xticklabels([fn.replace("_", "\n") for fn in _FEATURE_NAMES],
                       fontsize=7)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel(f"Spearman ρ vs K={ref_K}", fontsize=10)
    ax.set_title(f"C8 — K-Sensitivity: Attribution Rank Stability vs K={ref_K}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "E8_k_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_e8(df: pd.DataFrame, out_dir: str) -> None:
    """Generate E8 diagnostic plots.

    Produces:
      E8_channel_heatmap.png  : feature × ERA5 channel attribution heatmap
      E8_window_heatmap.png   : feature × time window attribution heatmap
      E8_regime_profile.png   : mean feature values grouped by track regime
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("[E8] matplotlib not available — skipping plots.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ── Reload attribution matrices ───────────────────────────────────────────
    ch_path  = os.path.join(out_dir, "E8_channel_attribution_matrix.csv")
    win_path = os.path.join(out_dir, "E8_window_attribution_matrix.csv")
    if not (os.path.exists(ch_path) and os.path.exists(win_path)):
        print("[E8] Attribution matrices not found — run run_e8() first.")
        return

    ch_mat  = pd.read_csv(ch_path,  index_col=0).values   # [N_FEATURES, 13]
    win_mat = pd.read_csv(win_path, index_col=0).values   # [N_FEATURES, 8]

    # Short feature labels
    short_labels = [
        "speed_mean", "speed_peak", "curvature_mean", "curvature_total",
        "spread_mean", "spread_final", "P(recurv)", "P(decel)",
        "bifurcation", "poleward_disp", "zonal_disp",
    ]

    def _heatmap(matrix, row_labels, col_labels, title, path, figsize=(12, 6)):
        vmax = np.abs(matrix).max() + 1e-6
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        plt.colorbar(im, ax=ax, label="Δ feature (+ = amplifies)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(j, i, f"{matrix[i, j]:+.3f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if abs(matrix[i, j]) > vmax * 0.5 else "black")
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    _heatmap(ch_mat,  short_labels, _CH_NAMES,
             "C8 — Track Feature × ERA5 Channel Attribution",
             os.path.join(out_dir, "E8_channel_heatmap.png"),
             figsize=(14, 7))

    _heatmap(win_mat, short_labels,
             _WINDOW_LABELS[:win_mat.shape[1]],
             "C8 — Track Feature × Time Window Attribution",
             os.path.join(out_dir, "E8_window_heatmap.png"),
             figsize=(10, 7))

    # ── Regime profile ────────────────────────────────────────────────────────
    if "track_type" in df.columns:
        regimes = df["track_type"].unique()
        feat_cols = [f for f in _FEATURE_NAMES if f in df.columns]

        fig, axes = plt.subplots(
            1, len(feat_cols), figsize=(3 * len(feat_cols), 4), sharey=False
        )
        if len(feat_cols) == 1:
            axes = [axes]

        for ax, fname in zip(axes, feat_cols):
            regime_means = [df[df["track_type"] == r][fname].mean() for r in regimes]
            colors = plt.cm.Set2(np.linspace(0, 1, len(regimes)))
            ax.bar(range(len(regimes)), regime_means, color=colors)
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=7)
            ax.set_title(fname.replace("_", "\n"), fontsize=7)
            ax.axhline(0, color="gray", linewidth=0.5)

        fig.suptitle("C8 — Mean Track Features by Regime", fontsize=11, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, "E8_regime_profile.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
