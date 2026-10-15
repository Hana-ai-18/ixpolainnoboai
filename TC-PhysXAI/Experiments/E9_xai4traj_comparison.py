"""
TC-PhysXAI — Experiment E9: comparison with the XAI4Traj baseline across models
================================================================================
Produces a table in the style of Tables 2-3 of Tung et al. (Neurocomputing 2025,
"XAI4Traj"): rows = segmentation x perturbation, columns = black-box model,
cells = Recall R / Precision P / Harmonic mean H, using the SAME evaluation
protocol (see PhysXAI/xai4traj_baseline.py for what is kept / adapted).

Rankings compared on the SAME segments and the SAME perturbation family
------------------------------------------------------------------------
  xai4traj     surrogate theta of the official method (the baseline of the paper)
  physx_dcrps  TC-PhysXAI C7 signal: delta-CRPS (vs best track) when the segment
               is neutral-masked (uses ground truth, like C7)
  physx_shift  ground-truth-free variant: displacement of the ensemble-mean
               prediction when the segment is neutral-masked
  random       expected AP of a random ranking (lower bound, sanity)
  oracle       ranking by the true single-segment displacement (upper bound)
and two extra segmentations: `phys_abc` (fixed time phases 0-3/4-5/6-7, a simple
reference) and `phys_regime` (data-driven motion-regime change points: heading / speed).

Usage
-----
  # 1) see which checkpoints exist and what they are
  python E9_xai4traj_comparison.py --scan /kaggle/input

  # 2) quick test (3 storms)
  python E9_xai4traj_comparison.py --quick --dataset_root ... --output_dir ... \
      --model fm:/kaggle/input/.../best_model_fm_seed0_new_ver.pth \
      --model st_trans:/path/st_trans.pth --model lstm:/path/lstm.pth

  # 3) full run: remove --quick, set --n_storms 20
Kinds: fm, st_trans, lstm, gru, rnn, mmstn, physdiff, tcdiffuser, mgtcf
(or `--auto DIR` to detect kinds from the checkpoints' `model_type`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TC_XFLOW_ROOT = Path("/kaggle/working/Tropical_Typhoon_Architecture")
sys.path.insert(0, str(TC_XFLOW_ROOT))

from PhysXAI import xai4traj_baseline as x4  # noqa: E402

SEGS = ("mdl", "rdp", "sliding", "phys_abc", "phys_regime")
RANKINGS = ("xai4traj", "physx_dcrps", "physx_shift", "random", "oracle")


# ─── Checkpoint discovery ───────────────────────────────────────────────────

def _kind_from_ckpt(ck: dict) -> str | None:
    mt = str(ck.get("model_type", "")).lower()
    table = {"lstm": "lstm", "gru": "gru", "rnn": "rnn", "mgtcf": "mgtcf",
             "physdiff": "physdiff", "tcdiffuser": "tcdiffuser", "mmstn": "mmstn",
             "st_trans": "st_trans", "sttrans": "st_trans"}
    if mt in table:
        return table[mt]
    if "model" in ck and ("model_cfg" in ck or "ema" in ck):
        return "fm"
    return None


def scan_checkpoints(root: str) -> list[tuple[str, str | None, dict]]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.pth"), recursive=True)):
        try:
            ck = torch.load(p, map_location="cpu")
            if not isinstance(ck, dict):
                out.append((p, None, {"note": "not a dict"}))
                continue
            info = {"model_type": ck.get("model_type"), "seed": ck.get("seed"),
                    "keys": sorted(ck.keys())[:12],
                    "has_model_cfg": bool(ck.get("model_cfg"))}
            out.append((p, _kind_from_ckpt(ck), info))
        except Exception as e:  # noqa: BLE001
            out.append((p, None, {"error": repr(e)}))
    return out


# ─── Model loading (re-uses the repo's own loaders) ─────────────────────────

def load_model(kind: str, path: str, device: torch.device):
    import evaluate_multi_model as emm
    if kind == "fm":
        from PhysXAI.model_io import load_tcxflow
        return load_tcxflow(path, device)
    if kind == "st_trans":
        return emm.load_st_trans(path, device)[0]
    if kind in ("lstm", "gru", "rnn"):
        return emm.load_paper_baseline(path, kind, device)[0]
    if kind == "mmstn":
        return emm.load_mmstn(path, device)[0]
    if kind == "physdiff":
        return emm.load_phys_diff(path, device)[0]
    if kind == "tcdiffuser":
        return emm.load_tc_diffuser(path, device)[0]
    if kind == "mgtcf":
        from Model.mgtcf_model import MGTCFModel
        ck = torch.load(path, map_location="cpu")
        cfg = ck.get("model_cfg")
        if not cfg:
            raise RuntimeError("MGTCF checkpoint has no model_cfg — cannot rebuild the architecture")
        model = MGTCFModel(**cfg).to(device)
        missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
        if missing or unexpected:
            print(f"  ⚠ MGTCF load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        return model.eval()
    raise ValueError(f"unknown model kind '{kind}'")


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


# ─── Black-box wrapper (all models share sample(batch_list, num_ensemble=K)) ─

class BlackBox:
    """Ensemble-mean prediction in degrees. Same seed before every call, so a
    stochastic model sees the same noise for every perturbed input (common
    random numbers): differences come from the input perturbation only."""

    def __init__(self, model, device, K: int, seed: int):
        self.model, self.device, self.K, self.seed = model, device, K, seed
        self.model.eval()
        self.n_calls = 0

    @torch.no_grad()
    def ensemble_deg(self, bl) -> np.ndarray:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        out = self.model.sample(bl, num_ensemble=self.K)
        self.n_calls += 1
        all_traj = out[2]
        if all_traj.dim() == 3:               # [T, B, 2] -> [1, T, B, 2]
            all_traj = all_traj.unsqueeze(0)
        arr = all_traj[:, :, 0, :2].float().cpu().numpy()      # [K, T, 2] normalised
        return x4.norm_to_deg(arr)

    @staticmethod
    def mean_displacement_km(ens_a: np.ndarray, ens_b: np.ndarray) -> float:
        T = min(ens_a.shape[1], ens_b.shape[1])
        return float(x4.haversine_km(ens_a.mean(0)[:T], ens_b.mean(0)[:T]).mean())


def crps_km(ens_deg: np.ndarray, gt_deg: np.ndarray) -> float:
    """CRPS in km for [K,T,2] ensemble vs [T,2] truth (K=1 -> MAE)."""
    T = min(ens_deg.shape[1], gt_deg.shape[0])
    e, g = ens_deg[:, :T], gt_deg[:T]
    term1 = x4.haversine_km(e, g[None]).mean(0)                          # [T]
    K = e.shape[0]
    if K > 1:
        d = x4.haversine_km(e[:, None], e[None, :]).mean((0, 1))         # [T]
    else:
        d = np.zeros(T)
    return float((term1 - 0.5 * d).mean())


# ─── Batch editing ──────────────────────────────────────────────────────────

def with_obs(bl, obs_norm: np.ndarray):
    """Copy of the batch with new observed coordinates (idx 0) and the matching
    relative displacements (idx 2, rel[0]=0 like the dataset)."""
    new = list(bl)
    o = bl[0].clone()
    o[:, 0, :2] = torch.as_tensor(obs_norm, dtype=o.dtype, device=o.device)
    new[0] = o
    r = bl[2].clone()
    r[0, 0, :2] = 0.0
    r[1:, 0, :2] = o[1:, 0, :2] - o[:-1, 0, :2]
    new[2] = r
    return new


def _interp_time(arr: np.ndarray, steps: list[int]) -> np.ndarray:
    """Replace rows `steps` by linear interpolation of the remaining rows."""
    T = arr.shape[0]
    keep = [i for i in range(T) if i not in set(steps)]
    out = arr.copy()
    if not keep:
        return out
    for c in range(arr.shape[1]):
        out[steps, c] = np.interp(steps, keep, arr[keep, c])
    return out


def masked_batch(bl, steps: list[int]):
    """Neutral masking of observation steps: coordinates and Me are linearly
    interpolated from the unmasked steps; ERA5 slices are set to the
    climatological mean (0 in normalised space)."""
    obs = bl[0][:, 0, :2].float().cpu().numpy()
    new = with_obs(bl, _interp_time(obs, steps))
    if torch.is_tensor(bl[7]) and bl[7].dim() == 3:
        me = bl[7].clone()
        me_np = me[:, 0, :].float().cpu().numpy()
        me[:, 0, :] = torch.as_tensor(_interp_time(me_np, steps), dtype=me.dtype, device=me.device)
        new[7] = me
    if torch.is_tensor(bl[11]):
        img = bl[11].clone()
        for s in steps:
            if s < img.shape[2]:
                img[:, :, s] = 0.0
        new[11] = img
    return new


# ─── Per-storm experiment ───────────────────────────────────────────────────

def explain_storm(bb: BlackBox, bl, args, storm_idx: int, taus: list[float]) -> list[dict]:
    obs_norm = bl[0][:, 0, :2].float().cpu().numpy()
    gt_deg = x4.norm_to_deg(bl[1][:, 0, :2].float().cpu().numpy())
    obs_deg = x4.norm_to_deg(obs_norm)
    frame = x4.LocalFrame(obs_deg.mean(0))
    x_km = frame.to_km(obs_deg)

    ens0 = bb.ensemble_deg(bl)
    crps0 = crps_km(ens0, gt_deg)
    cfg = x4.PerturbConfig(args.gaussian_sigma_km, args.scale_factor, np.deg2rad(args.rotation_deg))
    seg_params = {"mdl": dict(epsilon_km=args.mdl_eps_km),
                  "rdp": dict(epsilon_km=args.rdp_eps_km),
                  "sliding": dict(window=args.slide_window, step=args.slide_step),
                  "phys_abc": {}, "phys_regime": {}}

    def delta_fn(pts_km: np.ndarray) -> float:
        changed = np.abs(pts_km - x_km).max(1) > 1e-9
        new_norm = obs_norm.copy()
        new_norm[changed] = x4.deg_to_norm(frame.to_deg(pts_km[changed]))
        ens = bb.ensemble_deg(with_obs(bl, new_norm))
        return bb.mean_displacement_km(ens, ens0)

    rows: list[dict] = []
    for seg_name in SEGS:
        segments = x4.SEGMENTERS[seg_name](x_km, **seg_params[seg_name])
        n_seg = len(segments)
        # TC-PhysXAI rankings: neutral masking of each segment (once per segmentation)
        dcrps = np.zeros(n_seg)
        shift = np.zeros(n_seg)
        for i, idx in enumerate(segments):
            ens_m = bb.ensemble_deg(masked_batch(bl, [int(s) for s in idx]))
            dcrps[i] = crps_km(ens_m, gt_deg) - crps0
            shift[i] = bb.mean_displacement_km(ens_m, ens0)

        for p_i, pert in enumerate(x4.PERTURBATIONS):
            rng = np.random.default_rng(args.seed * 100003 + storm_idx * 31 + p_i)
            noisy = x4.make_noisy(x_km, pert, rng, cfg)
            rng_z = np.random.default_rng(args.seed * 7919 + storm_idx)
            d_eval, thetas, n_z = x4.run_config(
                x_km, segments, noisy, delta_fn, taus, args.n_perturb, rng_z, args.surrogate)
            for tau in taus:
                flags = d_eval > tau
                base = dict(storm=storm_idx, seg=seg_name, pert=pert, tau=tau,
                            n_seg=n_seg, n_flips=int(flags.sum()), delta_max=float(d_eval.max()))
                th = thetas[tau]
                if th is None:
                    ch, ap = 0, 0.0
                else:
                    ch, ap = x4.storm_metrics(np.argsort(-np.abs(th), kind="stable"), flags)
                rows.append({**base, "ranking": "xai4traj", "change": ch, "ap": ap})
                for name, score in (("physx_dcrps", dcrps), ("physx_shift", shift), ("oracle", d_eval)):
                    ch, ap = x4.storm_metrics(np.argsort(-score, kind="stable"), flags)
                    rows.append({**base, "ranking": name, "change": ch, "ap": ap})
                ch, ap = x4.expected_random_ap(flags, np.random.default_rng(storm_idx + 17))
                rows.append({**base, "ranking": "random", "change": ch, "ap": ap})
    return rows


# ─── Reporting ──────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.3f}"


def agg_df(df: pd.DataFrame) -> dict:
    return x4.aggregate(df["change"].values, df["ap"].values)


def paper_table(df: pd.DataFrame, tau: float, ranking: str = "xai4traj") -> pd.DataFrame:
    sub = df[(df.tau == tau) & (df.ranking == ranking)]
    recs = []
    for (seg, pert), g in sub.groupby(["seg", "pert"], sort=False):
        row = {"Segment": seg, "Perturb": pert}
        for label, gm in g.groupby("label", sort=False):
            a = agg_df(gm)
            row[f"{label}|R"], row[f"{label}|P"], row[f"{label}|H"] = a["R"], a["P"], a["H"]
        recs.append(row)
    return pd.DataFrame(recs)


def table_markdown(t: pd.DataFrame) -> str:
    labels = list(dict.fromkeys(c.split("|")[0] for c in t.columns if "|" in c))
    head1 = "| Segment | Perturb | " + " | ".join(f"{l} R | {l} P | {l} H" for l in labels) + " |"
    sep = "|" + "---|" * (2 + 3 * len(labels))
    lines = [head1, sep]
    for _, r in t.iterrows():
        cells = [str(r["Segment"]), str(r["Perturb"])]
        for l in labels:
            cells += [_fmt(r[f"{l}|R"]), _fmt(r[f"{l}|P"]), _fmt(r[f"{l}|H"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_latex(t: pd.DataFrame, caption: str) -> str:
    labels = list(dict.fromkeys(c.split("|")[0] for c in t.columns if "|" in c))
    cols = "ll" + "ccc" * len(labels)
    out = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}",
           f"\\begin{{tabular}}{{{cols}}}", "\\hline"]
    out.append("Segment & Perturb & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{l}}}" for l in labels) + " \\\\")
    out.append(" & & " + " & ".join("R & P & H" for _ in labels) + " \\\\ \\hline")
    for _, r in t.iterrows():
        cells = [str(r["Segment"]), str(r["Perturb"])]
        for l in labels:
            cells += [_fmt(r[f"{l}|R"]), _fmt(r[f"{l}|P"]), _fmt(r[f"{l}|H"])]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def ranking_summary(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Mean AP-with-change (P) and H per model x ranking, pooled over seg x pert."""
    recs = []
    for (label, ranking), g in df[df.tau == tau].groupby(["label", "ranking"], sort=False):
        a = agg_df(g)
        recs.append({"model": label, "ranking": ranking, "R": a["R"], "P": a["P"], "H": a["H"],
                     "mean_AP_all_cases": float(g["ap"].mean())})
    return pd.DataFrame(recs)


def paired_bootstrap(df: pd.DataFrame, tau: float, a: str, b: str, n_boot: int = 2000,
                     seed: int = 0) -> pd.DataFrame:
    """95% CI of mean(AP_a - AP_b) over storms (AP=0 when no change), per model."""
    rng = np.random.default_rng(seed)
    recs = []
    for label, g in df[df.tau == tau].groupby("label", sort=False):
        pa = g[g.ranking == a].groupby(["ckpt", "storm"])["ap"].mean()
        pb = g[g.ranking == b].groupby(["ckpt", "storm"])["ap"].mean()
        d = (pa - pb).dropna().values
        if len(d) < 2:
            continue
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        recs.append({"model": label, "A": a, "B": b, "mean_diff": float(d.mean()),
                     "ci95_lo": float(lo), "ci95_hi": float(hi), "n_storms": int(len(d))})
    return pd.DataFrame(recs)


def write_reports(df: pd.DataFrame, out_dir: str, taus: list[float], meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "E9_records.csv"), index=False)
    summary = {"meta": meta, "by_tau": {}}
    for tau in taus:
        tag = f"tau{int(tau)}"
        t = paper_table(df, tau, "xai4traj")
        t.to_csv(os.path.join(out_dir, f"E9_table_xai4traj_{tag}.csv"), index=False)
        with open(os.path.join(out_dir, f"E9_table_xai4traj_{tag}.md"), "w") as f:
            f.write(table_markdown(t))
        with open(os.path.join(out_dir, f"E9_table_xai4traj_{tag}.tex"), "w") as f:
            f.write(table_latex(t, f"XAI4Traj protocol on TC track models (change threshold {tau:g} km)."))
        rs = ranking_summary(df, tau)
        rs.to_csv(os.path.join(out_dir, f"E9_ranking_summary_{tag}.csv"), index=False)
        bs = pd.concat([paired_bootstrap(df, tau, "physx_dcrps", "xai4traj"),
                        paired_bootstrap(df, tau, "physx_shift", "xai4traj"),
                        paired_bootstrap(df, tau, "xai4traj", "random")], ignore_index=True)
        bs.to_csv(os.path.join(out_dir, f"E9_paired_bootstrap_{tag}.csv"), index=False)
        summary["by_tau"][tag] = {"ranking_summary": rs.to_dict("records"),
                                  "paired_bootstrap": bs.to_dict("records")}
        print(f"\n=== tau = {tau:g} km — ranking summary (pooled over segmentations x perturbations) ===")
        print(rs.round(3).to_string(index=False))
        if len(bs):
            print("\npaired bootstrap of mean AP difference (A - B):")
            print(bs.round(3).to_string(index=False))
    with open(os.path.join(out_dir, "E9_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)


# ─── Main ───────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan", default=None, help="list checkpoints under DIR and exit")
    p.add_argument("--auto", default=None, help="detect model kinds from all *.pth under DIR")
    p.add_argument("--model", action="append", default=[], help="KIND:PATH[:LABEL] (repeatable)")
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--output_dir", default="/kaggle/working/TC-PhysXAI/results/E9")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_storms", type=int, default=20)
    p.add_argument("--K", type=int, default=10, help="ensemble size for stochastic models")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--taus", default="10,25,50", help="'prediction changed' thresholds in km")
    p.add_argument("--n_perturb", type=int, default=64, help="max perturbation vectors per surrogate")
    p.add_argument("--surrogate", choices=["ridge", "logistic"], default="ridge")
    p.add_argument("--gaussian_sigma_km", type=float, default=15.0)
    p.add_argument("--scale_factor", type=float, default=1.2)
    p.add_argument("--rotation_deg", type=float, default=10.0)
    p.add_argument("--rdp_eps_km", type=float, default=20.0)
    p.add_argument("--mdl_eps_km", type=float, default=30.0)
    p.add_argument("--slide_window", type=int, default=2)
    p.add_argument("--slide_step", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="3 storms, 16 perturbation vectors")
    return p.parse_args()


def main():
    args = get_args()
    if args.scan:
        print(f"{'kind':<11} path  |  info")
        for p, kind, info in scan_checkpoints(args.scan):
            print(f"{str(kind):<11} {p}\n{'':<11} {info}")
        return
    if args.quick:
        args.n_storms, args.n_perturb = 3, 16
    if not args.dataset_root:
        sys.exit("--dataset_root is required")

    specs = []                                   # (kind, path, label)
    for m in args.model:
        parts = m.split(":")
        specs.append((parts[0], parts[1], parts[2] if len(parts) > 2 else parts[0]))
    if args.auto:
        for p, kind, info in scan_checkpoints(args.auto):
            if kind is None:
                print(f"[auto] skipped (unknown kind): {p}  {info}")
            else:
                specs.append((kind, p, kind))
    if not specs:
        sys.exit("No model given: use --model KIND:PATH or --auto DIR (see --scan DIR)")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    taus = [float(t) for t in args.taus.split(",")]
    print("=" * 70, f"\nE9 XAI4Traj comparison | device {device} | storms {args.n_storms} | "
          f"K {args.K} | n_perturb {args.n_perturb} | taus {taus} | surrogate {args.surrogate}\n", "=" * 70)

    from torch.utils.data import DataLoader
    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    ds = TrajectoryDataset(args.dataset_root, split="test")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=seq_collate, num_workers=2)
    batches = []
    for b in loader:
        if b[0].shape[1] != 1:
            print(f"[data] skipping batch with B={b[0].shape[1]} (E9 needs one storm per batch)")
            continue
        batches.append(b)
        if len(batches) >= args.n_storms:
            break
    print(f"[data] {len(ds)} test storms -> using {len(batches)}")

    all_rows: list[dict] = []
    failures: dict[str, int] = {}
    for kind, path, label in specs:
        ckpt_name = os.path.basename(path)
        print(f"\n{'=' * 70}\n[{label}] loading {kind}: {path}\n{'=' * 70}")
        try:
            model = load_model(kind, path, device)
        except Exception:  # noqa: BLE001
            print(f"[{label}] LOAD FAILED — model skipped:\n{traceback.format_exc()}")
            failures[f"{label}/{ckpt_name}"] = -1
            continue
        bb = BlackBox(model, device, args.K, args.seed)
        n_fail, t0 = 0, time.time()
        for s, batch in enumerate(batches):
            try:
                bl = to_device(list(batch), device)
                rows = explain_storm(bb, bl, args, s, taus)
                for r in rows:
                    r.update(label=label, kind=kind, ckpt=ckpt_name)
                all_rows.extend(rows)
                print(f"  [{label}] storm {s + 1}/{len(batches)} ok  "
                      f"({bb.n_calls} model calls, {time.time() - t0:.0f}s)")
            except Exception:  # noqa: BLE001
                n_fail += 1
                print(f"  [{label}] storm {s + 1} FAILED:\n{traceback.format_exc()}")
        failures[f"{label}/{ckpt_name}"] = n_fail
        del model, bb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if all_rows:                                   # incremental save (Kaggle timeouts)
            os.makedirs(args.output_dir, exist_ok=True)
            pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, "E9_records_partial.csv"), index=False)

    print("\n" + "=" * 70 + "\nFAILURE REPORT (storms failed per checkpoint; -1 = model failed to load)")
    for k, v in failures.items():
        print(f"  {k}: {v}")
    if not all_rows:
        sys.exit("E9: no results at all — every model/storm failed (see tracebacks above)")

    df = pd.DataFrame(all_rows)
    meta = {"n_storms": len(batches), "K": args.K, "n_perturb": args.n_perturb, "taus": taus,
            "surrogate": args.surrogate, "failures": failures,
            "models": [{"kind": k, "path": p, "label": l} for k, p, l in specs],
            "mean_n_segments": df.groupby("seg")["n_seg"].mean().round(2).to_dict(),
            "perturbation": {"gaussian_sigma_km": args.gaussian_sigma_km,
                             "scale_factor": args.scale_factor, "rotation_deg": args.rotation_deg},
            "segmentation": {"rdp_eps_km": args.rdp_eps_km, "mdl_eps_km": args.mdl_eps_km,
                             "slide_window": args.slide_window, "slide_step": args.slide_step}}
    print("\nmean #segments per segmentation:", meta["mean_n_segments"])
    write_reports(df, args.output_dir, taus, meta)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
