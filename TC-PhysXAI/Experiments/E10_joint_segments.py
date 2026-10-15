"""
TC-PhysXAI — Experiment E10: PHYSICAL-UNIT explanations (phase x level x variable) of the
forecast DISTRIBUTION, with deletion faithfulness and a steering-flow physics check
=====================================================================================
One-sentence idea (to be compared with XAI4Traj, which explains with trajectory segments):

    We explain a typhoon-track forecast with physically meaningful units that span the
    track AND the atmosphere - (time segment of the storm's motion regime) x (TRACK or an
    ERA5 level/variable group) - and measure how removing a unit changes the forecast
    DISTRIBUTION (position along / across track, spread, per-horizon shift, energy distance).

What is measured
----------------
Units       time segments (default `phys_regime`: change points of heading/speed of the
            OBSERVED track) x {TRACK, ERA5 groups}. ERA5 groups: `level` (200 / 500 / 850-925
            hPa / SST), `var` (GPH / U / V / SST) or `grid` (every channel: 4 levels x
            3 variables + SST = full "level x variable" product; no greedy ranking then).
Phase       label of each STORM = motion class of the MODEL'S OWN clean forecast
            (geo.classify_track_type) - never the ground-truth future (no leakage).
            The ground-truth class is kept in a separate column (`regime_gt`) for reference.
Operators   neutral : track/Me interpolated from the other steps, ERA5 = dataset mean (0)
            noise   : Gaussian (sigma km on the track, N(0,1) on normalised Me / ERA5)
            swap    : ERA5 of the SAME time steps/channels replaced by that of ANOTHER storm
                      with the same phase (real, physically consistent atmosphere);
                      track units get noise in this mode.
            The last observed position is never touched (it anchors the forecast).
Effects     per removed set: disp (km between ensemble means), dCRPS vs best track, along- /
            cross-track shift of the mean, change of ensemble spread, energy distance between
            the two ensembles, shift at 24 / 48 / 72 h.
Rankings    joint_greedy / joint_shapley / joint / joint_dcrps  (ours: all units)
            track_only(_greedy)                                  (trajectory-only explainer)
            era5_only(_greedy)                                   (ERA5 units only)
            random (mean of --n_random orders), lerf (reverse of `joint`)
Faithfulness cumulative deletion curve in each operator -> AOPC; scopes `all`, `era5`
            (what a track-only explainer cannot rank) and `track`. Paired bootstrap CI +
            Wilcoxon over storms.
Physics     steering check: removing the U,V of the deep layer (850/500/200 hPa) must shift
            the forecast opposite to the steering-wind ANOMALY around the storm. Reported as
            cosine similarity (also per level). Also: reliance on track vs ERA5.

Cost control for a single T4: every number below is a model call (K-member ensemble).
Use --scopes / --modes / --skip_greedy / --n_shapley_perm 0 / --n_storms to scale, and
--resume (per-model partial files) to survive the Kaggle session limit.

Outputs (output_dir): E10_curves.csv  E10_units.csv  E10_reliance.csv  E10_steering.csv
  E10_aopc_*.md/csv  E10_paired_bootstrap.csv  E10_units_by_phase.csv  E10_hypotheses.md
  E10_unit_heatmap_<model>.png  E10_notes.md  E10_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/kaggle/working/Tropical_Typhoon_Architecture")

from PhysXAI import xai4traj_baseline as x4  # noqa: E402
from PhysXAI.geo import classify_track_type  # noqa: E402
from E9_xai4traj_comparison import (  # noqa: E402
    BlackBox, crps_km, load_model, scan_checkpoints, to_device, with_obs, _interp_time,
)

# ERA5 channel order: GPH 200/500/850/925 (0-3), U 200/500/850/925 (4-7), V (8-11), SST (12)
ERA5_GROUPS: Dict[str, Dict[str, List[int]]] = {
    "level": {"ERA5_200hPa": [0, 4, 8], "ERA5_500hPa": [1, 5, 9],
              "ERA5_850_925hPa": [2, 3, 6, 7, 10, 11], "ERA5_SST": [12]},
    "var": {"ERA5_GPH": [0, 1, 2, 3], "ERA5_U": [4, 5, 6, 7],
            "ERA5_V": [8, 9, 10, 11], "ERA5_SST": [12]},
    "grid": {f"{v}_{l}": [vi * 4 + li] for vi, v in enumerate(("GPH", "U", "V"))
             for li, l in enumerate(("200", "500", "850", "925"))} | {"SST": [12]},
}
# ERA5 normalisation (Model/data/...: (x - mean) / std); only U, V are needed here
_STD_U = np.array([13.315, 8.042, 7.911, 7.494])
_STD_V = np.array([8.377, 5.999, 6.203, 6.555])
# trapezoid pressure weights of the deep-layer mean over 850 / 500 / 200 hPa
_DLM_W = {"850": 175.0 / 650, "500": 325.0 / 650, "200": 150.0 / 650}
_LEVEL_IDX = {"200": 0, "500": 1, "850": 2, "925": 3}
HORIZONS = {"24h": 3, "48h": 7, "72h": 11}      # 6-hourly steps: index of the 4th / 8th / 12th

PHASE_ORDER = ["STRAIGHT_WEST", "RECURVING_NE", "POLEWARD_TRACK", "STALLED", "MIXED"]


@dataclass(frozen=True)
class Unit:
    name: str
    kind: str                 # "track" | "era5"
    group: str                # "TRACK" | ERA5 group name
    seg: int
    steps: Tuple[int, ...]
    channels: Tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.steps) * max(len(self.channels), 1)


def build_units(segments: Sequence[np.ndarray], groups: Dict[str, List[int]]) -> List[Unit]:
    units: List[Unit] = []
    for si, idx in enumerate(segments):
        st = tuple(int(i) for i in idx)
        units.append(Unit(f"S{si}:TRACK", "track", "TRACK", si, st, ()))
        for g, ch in groups.items():
            units.append(Unit(f"S{si}:{g}", "era5", g, si, st, tuple(ch)))
    return units


# ─── Editing the batch ──────────────────────────────────────────────────────

def _fill_or_zero(arr: np.ndarray, steps: List[int]) -> np.ndarray:
    if len(steps) >= arr.shape[0]:
        return np.zeros_like(arr)
    return _interp_time(arr, steps)


def edit_batch(bl, units: Sequence[Unit], mode: str, ctx: dict):
    """Remove (`neutral`), corrupt (`noise`) or replace by another storm (`swap`) the units."""
    T = bl[0].shape[0]
    noise, frame, donor = ctx["noise"], ctx["frame"], ctx.get("donor_img")
    new = list(bl)

    trk = sorted({s for u in units if u.kind == "track" for s in u.steps})
    if trk:
        c_steps = [s for s in trk if s != T - 1]                 # the anchor stays
        if c_steps:
            obs = bl[0][:, 0, :2].float().cpu().numpy()
            if mode == "neutral":
                obs_new = _interp_time(obs, c_steps)
            else:                                                # noise / swap -> noise on track
                km = frame.to_km(x4.norm_to_deg(obs))
                km[c_steps] += noise["track_km"][c_steps]
                obs_new = x4.deg_to_norm(frame.to_deg(km))
            new = with_obs(new, obs_new)
        if torch.is_tensor(bl[7]) and bl[7].dim() == 3:
            me = bl[7].clone()
            me_np = me[:, 0, :].float().cpu().numpy()
            if mode == "neutral":
                me_np = _fill_or_zero(me_np, trk)
            else:
                me_np = me_np.copy()
                me_np[trk] += noise["me"][trk]
            me[:, 0, :] = torch.as_tensor(me_np, dtype=me.dtype, device=me.device)
            new[7] = me

    era = [u for u in units if u.kind == "era5"]
    if era and torch.is_tensor(bl[11]):
        img = bl[11].clone()
        for u in era:
            st = [s for s in u.steps if s < img.shape[2]]
            if not st:
                continue
            for c in u.channels:
                if mode == "neutral":
                    img[:, c, st] = 0.0
                elif mode == "noise":
                    img[:, c, st] = img[:, c, st] + noise["img"][:, c, st]
                else:                                            # swap
                    img[:, c, st] = donor[:, c, st]
        new[11] = img
    return new


def make_noise(bl, args, storm_idx: int) -> dict:
    rng = np.random.default_rng(args.seed * 1009 + storm_idx)
    T = bl[0].shape[0]
    d_me = bl[7].shape[-1] if torch.is_tensor(bl[7]) and bl[7].dim() == 3 else 1
    img = bl[11]
    return {
        "track_km": rng.normal(0.0, args.gaussian_sigma_km, size=(T, 2)),
        "me": rng.normal(0.0, 1.0, size=(T, d_me)).astype(np.float32),
        "img": torch.as_tensor(rng.standard_normal(tuple(img.shape)).astype(np.float32),
                               dtype=img.dtype, device=img.device) if torch.is_tensor(img) else None,
    }


# ─── Distribution-level effect metrics ──────────────────────────────────────

def ens_spread(ens: np.ndarray) -> float:
    if ens.shape[0] < 2:
        return 0.0
    return float(x4.haversine_km(ens, ens.mean(0, keepdims=True)).mean())


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over horizons of 2E|A-B| - E|A-A'| - E|B-B'| (km); 0 iff same distribution."""
    T = min(a.shape[1], b.shape[1])
    a, b = a[:, :T], b[:, :T]
    ab = x4.haversine_km(a[:, None], b[None, :]).mean((0, 1))
    aa = x4.haversine_km(a[:, None], a[None, :]).mean((0, 1))
    bb_ = x4.haversine_km(b[:, None], b[None, :]).mean((0, 1))
    return float((2 * ab - aa - bb_).mean())


def effect_metrics(ens1: np.ndarray, base: dict, gt_deg: np.ndarray, frame) -> Dict[str, float]:
    ens0, m0 = base["ens"], base["mean"]
    m1 = ens1.mean(0)
    T = min(m0.shape[0], m1.shape[0])
    disp_t = x4.haversine_km(m1[:T], m0[:T])
    d = frame.to_km(m1[:T]) - frame.to_km(m0[:T])                 # [T, 2] east / north km
    out = dict(
        disp=float(disp_t.mean()),
        dcrps=crps_km(ens1, gt_deg) - base["crps"],
        along=float((d @ base["u"]).mean()),
        cross=float((d @ base["perp"]).mean()),
        spread_delta=ens_spread(ens1) - base["spread"],
        energy=energy_distance(ens1, ens0),
        shift_e=float(d[:, 0].mean()), shift_n=float(d[:, 1].mean()),
    )
    for name, h in HORIZONS.items():
        out[f"d{name}"] = float(disp_t[min(h, T - 1)])
    return out


# ─── Steering-flow physics check ────────────────────────────────────────────

def steering_anomaly(bl, radius_px: int) -> Dict[str, np.ndarray]:
    """Wind ANOMALY (m/s, relative to the dataset mean) averaged over a window centred on
    the storm at the LAST observed step: per level and deep-layer mean. -> (east, north)."""
    img = bl[11][0, :, -1].float().cpu().numpy()                 # [13, H, W]
    H, W = img.shape[-2:]
    r = int(min(radius_px, H // 2, W // 2))
    win = img[:, H // 2 - r:H // 2 + r + 1, W // 2 - r:W // 2 + r + 1].mean((1, 2))
    res = {}
    for lev in ("200", "500", "850"):
        i = _LEVEL_IDX[lev]
        res[lev] = np.array([win[4 + i] * _STD_U[i], win[8 + i] * _STD_V[i]])
    res["DLM"] = sum(_DLM_W[l] * res[l] for l in ("200", "500", "850"))
    return res


def steering_diag(effect_fn, units_cls, bl, args, storm_idx: int, regime_info: dict) -> List[dict]:
    """Remove U,V (all observed steps, neutral) and compare the forecast shift with the
    direction expected from physics: the storm moves with the steering flow, so replacing
    the wind by its mean must shift it by ~ -anomaly."""
    anom = steering_anomaly(bl, args.steer_radius_px)
    T = bl[0].shape[0]
    steps = tuple(range(T))
    sets = {"DLM": [4 + _LEVEL_IDX[l] for l in ("200", "500", "850")]
                   + [8 + _LEVEL_IDX[l] for l in ("200", "500", "850")],
            "200": [4, 8], "500": [5, 9], "850": [6, 10]}
    rows = []
    for key, ch in sets.items():
        u = units_cls(f"diag:UV_{key}", "era5", "DIAG", -1, steps, tuple(ch))
        e = effect_fn([u], "neutral")
        exp = -anom[key]
        sv = np.array([e["shift_e"], e["shift_n"]])
        na, ns = np.linalg.norm(exp), np.linalg.norm(sv)
        cos = float(sv @ exp / (na * ns)) if na > 0.5 and ns > 1.0 else float("nan")
        rows.append(dict(storm=storm_idx, diag=f"UV_{key}", cos=cos, shift_km=float(ns),
                         anomaly_ms=float(na), shift_e=float(sv[0]), shift_n=float(sv[1]),
                         exp_e=float(exp[0]), exp_n=float(exp[1]), **regime_info))
    g = units_cls("diag:GPH_all", "era5", "DIAG", -1, steps, (0, 1, 2, 3))
    e = effect_fn([g], "neutral")
    rows.append(dict(storm=storm_idx, diag="GPH_all", cos=float("nan"),
                     shift_km=float(np.hypot(e["shift_e"], e["shift_n"])), anomaly_ms=float("nan"),
                     shift_e=e["shift_e"], shift_n=e["shift_n"], exp_e=float("nan"),
                     exp_n=float("nan"), **regime_info))
    return rows


# ─── Per-storm experiment ───────────────────────────────────────────────────

def explain_storm(bb: BlackBox, bl, args, storm_idx: int, groups: Dict[str, List[int]],
                  phase: str, donor_img=None, donor_same_phase: bool = True):
    obs_norm = bl[0][:, 0, :2].float().cpu().numpy()
    gt_norm = bl[1][:, 0, :2].float().cpu().numpy()
    gt_deg = x4.norm_to_deg(gt_norm)
    frame = x4.LocalFrame(x4.norm_to_deg(obs_norm).mean(0))
    x_km = frame.to_km(x4.norm_to_deg(obs_norm))
    regime_gt = classify_track_type(gt_norm)
    info = dict(phase=phase, regime_gt=regime_gt)

    ens0 = bb.ensemble_deg(bl)
    m0 = ens0.mean(0)
    km0 = frame.to_km(m0)
    uvec = km0[-1] - km0[0]
    nrm = np.linalg.norm(uvec)
    uvec = uvec / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
    base = dict(ens=ens0, mean=m0, crps=crps_km(ens0, gt_deg), spread=ens_spread(ens0),
                u=uvec, perp=np.array([-uvec[1], uvec[0]]))

    segments = x4.SEGMENTERS[args.segmentation](x_km)
    units = build_units(segments, groups)
    n = len(units)
    ctx = {"noise": make_noise(bl, args, storm_idx), "frame": frame, "donor_img": donor_img}
    modes = [m for m in args.modes if m != "swap" or donor_img is not None]
    rng = np.random.default_rng(args.seed * 7919 + storm_idx)
    cache: Dict[tuple, Dict[str, float]] = {}

    def effect_of(us: Sequence[Unit], mode: str) -> Dict[str, float]:
        key = (tuple(u.name for u in us), mode)
        if key not in cache:
            ens = bb.ensemble_deg(edit_batch(bl, us, mode, ctx))
            cache[key] = effect_metrics(ens, base, gt_deg, frame)
        return cache[key]

    def effect(sel: Sequence[int], mode: str) -> Dict[str, float]:
        return effect_of([units[i] for i in sel], mode)

    # ---- single-unit neutral effects (ranking signal + descriptive importance) ----
    single = [effect([i], "neutral") for i in range(n)]
    disp1 = np.array([s["disp"] for s in single])
    dcrps1 = np.array([s["dcrps"] for s in single])

    unit_rows = []
    for i, u in enumerate(units):
        desc = x4.describe_segment(x_km, np.array(u.steps))
        unit_rows.append(dict(
            storm=storm_idx, unit=u.name, kind=u.kind, group=u.group, seg=u.seg,
            n_seg=len(segments), seg_pos=(u.seg / max(len(segments) - 1, 1)), n_steps=len(u.steps),
            size=u.size, disp_per_cell=float(disp1[i] / u.size), **info, **desc,
            **{k: float(v) for k, v in single[i].items()}))

    def order_by(score: np.ndarray, idx: Sequence[int]) -> List[int]:
        idx = list(idx)
        sc = np.array([score[i] for i in idx])
        return [idx[j] for j in np.lexsort((rng.random(len(idx)), -sc))]

    def greedy(cands: Sequence[int]) -> List[int]:
        chosen: List[int] = []
        remaining = list(cands)
        while remaining:
            best = max(remaining, key=lambda i: (effect(chosen + [i], "neutral")["disp"], rng.random()))
            chosen.append(best)
            remaining.remove(best)
        return chosen

    trk_idx = [i for i, u in enumerate(units) if u.kind == "track"]
    era_idx = [i for i, u in enumerate(units) if u.kind == "era5"]
    joint = order_by(disp1, range(n))
    orders: Dict[str, List[int]] = {
        "joint": joint,
        "joint_dcrps": order_by(dcrps1, range(n)),
        "track_only": order_by(disp1, trk_idx) + list(rng.permutation(era_idx)),
        "era5_only": order_by(disp1, era_idx) + list(rng.permutation(trk_idx)),
        "lerf": joint[::-1],
    }
    do_greedy = (not args.skip_greedy) and n <= args.max_greedy_units
    if do_greedy:
        orders["joint_greedy"] = greedy(range(n))
        orders["track_only_greedy"] = greedy(trk_idx) + list(rng.permutation(era_idx))
        orders["era5_only_greedy"] = greedy(era_idx) + list(rng.permutation(trk_idx))
    if args.n_shapley_perm > 0:
        phi = np.zeros(n)
        for _ in range(args.n_shapley_perm):                    # permutation-sampling Shapley
            prev, chosen = 0.0, []
            for i in rng.permutation(n):
                chosen.append(int(i))
                v = effect(chosen, "neutral")["disp"]
                phi[i] += v - prev
                prev = v
        phi /= args.n_shapley_perm
        orders["joint_shapley"] = order_by(phi, range(n))
        for i, r in enumerate(unit_rows):
            r["shapley_disp"] = float(phi[i])
    for r in range(args.n_random):
        orders[f"random_{r}"] = [int(i) for i in rng.permutation(n)]

    # ---- deletion curves ----
    curve_rows = []
    scopes = {"all": None, "era5": "era5", "track": "track"}
    for scope in args.scopes:
        kfilter = scopes[scope]
        for name, order in orders.items():
            if scope == "era5" and name.startswith("track_only"):
                continue
            if scope == "track" and name.startswith("era5_only"):
                continue
            order = [int(i) for i in order if (kfilter is None or units[int(i)].kind == kfilter)]
            if not order:
                continue
            m = len(order)
            top3_era = float(np.mean([units[i].kind == "era5" for i in order[:3]]))
            for mode in modes:
                ds, cs = [], []
                for k in range(1, m + 1):
                    e = effect(order[:k], mode)
                    ds.append(e["disp"])
                    cs.append(e["dcrps"])
                curve_rows.append(dict(
                    storm=storm_idx, scope=scope,
                    ranking=name.split("_")[0] if name.startswith("random") else name,
                    perm=name, mode=mode, n_units=m, top3_era5_share=top3_era,
                    aopc_disp=float(np.mean(ds)), aopc_crps=float(np.mean(cs)),
                    final_disp=float(ds[-1]), donor_same_phase=donor_same_phase,
                    curve_disp=json.dumps([round(v, 3) for v in ds]), **info))

    rel_rows = []
    for mode in modes:
        et, ee = effect(trk_idx, mode), effect(era_idx, mode)
        rel_rows.append(dict(storm=storm_idx, mode=mode, track_all_disp=et["disp"],
                             era5_all_disp=ee["disp"], track_all_dcrps=et["dcrps"],
                             era5_all_dcrps=ee["dcrps"], era5_all_spread=ee["spread_delta"],
                             **info))

    steer_rows = steering_diag(effect_of, Unit, bl, args, storm_idx, info) if args.steering else []
    return curve_rows, unit_rows, rel_rows, steer_rows


# ─── Reporting ──────────────────────────────────────────────────────────────

def _ci(d: np.ndarray, rng: np.random.Generator, n_boot: int = 2000):
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def _wilcoxon_p(d: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon
        d = d[np.abs(d) > 1e-9]
        return float(wilcoxon(d).pvalue) if len(d) >= 6 else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


RANK_ORDER = ["joint_greedy", "joint_shapley", "joint", "joint_dcrps", "track_only_greedy",
              "track_only", "era5_only_greedy", "era5_only", "random", "lerf"]


def report(curves, units, rel, steer, out: str, meta: dict):
    os.makedirs(out, exist_ok=True)
    curves.to_csv(os.path.join(out, "E10_curves.csv"), index=False)
    units.to_csv(os.path.join(out, "E10_units.csv"), index=False)
    rel.to_csv(os.path.join(out, "E10_reliance.csv"), index=False)
    steer.to_csv(os.path.join(out, "E10_steering.csv"), index=False)

    c = (curves.groupby(["label", "storm", "phase", "scope", "ranking", "mode"], as_index=False)
         [["aopc_disp", "aopc_crps", "top3_era5_share"]].mean())
    c.to_csv(os.path.join(out, "E10_aopc_per_storm.csv"), index=False)

    summary: dict = {"meta": meta}
    for metric in ("aopc_disp", "aopc_crps"):
        t = (c.groupby(["label", "scope", "mode", "ranking"])[metric].agg(["mean", "std", "count"])
             .reset_index())
        t.to_csv(os.path.join(out, f"E10_{metric}_by_ranking.csv"), index=False)
        lines = []
        for scope, what in (("all", "ALL units"),
                            ("era5", "ERA5 units only (what a track-only explainer cannot rank)"),
                            ("track", "TRACK units only")):
            for mode in sorted(t["mode"].unique()):
                sub = t[(t["scope"] == scope) & (t["mode"] == mode)]
                if sub.empty:
                    continue
                piv = sub.pivot(index="label", columns="ranking", values="mean")
                piv = piv[[r for r in RANK_ORDER if r in piv.columns]]
                lines.append(f"\n**{metric} — {what} — deletion operator: {mode}** "
                             f"(km, higher = finds influential units sooner)\n")
                lines.append("| model | " + " | ".join(piv.columns) + " |")
                lines.append("|" + "---|" * (1 + len(piv.columns)))
                for lab, r_ in piv.iterrows():
                    lines.append(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in r_.values) + " |")
        with open(os.path.join(out, f"E10_{metric}.md"), "w") as f:
            f.write("\n".join(lines))
        print("\n".join(lines))
        summary[metric] = t.to_dict("records")

    rng = np.random.default_rng(0)
    pairs = {"all": (("joint_greedy", "track_only_greedy"), ("joint_greedy", "random"),
                     ("joint_shapley", "random"), ("joint", "track_only"),
                     ("track_only_greedy", "random")),
             "era5": (("era5_only_greedy", "random"), ("joint_shapley", "random"),
                      ("joint_greedy", "random"), ("era5_only", "random"),
                      ("era5_only_greedy", "lerf")),
             "track": (("joint_greedy", "random"), ("track_only_greedy", "random"))}
    recs = []
    for (label, scope, mode), g in c.groupby(["label", "scope", "mode"]):
        piv = g.pivot(index="storm", columns="ranking", values="aopc_disp")
        for a, b in pairs[scope]:
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna().values
                if len(d) >= 2:
                    lo, hi = _ci(d, rng)
                    recs.append(dict(model=label, scope=scope, mode=mode, A=a, B=b,
                                     mean_diff_km=float(d.mean()), ci95_lo=float(lo),
                                     ci95_hi=float(hi), wilcoxon_p=_wilcoxon_p(d),
                                     n_storms=len(d), frac_A_better=float((d > 0).mean())))
    pb = pd.DataFrame(recs)
    pb.to_csv(os.path.join(out, "E10_paired_bootstrap.csv"), index=False)
    if len(pb):
        print("\npaired bootstrap of AOPC_disp difference (A - B), 95% CI over storms:")
        print(pb.round(3).to_string(index=False))
    summary["paired_bootstrap"] = pb.to_dict("records")

    r = rel.groupby(["label", "mode"])[["track_all_disp", "era5_all_disp"]].mean().reset_index()
    r["era5_share"] = r["era5_all_disp"] / (r["era5_all_disp"] + r["track_all_disp"] + 1e-9)
    r.to_csv(os.path.join(out, "E10_reliance_summary.csv"), index=False)
    print("\nreliance (km shift when ALL track / ALL ERA5 units are removed):")
    print(r.round(2).to_string(index=False))
    summary["reliance"] = r.to_dict("records")

    if len(steer):
        s = (steer[steer["diag"] != "GPH_all"].groupby(["label", "diag"])
             .agg(cos_mean=("cos", "mean"), frac_cos_pos=("cos", lambda x: float((x.dropna() > 0).mean()) if x.notna().any() else float("nan")),
                  n_valid=("cos", lambda x: int(x.notna().sum())), shift_km=("shift_km", "mean"))
             .reset_index())
        s.to_csv(os.path.join(out, "E10_steering_summary.csv"), index=False)
        print("\nsteering-flow check (cosine between forecast shift and -wind anomaly; +1 = physical):")
        print(s.round(3).to_string(index=False))
        summary["steering"] = s.to_dict("records")

    _phase_tables(units, out)
    _heatmaps(units, out)
    _notes(out, meta)
    with open(os.path.join(out, "E10_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)


def _phase_tables(units: pd.DataFrame, out: str):
    cols = ["disp", "dcrps", "along", "cross", "spread_delta", "energy", "d24h", "d48h", "d72h"]
    cols = [c for c in cols if c in units]
    ph = units.groupby(["label", "phase", "group"])[cols].mean().reset_index()
    ph["n_storms"] = units.groupby(["label", "phase", "group"])["storm"].nunique().values
    ph.to_csv(os.path.join(out, "E10_units_by_phase.csv"), index=False)
    L = ["# Descriptive tables for the hypotheses (no verdicts are drawn automatically)\n",
         "Phase = motion class of the model's own forecast. Values = mean forecast shift (km) when "
         "the unit is removed (neutral), averaged over time segments.\n"]
    era = units[units["kind"] == "era5"]
    for label, g in era.groupby("label"):
        L.append(f"\n## {label}\n")
        L.append("**H1/H2 — top ERA5 groups per phase (disp, km)**\n")
        L.append("| phase | n storms | top-1 | top-2 | top-3 |\n|---|---|---|---|---|")
        for phase, gp in g.groupby("phase"):
            m = gp.groupby("group")["disp"].mean().sort_values(ascending=False)
            top = [f"{k} ({v:.1f})" for k, v in m.head(3).items()] + ["-"] * 3
            L.append(f"| {phase} | {gp['storm'].nunique()} | {top[0]} | {top[1]} | {top[2]} |")
        L.append("\n**H3 — mean |spread change| (km) and shift by time-segment position**\n")
        L.append("| segment position | disp | |spread_delta| |\n|---|---|---|")
        gg = g.copy()
        gg["pos"] = pd.cut(gg["seg_pos"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["early", "middle", "late"])
        for pos, gp in gg.groupby("pos", observed=True):
            L.append(f"| {pos} | {gp['disp'].mean():.2f} | {gp['spread_delta'].abs().mean():.3f} |")
        L.append("\n**H5 — horizon dependence: mean shift at 24 / 48 / 72 h per ERA5 group (km)**\n")
        L.append("| group | 24h | 48h | 72h | 72h/24h |\n|---|---|---|---|---|")
        for grp, gp in g.groupby("group"):
            a, b, c3 = gp["d24h"].mean(), gp["d48h"].mean(), gp["d72h"].mean()
            L.append(f"| {grp} | {a:.2f} | {b:.2f} | {c3:.2f} | {c3 / a if a > 1e-9 else float('nan'):.2f} |")
    with open(os.path.join(out, "E10_hypotheses.md"), "w") as f:
        f.write("\n".join(L))


def _heatmaps(units: pd.DataFrame, out: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[E10] heatmaps skipped: {e}")
        return
    for label, g in units.groupby("label"):
        g = g.copy()
        g["pos"] = pd.cut(g["seg_pos"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["early", "middle", "late"])
        p1 = g.pivot_table(index="group", columns="pos", values="disp", aggfunc="mean", observed=False)
        p2 = g.pivot_table(index="group", columns="phase", values="disp", aggfunc="mean", observed=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(p1) + 2.4))
        for ax, piv, ttl in ((axes[0], p1, "by time segment"), (axes[1], p2, "by storm phase (model forecast)")):
            im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns], rotation=20, fontsize=8)
            ax.set_yticks(range(piv.shape[0]), list(piv.index), fontsize=8)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if v == v:
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="w", fontsize=7)
            ax.set_title(f"{label}: shift (km) when unit removed — {ttl}", fontsize=8)
            fig.colorbar(im, ax=ax, label="km")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"E10_unit_heatmap_{label}.png"), dpi=150)
        plt.close(fig)


def _notes(out: str, meta: dict):
    txt = f"""# E10 — definitions and caveats (read before quoting numbers)
- Units: time segments of the OBSERVED window ({meta.get('segmentation')}) x TRACK / ERA5 groups ({meta.get('era5_groups')}).
- Phase of a storm = motion class of the model's own clean ensemble-mean forecast. The ground-truth class is only in `regime_gt`.
- neutral = interpolation (track, Me) / dataset-mean ERA5 (0 after normalisation; NOT a seasonal climatology).
- swap = ERA5 of another storm with the same phase (same time steps / channels); track units get Gaussian noise in this mode.
- The last observed position is never perturbed. Env-feature tensors (batch idx 13) are NOT edited.
- Steering check: wind anomaly = window mean (radius {meta.get('steer_radius_px')} px) of the normalised U,V at the last observed step x std. The pixel
  size of the ERA5 patch was not verified - treat the radius as a parameter. Cosine is NaN when the anomaly < 0.5 m/s or the shift < 1 km.
- Models that ignore ERA5 give ERA5 effects of exactly 0 (not an error).
- The forecast ensemble of TC-XFlow is generated from small noise (sigma_inference = 0.04), so spread effects are small by construction.
- Prediction horizon is 12 steps x 6 h = 72 h.
- Segmentation thresholds (turn_deg / speed_ratio) are heuristics: tune only on validation storms.
"""
    with open(os.path.join(out, "E10_notes.md"), "w") as f:
        f.write(txt)


# ─── Main ───────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--auto", default=None)
    p.add_argument("--model", action="append", default=[], help="KIND:PATH[:LABEL] (repeatable)")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--output_dir", default="/kaggle/working/TC-PhysXAI/results/E10")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_storms", type=int, default=20)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--segmentation", default="phys_regime", choices=sorted(x4.SEGMENTERS))
    p.add_argument("--era5_groups", default="level", choices=sorted(ERA5_GROUPS))
    p.add_argument("--gaussian_sigma_km", type=float, default=15.0)
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--n_shapley_perm", type=int, default=4, help="0 disables Shapley")
    p.add_argument("--skip_greedy", action="store_true")
    p.add_argument("--max_greedy_units", type=int, default=18)
    p.add_argument("--modes", default="neutral,noise,swap")
    p.add_argument("--scopes", default="all,era5,track")
    p.add_argument("--steering", type=int, default=1)
    p.add_argument("--steer_radius_px", type=int, default=16)
    p.add_argument("--resume", action="store_true", help="skip models whose partial files exist")
    p.add_argument("--quick", action="store_true", help="3 storms, 2 random orders, 2 Shapley perms")
    a = p.parse_args()
    a.modes = [m for m in a.modes.split(",") if m]
    a.scopes = [s for s in a.scopes.split(",") if s]
    return a


def _part(out, label, name):
    return os.path.join(out, f"E10_part_{label}_{name}.csv")


def main():
    args = get_args()
    if args.quick:
        args.n_storms, args.n_random, args.n_shapley_perm = 3, 2, 2
    specs = []
    for m in args.model:
        parts = m.split(":")
        specs.append((parts[0], parts[1], parts[2] if len(parts) > 2 else parts[0]))
    if args.auto:
        for pth, kind, info in scan_checkpoints(args.auto):
            if kind is None:
                print(f"[auto] skipped (unknown kind): {pth}  {info}")
            else:
                specs.append((kind, pth, kind))
    if not specs:
        sys.exit("No model given: use --model KIND:PATH or --auto DIR")
    if "swap" in args.modes and args.n_storms < 2:
        print("[E10] swap needs >= 2 storms; dropping it")
        args.modes = [m for m in args.modes if m != "swap"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    groups = ERA5_GROUPS[args.era5_groups]
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70, f"\nE10 | device {device} | storms {args.n_storms} | K {args.K} | segmentation "
          f"{args.segmentation} | ERA5 groups {args.era5_groups} | modes {args.modes} | scopes {args.scopes}"
          f"\n", "=" * 70)

    from torch.utils.data import DataLoader
    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    ds = TrajectoryDataset(args.dataset_root, split="test")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=seq_collate, num_workers=2)
    batches = []
    for b in loader:
        if b[0].shape[1] != 1:
            continue
        batches.append(b)
        if len(batches) >= args.n_storms:
            break
    print(f"[data] {len(ds)} test storms -> using {len(batches)}")

    names = ("curves", "units", "reliance", "steering")
    store = {n: [] for n in names}
    failures: Dict[str, int] = {}
    for kind, path, label in specs:
        ckpt = os.path.basename(path)
        if args.resume and all(os.path.exists(_part(args.output_dir, label, n)) for n in names):
            print(f"[{label}] resume: partial files found, skipping")
            for n in names:
                try:
                    store[n].append(pd.read_csv(_part(args.output_dir, label, n)))
                except pd.errors.EmptyDataError:
                    pass
            failures[f"{label}/{ckpt}"] = 0
            continue
        print(f"\n{'=' * 70}\n[{label}] loading {kind}: {path}\n{'=' * 70}")
        try:
            model = load_model(kind, path, device)
        except Exception:  # noqa: BLE001
            print(f"[{label}] LOAD FAILED — skipped:\n{traceback.format_exc()}")
            failures[f"{label}/{ckpt}"] = -1
            continue
        bb = BlackBox(model, device, args.K, args.seed)

        # pre-pass: phase of every storm from the model's own clean forecast (no GT)
        phases: List[str] = []
        for batch in batches:
            ens = bb.ensemble_deg(to_device(list(batch), device))
            phases.append(classify_track_type(x4.deg_to_norm(ens.mean(0))))
        print(f"[{label}] phases: " + ", ".join(f"{p}={phases.count(p)}" for p in PHASE_ORDER if p in phases))

        rows = {n: [] for n in names}
        n_fail, t0 = 0, time.time()
        rng = np.random.default_rng(args.seed)
        for s, batch in enumerate(batches):
            try:
                bl = to_device(list(batch), device)
                donor, same = None, True
                if "swap" in args.modes and torch.is_tensor(bl[11]):
                    cand = [j for j in range(len(batches)) if j != s and phases[j] == phases[s]]
                    if not cand:
                        cand, same = [j for j in range(len(batches)) if j != s], False
                    donor = to_device(batches[int(rng.choice(cand))][11], device)
                out_ = explain_storm(bb, bl, args, s, groups, phases[s], donor, same)
                for n, r_ in zip(names, out_):
                    for r in r_:
                        r.update(label=label, kind_model=kind, ckpt=ckpt)
                    rows[n] += r_
                el = time.time() - t0
                print(f"  [{label}] storm {s + 1}/{len(batches)} ok  ({bb.n_calls} calls, {el:.0f}s, "
                      f"ETA {el / (s + 1) * (len(batches) - s - 1):.0f}s)")
            except Exception:  # noqa: BLE001
                n_fail += 1
                print(f"  [{label}] storm {s + 1} FAILED:\n{traceback.format_exc()}")
        failures[f"{label}/{ckpt}"] = n_fail
        del model, bb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if rows["curves"]:
            for n in names:
                df_ = pd.DataFrame(rows[n])
                df_.to_csv(_part(args.output_dir, label, n), index=False)
                store[n].append(df_)

    print("\n" + "=" * 70 + "\nFAILURE REPORT (storms failed per checkpoint; -1 = model failed to load)")
    for k, v in failures.items():
        print(f"  {k}: {v}")
    if not store["curves"]:
        sys.exit("E10: no results at all — every model/storm failed (see tracebacks above)")
    meta = {"n_storms": len(batches), "K": args.K, "segmentation": args.segmentation,
            "era5_groups": args.era5_groups, "n_random": args.n_random, "modes": args.modes,
            "scopes": args.scopes, "n_shapley_perm": args.n_shapley_perm,
            "steer_radius_px": args.steer_radius_px, "failures": failures,
            "models": [{"kind": k, "path": p, "label": lab} for k, p, lab in specs]}
    cat = {n: pd.concat(store[n], ignore_index=True) if store[n] else pd.DataFrame() for n in names}
    report(cat["curves"], cat["units"], cat["reliance"], cat["steering"], args.output_dir, meta)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
