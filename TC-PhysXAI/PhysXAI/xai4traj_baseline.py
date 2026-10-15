"""
TC-PhysXAI — XAI4Traj baseline (numpy core, no torch dependency)
=================================================================
Re-implementation, adapted to continuous typhoon-track prediction, of the
segment-based perturbation explainer of

    Tung, Phuc, Le Duy. "Towards explainable trajectory classification:
    A segment-based perturbation approach", Neurocomputing 658 (2025) 131691.
    Official code: https://github.com/DAIR-Group/XAI4Traj  (MIT licence)

What is KEPT from the official code
-----------------------------------
  * segmentation: RDP, MDL, sliding window (same algorithms)
  * perturbation: Gaussian noise, scaling, rotation (applied to a segment)
  * surrogate: random binary perturbation vectors z in {0,1}^n_seg, perturbed
    trajectory built from clean/noisy segments, DTW-based sample weights
    exp(-|d - mean(d)| / std(d)), linear surrogate whose coefficients are the
    segment importances theta
  * evaluation protocol: perturb ONE segment at a time (identity matrix Z_eval),
    sort the results by |theta| (descending), "change" = at least one single
    segment perturbation changes the prediction, precision = AP@K of the sorted
    "changed" flags, recall R = fraction of trajectories with change,
    H = harmonic mean(R, P).

What is ADAPTED (the task is regression, not classification)
-------------------------------------------------------------
  * "class label changed"  ->  "predicted track moved by more than tau km"
    (mean great-circle distance between the perturbed-input prediction and
    the original-input prediction, ensemble mean, common random numbers).
  * surrogate: weighted Ridge on the continuous displacement (default) or the
    original LogisticRegression on the binarised label (tau-dependent).
  * the typhoon observation window has only 8 points, so segmentation
    parameters are given in km and sliding windows tile the 8 points.
  * perturbations act in a local tangent plane (km), origin = centroid of the
    observed points; scaling / rotation are about that origin.
  * the perturbed trajectory is built point-wise: a point is replaced by its
    noisy version iff it belongs to at least one selected segment (identical to
    the official code for non-overlapping segments; well-defined for overlaps).

Everything here is pure numpy so it can be unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

KM_PER_DEG = 111.195  # R_earth * pi / 180


# ─── Coordinates ─────────────────────────────────────────────────────────────

def norm_to_deg(xn: np.ndarray) -> np.ndarray:
    """Dataset-normalised (lon_n, lat_n) -> (lon°, lat°).

    Same convention as Model/flow_matching_model.py::_norm_to_deg.
    """
    xn = np.asarray(xn, dtype=np.float64)
    return np.stack([(xn[..., 0] * 50.0 + 1800.0) / 10.0,
                     (xn[..., 1] * 50.0) / 10.0], axis=-1)


def deg_to_norm(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    return np.stack([(d[..., 0] * 10.0 - 1800.0) / 50.0,
                     (d[..., 1] * 10.0) / 50.0], axis=-1)


def haversine_km(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between (lon°, lat°) arrays [..., 2]."""
    lon1, lat1 = np.deg2rad(a_deg[..., 0]), np.deg2rad(a_deg[..., 1])
    lon2, lat2 = np.deg2rad(b_deg[..., 0]), np.deg2rad(b_deg[..., 1])
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


class LocalFrame:
    """Local tangent plane (km) around a (lon°, lat°) origin."""

    def __init__(self, origin_deg: np.ndarray):
        self.lon0, self.lat0 = float(origin_deg[0]), float(origin_deg[1])
        self._cx = KM_PER_DEG * np.cos(np.deg2rad(self.lat0))

    def to_km(self, deg: np.ndarray) -> np.ndarray:
        return np.stack([(deg[..., 0] - self.lon0) * self._cx,
                         (deg[..., 1] - self.lat0) * KM_PER_DEG], axis=-1)

    def to_deg(self, km: np.ndarray) -> np.ndarray:
        return np.stack([km[..., 0] / self._cx + self.lon0,
                         km[..., 1] / KM_PER_DEG + self.lat0], axis=-1)


# ─── Segmentation (returns lists of point-index arrays) ─────────────────────

def _perp_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    n = np.linalg.norm(ab)
    if n == 0:
        return float(np.linalg.norm(p - a))
    return float(abs(ab[0] * (a[1] - p[1]) - ab[1] * (a[0] - p[0])) / n)


def _rdp_keep(pts: np.ndarray, eps: float) -> List[int]:
    """Indices kept by Ramer-Douglas-Peucker (always includes both ends)."""
    keep = {0, len(pts) - 1}

    def rec(lo: int, hi: int) -> None:
        if hi <= lo + 1:
            return
        d = [_perp_dist(pts[i], pts[lo], pts[hi]) for i in range(lo + 1, hi)]
        k = int(np.argmax(d))
        if d[k] > eps:
            m = lo + 1 + k
            keep.add(m)
            rec(lo, m)
            rec(m, hi)

    rec(0, len(pts) - 1)
    return sorted(keep)


def seg_rdp(pts: np.ndarray, epsilon_km: float = 20.0) -> List[np.ndarray]:
    """Official RDP segmentation: a segment ends at every reduced point."""
    keep = set(_rdp_keep(pts, epsilon_km))
    segs, cur, last = [], [], -1
    for i in range(len(pts)):
        cur.append(i)
        if i in keep and i > last:
            segs.append(np.asarray(cur))
            cur, last = [], i
    if cur:
        segs.append(np.asarray(cur))
    return segs


def _mdl_cost(seg: np.ndarray) -> float:
    if len(seg) < 2:
        return float("inf")
    start, end = seg[0], seg[-1]
    line = end - start
    n = np.linalg.norm(line)
    if n == 0:
        return float("inf")
    u = line / n
    proj = (seg - start) @ u
    recon = start + np.outer(proj, u)
    return float(np.linalg.norm(seg - recon, axis=1).sum())


def seg_mdl(pts: np.ndarray, epsilon_km: float = 30.0) -> List[np.ndarray]:
    """Official MDL segmentation (error threshold in km)."""
    segs, cur = [], [0]
    for i in range(1, len(pts)):
        cur.append(i)
        if _mdl_cost(pts[cur]) > epsilon_km:
            segs.append(np.asarray(cur[:-1]))
            cur = [i]
    if cur:
        segs.append(np.asarray(cur))
    return segs


def seg_sliding(pts: np.ndarray, window: int = 2, step: int = 2) -> List[np.ndarray]:
    """Official sliding window (window=step tiles the trajectory)."""
    n = len(pts)
    window = max(1, int(window))
    return [np.arange(i, i + window) for i in range(0, n - window + 1, step)]


def seg_phys_abc(pts: np.ndarray) -> List[np.ndarray]:
    """TC-PhysXAI fixed phases A/B/C (steps 0-3 / 4-5 / 6-7 for 8 obs steps)."""
    n = len(pts)
    if n == 8:
        return [np.arange(0, 4), np.arange(4, 6), np.arange(6, 8)]
    return [np.asarray(a) for a in np.array_split(np.arange(n), 3)]


def _headings_speeds(pts: np.ndarray):
    d = np.diff(pts, axis=0)
    return np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0, np.hypot(d[:, 0], d[:, 1])


def seg_phys_regime(pts: np.ndarray, turn_deg: float = 25.0, speed_ratio: float = 0.35,
                    min_len: int = 2, max_segments: int = 4) -> List[np.ndarray]:
    """DATA-DRIVEN motion-regime segmentation (TC-PhysXAI).

    A new segment starts at an interior point where the storm's motion changes regime:
        score(j) = |heading change at j| / turn_deg  +  |relative speed change at j| / speed_ratio
    (heading = compass angle of the step, speed = km per step; both from the observed
    track only). Change points with score > 1 are taken greedily (largest first) subject to
    a minimum segment length and at most `max_segments` segments. If no point exceeds the
    threshold the track is a single steady regime and is still split at the strongest
    change point, so that every storm has >= 2 segments (otherwise nothing to rank).
    turn_deg / speed_ratio are heuristics: confirm / tune on the validation split only.
    """
    n = len(pts)
    if n < 2 * min_len:
        return [np.arange(n)]
    th, v = _headings_speeds(pts)                    # per step (n-1)
    score = np.zeros(n)
    for j in range(1, n - 1):                        # point j sits between step j-1 and step j
        dth = abs((th[j] - th[j - 1] + 180.0) % 360.0 - 180.0)
        dv = abs(v[j] - v[j - 1]) / max(0.5 * (v[j] + v[j - 1]), 1e-6)
        score[j] = dth / turn_deg + dv / speed_ratio
    cuts: List[int] = []
    order = np.lexsort((np.abs(np.arange(n) - n / 2.0), -score))   # strongest first, ties -> middle
    for j in order:
        j = int(j)
        if j < min_len or j > n - min_len:
            continue
        if any(abs(j - c) < min_len for c in cuts):
            continue
        if score[j] <= 1.0 and cuts:
            break
        cuts.append(j)
        if len(cuts) >= max_segments - 1:
            break
    if not cuts:                                     # fallback: middle
        cuts = [n // 2]
    cuts = sorted(cuts)
    bounds = [0] + cuts + [n]
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def describe_segment(pts_km: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    """Net heading change (deg, + = clockwise = NH recurving) and mean speed (km/step)."""
    seg = pts_km[np.asarray(idx)]
    if len(seg) < 3:
        return {"turn_deg": float("nan"), "speed_km": float("nan")}
    th, v = _headings_speeds(seg)
    turn = float(((th[-1] - th[0] + 180.0) % 360.0) - 180.0)
    return {"turn_deg": turn, "speed_km": float(v.mean())}


SEGMENTERS: Dict[str, Callable[..., List[np.ndarray]]] = {
    "mdl": seg_mdl, "rdp": seg_rdp, "sliding": seg_sliding, "phys_abc": seg_phys_abc,
    "phys_regime": seg_phys_regime,
}


# ─── Perturbation (on km points, origin = frame origin) ─────────────────────

@dataclass
class PerturbConfig:
    gaussian_sigma_km: float = 15.0
    scale_factor: float = 1.2
    angle_rad: float = float(np.pi / 18)   # 10 degrees, official default


def make_noisy(pts_km: np.ndarray, kind: str, rng: np.random.Generator,
               cfg: PerturbConfig) -> np.ndarray:
    """Perturbed copy of ALL points; segments later pick which points to use."""
    if kind == "gaussian":
        return pts_km + rng.normal(0.0, cfg.gaussian_sigma_km, size=pts_km.shape)
    if kind == "scaling":
        return pts_km * cfg.scale_factor
    if kind == "rotation":
        c, s = np.cos(cfg.angle_rad), np.sin(cfg.angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return pts_km @ rot.T
    raise ValueError(f"unknown perturbation '{kind}'")


PERTURBATIONS = ("gaussian", "rotation", "scaling")


def apply_selection(x_km: np.ndarray, noisy_km: np.ndarray,
                    segments: Sequence[np.ndarray], select: Sequence[int]) -> np.ndarray:
    """Point-wise union of the selected segments take the noisy value."""
    out = x_km.copy()
    for j in select:
        idx = segments[j]
        out[idx] = noisy_km[idx]
    return out


# ─── DTW (exact, tiny trajectories) ─────────────────────────────────────────

def dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def dtw_weights(x_km: np.ndarray, perturbed: List[np.ndarray]) -> np.ndarray:
    d = np.array([dtw(x_km, p) for p in perturbed])
    std = d.std()
    if std == 0:
        return np.ones_like(d)
    return np.exp(-np.abs((d - d.mean()) / (std + 1e-10)))


# ─── Surrogate importance ───────────────────────────────────────────────────

def perturbation_vectors(n_seg: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """All 2^n vectors if 2^n <= cap (deterministic), else `cap` random ones."""
    if n_seg <= 30 and 2 ** n_seg <= cap:
        v = ((np.arange(2 ** n_seg)[:, None] >> np.arange(n_seg)[None, :]) & 1)
        return v.astype(int)
    return rng.integers(0, 2, size=(cap, n_seg))


def surrogate_theta(Z: np.ndarray, delta: np.ndarray, weights: np.ndarray,
                    tau_km: float, kind: str = "ridge",
                    alpha: float = 1.0) -> Optional[np.ndarray]:
    """Segment importances theta, or None if no perturbation changed the output
    (official behaviour: 'only one class detected, skipping explanation')."""
    y = (delta > tau_km).astype(int)
    if len(np.unique(y)) == 1:
        return None
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(Z, y, sample_weight=weights)
        return clf.coef_[0]
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=alpha)
    reg.fit(Z, delta, sample_weight=weights)
    return reg.coef_


# ─── Metrics ────────────────────────────────────────────────────────────────

def ap_at_k(flags_sorted: Sequence[int]) -> float:
    """Average precision of the sorted 'changed' flags (official ap_at_k, k=len)."""
    hits, s = 0, 0.0
    for i, f in enumerate(flags_sorted):
        if f:
            hits += 1
            s += hits / (i + 1)
    return s / hits if hits > 0 else 0.0


def storm_metrics(order: np.ndarray, flags: np.ndarray) -> Tuple[int, float]:
    """(change, AP) for a ranking `order` (segment indices, most important
    first) and single-segment 'changed' flags."""
    f = np.asarray(flags)[np.asarray(order)]
    change = int(f.any())
    return change, (ap_at_k(f) if change else 0.0)


def expected_random_ap(flags: np.ndarray, rng: np.random.Generator,
                       n_perm: int = 200) -> Tuple[int, float]:
    """Expected AP of a uniformly random ranking (sanity lower bound)."""
    flags = np.asarray(flags)
    if not flags.any():
        return 0, 0.0
    aps = [ap_at_k(flags[rng.permutation(len(flags))]) for _ in range(n_perm)]
    return 1, float(np.mean(aps))


def aggregate(changes: Sequence[int], aps: Sequence[float]) -> Dict[str, float]:
    """R = mean(change); P = mean AP over storms with change; H = harmonic mean.
    P/H are NaN when R == 0 (same as the 'nan' cells of the XAI4Traj tables)."""
    changes = np.asarray(changes, dtype=float)
    aps = np.asarray(aps, dtype=float)
    if len(changes) == 0:
        return {"R": float("nan"), "P": float("nan"), "H": float("nan"), "n": 0}
    R = float(changes.mean())
    if changes.sum() == 0:
        return {"R": R, "P": float("nan"), "H": float("nan"), "n": int(len(changes))}
    P = float(aps[changes == 1].mean())
    H = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
    return {"R": R, "P": P, "H": float(H), "n": int(len(changes))}


# ─── One storm, one (segmentation, perturbation) configuration ──────────────

@dataclass
class ConfigResult:
    n_seg: int
    delta_eval: np.ndarray          # [n_seg] single-segment displacement (km)
    theta: Optional[np.ndarray]     # surrogate importances (None = skipped)
    n_surrogate: int


def run_config(
    x_km: np.ndarray,
    segments: List[np.ndarray],
    noisy_km: np.ndarray,
    delta_fn: Callable[[np.ndarray], float],
    tau_list: Sequence[float],
    n_perturb: int,
    rng: np.random.Generator,
    surrogate: str = "ridge",
) -> Tuple[np.ndarray, Dict[float, Optional[np.ndarray]], int]:
    """Returns (delta_eval[n_seg], {tau: theta or None}, n_surrogate_samples).

    delta_fn(perturbed_pts_km) -> displacement (km) of the prediction w.r.t. the
    prediction of the unperturbed input. Results are cached by selected-point set.
    """
    n = len(segments)
    cache: Dict[tuple, float] = {}

    def delta_of(select: Sequence[int]) -> float:
        pts = apply_selection(x_km, noisy_km, segments, select)
        if np.allclose(pts, x_km):
            return 0.0                     # identical input -> identical output
        key = tuple(np.round(pts, 6).ravel())
        if key not in cache:
            cache[key] = float(delta_fn(pts))
        return cache[key]

    delta_eval = np.array([delta_of([i]) for i in range(n)])

    Z = perturbation_vectors(n, n_perturb, rng)
    perturbed, delta = [], np.zeros(len(Z))
    for r, z in enumerate(Z):
        sel = np.flatnonzero(z)
        perturbed.append(apply_selection(x_km, noisy_km, segments, sel))
        delta[r] = delta_of(sel)
    w = dtw_weights(x_km, perturbed)
    thetas = {t: surrogate_theta(Z, delta, w, t, surrogate) for t in tau_list}
    return delta_eval, thetas, len(Z)
