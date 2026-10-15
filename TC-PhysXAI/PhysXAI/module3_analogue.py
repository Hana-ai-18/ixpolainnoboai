"""
TC-PhysXAI — Module 3: Analogue Uncertainty XAI
================================================
Physics Motivation
------------------
A flow-matching model generates K=30 ensemble trajectories by sampling different
initial noise z0 ~ N(0,I). These trajectories can cluster into distinct physical
scenarios: (a) continued westward propagation under steady ridge, (b) recurvature
under approaching trough, (c) stalled/looping under weak steering.

This module provides two complementary tools:

Tool A — DTW Cluster Attribution:
  1. Cluster K ensemble trajectories into M scenarios using DTW distance
  2. For each cluster, compute the cluster's MEAN contribution to CRPS degradation
  3. Attach physical label to each cluster: "STRAIGHT_WEST", "RECURVING_NE",
     "LOOPING", "UNKNOWN" based on endpoint displacement vector
  4. Attribution: which scenario is MOST LIKELY and how does each scenario
     contribute to prediction uncertainty?

Tool B — Latent Analogue Retrieval:
  1. Extract the latent context vector c = ContextEncoder(batch_list) ∈ R^d_cond
  2. Retrieve the K nearest analogues from a latent code library
  3. Overlay analogue tracks to show "what happened in similar past cases"
  4. Physical interpretation: are analogue typhoons also heading toward recurvature?

Reference: Analogous to "analogue forecasting" in NWP, but operating in the
learned latent space of TC-XFlow rather than raw ERA5 pattern space.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .utils import dtw_distance, dtw_distance_batch, crps_ensemble_fast, denorm_latlon


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryCluster:
    """One cluster of ensemble trajectories."""
    cluster_id: int
    member_indices: List[int]               # which ensemble members belong here
    centroid: np.ndarray                    # [T_pred, 2] mean trajectory
    probability: float                      # fraction of ensemble members
    physical_label: str                     # "STRAIGHT_WEST", "RECURVING_NE", etc.
    endpoint_displacement: Tuple[float, float]  # (Δlat, Δlon) in degrees
    intracluster_spread: float              # mean L2 spread of members around cluster centroid
    description: str                        # meteorological description


@dataclass
class AnalogueResult:
    """Result of latent-space analogue retrieval."""
    analogue_id: str                        # storm ID or index
    latent_distance: float                  # distance in context space
    analogue_trajectory: np.ndarray         # [T, 2] historical track (normalised)
    similarity_score: float                 # [0, 1] — 1 = identical context
    historical_outcome: str                 # "STRAIGHT", "RECURVED", "LOOPED"


# ─── DTW Clustering ──────────────────────────────────────────────────────────

def _cluster_trajectories_dtw(
    trajectories: np.ndarray,   # [K, T, 2]
    n_clusters: int = 3,
    max_iter: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-medoids clustering using DTW distances.

    K-medoids is preferred over K-means for trajectories because:
    - DTW distance is non-Euclidean (triangle inequality may fail)
    - Medoids are actual trajectories (interpretable)
    - More robust to outliers (looping/erratic trajectories)

    Returns
    -------
    labels    : [K] cluster assignments (0..n_clusters-1)
    medoids   : [n_clusters] indices of medoid members
    dist_mat  : [K, K] pairwise DTW distance matrix
    """
    K = len(trajectories)
    rng = np.random.default_rng(seed)

    # Compute full pairwise DTW matrix
    dist_mat = dtw_distance_batch(trajectories)  # [K, K]

    # Initialise medoids randomly
    medoids = rng.choice(K, size=min(n_clusters, K), replace=False)
    labels  = np.zeros(K, dtype=int)

    for _ in range(max_iter):
        # Assignment step
        for k in range(K):
            labels[k] = int(np.argmin(dist_mat[k, medoids]))

        # Update step: find best medoid within each cluster
        new_medoids = medoids.copy()
        for m in range(len(medoids)):
            cluster_idx = np.where(labels == m)[0]
            if len(cluster_idx) == 0:
                continue
            # Pick member minimising total DTW to others in cluster
            sub = dist_mat[np.ix_(cluster_idx, cluster_idx)]
            best_local = int(sub.sum(0).argmin())
            new_medoids[m] = cluster_idx[best_local]

        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids

    return labels, medoids, dist_mat


def _classify_trajectory_motion(
    traj: np.ndarray,   # [T, 2] ABSOLUTE normalised (lon_n, lat_n)
) -> Tuple[str, Tuple[float, float]]:
    """Classify track type from endpoint displacement.

    Uses denormalised lat/lon differences to identify motion regime.
    Degree thresholds below are heuristics (confirm with the advisor).
    """
    # (lon_n, lat_n), 1 unit = 5 deg  (see geo.py). Rules below use degrees.
    from .geo import norm_to_lonlat
    ll  = norm_to_lonlat(np.asarray(traj))
    lat = ll[:, 1]
    lon = ll[:, 0]

    # Total displacement from start to end
    d_lat = lat[-1] - lat[0]
    d_lon = lon[-1] - lon[0]

    # Curvature: check if trajectory bends poleward significantly
    if len(traj) >= 4:
        mid = len(traj) // 2
        d_lat_early = lat[mid] - lat[0]
        d_lat_late  = lat[-1] - lat[mid]
        curvature   = d_lat_late - d_lat_early  # positive = accelerating northward
    else:
        curvature = 0.0

    # Classify
    if d_lon < -5.0 and d_lat < 3.0 and curvature < 2.0:
        label = "STRAIGHT_WEST"
    elif curvature > 3.0 or (d_lat > 5.0 and d_lon > -3.0):
        label = "RECURVING_NE"
    elif abs(d_lon) < 2.0 and abs(d_lat) < 2.0:
        label = "STALLED"
    elif d_lat > 4.0 and d_lon < -5.0:
        label = "POLEWARD_TRACK"
    else:
        label = "MIXED"

    return label, (float(d_lat), float(d_lon))


def _cluster_to_physical_object(
    cluster: TrajectoryCluster,
) -> str:
    """Generate meteorological description for a track cluster."""
    label = cluster.physical_label
    p     = cluster.probability
    d_lat, d_lon = cluster.endpoint_displacement

    desc_map = {
        "STRAIGHT_WEST": (
            f"CONTINUED WESTWARD PROPAGATION ({p:.0%} probability): "
            f"Ensemble members follow a steady WNW track (Δlat={d_lat:+.1f}°, "
            f"Δlon={d_lon:+.1f}°). This scenario is consistent with a dominant "
            "500 hPa subtropical ridge maintaining strong easterly steering flow. "
            "The ridge axis remains southeast of the typhoon, providing sustained "
            "WNW guidance with minimal curvature. Track error is mainly in speed."
        ),
        "RECURVING_NE": (
            f"NORTHEAST RECURVATURE ({p:.0%} probability): "
            f"Ensemble members execute a recurvature into the mid-latitudes "
            f"(Δlat={d_lat:+.1f}°, Δlon={d_lon:+.1f}°). This requires the "
            "subtropical ridge to weaken or the typhoon to round its western "
            "terminus. An approaching 200 hPa trough from the west reduces the "
            "ridge and channels the storm into the mid-latitude westerlies. "
            "This is the highest-uncertainty scenario — small errors in ridge "
            "position translate to large track deviations."
        ),
        "STALLED": (
            f"STALLED / LOOPING ({p:.0%} probability): "
            "Ensemble members show near-zero net displacement — the typhoon is "
            "caught in a weak steering environment (col point between ridge and "
            "approaching trough). This typically precedes recurvature but can "
            "also indicate a bifurcation. Large uncertainty in this regime as "
            "small perturbations decide the eventual track."
        ),
        "POLEWARD_TRACK": (
            f"POLEWARD ACCELERATION ({p:.0%} probability): "
            f"Ensemble members show rapid northward motion (Δlat={d_lat:+.1f}°). "
            "This pattern is associated with the typhoon approaching the axis of "
            "a mid-latitude trough while simultaneously being deflected by an "
            "upper-level jet stream. Often associated with rapid motion increase "
            "to 40-60 kt translation speed."
        ),
        "MIXED": (
            f"MIXED STEERING ({p:.0%} probability): "
            f"Ensemble shows varied tracks (Δlat={d_lat:+.1f}°, Δlon={d_lon:+.1f}°). "
            "This multi-scenario cluster indicates competing synoptic influences "
            "not yet resolved by the flow field at current lead time."
        ),
    }

    return desc_map.get(label, f"Cluster type {label} ({p:.0%} prob).")


# ─── Main Module 3 ───────────────────────────────────────────────────────────

class AnalogueXAI:
    """Module 3: Analogue Uncertainty XAI for ensemble track prediction.

    Provides two complementary XAI tools:
      A. DTW-based clustering of ensemble trajectories into physical scenarios
      B. Latent-space analogue retrieval (requires analogue library)

    Usage
    -----
    >>> xai = AnalogueXAI(model, device)
    >>> result = xai.compute(batch_list, n_ensemble=30)
    >>> print(result['cluster_summary'])

    For analogue retrieval, build the library first:
    >>> xai.build_analogue_library(historical_batch_list, storm_ids)
    >>> result = xai.compute(batch_list, use_analogues=True)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_clusters: int = 3,
        n_analogues: int = 5,
    ):
        self.model       = model
        self.device      = device
        self.n_clusters  = n_clusters
        self.n_analogues = n_analogues
        self.model.eval()

        # Analogue library (built lazily)
        self._analogue_contexts: Optional[torch.Tensor] = None  # [N, d_cond]
        self._analogue_trajs:    Optional[np.ndarray]   = None  # [N, T, 2]
        self._analogue_ids:      Optional[List[str]]    = None

    # ─── Tool A: DTW Clustering ──────────────────────────────────────────────

    def _generate_ensemble(
        self, batch_list: list, K: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate K ensemble trajectories.

        Returns
        -------
        all_traj  : [K, T_pred, B, 2]
        pred_traj : [T_pred, B, 2]
        """
        with torch.no_grad():
            _, _, all_traj = self.model.sample(
                batch_list,
                num_ensemble=K,
                use_speed_calibration=True,
                return_xai=False,
            )
        pred_traj = batch_list[1]  # [T_pred, B, 2] ground truth
        return all_traj, pred_traj

    def cluster_ensemble(
        self,
        batch_list: list,
        K: int = 30,
        storm_idx: int = 0,
    ) -> List[TrajectoryCluster]:
        """Cluster K ensemble trajectories for a single storm.

        Parameters
        ----------
        storm_idx : which storm in the batch to analyse

        Returns
        -------
        List of TrajectoryCluster objects sorted by probability descending.
        """
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        all_traj, pred_traj = self._generate_ensemble(bl, K)

        # Extract for this storm: [K, T_pred, 2]
        traj_np = all_traj[:, :, storm_idx, :].cpu().numpy()
        gt_np   = pred_traj[:, storm_idx, :].cpu().numpy()

        # DTW clustering
        n_clust = min(self.n_clusters, K)
        labels, medoids, dist_mat = _cluster_trajectories_dtw(
            traj_np, n_clusters=n_clust
        )

        clusters = []
        for m_local, med_global in enumerate(medoids):
            member_idx = list(np.where(labels == m_local)[0])
            if not member_idx:
                continue

            member_trajs = traj_np[member_idx]          # [n_m, T, 2]
            centroid     = member_trajs.mean(0)         # [T, 2]
            prob         = len(member_idx) / K
            phys_label, disp = _classify_trajectory_motion(centroid)

            # Intra-cluster spread: mean L2 deviation of members from centroid.
            # (Note: this is NOT CRPS; it measures internal cluster compactness.)
            member_tensor = torch.tensor(member_trajs, dtype=torch.float32)  # [n_m, T, 2]
            if len(member_idx) > 1:
                spread_frac = float(
                    (member_tensor - centroid).norm(dim=-1).mean()
                )
            else:
                spread_frac = 0.0

            c = TrajectoryCluster(
                cluster_id=m_local,
                member_indices=member_idx,
                centroid=centroid,
                probability=prob,
                physical_label=phys_label,
                endpoint_displacement=disp,
                intracluster_spread=spread_frac,
                description="",  # fill below
            )
            c.description = _cluster_to_physical_object(c)
            clusters.append(c)

        clusters.sort(key=lambda c: c.probability, reverse=True)
        return clusters

    # ─── Tool B: Latent Analogue Retrieval ───────────────────────────────────

    @torch.no_grad()
    def _encode_context(self, batch_list: list) -> torch.Tensor:
        """Extract latent context vector c from ContextEncoder (model.encoder).

        Uses the same hard_score pattern as module2_spatial (C2) to ensure
        consistent access to TC-XFlow's ContextEncoder.

        Returns
        -------
        cond : [B, d_cond]
        """
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]
        from Model.flow_matching_model import hard_score_from_obs
        h_score = hard_score_from_obs(
            bl[0][:, :, :2], weight_logits=self.model.hard_score_weight_logits)
        cond = self.model.encoder(bl, hard_score=h_score)   # [B, d_cond]
        return cond

    def build_analogue_library(
        self,
        historical_batches: List[list],
        storm_ids: List[str],
        historical_trajs: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Build the analogue library from historical cases.

        Parameters
        ----------
        historical_batches : list of batch_list for historical storms
        storm_ids          : corresponding storm identifiers
        historical_trajs   : [T, 2] ground truth tracks (optional, for labelling)
        """
        contexts = []
        trajs    = []

        for i, bl in enumerate(historical_batches):
            try:
                c = self._encode_context(bl)      # [B, d_cond]
                contexts.append(c[0].cpu())       # take first storm

                if historical_trajs is not None:
                    trajs.append(historical_trajs[i])
                else:
                    # Use prediction trajectory as proxy
                    pred_t = bl[1][:, 0, :].cpu().numpy()  # [T, 2]
                    trajs.append(pred_t)
            except Exception as e:
                print(f"[AnalogueXAI] Skipped storm {storm_ids[i]}: {e}")

        if contexts:
            self._analogue_contexts = torch.stack(contexts, dim=0)  # [N, d_cond]
            self._analogue_trajs    = np.stack(trajs, axis=0)       # [N, T, 2]
            self._analogue_ids      = storm_ids[:len(contexts)]
            print(f"[AnalogueXAI] Library built: {len(contexts)} analogues.")

    def retrieve_analogues(
        self,
        batch_list: list,
        storm_idx: int = 0,
        K: Optional[int] = None,
    ) -> List[AnalogueResult]:
        """Retrieve top-K latent analogues for a query storm.

        Returns
        -------
        List of AnalogueResult sorted by similarity descending.
        """
        if self._analogue_contexts is None:
            return []

        K = K or self.n_analogues
        cond = self._encode_context(batch_list)     # [B, d_cond]
        query = cond[storm_idx].cpu()               # [d_cond]

        # L2 distances in context space
        dists = (self._analogue_contexts - query.unsqueeze(0)).norm(dim=-1)  # [N]
        top_k = dists.argsort()[:K]

        results = []
        max_dist = float(dists.max()) + 1e-8
        for idx in top_k.tolist():
            traj    = self._analogue_trajs[idx]  # [T, 2]
            dist    = float(dists[idx])
            sim     = 1.0 - dist / max_dist

            # Classify this analogue's outcome
            hist_label, _ = _classify_trajectory_motion(traj)
            label_map = {
                "STRAIGHT_WEST": "STRAIGHT",
                "RECURVING_NE":  "RECURVED",
                "STALLED":       "LOOPED",
                "POLEWARD_TRACK":"RECURVED",
                "MIXED":         "AMBIGUOUS",
            }
            outcome = label_map.get(hist_label, "UNKNOWN")

            results.append(AnalogueResult(
                analogue_id=self._analogue_ids[idx],
                latent_distance=dist,
                analogue_trajectory=traj,
                similarity_score=sim,
                historical_outcome=outcome,
            ))

        return results

    # ─── Main API ────────────────────────────────────────────────────────────

    def compute(
        self,
        batch_list: list,
        K: int = 30,
        storm_idx: int = 0,
        use_analogues: bool = False,
        verbose: bool = False,
    ) -> Dict:
        """Run complete Module 3 analysis.

        Returns
        -------
        dict with keys:
          "clusters"          : List[TrajectoryCluster]
          "cluster_summary"   : str
          "n_scenarios"       : int
          "dominant_scenario" : str
          "uncertainty_entropy": float  — Shannon entropy of cluster probs
          "analogues"         : List[AnalogueResult] (if use_analogues=True)
          "analogue_consensus": str     (if use_analogues=True)
          "physical_interpretation": str
        """
        bl = [t.to(self.device) if torch.is_tensor(t) else t for t in batch_list]

        if verbose:
            print("[AnalogueXAI] Clustering ensemble trajectories ...")
        clusters = self.cluster_ensemble(bl, K=K, storm_idx=storm_idx)

        # Shannon entropy of scenario probabilities
        probs   = np.array([c.probability for c in clusters])
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        max_entropy = np.log(len(clusters) + 1e-8)
        entropy_norm = entropy / max_entropy  # [0, 1]

        # Dominant scenario
        dominant = clusters[0].physical_label if clusters else "UNKNOWN"

        # Cluster summary string
        summary_lines = ["=== Ensemble Scenario Decomposition ==="]
        for c in clusters:
            summary_lines.append(
                f"\n[Scenario {c.cluster_id + 1}: {c.physical_label}] "
                f"p={c.probability:.1%}, n={len(c.member_indices)} members"
            )
            summary_lines.append(f"  {c.description}")
        cluster_summary = "\n".join(summary_lines)

        # Analogues
        analogues = []
        analogue_consensus = ""
        if use_analogues:
            if verbose:
                print("[AnalogueXAI] Retrieving latent analogues ...")
            analogues = self.retrieve_analogues(bl, storm_idx=storm_idx)
            if analogues:
                outcomes = [a.historical_outcome for a in analogues]
                outcome_counts: Dict[str, int] = {}
                for o in outcomes:
                    outcome_counts[o] = outcome_counts.get(o, 0) + 1
                majority = max(outcome_counts, key=outcome_counts.get)
                n_maj    = outcome_counts[majority]
                analogue_consensus = (
                    f"ANALOGUE CONSENSUS ({n_maj}/{len(analogues)} most similar historical storms): "
                    f"{majority}. "
                    f"Top analogue (sim={analogues[0].similarity_score:.2%}): "
                    f"{analogues[0].analogue_id} → {analogues[0].historical_outcome}."
                )

        # Physical interpretation
        phys_interp = self._interpret(clusters, entropy_norm, analogues)

        return {
            "clusters":              clusters,
            "cluster_summary":       cluster_summary,
            "n_scenarios":           len(clusters),
            "dominant_scenario":     dominant,
            "uncertainty_entropy":   entropy_norm,
            "analogues":             analogues,
            "analogue_consensus":    analogue_consensus,
            "physical_interpretation": phys_interp,
        }

    def _interpret(
        self,
        clusters: List[TrajectoryCluster],
        entropy_norm: float,
        analogues: List[AnalogueResult],
    ) -> str:
        """Generate physical interpretation of the ensemble scenario decomposition."""
        if not clusters:
            return "No distinct scenarios identified."

        dominant = clusters[0]
        lines    = []

        lines.append(
            f"PRIMARY SCENARIO: {dominant.physical_label} "
            f"({dominant.probability:.0%} of ensemble members)."
        )

        if entropy_norm < 0.3:
            lines.append(
                "LOW UNCERTAINTY REGIME: Ensemble strongly converges to a single "
                "track scenario. The atmospheric steering environment is unambiguous, "
                "likely a deep and persistent subtropical ridge with a well-defined "
                "steering flow. CRPS-based track errors will be dominated by "
                "intensity-dependent speed biases rather than direction uncertainty."
            )
        elif entropy_norm < 0.6:
            lines.append(
                "MODERATE UNCERTAINTY: Two competing scenarios exist with significant "
                "probability. This typically arises when the subtropical ridge is "
                "weakening on one flank and an approaching mid-latitude trough has "
                "not yet dominated the steering channel. 24-48h track errors will "
                "be large. Ensemble spread is meteorologically meaningful."
            )
        else:
            lines.append(
                "HIGH UNCERTAINTY / SCENARIO BIFURCATION: The ensemble splits into "
                "multiple distinct scenarios with nearly equal probability. This is "
                "the classic recurvature bifurcation problem — whether the typhoon "
                "rounds the western end of the ridge or continues westward depends "
                "sensitively on the exact position and intensity of the subtropical "
                "anticyclone and the timing of the approaching upper-level trough. "
                "Operational forecasters should weight the ensemble spread strongly."
            )

        if len(clusters) >= 2:
            s1, s2 = clusters[0], clusters[1]
            lines.append(
                f"\nSCENARIO CONTRAST: "
                f"{s1.physical_label} ({s1.probability:.0%}) vs "
                f"{s2.physical_label} ({s2.probability:.0%}). "
                f"Endpoint lat difference: "
                f"{abs(s1.endpoint_displacement[0] - s2.endpoint_displacement[0]):.1f}°."
            )

        if analogues:
            outcomes = [a.historical_outcome for a in analogues]
            outcome_counts: Dict[str, int] = {}
            for o in outcomes:
                outcome_counts[o] = outcome_counts.get(o, 0) + 1
            lines.append(
                f"\nHISTORICAL ANALOGUE EVIDENCE: Among {len(analogues)} nearest "
                f"latent-space analogues: "
                + ", ".join(f"{v} × {k}" for k, v in outcome_counts.items()) + "."
            )

        return "\n".join(lines)

    def save_analogue_library(self, path: str) -> None:
        """Save analogue library to disk."""
        if self._analogue_contexts is None:
            print("[AnalogueXAI] No library to save.")
            return
        np.savez(
            path,
            contexts=self._analogue_contexts.numpy(),
            trajs=self._analogue_trajs,
            ids=np.array(self._analogue_ids),
        )
        print(f"[AnalogueXAI] Library saved to {path}.")

    def load_analogue_library(self, path: str) -> None:
        """Load analogue library from disk."""
        data = np.load(path, allow_pickle=True)
        self._analogue_contexts = torch.tensor(data["contexts"])
        self._analogue_trajs    = data["trajs"]
        self._analogue_ids      = list(data["ids"])
        print(f"[AnalogueXAI] Library loaded: {len(self._analogue_ids)} analogues.")
