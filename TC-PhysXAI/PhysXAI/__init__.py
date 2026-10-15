"""
TC-PhysXAI: Physics-Guided Explainable AI for Tropical Cyclone Track Prediction
=================================================================================
Built on TC-XFlow (ESWA 2025) — Nguyen Hong Gam et al.

Architecture:
  Module 1  — Temporal Attribution         (C1: climatological-baseline perturbation)
  Module 2  — Spatial Object Attribution   (C2: Integrated Gradients + physical object detection)
  Module 3  — Analogue Uncertainty XAI     (DTW clustering + latent analogue retrieval)
  C3        — Probabilistic Fidelity Metric (H-mean-F)
  C4        — FM-native integration-path attribution (FreeFM principle)
  C5        — Noise-Context Variance Decomposition (law of total variance)
  C6        — FM-native counterfactual via ODE adjoint (projected gradient)
  C7        — Segment-Level Temporal XAI + AOPC Faithfulness (XAI4Traj-inspired)
  C8        — Track Feature Attribution: predict features → physical drivers
"""

__version__ = "1.4.3"
__author__  = "Nguyen Hong Gam"

from .module1_temporal        import TemporalAttributor
from .module2_spatial         import SpatialAttributor
from .module3_analogue        import AnalogueXAI
from .c4_fm_attribution       import FMIntegrationAttribution
from .c5_variance_decomp      import VarianceDecomposition
from .c6_counterfactual       import FMCounterfactual
from .c7_segment_perturbation import (
    SegmentPerturbationXAI,
    AdaptivePhaseDiscovery,
    GlobalAttributionMiner,
    compute_segment_xai,
    compute_adaptive_and_fixed,
    # dataclasses
    SegmentXAIResult,
    AdaptivePhaseConfig,
    AdaptiveDiscoveryResult,
    TrackTypeStats,
    GlobalAttributionResult,
)
from .c8_track_feature_attribution import (
    TrackFeatureAttributor,
    TrackFeatureExtractor,
    compute_track_feature_attribution,
    TrackFeatures,
    FeatureAttributionResult,
)
from .fidelity_metric         import ProbabilisticFidelity
from .meteorological_interp   import MeteorologyInterpreter

__all__ = [
    # Modules
    "TemporalAttributor",
    "SpatialAttributor",
    "AnalogueXAI",
    "FMIntegrationAttribution",
    "VarianceDecomposition",
    "FMCounterfactual",
    "ProbabilisticFidelity",
    "MeteorologyInterpreter",
    # C7 — Segment-Level XAI (enhanced)
    "SegmentPerturbationXAI",
    "AdaptivePhaseDiscovery",
    "GlobalAttributionMiner",
    "compute_segment_xai",
    "compute_adaptive_and_fixed",
    # C8 — Track Feature Attribution
    "TrackFeatureAttributor",
    "TrackFeatureExtractor",
    "compute_track_feature_attribution",
    "TrackFeatures",
    "FeatureAttributionResult",
    # C7 dataclasses
    "SegmentXAIResult",
    "AdaptivePhaseConfig",
    "AdaptiveDiscoveryResult",
    "TrackTypeStats",
    "GlobalAttributionResult",
]
