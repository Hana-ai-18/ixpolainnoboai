"""
TC-PhysXAI — Meteorological Interpreter
========================================
Synthesises outputs from all modules (M1, M2, M3, C3-C6) into a coherent,
publication-quality meteorological narrative.

The interpreter follows the diagnostic framework of tropical cyclone track
prediction guidance used by JMA, NHC, and ECMWF:

  Level 1 — Synoptic Situation:
    What is the large-scale steering environment?
    (500 hPa ridge position, approaching trough, monsoon trough)

  Level 2 — Multi-level Steering Analysis:
    Which pressure levels are most influential?
    (Deep steering: 500/300 hPa, Surface-level: 850/925 hPa)

  Level 3 — Temporal Evolution:
    Which observation window most determines the track?
    (Recent 0-12h: stable regime; Remote 30-48h: approaching trough)

  Level 4 — Ensemble Scenario Decomposition:
    What are the competing track scenarios and their probabilities?
    (Straight-west vs. recurvature bifurcation)

  Level 5 — Counterfactual Analysis:
    What change in the atmosphere would alter the forecast?
    (Minimum ridge weakening required for recurvature)

  Level 6 — Forecast Confidence Assessment:
    How reliable is the probabilistic forecast?
    (H-mean-F score, spread-error ratio)

Output Format:
  The interpreter produces a structured meteorological bulletin,
  following the style of JMA Tropical Cyclone Advisory text products.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .physical_objects import PhysicalObject


@dataclass
class MeteoBulletin:
    """Complete meteorological bulletin for a TC track case."""
    typhoon_id:       str
    analysis_time:    str      # e.g. "2019-09-07 00Z"
    synoptic_header:  str
    level1_synoptic:  str
    level2_steering:  str
    level3_temporal:  str
    level4_scenarios: str
    level5_cf:        str
    level6_confidence:str
    full_text:        str       # concatenated bulletin


_STEERING_LEVEL_DESCRIPTIONS = {
    "200hPa": "upper-tropospheric (200 hPa) divergent outflow layer",
    "500hPa": "mid-tropospheric (500 hPa) primary steering layer",
    "850hPa": "lower-tropospheric (850 hPa) boundary layer flow",
    "925hPa": "near-surface (925 hPa) boundary layer flow",
    "SST":    "sea-surface temperature (SST) field",
}

_REGIME_TEMPLATES = {
    "DEEP-TROPOSPHERIC STEERING": (
        "The typhoon track is primarily governed by DEEP-TROPOSPHERIC STEERING. "
        "Attribution weights are broadly distributed across 200-500 hPa, indicating "
        "that the entire tropospheric column participates in steering. This is typical "
        "of a well-developed, deep typhoon (Category 3+) where the vortex extends from "
        "the planetary boundary layer to the tropopause. The steering layer depth "
        "averages over multiple pressure levels, reducing track sensitivity to "
        "individual level-specific features."
    ),
    "UPPER-TROPOSPHERE DOMINANT": (
        "UPPER-TROPOSPHERIC PROCESSES (200 hPa) dominate the track attribution. "
        "This pattern suggests the presence of an upper-level outflow channel or "
        "approaching upper trough. The 200 hPa flow directly influences typhoon "
        "venting and divergent outflow — when an upper trough approaches, the "
        "divergent outflow is enhanced downshear, modifying intensity and potentially "
        "triggering recurvature. The track model correctly identifies this upper-level "
        "signal as the primary steering indicator."
    ),
    "BOUNDARY-LAYER DOMINATED": (
        "BOUNDARY-LAYER CIRCULATION (850-925 hPa) dominates the track attribution. "
        "This low-level steering dominance is characteristic of: (a) a shallow typhoon "
        "early in its development phase, (b) a storm interacting with the monsoon trough "
        "at low levels, or (c) strong land interaction where terrain-induced low-level "
        "jets modify the immediate steering current. The low attribution from upper "
        "levels may indicate a decoupled upper/lower troposphere environment, which "
        "typically accompanies high vertical wind shear."
    ),
    "SST-MODULATED": (
        "SEA-SURFACE TEMPERATURE (SST) PLAYS A SIGNIFICANT ROLE in the track "
        "attribution — an unusual but physically meaningful result. SST modulates "
        "track prediction through: (1) differential surface heat fluxes affecting "
        "boundary layer stability and storm intensity (which in turn affects translation "
        "speed); (2) SST gradient zones (fronts) causing organised convective "
        "asymmetries that deflect the track through ITCZ-like channelling effects; "
        "(3) the warm pool extent determining how long the storm can maintain "
        "intensity before encountering cool SST and weakening. The model's attention "
        "to SST is physically justified for typhoons crossing the Kuroshio Extension."
    ),
}


class MeteorologyInterpreter:
    """Synthesises XAI outputs into meteorological narratives.

    This class takes outputs from all TC-PhysXAI modules and produces
    a structured meteorological bulletin in the style of operational advisories.

    Usage
    -----
    >>> interp = MeteorologyInterpreter()
    >>> bulletin = interp.synthesise(
    ...     typhoon_id="HAGIBIS_2019",
    ...     temporal_result=temporal_result,
    ...     spatial_result=spatial_result,
    ...     analogue_result=analogue_result,
    ...     fidelity_result=fidelity_result,
    ...     cf_result=cf_result,
    ... )
    >>> print(bulletin.full_text)
    """

    def __init__(self):
        pass

    # ─── Level 1: Synoptic Situation ─────────────────────────────────────────

    def _level1_synoptic(
        self,
        spatial_result: Optional[Dict],
        temporal_result: Optional[Dict],
    ) -> str:
        """Generate synoptic situation header from dominant physical objects."""
        lines = ["[LEVEL 1 — SYNOPTIC SITUATION]\n"]

        if spatial_result and "physical_objects" in spatial_result:
            objects: List[PhysicalObject] = spatial_result["physical_objects"]
            if objects:
                top_obj = objects[0]
                lines.append(
                    f"PRIMARY STEERING FEATURE: {top_obj.type.upper()} at "
                    f"{top_obj.level} (attribution weight: {top_obj.attribution_weight:.1%})"
                )
                lines.append(top_obj.description)
                lines.append("")

                if len(objects) >= 2:
                    lines.append(f"SECONDARY FEATURE: {objects[1].type.upper()} at {objects[1].level}")
                    lines.append(objects[1].description[:200] + "..." if len(objects[1].description) > 200 else objects[1].description)

        if temporal_result and "physical_interpretation" in temporal_result:
            lines.append("")
            lines.append("TEMPORAL CONTEXT:")
            lines.append(temporal_result["physical_interpretation"])

        return "\n".join(lines)

    # ─── Level 2: Multi-level Steering Analysis ───────────────────────────────

    def _level2_steering(self, spatial_result: Optional[Dict]) -> str:
        """Generate multi-level steering analysis from IG attribution."""
        lines = ["\n[LEVEL 2 — MULTI-LEVEL STEERING ANALYSIS]\n"]

        if spatial_result is None:
            lines.append("Spatial attribution not available.")
            return "\n".join(lines)

        level_imp = spatial_result.get("level_importance", {})
        if level_imp:
            lines.append("ERA5 Level Attribution (% of total):")
            sorted_levels = sorted(level_imp.items(), key=lambda x: x[1], reverse=True)
            for level, imp in sorted_levels:
                bar_len = int(imp * 40)
                bar     = "█" * bar_len + "░" * (40 - bar_len)
                desc    = _STEERING_LEVEL_DESCRIPTIONS.get(level, level)
                lines.append(f"  {level:8s} [{bar}] {imp:.1%}")
                lines.append(f"           → {desc}")

        if "physical_interpretation" in spatial_result:
            lines.append("")
            regime = spatial_result.get("steering_regime", "")
            if regime in _REGIME_TEMPLATES:
                lines.append(f"STEERING REGIME: {regime}")
                lines.append(_REGIME_TEMPLATES[regime])
            else:
                lines.append(spatial_result["physical_interpretation"])

        return "\n".join(lines)

    # ─── Level 3: Temporal Evolution ─────────────────────────────────────────

    def _level3_temporal(self, temporal_result: Optional[Dict]) -> str:
        """Describe which observation windows drive track predictability."""
        lines = ["\n[LEVEL 3 — TEMPORAL ATTRIBUTION]\n"]

        if temporal_result is None:
            lines.append("Temporal attribution not available.")
            return "\n".join(lines)

        labels = temporal_result.get("window_time_labels", [])
        scores = temporal_result.get("mean_attribution_per_window", np.array([]))
        best_w = temporal_result.get("most_important_window", 0)

        if len(scores) > 0:
            lines.append("Attribution by observation window:")
            max_s = max(scores.max(), 1e-8)
            for i, (lab, sc) in enumerate(zip(labels, scores)):
                bar_len = int(sc / max_s * 30)
                marker  = " ← MOST IMPORTANT" if i == best_w else ""
                lines.append(
                    f"  {lab:8s} [{'█'*bar_len}{'░'*(30-bar_len)}] {sc:.4f}{marker}"
                )

        lines.append("")
        lines.append(temporal_result.get("physical_interpretation", ""))
        return "\n".join(lines)

    # ─── Level 4: Ensemble Scenarios ─────────────────────────────────────────

    def _level4_scenarios(self, analogue_result: Optional[Dict]) -> str:
        """Describe ensemble scenario decomposition."""
        lines = ["\n[LEVEL 4 — ENSEMBLE SCENARIO ANALYSIS]\n"]

        if analogue_result is None:
            lines.append("Ensemble scenario analysis not available.")
            return "\n".join(lines)

        lines.append(
            f"Forecast Uncertainty (entropy): "
            f"{analogue_result.get('uncertainty_entropy', 0):.3f} "
            f"(0=certain, 1=maximally uncertain)"
        )
        lines.append(
            f"Dominant scenario: {analogue_result.get('dominant_scenario', 'UNKNOWN')}"
        )
        lines.append("")
        lines.append(analogue_result.get("cluster_summary", ""))

        if analogue_result.get("analogue_consensus"):
            lines.append("")
            lines.append("HISTORICAL ANALOGUES:")
            lines.append(analogue_result["analogue_consensus"])

        return "\n".join(lines)

    # ─── Level 5: Counterfactual ──────────────────────────────────────────────

    def _level5_counterfactual(self, cf_result) -> str:
        """Describe counterfactual atmospheric change."""
        lines = ["\n[LEVEL 5 — COUNTERFACTUAL ANALYSIS]\n"]

        if cf_result is None:
            lines.append("Counterfactual analysis not available.")
            return "\n".join(lines)

        lines.append(
            f"Counterfactual type: {cf_result.counterfactual_type}"
        )
        lines.append(
            f"Maximum implied track shift: {cf_result.track_delta_km:.0f} km"
        )
        lines.append(
            f"Minimum context change required: {cf_result.context_change_norm:.1%}"
        )
        lines.append("")
        lines.append(cf_result.physical_explanation)

        return "\n".join(lines)

    # ─── Level 6: Forecast Confidence ─────────────────────────────────────────

    def _level6_confidence(self, fidelity_result) -> str:
        """Report forecast confidence and explanation reliability."""
        lines = ["\n[LEVEL 6 — FORECAST CONFIDENCE ASSESSMENT]\n"]

        if fidelity_result is None:
            lines.append("Fidelity metric not available.")
            return "\n".join(lines)

        lines.append(
            f"Probabilistic Fidelity (H-mean-F): {fidelity_result.h_mean_f:.3f}"
        )
        lines.append(
            f"  Precision-F  = {fidelity_result.precision_f:.3f} "
            f"(feature necessity)"
        )
        lines.append(
            f"  Recall-F     = {fidelity_result.recall_f:.3f} "
            f"(feature sufficiency)"
        )
        lines.append(
            f"  Uncertainty-F = {fidelity_result.uncertainty_f:.3f} "
            f"(calibration)"
        )
        lines.append("")
        lines.append(fidelity_result.interpretation)

        return "\n".join(lines)

    # ─── Main API ─────────────────────────────────────────────────────────────

    def synthesise(
        self,
        typhoon_id: str,
        analysis_time: str = "N/A",
        temporal_result: Optional[Dict] = None,
        spatial_result: Optional[Dict] = None,
        analogue_result: Optional[Dict] = None,
        fidelity_result=None,
        cf_result=None,
        variance_result=None,
        fm_attr_result=None,
    ) -> MeteoBulletin:
        """Synthesise all XAI module outputs into a complete meteorological bulletin.

        Parameters
        ----------
        typhoon_id      : TC identifier (e.g. "HAGIBIS_2019" or "2019-17W")
        analysis_time   : ISO datetime string for the analysis
        temporal_result : output of TemporalAttributor.compute()
        spatial_result  : output of SpatialAttributor.compute()
        analogue_result : output of AnalogueXAI.compute()
        fidelity_result : output of ProbabilisticFidelity.compute()
        cf_result       : output of FMCounterfactual.compute()
        variance_result : output of VarianceDecomposition.compute()
        fm_attr_result  : output of FMIntegrationAttribution.compute()

        Returns
        -------
        MeteoBulletin with all six levels plus full_text
        """
        header = (
            f"\n{'='*70}\n"
            f"TC-PhysXAI METEOROLOGICAL EXPLANATION BULLETIN\n"
            f"{'='*70}\n"
            f"Typhoon ID    : {typhoon_id}\n"
            f"Analysis Time : {analysis_time}\n"
            f"{'='*70}\n"
            f"Generated by TC-PhysXAI v1.1 (Nguyen Hong Gam, 2025)\n"
            f"Based on TC-XFlow (ESWA 2025): Conditional Flow Matching for\n"
            f"Typhoon Track Prediction with Physics-Guided XAI\n"
            f"{'='*70}\n"
        )

        l1 = self._level1_synoptic(spatial_result, temporal_result)
        l2 = self._level2_steering(spatial_result)
        l3 = self._level3_temporal(temporal_result)
        l4 = self._level4_scenarios(analogue_result)
        l5 = self._level5_counterfactual(cf_result)
        l6 = self._level6_confidence(fidelity_result)

        # Optional C5 appendix
        appendix = ""
        if variance_result is not None:
            appendix += (
                "\n[APPENDIX A — VARIANCE DECOMPOSITION (C5)]\n"
                + variance_result.physical_interpretation
            )
        if fm_attr_result is not None:
            appendix += (
                "\n[APPENDIX B — FM INTEGRATION-PATH ATTRIBUTION (C4)]\n"
                + fm_attr_result.physical_interpretation
            )

        footer = (
            f"\n{'='*70}\n"
            f"END OF BULLETIN — {typhoon_id}\n"
            f"{'='*70}\n"
        )

        full_text = header + l1 + l2 + l3 + l4 + l5 + l6 + appendix + footer

        return MeteoBulletin(
            typhoon_id=typhoon_id,
            analysis_time=analysis_time,
            synoptic_header=header,
            level1_synoptic=l1,
            level2_steering=l2,
            level3_temporal=l3,
            level4_scenarios=l4,
            level5_cf=l5,
            level6_confidence=l6,
            full_text=full_text,
        )

    def quick_summary(
        self,
        spatial_result: Optional[Dict] = None,
        temporal_result: Optional[Dict] = None,
        analogue_result: Optional[Dict] = None,
    ) -> str:
        """Generate a compact 3-sentence summary for figure captions."""
        parts = []

        if spatial_result:
            regime = spatial_result.get("steering_regime", "")
            objs   = spatial_result.get("physical_objects", [])
            if objs:
                top_obj = objs[0]
                parts.append(
                    f"The track prediction is primarily controlled by "
                    f"{top_obj.type.replace('_', ' ')} at {top_obj.level} "
                    f"({top_obj.attribution_weight:.0%} of total attribution)."
                )

        if temporal_result:
            best_w = temporal_result.get("most_important_window", 7)
            labels = temporal_result.get("window_time_labels", [])
            label  = labels[best_w] if labels else "t-0h"
            parts.append(
                f"Temporal attribution identifies {label} as the most influential "
                "observation window, indicating the immediate steering environment "
                "dominates track predictability."
            )

        if analogue_result:
            dom = analogue_result.get("dominant_scenario", "UNKNOWN")
            ent = analogue_result.get("uncertainty_entropy", 0.5)
            parts.append(
                f"Ensemble scenario decomposition shows {dom} as the dominant "
                f"trajectory pattern with uncertainty entropy {ent:.2f}."
            )

        return " ".join(parts) if parts else "Insufficient data for summary."
