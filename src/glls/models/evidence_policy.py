from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVisibilityDecision:
    prompt_visible: bool
    reason: str


class EvidencePromptPolicy:
    """Central prompt-visibility policy for derived evidence.

    Derived priors can be useful audit data, but they should not automatically
    become prompt-visible hints. Keeping the decision here prevents answer-option
    mappings or Phase-1 hallucinations from leaking into scattered prompt code.
    """

    @staticmethod
    def pvla_hypothesis_visibility(
        *,
        has_hypotheses: bool,
        local_crop_count: int,
        used_logic_engine: bool,
        allow_with_local_evidence: bool = False,
    ) -> PromptVisibilityDecision:
        if not has_hypotheses:
            return PromptVisibilityDecision(False, "no_pvla_visual_hypothesis")
        if used_logic_engine:
            return PromptVisibilityDecision(True, "logic_engine_verdict_source")
        if allow_with_local_evidence:
            return PromptVisibilityDecision(True, "source_backed_hypothesis_for_local_verification")
        if int(local_crop_count or 0) > 0:
            return PromptVisibilityDecision(False, "local_evidence_first_phase1_audit_only")
        return PromptVisibilityDecision(True, "no_local_evidence_fallback")

    @staticmethod
    def spatial_option_audit_visibility(has_option_audit: bool) -> PromptVisibilityDecision:
        if not has_option_audit:
            return PromptVisibilityDecision(False, "no_coordinate_option_audit")
        return PromptVisibilityDecision(False, "coordinate_option_mapping_audit_only")

    @staticmethod
    def spatial_localization_instruction() -> str:
        return (
            "Treat the red-contour coordinates as approximate localization descriptors only. "
            "Map location answer choices to the visible highlighted region yourself; do not rely "
            "on a precomputed option match. Always verify choices that mention visible parts, colors, "
            "text/imprints, seams, edges, caps, or ends; select the option whose visible part is "
            "actually highlighted."
        )
