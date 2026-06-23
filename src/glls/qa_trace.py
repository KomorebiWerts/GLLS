"""Compact method trace helpers for the MVTec/VisA QA pipeline."""

from __future__ import annotations

from typing import Any

from glls.rag.semantic_grounding import is_normal_or_no_defect_report


def summarize_method_participation(debug_meta: dict[str, Any]) -> dict[str, Any]:
    task_policy = debug_meta.get("task_policy") or {}
    mcts_action_trace = debug_meta.get("mcts_action_trace") or []
    region_count = int(debug_meta.get("region_proposal_count") or 0)
    crop_count = int(debug_meta.get("crop_count") or 0)
    if mcts_action_trace:
        mcts_status = "searched"
    elif task_policy.get("use_mcts_search") is False:
        mcts_status = "disabled_by_task_policy"
    elif region_count <= 0:
        mcts_status = "skipped_no_region_proposals"
    elif crop_count <= 0:
        mcts_status = "skipped_no_selected_crops"
    else:
        mcts_status = "missing_action_trace"

    sam_scores = debug_meta.get("sam_mask_scores") or []
    sam_score_dicts = [item for item in sam_scores if isinstance(item, dict)]
    if debug_meta.get("sam_text_prompt_hits"):
        sam_status = "text_prompt_refinement"
    elif (debug_meta.get("sam3_normal_part_veto") or {}).get("status") == "attempted":
        sam_status = "normal_part_veto"
    elif int(debug_meta.get("sam_refined_crop_count") or 0) > 0:
        sam_status = "box_prompt_refinement"
    elif sam_score_dicts:
        sam_status = "attempted_no_selected_mask"
    elif debug_meta.get("sam_local_refinement_enabled") is False:
        sam_status = "disabled_by_policy"
    else:
        sam_status = "not_attempted"

    rag_blocks = debug_meta.get("rag_block_provenance") or []
    rag_dicts = [item for item in rag_blocks if isinstance(item, dict)]
    graph_path_count = sum(1 for item in rag_dicts if item.get("graph_cache_path"))
    text_json_count = sum(1 for item in rag_dicts if item.get("source_json_path"))
    visual_source_backed_count = sum(1 for item in rag_dicts if item.get("visual_reference_source_backed"))
    return {
        "mcts_participation_status": mcts_status,
        "sam_participation_status": sam_status,
        "rag_source_backed_summary": {
            "block_count": len(rag_dicts),
            "graph_cache_path_count": graph_path_count,
            "source_json_path_count": text_json_count,
            "visual_reference_source_backed_count": visual_source_backed_count,
            "all_blocks_have_graph_and_text_source": bool(
                rag_dicts and graph_path_count == len(rag_dicts) and text_json_count == len(rag_dicts)
            ),
        },
    }


def build_method_trace(
    debug_meta: dict[str, Any],
    mcts_output: dict[str, Any],
    strategy: str,
    instruction_text: str | None = None,
    response: str | None = None,
    *,
    trace_level: str = "method",
) -> dict[str, Any]:
    """Build the stored QA trace.

    The default trace keeps only method-relevant provenance: heatmap/proposal
    evidence, MCTS search, SAM3 refinement, PVLA/RAG sources, prompt-visible
    block selection, and the final binary anomaly gate. Full prompt and raw
    model text are opt-in because they are bulky and are not needed to prove
    method participation.
    """

    participation = summarize_method_participation(debug_meta)
    trace = {
        "verification_strategy": strategy,
        "heatmap_score": debug_meta.get("heatmap_peak_score", 0),
        "threshold_used": debug_meta.get("anomaly_threshold", 0),
        "pixel_threshold": debug_meta.get("pixel_threshold", 0),
        "threshold_source": debug_meta.get("threshold_source", "unknown"),
        "category": debug_meta.get("category", ""),
        "dataset_category": debug_meta.get("dataset_category", ""),
        "localizer_detected_category": debug_meta.get("localizer_detected_category", ""),
        "category_source": debug_meta.get("category_source", ""),
        "region_proposal_count": debug_meta.get("region_proposal_count", 0),
        "region_proposal_sources": debug_meta.get("region_proposal_sources", []),
        "region_proposal_audit": debug_meta.get("region_proposal_audit", []),
        "crop_candidate_audit": debug_meta.get("crop_candidate_audit", []),
        "skipped_crop_candidates": debug_meta.get("skipped_crop_candidates", []),
        "crop_evidence_audit": debug_meta.get("crop_evidence_audit", []),
        "raw_proposal_fill_reasons": debug_meta.get("raw_proposal_fill_reasons", []),
        "show_crop_images_in_final_prompt": debug_meta.get("show_crop_images_in_final_prompt", True),
        "prompt_visible_crop_count": debug_meta.get("prompt_visible_crop_count", debug_meta.get("crop_count", 0)),
        "prompt_crop_selection_audit": debug_meta.get("prompt_crop_selection_audit", []),
        "prompt_crop_labels": debug_meta.get("prompt_crop_labels", []),
        "local_evidence_prompt_policy": debug_meta.get("local_evidence_prompt_policy", ""),
        "mcts_budget_config": debug_meta.get("mcts_budget_config", {}),
        "mcts_action_trace": debug_meta.get("mcts_action_trace", []),
        "mcts_search_summary": debug_meta.get("mcts_search_summary", {}),
        "sam_refined_crop_count": debug_meta.get("sam_refined_crop_count", 0),
        "sam_text_refined_crop_count": debug_meta.get("sam_text_refined_crop_count", 0),
        "sam_local_refinement_enabled": debug_meta.get("sam_local_refinement_enabled", True),
        "sam_mask_scores": debug_meta.get("sam_mask_scores", []),
        "sam_profile": debug_meta.get("sam_profile", {}),
        "sam_prompt_selection_audit": debug_meta.get("sam_prompt_selection_audit", []),
        "sam_text_prompt_attempts": debug_meta.get("sam_text_prompt_attempts", []),
        "sam_text_prompt_hits": debug_meta.get("sam_text_prompt_hits", []),
        "sam3_normal_part_veto": debug_meta.get("sam3_normal_part_veto", {}),
        "sam3_crop_artifact_rule": debug_meta.get("sam3_crop_artifact_rule", ""),
        "show_rag_blocks_in_final_prompt": debug_meta.get("show_rag_blocks_in_final_prompt", True),
        "prompt_rag_selection_audit": debug_meta.get("prompt_rag_selection_audit", {}),
        "prompt_rag_text_blocks": debug_meta.get("prompt_rag_text_blocks", []),
        "rag_retrieval_mode": debug_meta.get("rag_retrieval_mode", "full"),
        "rag_block_provenance": debug_meta.get("rag_block_provenance", []),
        "rag_source_backed_summary": participation["rag_source_backed_summary"],
        "pvla_hypothesis_prompt_visible": debug_meta.get("pvla_hypothesis_prompt_visible", False),
        "pvla_hypothesis_prompt_reason": debug_meta.get("pvla_hypothesis_prompt_reason", ""),
        "show_pvla_hypothesis": debug_meta.get("show_pvla_hypothesis", False),
        "weak_localization_phase1": debug_meta.get("weak_localization_phase1", False),
        "spatial_candidate_report": debug_meta.get("spatial_candidate_report", ""),
        "spatial_option_prior_prompt_visible": debug_meta.get("spatial_option_prior_prompt_visible", False),
        "spatial_option_prior_prompt_reason": debug_meta.get("spatial_option_prior_prompt_reason", ""),
        "binary_anomaly_decision_override": debug_meta.get("binary_anomaly_decision_override", {}),
        "used_logic_engine": debug_meta.get("used_logic_engine", False),
        "task_policy": debug_meta.get("task_policy", {}),
        "method_participation_summary": participation,
        "mcts_participation_status": participation["mcts_participation_status"],
        "sam_participation_status": participation["sam_participation_status"],
        "phase_1_result": debug_meta.get("phase_1_conclusion", "N/A"),
        "did_crop": debug_meta.get("is_crop_triggered", False),
        "crop_count": debug_meta.get("crop_count", 0),
    }

    if str(trace_level or "method").lower() == "full":
        trace.update({
            "suppress_prompt_crop_labels": debug_meta.get("suppress_prompt_crop_labels", False),
            "sam3_debug_items": mcts_output.get("sam3_debug_items", []),
            "pvla_option_grounding_summary": debug_meta.get("pvla_option_grounding_summary", ""),
            "pvla_evidence_hypothesis": debug_meta.get("pvla_evidence_hypothesis", ""),
            "pvla_evidence_hypothesis_audit": debug_meta.get("pvla_evidence_hypothesis_audit", []),
            "spatial_option_prior": debug_meta.get("spatial_option_prior", ""),
            "spatial_primary_phrase": debug_meta.get("spatial_primary_phrase", ""),
            "spatial_primary_terms": debug_meta.get("spatial_primary_terms", []),
            "phase1_option_prior": debug_meta.get("phase1_option_prior", ""),
            "phase1_analysis_prior": debug_meta.get("phase1_analysis_prior", ""),
            "phase1_visual_primitive_hint": debug_meta.get("phase1_visual_primitive_hint", ""),
            "phase1_visual_primitive_hint_audit": debug_meta.get("phase1_visual_primitive_hint_audit", []),
            "crop_labels": debug_meta.get("crop_labels", []),
            "final_prompt": instruction_text or "",
            "raw_response": response or "",
        })
    return trace


def _yes_no_option_keys(options: dict[str, Any] | None) -> tuple[str | None, str | None]:
    yes_key = None
    no_key = None
    for key, value in (options or {}).items():
        text = str(value or "").strip().lower().rstrip(".")
        if text == "yes":
            yes_key = str(key)
        elif text == "no":
            no_key = str(key)
    return yes_key, no_key


def binary_anomaly_decision_override(
    task_type: str | None,
    options: dict[str, Any] | None,
    debug_meta: dict[str, Any],
    pred_key: str,
) -> dict[str, Any] | None:
    if str(task_type or "").strip().lower() != "anomaly detection":
        return None
    yes_key, no_key = _yes_no_option_keys(options)
    if not yes_key or not no_key:
        return None

    region_count = int(debug_meta.get("region_proposal_count", 0) or 0)
    crop_count = int(debug_meta.get("crop_count", 0) or 0)
    try:
        heatmap_score = float(debug_meta.get("heatmap_peak_score", 0.0) or 0.0)
        threshold = float(debug_meta.get("anomaly_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        heatmap_score = 0.0
        threshold = 0.0

    local_evidence = bool(region_count > 0 and crop_count > 0 and heatmap_score >= threshold)
    phase1_normal = is_normal_or_no_defect_report(str(debug_meta.get("phase_1_conclusion", "") or ""))
    score_gap = threshold - heatmap_score
    uncertainty_margin = max(0.01, 0.01 * threshold) if threshold > 0 else 0.0
    clearly_below_threshold = threshold > 0 and score_gap >= uncertainty_margin
    sam3_normal_part_veto = debug_meta.get("sam3_normal_part_veto") or {}

    base = {
        "policy": "binary_anomaly_local_evidence_gate_uncertainty_band",
        "original_pred_key": pred_key,
        "yes_key": yes_key,
        "no_key": no_key,
        "local_evidence": local_evidence,
        "region_proposal_count": region_count,
        "crop_count": crop_count,
        "heatmap_score": round(float(heatmap_score), 6),
        "threshold": round(float(threshold), 6),
        "score_gap": round(float(score_gap), 6),
        "uncertainty_margin": round(float(uncertainty_margin), 6),
        "phase1_normal": phase1_normal,
    }

    if not local_evidence and pred_key == yes_key and sam3_normal_part_veto.get("veto_yes_to_no"):
        return {
            **base,
            "applied": True,
            "override_pred_key": no_key,
            "sam3_normal_part_veto": sam3_normal_part_veto,
            "reason": "sam3_normal_part_veto_corrects_yes_to_no",
        }
    if not local_evidence and pred_key == yes_key and phase1_normal and clearly_below_threshold:
        return {
            **base,
            "applied": True,
            "override_pred_key": no_key,
            "reason": "clearly_below_threshold_and_normal_phase1_corrects_yes_to_no",
        }

    if not local_evidence or pred_key == yes_key:
        reason = "already_yes_with_local_evidence"
        if not local_evidence:
            reason = "no_override_without_stable_local_evidence"
        if (
            not local_evidence
            and pred_key == yes_key
            and phase1_normal
            and threshold > 0
            and heatmap_score < threshold
            and not clearly_below_threshold
        ):
            reason = "near_threshold_uncertainty_keeps_original_yes"
        return {
            **base,
            "applied": False,
            "override_pred_key": pred_key,
            "reason": reason,
        }

    return {
        **base,
        "applied": True,
        "override_pred_key": yes_key,
        "reason": "stable_local_evidence_corrects_no_to_yes",
    }
