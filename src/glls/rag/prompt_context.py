from __future__ import annotations

import re
from typing import Any


def _is_global_or_whole_region(region_name: Any) -> bool:
    region_l = str(region_name or "").lower()
    return "whole" in region_l or "global" in region_l


def _normalize_region_key(region_name: Any) -> str:
    return re.sub(r"\s+", "_", str(region_name or "").strip().lower())


def _hypothesis_region_names(hypotheses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[str]:
    regions = []
    seen = set()
    for item in hypotheses or []:
        if not isinstance(item, dict):
            continue
        region = item.get("region", "")
        if not region and item.get("graph_path"):
            path_parts = [
                part.strip()
                for part in str(item.get("graph_path", "")).split("->")
                if part.strip()
            ]
            if len(path_parts) >= 2:
                region = path_parts[1]
        key = _normalize_region_key(region)
        if key and key not in seen:
            seen.add(key)
            regions.append(str(region))
    return regions


class PromptRAGSelector:
    """Select prompt-visible graph blocks from source-grounded visual evidence.

    This module deliberately does not inspect answer-option text. Option-based
    graph matches can be retained in audit metadata, but prompt-visible region
    filtering is driven by PVLA visual hypotheses from image evidence.
    """

    def select(
        self,
        rag_blocks: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        pvla_evidence_hypothesis_audit: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        visible_blocks = [
            block for block in rag_blocks or []
            if isinstance(block, dict) and block.get("prompt_visible") is not False
        ]
        visible_regions = [str(block.get("region", "")) for block in visible_blocks]
        hypothesis_regions = _hypothesis_region_names(pvla_evidence_hypothesis_audit)
        local_hypothesis_keys = {
            _normalize_region_key(region)
            for region in hypothesis_regions
            if not _is_global_or_whole_region(region)
        }

        global_anchor_regions = [
            region for region in visible_regions
            if _is_global_or_whole_region(region)
        ]

        if not local_hypothesis_keys:
            return visible_blocks, {
                "mode": "full_fallback_no_local_pvla_hypothesis",
                "policy": "pvla_visual_hypothesis_first",
                "hypothesis_regions": hypothesis_regions,
                "global_anchor_regions": global_anchor_regions,
                "selected_regions": visible_regions,
                "skipped_regions": [],
                "input_visible_block_count": len(visible_blocks),
                "selected_block_count": len(visible_blocks),
                "skipped_block_count": 0,
            }

        selected = []
        skipped_regions = []
        for block in visible_blocks:
            region = str(block.get("region", ""))
            region_key = _normalize_region_key(region)
            if _is_global_or_whole_region(region) or region_key in local_hypothesis_keys:
                selected.append(block)
            else:
                skipped_regions.append(region)

        if not selected:
            return visible_blocks, {
                "mode": "full_fallback_unmatched_pvla_hypothesis",
                "policy": "pvla_visual_hypothesis_first",
                "hypothesis_regions": hypothesis_regions,
                "global_anchor_regions": global_anchor_regions,
                "selected_regions": visible_regions,
                "skipped_regions": [],
                "input_visible_block_count": len(visible_blocks),
                "selected_block_count": len(visible_blocks),
                "skipped_block_count": 0,
            }

        return selected, {
            "mode": "pvla_hypothesis_region_filter",
            "policy": "pvla_visual_hypothesis_first",
            "hypothesis_regions": hypothesis_regions,
            "global_anchor_regions": global_anchor_regions,
            "selected_regions": [str(block.get("region", "")) for block in selected],
            "skipped_regions": skipped_regions,
            "input_visible_block_count": len(visible_blocks),
            "selected_block_count": len(selected),
            "skipped_block_count": len(skipped_regions),
        }


def select_prompt_rag_blocks(
    rag_blocks: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    pvla_evidence_hypothesis_audit: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return PromptRAGSelector().select(rag_blocks, pvla_evidence_hypothesis_audit)
