from __future__ import annotations

from collections import Counter
from typing import Any, Optional


def _is_valid_text(value: Any) -> bool:
    if not value:
        return False
    if not isinstance(value, str):
        return True
    return bool(value.strip()) and value.strip().upper() != "N/A"


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


class PVLARetrievalPlanner:
    """Plan topology-scoped PVLA region retrieval with auditable reasons."""

    def __init__(self, rag: Any):
        self.rag = rag

    def _region_info(self, region: str) -> dict[str, Any]:
        graph = self.rag.G
        node_data = graph.nodes[region] if graph.has_node(region) else {}
        defects = []
        for defect in self.rag.iter_region_defects([region]):
            defects.append({
                "defect": defect.get("defect", ""),
                "defect_node": defect.get("defect_node", ""),
                "graph_path": defect.get("graph_path", ""),
            })
        return {
            "region": region,
            "graph_path": f"{self.rag.root_name} -> {region}",
            "definition_available": _is_valid_text(node_data.get("definition", "")),
            "normal_standard_available": _is_valid_text(node_data.get("normal_standard", "")),
            "critical_check_available": _is_valid_text(node_data.get("critical_check", "")),
            "visual_reference_count": len(self.rag.get_image_paths(region)),
            "defect_count": len(defects),
            "defect_paths": [item["graph_path"] for item in defects if item.get("graph_path")],
        }

    def plan(
        self,
        *,
        all_regions: list[str],
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        include_global: bool = True,
        retrieval_mode: str = "full",
        options: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        retrieval_mode = (retrieval_mode or "full").lower()
        all_regions = list(all_regions or [])
        top_k = int(top_k or 0)

        audit_by_region = {region: self._region_info(region) for region in all_regions}
        for item in audit_by_region.values():
            item["selection_reasons"] = []
            item["matched_options"] = []
            item["matched_option_evidence"] = []

        global_regions = [
            region for region in all_regions
            if include_global and ("whole" in region.lower() or "global" in region.lower())
        ]
        for region in global_regions:
            audit_by_region[region]["selection_reasons"].append("global_or_whole_standard")

        critical_regions = [
            region for region in all_regions
            if audit_by_region[region]["critical_check_available"]
        ]
        for region in critical_regions:
            audit_by_region[region]["selection_reasons"].append("critical_check_available")

        option_regions = []
        option_matches = []
        if options:
            try:
                option_matches = self.rag.match_options_to_defects(options, region_names=all_regions)
            except Exception:
                option_matches = []
        for match in option_matches:
            matched_defects = match.get("matched_defects") or [match]
            for item in matched_defects:
                region = item.get("region")
                if region in audit_by_region:
                    option_regions.append(region)
                    option_key = match.get("option_key")
                    if option_key and option_key not in audit_by_region[region]["matched_options"]:
                        audit_by_region[region]["matched_options"].append(str(option_key))
                    audit_by_region[region]["matched_option_evidence"].append({
                        "option_key": str(option_key or ""),
                        "option_text": str(match.get("option_text", "")),
                        "defect": str(item.get("defect", "")),
                        "graph_path": str(item.get("graph_path", "")),
                        "support_channels": list(item.get("support_channels") or []),
                        "source_evidence_channels": list(item.get("source_evidence_channels") or []),
                        "label_only_match": bool(item.get("label_only_match", False)),
                        "evidence_consensus": bool(item.get("evidence_consensus", False)),
                    })

        semantic_ranked_regions = []
        if query and top_k > 0:
            try:
                semantic_ranked_regions = [
                    region for region in self.rag.search_regions(query, top_k=top_k)
                    if region in audit_by_region
                ]
            except Exception:
                semantic_ranked_regions = []
        for rank, region in enumerate(semantic_ranked_regions, start=1):
            audit_by_region[region]["selection_reasons"].append(f"semantic_rank_{rank}")

        if retrieval_mode == "semantic_topk" and query and top_k > 0:
            selected = _dedupe(global_regions + semantic_ranked_regions)
        elif retrieval_mode == "hybrid" and query and top_k > 0:
            selected = _dedupe(global_regions + critical_regions + semantic_ranked_regions)
        else:
            selected = all_regions
            for region in selected:
                audit_by_region[region]["selection_reasons"].append("full_graph_context")

        if not selected and all_regions:
            selected = all_regions[:top_k or len(all_regions)]
            for region in selected:
                audit_by_region[region]["selection_reasons"].append("fallback_region_order")

        reason_counts: Counter[str] = Counter()
        for region in selected:
            reasons = audit_by_region[region].get("selection_reasons") or ["selected_without_reason"]
            audit_by_region[region]["selection_reasons"] = _dedupe(reasons)
            reason_counts.update(audit_by_region[region]["selection_reasons"])

        return {
            "target_object": self.rag.root_name,
            "retrieval_mode": retrieval_mode,
            "selected_regions": selected,
            "selected_from": len(all_regions),
            "top_k": top_k,
            "semantic_ranked_regions": semantic_ranked_regions,
            "option_grounded_regions": _dedupe(option_regions),
            "option_grounding_audit": [
                {
                    "region": region,
                    "graph_path": audit_by_region[region].get("graph_path", ""),
                    "matched_options": audit_by_region[region].get("matched_options", []),
                    "matched_option_evidence": audit_by_region[region].get("matched_option_evidence", []),
                    "prompt_selection": "audit_only_not_used_for_region_selection",
                }
                for region in _dedupe(option_regions)
            ],
            "selection_audit": [audit_by_region[region] for region in selected],
            "selection_reason_counts": dict(sorted(reason_counts.items())),
        }
