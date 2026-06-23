from __future__ import annotations

import os
from typing import Any


KNOWLEDGE_SOURCE_TYPE = "text_knowledge_json"
TRUSTED_VISUAL_REFERENCE_SOURCE = "normal_reference_region_cutouts"
TRUSTED_VISUAL_REFERENCE_GENERATION = (
    "database img cutouts generated from normal/reference samples by SAM3 segmentation "
    "or direct-copy category profiles"
)
UNKNOWN_KNOWLEDGE_SOURCE_TYPE = "unknown"
UNKNOWN_VISUAL_REFERENCE_SOURCE = "unknown"
UNKNOWN_VISUAL_REFERENCE_GENERATION = "missing_or_untrusted_graph_source_metadata"


def same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return os.path.abspath(str(left)) == os.path.abspath(str(right))


def build_graph_source_metadata(
    *,
    dataset: str,
    category: str,
    source_json_path: str,
    text_knowledge_root: str,
    visual_reference_root: str,
    graph_output_root: str,
    visual_reference_builder: str,
    visual_reference_max_k_shot: int,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset).lower(),
        "category": str(category),
        "source_json_path": os.path.abspath(source_json_path),
        "text_knowledge_root": os.path.abspath(text_knowledge_root),
        "knowledge_source_type": KNOWLEDGE_SOURCE_TYPE,
        "source_type": KNOWLEDGE_SOURCE_TYPE,
        "visual_reference_source": TRUSTED_VISUAL_REFERENCE_SOURCE,
        "visual_reference_generation": TRUSTED_VISUAL_REFERENCE_GENERATION,
        "visual_reference_root": os.path.abspath(visual_reference_root),
        "visual_reference_builder": str(visual_reference_builder),
        "visual_reference_max_k_shot": int(visual_reference_max_k_shot),
        "graph_output_root": os.path.abspath(graph_output_root),
    }


def is_trusted_visual_reference(source: Any, generation: Any) -> bool:
    source_text = str(source or "").strip()
    generation_text = str(generation or "").strip().lower()
    if source_text != TRUSTED_VISUAL_REFERENCE_SOURCE:
        return False
    if not ("normal" in generation_text or "reference" in generation_text):
        return False
    return any(token in generation_text for token in ("sam3", "segmentation", "direct-copy", "direct copy"))


def visual_reference_trust_audit(
    *,
    source: Any,
    generation: Any,
    image_paths: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Audit whether runtime visual references are source-backed normal evidence."""

    paths = [str(path) for path in list(image_paths or []) if str(path)]
    existing_count = sum(1 for path in paths if os.path.exists(path))
    trusted_source = is_trusted_visual_reference(source, generation)
    return {
        "trusted_source": trusted_source,
        "path_count": len(paths),
        "existing_path_count": existing_count,
        "missing_path_count": max(0, len(paths) - existing_count),
        "source_backed": bool(trusted_source and paths and existing_count == len(paths)),
    }


def source_metadata_checks(
    metadata: dict[str, Any],
    *,
    expected_dataset: str,
    expected_category: str,
    expected_text_json_path: str,
    expected_visual_reference_root: str,
    expected_graph_output_root: str,
) -> dict[str, bool]:
    if not metadata:
        return {
            "present": False,
            "dataset_matches": False,
            "category_matches": False,
            "knowledge_source_type_trusted": False,
            "source_json_matches": False,
            "visual_reference_root_matches": False,
            "visual_reference_source_trusted": False,
            "visual_reference_generation_trusted": False,
            "builder_recorded": False,
            "graph_output_root_matches": False,
        }

    return {
        "present": True,
        "dataset_matches": str(metadata.get("dataset") or "").lower() == str(expected_dataset).lower(),
        "category_matches": str(metadata.get("category") or "") == str(expected_category),
        "knowledge_source_type_trusted": (
            metadata.get("knowledge_source_type") == KNOWLEDGE_SOURCE_TYPE
            or metadata.get("source_type") == KNOWLEDGE_SOURCE_TYPE
        ),
        "source_json_matches": same_path(metadata.get("source_json_path"), expected_text_json_path),
        "visual_reference_root_matches": same_path(metadata.get("visual_reference_root"), expected_visual_reference_root),
        "visual_reference_source_trusted": metadata.get("visual_reference_source") == TRUSTED_VISUAL_REFERENCE_SOURCE,
        "visual_reference_generation_trusted": is_trusted_visual_reference(
            metadata.get("visual_reference_source"),
            metadata.get("visual_reference_generation"),
        ),
        "builder_recorded": bool(str(metadata.get("visual_reference_builder") or "").strip()),
        "graph_output_root_matches": same_path(metadata.get("graph_output_root"), expected_graph_output_root),
    }
