from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _round_float(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _bbox_tuple(bbox_xyxy: Any) -> tuple[int, int, int, int] | None:
    if bbox_xyxy is None:
        return None
    values = list(bbox_xyxy)
    if len(values) != 4:
        return None
    return tuple(int(v) for v in values)


def _bbox_list(bbox_xyxy: Any) -> list[int] | None:
    bbox = _bbox_tuple(bbox_xyxy)
    return [int(v) for v in bbox] if bbox is not None else None


def _box_iou(box1: Any, box2: Any) -> float:
    b1 = _bbox_tuple(box1)
    b2 = _bbox_tuple(box2)
    if b1 is None or b2 is None:
        return 0.0
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    inter_area = inter_width * inter_height
    box1_area = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    box2_area = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _crop_candidate_audit_item(
    candidate: dict[str, Any],
    *,
    status: str,
    rank: int | None = None,
    selected_index: int | None = None,
    reason: str = "",
) -> dict[str, Any]:
    item = {
        "rank": rank,
        "selected_index": selected_index,
        "status": status,
        "reason": reason,
        "label": candidate.get("label", ""),
        "source": candidate.get("source", ""),
        "priority": _round_float(candidate.get("priority", 0.0)),
        "bbox_xyxy": _bbox_list(candidate.get("bbox_xyxy")),
        "sam_score": _round_float(candidate.get("sam_score", 0.0)) if candidate.get("sam_score") is not None else None,
        "sam_mask_quality": candidate.get("sam_mask_quality", {}),
        "prompt_role": candidate.get("prompt_role", ""),
        "prompt_text": candidate.get("prompt_text", ""),
    }
    for key in (
        "heatmap_score",
        "search_score",
        "selection_score",
        "proposal_source",
        "selection_reason",
        "proposal_index",
        "proposal_bbox_xyxy",
        "semantic_defect_priority_bonus",
        "spatial_detail",
    ):
        if key in candidate:
            value = candidate.get(key)
            if isinstance(value, float):
                value = _round_float(value)
            item[key] = value
    return item


def _is_duplicate_bbox(
    bbox_xyxy: Any,
    existing_boxes: list[Any] | tuple[Any, ...] | None,
    *,
    iou_threshold: float,
) -> bool:
    bbox = _bbox_tuple(bbox_xyxy)
    if bbox is None:
        return False
    for existing in existing_boxes or []:
        if existing is not None and _box_iou(bbox, existing) >= float(iou_threshold):
            return True
    return False


@dataclass(frozen=True)
class MCTSCropCandidateSelection:
    selected_candidates: list[dict[str, Any]]
    selected_boxes: list[tuple[int, int, int, int]]
    candidate_audit: list[dict[str, Any]]
    selected_audit: list[dict[str, Any]]
    skipped_audit: list[dict[str, Any]]


class MCTSCropCandidateSelector:
    """Select prompt-visible local evidence crops by source priority and NMS."""

    def __init__(
        self,
        *,
        max_candidates: int = 3,
        iou_threshold: float = 0.78,
        source_limits: dict[str, int] | None = None,
    ):
        self.max_candidates = max(1, int(max_candidates or 1))
        self.iou_threshold = float(iou_threshold)
        self.source_limits = {
            str(source): max(0, int(limit))
            for source, limit in (source_limits or {}).items()
        }

    def select(self, crop_candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> MCTSCropCandidateSelection:
        sorted_candidates = sorted(
            [candidate for candidate in crop_candidates or [] if isinstance(candidate, dict)],
            key=lambda item: item.get("priority", 0.0),
            reverse=True,
        )
        candidate_audit = [
            _crop_candidate_audit_item(candidate, status="candidate", rank=rank)
            for rank, candidate in enumerate(sorted_candidates, start=1)
        ]
        selected: list[dict[str, Any]] = []
        selected_boxes: list[tuple[int, int, int, int]] = []
        skipped: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}

        for rank, candidate in enumerate(sorted_candidates, start=1):
            source = str(candidate.get("source", ""))
            source_limit = self.source_limits.get(source)
            if source_limit is not None and source_counts.get(source, 0) >= source_limit:
                skipped.append(
                    _crop_candidate_audit_item(
                        candidate,
                        status="skipped",
                        rank=rank,
                        reason=f"source_limit:{source}",
                    )
                )
                continue
            if _is_duplicate_bbox(candidate.get("bbox_xyxy"), selected_boxes, iou_threshold=self.iou_threshold):
                skipped.append(
                    _crop_candidate_audit_item(
                        candidate,
                        status="skipped",
                        rank=rank,
                        reason="duplicate_bbox_iou",
                    )
                )
                continue
            selected.append(candidate)
            bbox = _bbox_tuple(candidate.get("bbox_xyxy"))
            if bbox is not None:
                selected_boxes.append(bbox)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) >= self.max_candidates:
                break

        selected_audit = [
            _crop_candidate_audit_item(
                candidate,
                status="selected",
                rank=idx,
                selected_index=idx,
                reason="selected_by_priority_nms",
            )
            for idx, candidate in enumerate(selected, start=1)
        ]
        return MCTSCropCandidateSelection(
            selected_candidates=selected,
            selected_boxes=selected_boxes,
            candidate_audit=candidate_audit,
            selected_audit=selected_audit,
            skipped_audit=skipped,
        )


def crop_candidate_audit_item(
    candidate: dict[str, Any],
    *,
    status: str,
    rank: int | None = None,
    selected_index: int | None = None,
    reason: str = "",
) -> dict[str, Any]:
    return _crop_candidate_audit_item(
        candidate,
        status=status,
        rank=rank,
        selected_index=selected_index,
        reason=reason,
    )


def is_duplicate_bbox(
    bbox_xyxy: Any,
    existing_boxes: list[Any] | tuple[Any, ...] | None,
    *,
    iou_threshold: float = 0.78,
) -> bool:
    return _is_duplicate_bbox(bbox_xyxy, existing_boxes, iou_threshold=iou_threshold)
