from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _round_float(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _bbox_list(coords: Any) -> list[int] | None:
    if not coords:
        return None
    return [int(v) for v in coords]


def _box_iou(box1: Any, box2: Any) -> float:
    if not box1 or not box2:
        return 0.0
    x1_min, y1_min, x1_max, y1_max = [float(v) for v in box1]
    x2_min, y2_min, x2_max, y2_max = [float(v) for v in box2]
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter_width = max(0.0, xi_max - xi_min)
    inter_height = max(0.0, yi_max - yi_min)
    inter_area = inter_width * inter_height
    box1_area = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    box2_area = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _node_final_score(node: Any) -> float:
    state = getattr(node, "state", {}) or {}
    components = state.get("mcts_final_score_components") if isinstance(state, dict) else None
    if isinstance(components, dict) and components.get("score") is not None:
        return float(components.get("score") or 0.0)
    return float(getattr(node, "leaf_reward", 0.0) or 0.0)


@dataclass(frozen=True)
class MCTSFinalCandidateSelection:
    selected_nodes: list[Any]
    audit: list[dict[str, Any]]


class MCTSFinalCandidateSelector:
    """Select final MCTS evidence nodes with posterior scoring and NMS.

    The selector owns the final evidence-node policy for the Fine-Grained &
    Actions stream: score each explored node using the action policy posterior,
    deduplicate identical boxes, apply NMS for diversity, and emit audit rows
    that explain why each node was selected or skipped.
    """

    def __init__(self, action_policy: Any, *, max_candidates: int = 3):
        self.action_policy = action_policy
        self.max_candidates = max(1, int(max_candidates or 1))

    def select(
        self,
        nodes: Iterable[Any],
        *,
        root: Any = None,
        image_width: int = 0,
        image_height: int = 0,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.1,
    ) -> MCTSFinalCandidateSelection:
        root_visits = int(getattr(root, "visits", 0) or 0)
        unique_nodes = self._dedupe_nodes(nodes, root_visits=root_visits)
        rounded_threshold = _round_float(score_threshold)

        if not unique_nodes:
            audit = [{
                "status": "selected" if root is not None else "missing",
                "reason": "fallback_root_no_mcts_nodes",
                "selected_index": 1 if root is not None else None,
                "bbox_xyxy": [0, 0, int(image_width), int(image_height)],
                "final_score": 0.0,
                "score_threshold": rounded_threshold,
                "threshold_filter_active": False,
                "score_components": {},
            }]
            return MCTSFinalCandidateSelection([root] if root is not None else [], audit)

        candidates = sorted(unique_nodes, key=_node_final_score, reverse=True)
        selected = self._select_diverse(candidates, iou_threshold=float(iou_threshold))
        audit = self._build_audit(
            unique_nodes,
            selected,
            score_threshold=rounded_threshold,
        )
        return MCTSFinalCandidateSelection(selected, audit)

    def _dedupe_nodes(self, nodes: Iterable[Any], *, root_visits: int) -> list[Any]:
        unique_map: dict[tuple[int, int, int, int], Any] = {}
        for node in nodes or []:
            state = getattr(node, "state", {}) or {}
            coords = state.get("region_coords")
            if not coords:
                continue
            components = self.action_policy.final_node_score_components(node, root_visits=root_visits)
            state["mcts_final_score_components"] = components
            state["mcts_final_score"] = components["score"]
            key = tuple(int(v) for v in coords)
            if key not in unique_map or components["score"] > _node_final_score(unique_map[key]):
                unique_map[key] = node
        return list(unique_map.values())

    def _select_diverse(self, candidates: list[Any], *, iou_threshold: float) -> list[Any]:
        selected = []
        for node in candidates:
            coords = (getattr(node, "state", {}) or {}).get("region_coords")
            if any(_box_iou(coords, (kept.state or {}).get("region_coords")) >= iou_threshold for kept in selected):
                continue
            selected.append(node)
            if len(selected) >= self.max_candidates:
                break
        return selected

    @staticmethod
    def _build_audit(
        unique_nodes: list[Any],
        selected: list[Any],
        *,
        score_threshold: float,
    ) -> list[dict[str, Any]]:
        selection_reason = "ranked_budgeted_mcts_nms_score"
        selected_ids = {id(node): idx for idx, node in enumerate(selected, start=1)}
        audit = []
        for node in sorted(unique_nodes, key=_node_final_score, reverse=True):
            state = getattr(node, "state", {}) or {}
            coords = state.get("region_coords")
            selected_index = selected_ids.get(id(node))
            audit.append({
                "status": "selected" if selected_index else "candidate",
                "reason": selection_reason if selected_index else "not_selected_by_ranked_budgeted_nms",
                "selected_index": selected_index,
                "bbox_xyxy": _bbox_list(coords),
                "depth": int(state.get("depth", 0) or 0),
                "final_score": _round_float(_node_final_score(node)),
                "score_threshold": score_threshold,
                "threshold_filter_active": False,
                "visits": int(getattr(node, "visits", 0) or 0),
                "value": _round_float(getattr(node, "value", 0.0)),
                "leaf_reward": _round_float(getattr(node, "leaf_reward", 0.0)),
                "search_score": _round_float(state.get("search_score", state.get("heatmap_score", 0.0))),
                "score_components": state.get("mcts_final_score_components", {}),
            })
        return audit
