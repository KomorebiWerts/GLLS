from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np

from .heatmap_regions import salient_heatmap_mask


DEFAULT_LOCAL_ACTIONS = ("zoom_in", "move_left", "move_right", "move_up", "move_down", "zoom_out")


def _round_float(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _bbox_list(coords: Any) -> list[int]:
    if not coords:
        return []
    return [int(v) for v in coords]


class MCTSActionPolicy:
    """Budgeted action policy for fine-grained local search.

    The policy keeps MCTS action ordering and scoring in one small module. It
    uses the anomaly heatmap as a heuristic prior, while the caller remains
    responsible for image crops and asynchronous execution.
    """

    def __init__(self, *, c_puct: float = 1.0, heatmap_prior_weight: float = 0.5):
        self.c_puct = float(c_puct)
        self.heatmap_prior_weight = float(heatmap_prior_weight)

    @staticmethod
    def root_actions(regions: list[dict[str, Any]]) -> list[str]:
        indexed = []
        for idx, region in enumerate(regions or []):
            try:
                score = float(region.get("score", 0.0))
            except Exception:
                score = 0.0
            indexed.append((score, idx))
        indexed.sort(key=lambda item: item[0], reverse=True)
        return [f"inspect_region_{idx}" for _, idx in indexed]

    @staticmethod
    def local_actions(state: dict[str, Any], *, pixel_threshold: float = 0.0) -> list[str]:
        heatmap = state.get("heatmap_array")
        if not isinstance(heatmap, np.ndarray) or heatmap.size == 0:
            return list(DEFAULT_LOCAL_ACTIONS)

        coords = state.get("region_coords") or (0, 0, 1, 1)
        width = max(1.0, float(coords[2] - coords[0]))
        height = max(1.0, float(coords[3] - coords[1]))
        mask, _ = salient_heatmap_mask(
            heatmap,
            tail_fraction=0.05,
            min_score=0.0,
            min_tail_pixels=max(1, int(float(heatmap.size) * 0.01)),
        )
        if not np.any(mask):
            return list(DEFAULT_LOCAL_ACTIONS)

        ys, xs = np.where(mask)
        cx = float(np.mean(xs)) / max(1.0, float(heatmap.shape[1] - 1))
        cy = float(np.mean(ys)) / max(1.0, float(heatmap.shape[0] - 1))
        area = width * height

        actions = ["zoom_in"]
        if cx < 0.42:
            actions.append("move_left")
        elif cx > 0.58:
            actions.append("move_right")
        else:
            actions.extend(["move_left", "move_right"])

        if cy < 0.42:
            actions.append("move_up")
        elif cy > 0.58:
            actions.append("move_down")
        else:
            actions.extend(["move_up", "move_down"])

        if area < 224 * 224:
            actions.append("zoom_out")
        else:
            actions.append("zoom_out")

        deduped = []
        for action in actions:
            if action not in deduped:
                deduped.append(action)
        for action in DEFAULT_LOCAL_ACTIONS:
            if action not in deduped:
                deduped.append(action)
        return deduped

    def ucb_score(self, child: Any, total_visits: int) -> float:
        if getattr(child, "visits", 0) == 0:
            return float("inf")
        total_visits = max(1, int(total_visits))
        exploitation = float(child.value) / float(child.visits)
        exploration = self.c_puct * math.sqrt(2.0 * math.log(total_visits) / float(child.visits))
        search_score = float(child.state.get("search_score", getattr(child, "heatmap_score", 0.0)))
        return exploitation + exploration + self.heatmap_prior_weight * search_score

    @staticmethod
    def node_snapshot(node: Any) -> dict[str, Any]:
        state = getattr(node, "state", {}) or {}
        return {
            "depth": int(state.get("depth", 0) or 0),
            "bbox_xyxy": _bbox_list(state.get("region_coords")),
            "visits": int(getattr(node, "visits", 0) or 0),
            "value": _round_float(getattr(node, "value", 0.0)),
            "leaf_reward": _round_float(getattr(node, "leaf_reward", 0.0)),
            "heatmap_score": _round_float(state.get("heatmap_score", getattr(node, "heatmap_score", 0.0))),
            "search_score": _round_float(state.get("search_score", state.get("heatmap_score", 0.0))),
            "remaining_actions": list(getattr(node, "untried_actions", []) or []),
        }

    @staticmethod
    def final_node_score_components(node: Any, *, root_visits: int = 0) -> dict[str, float]:
        """Score final evidence nodes with both heatmap prior and MCTS posterior."""

        state = getattr(node, "state", {}) or {}
        visits = int(getattr(node, "visits", 0) or 0)
        value = float(getattr(node, "value", 0.0) or 0.0)
        mean_value = value / float(visits) if visits > 0 else 0.0
        leaf_reward = float(getattr(node, "leaf_reward", 0.0) or 0.0)
        search_score = float(state.get("search_score", state.get("heatmap_score", leaf_reward)) or 0.0)
        visit_confidence = 0.0
        if root_visits > 0 and visits > 0:
            visit_confidence = math.log1p(visits) / math.log1p(max(1, int(root_visits)))
        final_score = (
            0.45 * search_score
            + 0.25 * mean_value
            + 0.20 * leaf_reward
            + 0.10 * visit_confidence
        )
        return {
            "score": _round_float(final_score),
            "search_score": _round_float(search_score),
            "mean_value": _round_float(mean_value),
            "leaf_reward": _round_float(leaf_reward),
            "visit_confidence": _round_float(visit_confidence),
            "visits": float(visits),
        }

    @staticmethod
    def summarize_trace(trace: Iterable[dict[str, Any]], *, budget: int) -> dict[str, Any]:
        rows = list(trace or [])
        action_counts = Counter(str(row.get("expanded_action", "")) for row in rows if row.get("expanded_action"))
        expansion_status_counts = Counter(str(row.get("expansion_status", "")) for row in rows if row.get("expansion_status"))
        reward_sources = Counter(
            str((row.get("simulation") or {}).get("reward_source", "unknown"))
            for row in rows
        )
        terminal_count = sum(1 for row in rows if row.get("terminal"))
        rewards = [float(row.get("reward", 0.0) or 0.0) for row in rows]
        path_depths = [len(row.get("path_actions") or []) for row in rows]
        selection_depths = [
            int((row.get("selection") or {}).get("depth", 0) or 0)
            for row in rows
        ]
        return {
            "budget": int(budget),
            "iterations_recorded": len(rows),
            "terminal_iterations": terminal_count,
            "expanded_action_counts": dict(sorted(action_counts.items())),
            "expansion_status_counts": dict(sorted(expansion_status_counts.items())),
            "reward_source_counts": dict(sorted(reward_sources.items())),
            "max_reward": _round_float(max(rewards) if rewards else 0.0),
            "mean_reward": _round_float(sum(rewards) / len(rewards) if rewards else 0.0),
            "max_path_depth": int(max(path_depths) if path_depths else 0),
            "mean_path_depth": _round_float(sum(path_depths) / len(path_depths) if path_depths else 0.0),
            "max_selection_depth": int(max(selection_depths) if selection_depths else 0),
        }
