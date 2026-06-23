from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


VISUAL_TASK_TYPES = {
    "defect localization",
    "defect classification",
    "defect description",
    "defect analysis",
}
AD_NEAR_THRESHOLD_FLOOR_RATIO = 0.95


@dataclass(frozen=True)
class HeatmapThresholds:
    image: float
    pixel: float
    source: str = "configured"


@dataclass
class RegionProposal:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float
    source: str
    contour: Optional[np.ndarray] = None
    component_id: Optional[int] = None
    selection_score: Optional[float] = None
    selection_reason: str = ""
    pixel_threshold: Optional[float] = None
    selection_features: dict[str, float | str] = field(default_factory=dict)

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return x, y, x + w, y + h

    def normalized_box(self, image_width: int, image_height: int) -> list[float]:
        x, y, w, h = self.bbox
        return [
            (x + w / 2.0) / max(1, image_width),
            (y + h / 2.0) / max(1, image_height),
            w / max(1, image_width),
            h / max(1, image_height),
        ]


def adaptive_thresholds(
    heatmap: np.ndarray,
    *,
    default_image: float,
    default_pixel: float,
    source: str = "adaptive",
) -> HeatmapThresholds:
    values = np.asarray(heatmap, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return HeatmapThresholds(default_image, default_pixel, source=f"{source}:empty")

    max_score = float(values.max())
    if max_score <= 0:
        return HeatmapThresholds(default_image, default_pixel, source=f"{source}:zero")

    high_pixel = float(np.quantile(values, 0.985))
    high_image = float(np.quantile(values, 0.995))
    pixel = max(0.0, min(default_pixel, high_pixel, max_score * 0.85))
    image = max(pixel, min(default_image, high_image, max_score * 0.95))
    return HeatmapThresholds(image=image, pixel=pixel, source=source)


def resolve_heatmap_thresholds(
    localizer,
    category: str,
    heatmap: np.ndarray,
    *,
    default_image: float = 0.9,
    default_pixel: float = 0.9,
) -> HeatmapThresholds:
    if hasattr(localizer, "get_threshold_config"):
        cfg = localizer.get_threshold_config(category, heatmap=heatmap)
        if isinstance(cfg, HeatmapThresholds):
            return cfg

    if hasattr(localizer, "get_thresholds"):
        image_threshold, pixel_threshold = localizer.get_thresholds(category)
        return HeatmapThresholds(float(image_threshold), float(pixel_threshold), source="legacy")

    return adaptive_thresholds(
        heatmap,
        default_image=default_image,
        default_pixel=default_pixel,
        source="adaptive:fallback",
    )


def detect_heatmap_regions(
    heatmap: np.ndarray,
    thresholds: HeatmapThresholds,
    *,
    task_type: str = "",
    top_k: int = 3,
    min_area: int = 50,
    group_kernel: int = 15,
    fallback_min_score: float = 0.15,
    fallback_min_area: int = 20,
    tail_fraction: float = 0.015,
) -> list[RegionProposal]:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0:
        return []

    task_name = task_type.lower()
    is_visual_task = task_name in VISUAL_TASK_TYPES
    if not is_visual_task:
        strict = _collect_grouped_regions(
            heatmap,
            pixel_threshold=thresholds.pixel,
            image_threshold=thresholds.image,
            top_k=top_k,
            min_area=min_area,
            kernel_size=group_kernel,
            source="strict",
        )
        if strict:
            return strict
        if task_name == "anomaly detection":
            peak = float(np.max(heatmap[np.isfinite(heatmap)])) if np.any(np.isfinite(heatmap)) else 0.0
            image_threshold = float(thresholds.image)
            near_threshold_floor = (
                image_threshold * AD_NEAR_THRESHOLD_FLOOR_RATIO
                if image_threshold > 0
                else 0.0
            )
            if image_threshold > 0 and near_threshold_floor <= peak < image_threshold:
                near_regions = _collect_ranked_tail_regions(
                    heatmap,
                    top_k=top_k,
                    min_area=fallback_min_area,
                    kernel_size=5,
                    min_peak_score=max(fallback_min_score, near_threshold_floor),
                    tail_fraction=tail_fraction,
                    source="near_threshold_ad_ranked_tail",
                )
                if near_regions:
                    return near_regions
        return strict

    strict = _collect_grouped_regions(
        heatmap,
        pixel_threshold=thresholds.pixel,
        image_threshold=thresholds.image,
        top_k=top_k,
        min_area=min_area,
        kernel_size=group_kernel,
        source="strict",
    )
    if strict:
        return strict

    # Keep the old effective fallback semantics for visual QA: a weak local
    # hypothesis is allowed only when the calibrated pixel-threshold mask has
    # some support, but no component reaches the stricter image threshold. A
    # pure ranked-tail crop can fabricate local evidence from diffuse heatmaps
    # and has empirically hurt classification/localization prompts.
    near_threshold = _collect_grouped_regions(
        heatmap,
        pixel_threshold=thresholds.pixel,
        image_threshold=fallback_min_score,
        top_k=1,
        min_area=fallback_min_area,
        kernel_size=5,
        source="weak_pixel_threshold_fallback",
    )
    return near_threshold


def _tail_cutoff(
    heatmap: np.ndarray,
    *,
    tail_fraction: float,
    min_score: float,
    min_tail_pixels: int = 1,
) -> Optional[float]:
    finite = heatmap[np.isfinite(heatmap)]
    if finite.size == 0:
        return None
    max_score = float(finite.max())
    if max_score <= min_score:
        return None
    positive = finite[finite > 0]
    if positive.size == 0:
        return None
    tail_fraction = min(0.20, max(0.001, float(tail_fraction)))
    tail_count = max(int(min_tail_pixels or 1), int(round(float(positive.size) * tail_fraction)))
    if tail_count >= positive.size:
        cutoff = float(positive.min())
    else:
        cutoff = float(np.partition(positive, -tail_count)[-tail_count])
    return max(min_score, cutoff)


def salient_heatmap_mask(
    heatmap: np.ndarray,
    *,
    tail_fraction: float = 0.05,
    min_score: float = 0.0,
    min_tail_pixels: int = 1,
) -> tuple[np.ndarray, Optional[float]]:
    """Return the top-mass saliency mask without category-specific thresholds."""

    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return np.zeros_like(values, dtype=bool), None
    cutoff = _tail_cutoff(
        values,
        tail_fraction=tail_fraction,
        min_score=min_score,
        min_tail_pixels=min_tail_pixels,
    )
    if cutoff is None:
        return np.zeros_like(values, dtype=bool), None
    return np.isfinite(values) & (values >= cutoff), cutoff


def _positive_robust_stats(heatmap: np.ndarray) -> tuple[float, float]:
    positive = heatmap[np.isfinite(heatmap) & (heatmap > 0)]
    if positive.size == 0:
        return 0.0, 1.0
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median)))
    scale = max(1e-6, 1.4826 * mad)
    return median, scale


def _finite_robust_stats(heatmap: np.ndarray) -> tuple[float, float]:
    finite = heatmap[np.isfinite(heatmap)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1e-6, 1.4826 * mad)
    return median, scale


def _collect_grouped_regions(
    heatmap: np.ndarray,
    *,
    pixel_threshold: float,
    image_threshold: float,
    top_k: int,
    min_area: int,
    kernel_size: int,
    source: str,
) -> list[RegionProposal]:
    mask = ((heatmap > pixel_threshold) * 255).astype(np.uint8)
    if not np.any(mask):
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_candidates = []
    for component_id in range(1, num_labels):
        x, y, w, h, _ = stats[component_id]
        component_values = heatmap[y : y + h, x : x + w][labels[y : y + h, x : x + w] == component_id]
        if component_values.size == 0:
            continue
        peak = float(component_values.max())
        if peak >= image_threshold:
            component_candidates.append({"id": component_id, "score": peak})

    component_candidates.sort(key=lambda item: item["score"], reverse=True)
    component_candidates = component_candidates[:top_k]
    if not component_candidates:
        return []

    clean_mask = np.zeros_like(mask)
    component_scores = {}
    for candidate in component_candidates:
        component_id = candidate["id"]
        clean_mask[labels == component_id] = 255
        component_scores[component_id] = candidate["score"]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    grouped_mask = cv2.dilate(clean_mask, kernel)
    contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        region_score = float(np.max(heatmap[y : y + h, x : x + w]))
        proposals.append(
            RegionProposal(
                bbox=(int(x), int(y), int(w), int(h)),
                score=region_score,
                source=source,
                contour=contour,
                component_id=_best_component_id(labels, contour, component_scores),
                selection_score=region_score,
                selection_reason="strict_threshold_component",
                pixel_threshold=float(pixel_threshold),
                selection_features={
                    "peak": round(float(region_score), 6),
                    "pixel_threshold": round(float(pixel_threshold), 6),
                    "image_threshold": round(float(image_threshold), 6),
                    "selector": "absolute_calibrated_threshold",
                },
            )
        )

    proposals.sort(key=lambda proposal: proposal.score, reverse=True)
    return proposals[:top_k]


def _collect_ranked_tail_regions(
    heatmap: np.ndarray,
    *,
    top_k: int,
    min_area: int,
    kernel_size: int,
    min_peak_score: float,
    tail_fraction: float,
    source: str = "ranked_tail",
    exclude_proposals: Optional[list[RegionProposal]] = None,
) -> list[RegionProposal]:
    if exclude_proposals:
        heatmap = np.array(heatmap, dtype=np.float32, copy=True)
        height, width = heatmap.shape
        for proposal in exclude_proposals:
            x1, y1, x2, y2 = proposal.xyxy
            x1, x2 = max(0, int(x1)), min(width, int(x2))
            y1, y2 = max(0, int(y1)), min(height, int(y2))
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] = 0.0

    cutoff = _tail_cutoff(
        heatmap,
        tail_fraction=tail_fraction,
        min_score=0.0,
        min_tail_pixels=min_area,
    )
    if cutoff is None:
        return []

    mask = ((heatmap >= cutoff) * 255).astype(np.uint8)
    if not np.any(mask):
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    grouped_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    grouped_mask = cv2.dilate(grouped_mask, kernel)
    contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    image_area = max(1, int(heatmap.shape[0]) * int(heatmap.shape[1]))
    positive_median, positive_scale = _positive_robust_stats(heatmap)
    robust_median, robust_scale = _finite_robust_stats(heatmap)
    for component_id, contour in enumerate(contours, start=1):
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        roi = heatmap[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        tail_pixels = roi[roi >= cutoff]
        if tail_pixels.size == 0:
            continue
        peak = float(np.max(roi))
        tail_mean = float(np.mean(tail_pixels))
        coverage = min(1.0, float(tail_pixels.size) / float(max(1, w * h)) * 8.0)
        compactness = min(1.0, float(tail_pixels.size) / float(image_area) * 200.0)
        bbox_area_fraction = float(max(1, w * h)) / float(image_area)
        excess_tail_mean = max(0.0, tail_mean - robust_median)
        excess_peak = max(0.0, peak - robust_median)
        robust_peak_z = max(0.0, (peak - robust_median) / robust_scale)
        robust_tail_z = max(0.0, (tail_mean - robust_median) / robust_scale)
        tail_pixel_fraction = float(tail_pixels.size) / float(image_area)
        has_distributional_support = excess_peak > 0.0 and excess_tail_mean > 0.0
        is_scattered_full_image_tail = bbox_area_fraction > 0.65 and coverage < 0.25
        if not has_distributional_support or is_scattered_full_image_tail:
            continue
        selection_score = 0.60 * peak + 0.25 * tail_mean + 0.10 * coverage + 0.05 * compactness
        proposals.append(
            RegionProposal(
                bbox=(int(x), int(y), int(w), int(h)),
                score=peak,
                source=source,
                contour=contour,
                component_id=component_id,
                selection_score=float(selection_score),
                selection_reason=f"{source}_heatmap_component",
                pixel_threshold=float(cutoff),
                selection_features={
                    "selector": "rank_mass_robust_tail",
                    "rank_tail_fraction": round(float(tail_fraction), 6),
                    "rank_cutoff": round(float(cutoff), 6),
                    "peak": round(float(peak), 6),
                    "tail_mean": round(float(tail_mean), 6),
                    "excess_peak": round(float(excess_peak), 6),
                    "excess_tail_mean": round(float(excess_tail_mean), 6),
                    "coverage": round(float(coverage), 6),
                    "compactness": round(float(compactness), 6),
                    "bbox_area_fraction": round(float(bbox_area_fraction), 8),
                    "tail_pixel_fraction": round(float(tail_pixel_fraction), 8),
                    "positive_median": round(float(positive_median), 6),
                    "positive_scale": round(float(positive_scale), 6),
                    "robust_median": round(float(robust_median), 6),
                    "robust_scale": round(float(robust_scale), 6),
                    "robust_peak_z": round(float(robust_peak_z), 6),
                    "robust_tail_z": round(float(robust_tail_z), 6),
                    "fallback_min_peak_score": round(float(min_peak_score), 6),
                    "absolute_peak_gate_active": False,
                    "distributional_support": True,
                },
            )
        )

    proposals.sort(key=lambda proposal: (proposal.selection_score or proposal.score, proposal.score), reverse=True)
    return proposals[:top_k]


def _best_component_id(
    labels: np.ndarray,
    contour: np.ndarray,
    component_scores: dict[int, float],
) -> Optional[int]:
    if not component_scores:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    roi = labels[y : y + h, x : x + w]
    present_ids = set(int(v) for v in np.unique(roi) if int(v) in component_scores)
    if not present_ids:
        return None
    return max(present_ids, key=lambda component_id: component_scores[component_id])


def boxes_iou(box: Tuple[int, int, int, int], proposals: Iterable[RegionProposal]) -> float:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    best = 0.0
    for proposal in proposals:
        px1, py1, px2, py2 = proposal.xyxy
        inter_w = max(0, min(x2, px2) - max(x1, px1))
        inter_h = max(0, min(y2, py2) - max(y1, py1))
        inter = inter_w * inter_h
        prop_area = max(0, px2 - px1) * max(0, py2 - py1)
        union = area + prop_area - inter
        if union > 0:
            best = max(best, inter / union)
    return best
