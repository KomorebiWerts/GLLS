from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def analyze_global_weld_logic(
    image_path: str | Path,
    *,
    sam_engine: Any | None = None,
    station: str = "auto",
) -> dict[str, Any]:
    """Conservatively gate whole-image weld logic before local AD.

    SAM3 supplies the structural mask when its response passes geometric quality
    checks. A deterministic annular/arc response is retained as a fallback so a
    poor segmentation never becomes a confident logical NG decision.
    """
    image_path = Path(image_path).expanduser().resolve()
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    resolved_station = _resolve_station(width, height, station)
    prior, boxes, angle_span = _geometry_prior(width, height, resolved_station)
    fallback_mask = _appearance_mask(rgb, prior)
    selected_mask = fallback_mask
    mask_source = "geometry+appearance fallback"
    sam_score = None
    sam_quality = 0.0

    if sam_engine is not None:
        try:
            sam_engine.set_image(str(image_path))
            mask, score = sam_engine.predict_mask_with_boxes(boxes, threshold=0.2)
            candidate = np.squeeze(np.asarray(mask).astype(bool)) if mask is not None else None
            if candidate is not None and candidate.shape == prior.shape:
                candidate = _clean_mask(candidate & prior & fallback_mask)
                sam_quality = _mask_quality(candidate, prior, angle_span)
                if sam_quality >= 0.42:
                    selected_mask = candidate
                    mask_source = "SAM3 global boxes + weld geometry prior"
            sam_score = float(score)
        except Exception:
            sam_score = None

    metrics = _logic_metrics(selected_mask, prior, angle_span, resolved_station)
    reliable_structure = metrics["observed_angle_count"] >= (72 if resolved_station == "station1" else 36)
    rules = _logic_rules(metrics, resolved_station, reliable_structure)
    logic_ng = any(rule["triggered"] for rule in rules)
    overlay = _render_overlay(rgb, prior, selected_mask, rules, resolved_station)
    return {
        "station": resolved_station,
        "logic_ng": bool(logic_ng),
        "verdict": "NG" if logic_ng else "PASS_TO_LOCAL",
        "mask_source": mask_source,
        "sam3_score": sam_score,
        "sam3_quality": float(sam_quality),
        "reliable_structure": bool(reliable_structure),
        "metrics": metrics,
        "rules": rules,
        "overlay": overlay,
        "boxes": boxes,
    }


def weld_geometry_prior(width: int, height: int, station: str) -> np.ndarray:
    return _geometry_prior(width, height, station)[0]


def _resolve_station(width: int, height: int, station: str) -> str:
    if station in {"station1", "station2"}:
        return station
    ratio = width / max(1, height)
    return "station2" if ratio >= 1.42 else "station1"


def _geometry_prior(
    width: int,
    height: int,
    station: str,
) -> tuple[np.ndarray, list[list[float]], tuple[float, float]]:
    yy, xx = np.ogrid[:height, :width]
    if station == "station1":
        cx, cy = width / 2.0, height / 2.0
        scale = float(min(width, height))
        inner, outer = 0.34 * scale, 0.44 * scale
        angle_span = (0.0, 360.0)
        prompt_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    else:
        cx, cy = width / 2.0, 0.14 * height
        scale = float(height)
        inner, outer = 0.48 * scale, 0.72 * scale
        angle_span = (35.0, 145.0)
        prompt_angles = np.deg2rad(np.linspace(42.0, 138.0, 6))
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    angle = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0
    prior = (radius >= inner) & (radius <= outer)
    if station == "station2":
        prior &= (angle >= angle_span[0]) & (angle <= angle_span[1])
    box_side = 0.20 * scale if station == "station1" else 0.24 * scale
    prompt_radius = (inner + outer) / 2.0
    boxes = []
    for value in prompt_angles:
        px = cx + prompt_radius * float(np.cos(value))
        py = cy + prompt_radius * float(np.sin(value))
        boxes.append(
            [
                float(px / width),
                float(py / height),
                float(box_side / width),
                float(box_side / height),
            ]
        )
    return prior, boxes, angle_span


def _appearance_mask(rgb: np.ndarray, prior: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    values = gray[prior]
    if values.size == 0:
        return np.zeros_like(prior)
    threshold = float(np.percentile(values, 47.0))
    dark = (gray <= threshold) & prior
    scale = max(3, int(round(min(gray.shape) * 0.004)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
    return cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    scale = max(3, int(round(min(mask.shape) * 0.003)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def _mask_quality(mask: np.ndarray, prior: np.ndarray, angle_span: tuple[float, float]) -> float:
    if not np.any(mask):
        return 0.0
    area_ratio = float(mask.sum() / max(1, prior.sum()))
    angular = _angular_profile(mask, prior, angle_span)
    coverage = float(np.mean(angular > 0.025)) if angular.size else 0.0
    area_score = max(0.0, 1.0 - abs(area_ratio - 0.32) / 0.42)
    return float(0.55 * coverage + 0.45 * area_score)


def _logic_metrics(
    mask: np.ndarray,
    prior: np.ndarray,
    angle_span: tuple[float, float],
    station: str,
) -> dict[str, float | int]:
    profile = _angular_profile(mask, prior, angle_span)
    observed = profile > (0.025 if station == "station1" else 0.035)
    max_gap = _max_false_run(observed, circular=station == "station1")
    degrees_per_bin = (angle_span[1] - angle_span[0]) / max(1, len(profile))
    components, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    component_areas = sorted((int(row[cv2.CC_STAT_AREA]) for row in stats[1:]), reverse=True)
    significant = sum(area >= max(64, int(prior.sum() * 0.004)) for area in component_areas)
    active = profile[observed]
    thickness_cv = float(np.std(active) / max(1e-6, np.mean(active))) if active.size else 1.0
    return {
        "angular_coverage": float(np.mean(observed)) if observed.size else 0.0,
        "max_gap_degrees": float(max_gap * degrees_per_bin),
        "thickness_cv": thickness_cv,
        "mask_area_ratio": float(mask.sum() / max(1, prior.sum())),
        "significant_components": int(significant),
        "observed_angle_count": int(observed.sum()),
        "component_count": int(max(0, components - 1)),
    }


def _angular_profile(
    mask: np.ndarray,
    prior: np.ndarray,
    angle_span: tuple[float, float],
    bins: int = 180,
) -> np.ndarray:
    height, width = mask.shape
    if angle_span == (0.0, 360.0):
        cx, cy = width / 2.0, height / 2.0
    else:
        cx, cy = width / 2.0, 0.14 * height
    yy, xx = np.nonzero(prior)
    if yy.size == 0:
        return np.zeros(bins, dtype=np.float32)
    angle = (np.degrees(np.arctan2(yy.astype(np.float32) - cy, xx.astype(np.float32) - cx)) + 360.0) % 360.0
    low, high = angle_span
    valid = (angle >= low) & (angle < high)
    positions = np.floor((angle[valid] - low) * bins / max(1e-6, high - low)).astype(np.int32)
    positions = np.clip(positions, 0, bins - 1)
    totals = np.bincount(positions, minlength=bins).astype(np.float32)
    positives = np.bincount(
        positions,
        weights=mask[yy[valid], xx[valid]].astype(np.float32),
        minlength=bins,
    ).astype(np.float32)
    return positives / np.maximum(totals, 1.0)


def _max_false_run(values: np.ndarray, *, circular: bool) -> int:
    if values.size == 0 or np.all(values):
        return 0
    work = np.concatenate([values, values]) if circular else values
    best = current = 0
    for value in work:
        current = 0 if value else current + 1
        best = max(best, current)
    return min(best, len(values))


def _logic_rules(metrics: dict[str, Any], station: str, reliable: bool) -> list[dict[str, Any]]:
    if station == "station1":
        candidates = [
            ("closed_ring_continuity", metrics["max_gap_degrees"] > 42.0, "maximum ring gap <= 42 deg"),
            ("stable_radial_thickness", metrics["thickness_cv"] > 0.92, "angular thickness CV <= 0.92"),
            ("single_connected_weld_structure", metrics["significant_components"] > 4, "no fragmented global weld response"),
        ]
    else:
        candidates = [
            ("visible_arc_continuity", metrics["max_gap_degrees"] > 34.0, "maximum visible-arc gap <= 34 deg"),
            ("stable_arc_thickness", metrics["thickness_cv"] > 1.05, "visible-arc thickness CV <= 1.05"),
        ]
    return [
        {
            "name": name,
            "triggered": bool(reliable and triggered),
            "status": "NG" if reliable and triggered else "PASS" if reliable else "INSUFFICIENT_EVIDENCE",
            "criterion": criterion,
        }
        for name, triggered, criterion in candidates
    ]


def _render_overlay(
    rgb: np.ndarray,
    prior: np.ndarray,
    mask: np.ndarray,
    rules: list[dict[str, Any]],
    station: str,
) -> Image.Image:
    canvas = rgb.copy()
    tint = np.zeros_like(canvas)
    tint[prior] = (52, 82, 199)
    canvas = np.where(prior[..., None], (0.88 * canvas + 0.12 * tint).astype(np.uint8), canvas)
    mask_color = np.zeros_like(canvas)
    mask_color[mask] = (15, 170, 105)
    canvas = np.where(mask[..., None], (0.62 * canvas + 0.38 * mask_color).astype(np.uint8), canvas)
    outline = cv2.morphologyEx(prior.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    canvas[outline.astype(bool)] = (52, 82, 199)
    status = "NG" if any(row["triggered"] for row in rules) else "LOGIC PASS"
    color = (192, 57, 43) if status == "NG" else (15, 122, 77)
    cv2.rectangle(canvas, (18, 18), (390, 92), (250, 252, 255), -1)
    cv2.putText(canvas, f"{station.upper()} GLOBAL LOGIC", (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (27, 34, 48), 2)
    cv2.putText(canvas, status, (32, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)
    image = Image.fromarray(canvas)
    image.thumbnail((1500, 1000), Image.Resampling.LANCZOS)
    return image
