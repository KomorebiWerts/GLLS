from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from glls.weld.data import weld_annotation_reference
from glls.weld.logic import analyze_global_weld_logic, weld_geometry_prior
from glls.weld.pvla_prior import WeldPVLANormalPrior
from glls.weld.scoring import WeldAdaptCLIPScorer, WeldTileConfig, weld_heatmap_shape_policies


DEFAULT_LOCAL_ALERT_THRESHOLD = 0.1306
DEFAULT_TILE_SIZE = 1600


def is_local_weld_anomaly(compact_hotspot_score: float, threshold: float) -> bool:
    return float(compact_hotspot_score) >= float(threshold)


def run_weld_orchestra(
    image_path: str | Path,
    *,
    localizer: Any,
    sam_engine: Any | None,
    pvla_prior: WeldPVLANormalPrior,
    output_dir: str | Path,
    sample: dict[str, Any] | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = 0.25,
    local_threshold: float = DEFAULT_LOCAL_ALERT_THRESHOLD,
) -> dict[str, Any]:
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    station = _sample_station(sample, image_path)

    global_result = analyze_global_weld_logic(
        image_path,
        sam_engine=sam_engine,
        station=station,
    )
    scorer = WeldAdaptCLIPScorer(
        localizer,
        WeldTileConfig(
            tile_size=int(tile_size),
            overlap=float(overlap),
            heatmap_max_side=768,
            include_full_image=True,
        ),
    )
    local_result = scorer.score_path(image_path)
    heatmap = np.asarray(local_result.pop("heatmap"), dtype=np.float32)
    local_result["scores"].update(weld_heatmap_shape_policies(heatmap))
    with Image.open(image_path) as image:
        width, height = image.size
    region_prior = weld_geometry_prior(width, height, station)
    coarse_prior = cv2.resize(
        region_prior.astype(np.uint8),
        (heatmap.shape[1], heatmap.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    focused_heatmap = heatmap.copy()
    focused_heatmap[~coarse_prior] = float(np.median(heatmap[coarse_prior])) if np.any(coarse_prior) else 0.0
    residual = np.maximum(focused_heatmap - cv2.GaussianBlur(focused_heatmap, (0, 0), sigmaX=32), 0.0)
    weld_region_compact_score = (
        float(np.max(residual[coarse_prior])) if np.any(coarse_prior) else 0.0
    )
    local_result["scores"]["weld_region_compact_hotspot_sigma32_max"] = (
        weld_region_compact_score
    )

    # Keep the fine-grained stream independent from the global geometry prior.
    # DEFAULT_LOCAL_ALERT_THRESHOLD was established for the full tiled heatmap;
    # applying the SAM3/geometry mask here changes the score distribution and can
    # erase compact dx regions while still reusing the old operating point.
    compact_score = float(local_result["scores"]["compact_hotspot_sigma32_max"])
    prior = pvla_prior.score_heatmap(image_path, heatmap)
    normal_distance = float(prior.get("normal_distance", 1.0))
    pvla_adjusted_score = compact_score * (0.55 + 0.90 * float(np.clip(normal_distance, 0.0, 1.0)))
    decision_score = compact_score
    local_ng = is_local_weld_anomaly(decision_score, local_threshold)
    final_ng = bool(global_result["logic_ng"] or local_ng)

    heatmap_view = render_heatmap_overlay(image_path, heatmap)
    hotspot_gallery = write_hotspot_crops(
        image_path,
        heatmap,
        output_dir=output_dir / "hotspots",
        count=4,
    )
    annotation_view = render_annotation_overlay(image_path, sample)
    reference = weld_annotation_reference(sample)
    trace = {
        "schema": "glls_weld_orchestra_v1",
        "sample_id": (sample or {}).get("id") or image_path.stem,
        "data_status": (sample or {}).get("quality_status", "uploaded_unlabeled"),
        "official_ground_truth": None,
        "reference_ground_truth": reference["ground_truth"],
        "reference_label_role": reference["role"],
        "reference_label_reason": reference["reason"],
        "global_stream": {
            key: value
            for key, value in global_result.items()
            if key not in {"overlay"}
        },
        "local_stream": {
            "compact_hotspot_score": compact_score,
            "weld_region_compact_hotspot_score": weld_region_compact_score,
            "pvla_normal_similarity": prior.get("normal_similarity"),
            "pvla_normal_distance": normal_distance,
            "orchestra_score": decision_score,
            "pvla_adjusted_score": pvla_adjusted_score,
            "alert_threshold": float(local_threshold),
            "threshold_status": (
                "full tiled compact-hotspot operating point from the existing offline validation; "
                "PVLA similarity is reported as evidence and does not suppress the decision score"
            ),
            "local_ng": bool(local_ng),
            "top_tiles": local_result.get("top_tiles", []),
            "online_crops": [
                {"path": str(item[0]), "caption": str(item[1])}
                for item in hotspot_gallery
            ],
        },
        "fusion": {
            "rule": "global_logic_ng OR compact_hotspot_score >= threshold",
            "global_logic_ng": bool(global_result["logic_ng"]),
            "local_adaptclip_ng": bool(local_ng),
            "final_verdict": "NG" if final_ng else "OK",
        },
        "pvla": {
            **prior,
            "role": "PDF rules plus one trusted normal image suppress known-normal local responses",
        },
        "annotation_evidence": {
            "shape_labels": list((sample or {}).get("shape_labels") or []),
            "reference_ground_truth": reference["ground_truth"],
            "role": reference["role"],
            "note": reference["reason"],
        },
    }
    trace_path = output_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "verdict": "NG" if final_ng else "OK",
        "global_logic_ng": bool(global_result["logic_ng"]),
        "local_ng": bool(local_ng),
        "global": global_result,
        "local": local_result,
        "pvla": prior,
        "orchestra_score": float(decision_score),
        "local_threshold": float(local_threshold),
        "heatmap_view": heatmap_view,
        "hotspot_gallery": hotspot_gallery,
        "annotation_view": annotation_view,
        "trace": trace,
        "trace_path": str(trace_path),
        "reference": reference,
    }


def render_heatmap_overlay(image_path: str | Path, heatmap: np.ndarray) -> Image.Image:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    values = np.asarray(heatmap, dtype=np.float32)
    low, high = np.quantile(values, [0.55, 0.995])
    normalized = np.clip((values - low) / max(1e-6, high - low), 0.0, 1.0)
    resized = cv2.resize(normalized, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(np.uint8(resized * 255.0), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    alpha = (0.12 + 0.48 * resized)[..., None]
    overlay = np.uint8(np.clip(rgb * (1.0 - alpha) + color * alpha, 0, 255))
    result = Image.fromarray(overlay)
    result.thumbnail((1500, 1000), Image.Resampling.LANCZOS)
    return result


def write_hotspot_crops(
    image_path: str | Path,
    heatmap: np.ndarray,
    *,
    output_dir: Path,
    count: int,
    crop_size: int = 896,
) -> list[tuple[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    response = np.maximum(
        heatmap - cv2.GaussianBlur(np.asarray(heatmap, dtype=np.float32), (0, 0), sigmaX=32),
        0.0,
    )
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        selected: list[tuple[int, int, float]] = []
        work = response.copy()
        suppress = max(8, int(round(min(work.shape) * 0.10)))
        for _ in range(max(1, int(count))):
            y, x = np.unravel_index(int(np.argmax(work)), work.shape)
            score = float(work[y, x])
            if score <= 0 and selected:
                break
            selected.append((x, y, score))
            cv2.circle(work, (int(x), int(y)), suppress, 0.0, -1)
        gallery = []
        for index, (x, y, score) in enumerate(selected):
            center_x = int(round((x + 0.5) * width / response.shape[1]))
            center_y = int(round((y + 0.5) * height / response.shape[0]))
            half = min(crop_size, width, height) // 2
            x0 = max(0, min(width - 2 * half, center_x - half))
            y0 = max(0, min(height - 2 * half, center_y - half))
            crop = rgb.crop((x0, y0, x0 + 2 * half, y0 + 2 * half))
            draw = ImageDraw.Draw(crop)
            draw.rectangle((2, 2, crop.width - 3, crop.height - 3), outline=(192, 57, 43), width=8)
            output = output_dir / f"hotspot_{index + 1:02d}.png"
            crop.save(output)
            gallery.append(
                (
                    str(output),
                    f"Online crop {index + 1} | center=({center_x},{center_y}) | compact={score:.4f}",
                )
            )
    return gallery


def render_annotation_overlay(
    image_path: str | Path,
    sample: dict[str, Any] | None,
) -> Image.Image | None:
    if not sample or not sample.get("annotation_path"):
        return None
    root = Path(sample["_root"])
    annotation_path = root / sample["annotation_path"]
    if not annotation_path.exists():
        return None
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    line_width = max(5, int(round(min(rgb.size) * 0.003)))
    for shape in annotation.get("shapes") or []:
        points = [tuple(float(value) for value in point) for point in shape.get("points") or []]
        if len(points) < 2:
            continue
        label = str(shape.get("label") or "region")
        color = (192, 57, 43) if label == "dx" else (52, 82, 199)
        draw.line(points + [points[0]], fill=color, width=line_width, joint="curve")
        x, y = points[0]
        text_box = (x, max(0.0, y - 30.0), x + 88.0, y)
        draw.rectangle(text_box, fill=color)
        draw.text((x + 6.0, max(0.0, y - 27.0)), label, fill=(255, 255, 255))
    rgb.thumbnail((1500, 1000), Image.Resampling.LANCZOS)
    return rgb


def _sample_station(sample: dict[str, Any] | None, image_path: Path) -> str:
    if sample and int(sample.get("station") or 0) in {1, 2}:
        return f"station{int(sample['station'])}"
    with Image.open(image_path) as image:
        width, height = image.size
    return "station2" if width / max(1, height) >= 1.42 else "station1"
