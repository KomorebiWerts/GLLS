from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from glls.rag.source_chain import build_graph_source_metadata
from glls.weld.data import load_weld_manifest


def build_weld_assets(
    manifest_path: Path,
    *,
    sam_checkpoint: str,
    device: str | None = None,
    max_station1_refs: int = 1,
    max_station2_refs: int = 0,
) -> dict[str, Any]:
    from glls.rag.GraphRag import SimInspecGraphEngine
    from glls.seg.sam3_engine import Sam3Engine

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_weld_manifest(manifest_path)
    root = Path(manifest["output_root"])
    reference_root = root / "assets" / "normal_references" / "gear_weld"
    reference_root.mkdir(parents=True, exist_ok=True)
    sam_engine = Sam3Engine(sam_checkpoint, device=device)

    normal_meta = manifest.get("normal_reference") or {}
    source = Path(normal_meta.get("path") or "").expanduser()
    if not source.exists():
        raise FileNotFoundError(
            "Trusted PDF normal reference is missing. Expected 焊缝缺陷检测数据/reference/pdf_page4_normal.png"
        )
    source_copy = reference_root / "pdf_page4_normal_full.png"
    with Image.open(source) as image:
        image.convert("RGB").save(source_copy)
    ring_output = reference_root / "pdf_page4_normal_weld_ring.png"
    ring_audit = _write_station1_reference(sam_engine, source_copy, ring_output)
    station1_refs = [
        {
            "region": "station1_full_weld_ring",
            "path": str(ring_output),
            "trusted_source_path": str(source_copy),
            "source_caption": normal_meta.get("caption", "正常图像"),
            **ring_audit,
        }
    ]
    station2_refs = _write_arc_atlas(source_copy, reference_root, sectors=8)

    knowledge_path = Path(manifest["knowledge"]["knowledge_path"])
    knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))
    engine = SimInspecGraphEngine()
    engine.load_json(knowledge)
    references_by_region = {
        "station1_full_weld_ring": [row["path"] for row in station1_refs],
        "station2_weld_arc": [row["path"] for row in station2_refs],
        "mirror_and_surrounding_surface": [row["path"] for row in station1_refs[:2]],
    }
    for region, paths in references_by_region.items():
        if engine.G.has_node(region):
            engine.G.nodes[region]["image_paths"] = paths
    graph_path = root / "assets" / "graph" / "gear_weld_graph.pkl"
    engine.source_metadata.update(
        build_graph_source_metadata(
            dataset="weld",
            category="gear_weld",
            source_json_path=str(knowledge_path),
            text_knowledge_root=str(knowledge_path.parent),
            visual_reference_root=str(reference_root),
            graph_output_root=str(graph_path.parent),
            visual_reference_builder="glls.weld.assets",
            visual_reference_max_k_shot=1,
        )
    )
    engine.source_metadata.update(
        {
            "source_documents": ["焊缝缺陷检测讨论.pdf", "焊缝缺陷检测升级.pdf"],
            "annotation_label_policy": manifest.get("annotation_policy") or {},
            "normal_reference_policy": "one trusted PDF image; all arc crops are deterministic derivatives",
        }
    )
    engine.save_to_disk(str(graph_path))

    result = {
        "schema": "glls_weld_assets_v2",
        "manifest_path": str(manifest_path),
        "knowledge_path": str(knowledge_path),
        "graph_path": str(graph_path),
        "reference_root": str(reference_root),
        "trusted_normal_reference": str(source_copy),
        "references": station1_refs + station2_refs,
        "references_by_region": references_by_region,
    }
    assets_path = root / "assets" / "assets_manifest.json"
    assets_path.parent.mkdir(parents=True, exist_ok=True)
    assets_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["assets_manifest_path"] = str(assets_path)
    return result


def _write_arc_atlas(source: Path, output_root: Path, *, sectors: int) -> list[dict[str, Any]]:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        cx, cy = width / 2.0, height / 2.0
        radius = 0.39 * min(width, height)
        side = max(160, int(round(0.22 * min(width, height))))
        rows = []
        for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)):
            px = cx + radius * float(np.cos(angle))
            py = cy + radius * float(np.sin(angle))
            x0 = max(0, min(width - side, int(round(px - side / 2))))
            y0 = max(0, min(height - side, int(round(py - side / 2))))
            output = output_root / f"pdf_page4_normal_arc_{index:02d}.png"
            rgb.crop((x0, y0, x0 + side, y0 + side)).save(output)
            rows.append(
                {
                    "region": "station2_weld_arc",
                    "path": str(output),
                    "source_image_path": str(source),
                    "generation": "deterministic arc-sector crop from the single trusted PDF normal image",
                    "sector_index": index,
                    "crop_box": [x0, y0, x0 + side, y0 + side],
                    "one_shot_derivative": True,
                }
            )
    return rows


def _write_station1_reference(sam_engine: Any, source: Path, output: Path) -> dict[str, Any]:
    try:
        with Image.open(source) as image:
            width, height = image.size
        prior = _station1_annulus_prior(width, height)
        boxes = _station1_ring_boxes(width, height)
        sam_engine.set_image(str(source))
        mask, score = sam_engine.predict_mask_with_boxes(boxes, threshold=0.2)
        sam_mask = np.squeeze(np.asarray(mask).astype(bool)) if mask is not None else None
        if sam_mask is not None and sam_mask.shape == prior.shape:
            refined = sam_mask & prior
            coverage = float(refined.sum() / max(1, prior.sum()))
        else:
            refined = np.zeros_like(prior)
            coverage = 0.0
        angular_coverage = _annular_angular_coverage(refined, width=width, height=height)
        if coverage >= 0.35 and angular_coverage >= 0.70:
            kernel = np.ones((17, 17), dtype=np.uint8)
            cleaned = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            status = "sam3_verified_ring_with_geometry_unwrap"
        else:
            cleaned = prior
            status = "sam3_low_quality_geometry_unwrap_fallback"
        cutout = _unwrap_station1_ring(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        cutout.save(output)
        return {
            "source_image_path": str(source),
            "generation": "SAM3 ring box prompts constrained by the station-1 annular weld prior",
            "sam3_status": status,
            "sam3_score": float(score),
            "mask_area_ratio": float(cleaned.mean()),
            "sam3_prior_coverage": coverage,
            "sam3_angular_coverage": angular_coverage,
            "boxes": boxes,
        }
    except Exception as exc:
        result = _fallback_center_crop(source, output, crop_fraction=0.72, status="sam3_error_fallback")
        result["sam3_error"] = str(exc)
        return result


def _station1_annulus_prior(width: int, height: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    scale = float(min(width, height))
    return (radius >= 0.34 * scale) & (radius <= 0.44 * scale)


def _station1_ring_boxes(width: int, height: int) -> list[list[float]]:
    scale = float(min(width, height))
    radius = 0.39 * scale
    box_side = 0.18 * scale
    boxes = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        center_x = width / 2.0 + radius * float(np.cos(angle))
        center_y = height / 2.0 + radius * float(np.sin(angle))
        boxes.append(
            [
                float(center_x / width),
                float(center_y / height),
                float(box_side / width),
                float(box_side / height),
            ]
        )
    return boxes


def _unwrap_station1_ring(source: Path) -> Image.Image:
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = float(min(width, height))
    max_radius = 0.46 * scale
    circumference = max(720, int(round(2.0 * np.pi * 0.39 * scale)))
    polar = cv2.warpPolar(
        rgb,
        (int(round(max_radius)), circumference),
        (width / 2.0, height / 2.0),
        max_radius,
        cv2.WARP_POLAR_LINEAR,
    )
    inner = int(round(0.33 * scale))
    outer = min(polar.shape[1], int(round(0.45 * scale)))
    strip = polar[:, inner:outer]
    return Image.fromarray(cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE))


def _annular_angular_coverage(mask: np.ndarray, *, width: int, height: int) -> float:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return 0.0
    angles = (np.degrees(np.arctan2(yy - height / 2.0, xx - width / 2.0)) + 360.0) % 360.0
    bins = np.unique(np.floor(angles / 5.0).astype(np.int32))
    return float(len(bins) / 72.0)


def _write_station2_reference(
    sam_engine: Any,
    source: Path,
    output: Path,
    *,
    annotation: dict[str, Any],
) -> dict[str, Any]:
    width = int(annotation.get("imageWidth") or 0)
    height = int(annotation.get("imageHeight") or 0)
    box = _context_box(annotation, width=width, height=height)
    try:
        sam_engine.set_image(str(source))
        mask, score = sam_engine.predict_mask_with_boxes([box], threshold=0.2)
        cleaned = _largest_component(mask, close_kernel=13)
        if cleaned is None:
            return _fallback_box_crop(source, output, box=box, status="empty_box_mask_fallback")
        cutout = sam_engine.get_cutout_pil(cleaned, padding=24)
        if cutout is None:
            return _fallback_box_crop(source, output, box=box, status="empty_cutout_fallback")
        output.parent.mkdir(parents=True, exist_ok=True)
        cutout.save(output)
        return {
            "source_image_path": str(source),
            "generation": "SAM3 box-prompt station-2 weld-context cutout",
            "sam3_status": "box_prompt_cutout",
            "sam3_score": float(score),
            "box_cxcywh_norm": box,
            "mask_area_ratio": float(cleaned.mean()),
        }
    except Exception as exc:
        result = _fallback_box_crop(source, output, box=box, status="sam3_error_fallback")
        result["sam3_error"] = str(exc)
        return result


def _context_box(annotation: dict[str, Any], *, width: int, height: int) -> list[float]:
    points = []
    for shape in annotation.get("shapes") or []:
        if str(shape.get("label") or "").strip() == "ljtq":
            points.extend(shape.get("points") or [])
    if not points or width <= 0 or height <= 0:
        return [0.5, 0.62, 0.9, 0.65]
    array = np.asarray(points, dtype=np.float32)
    x0, y0 = array.min(axis=0)
    x1, y1 = array.max(axis=0)
    pad_x = max(80.0, (x1 - x0) * 0.35)
    pad_y = max(80.0, (y1 - y0) * 0.35)
    x0, x1 = max(0.0, x0 - pad_x), min(float(width), x1 + pad_x)
    y0, y1 = max(0.0, y0 - pad_y), min(float(height), y1 + pad_y)
    return [
        float((x0 + x1) / (2.0 * width)),
        float((y0 + y1) / (2.0 * height)),
        float((x1 - x0) / width),
        float((y1 - y0) / height),
    ]


def _largest_component(mask: Any, *, close_kernel: int) -> np.ndarray | None:
    if mask is None:
        return None
    array = np.asarray(mask).astype(np.uint8)
    array = np.squeeze(array)
    if array.ndim != 2 or not np.any(array):
        return None
    kernel_size = max(3, int(close_kernel) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    array = cv2.morphologyEx(array, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = labels == largest
    return cleaned if np.any(cleaned) else None


def _fallback_center_crop(source: Path, output: Path, *, crop_fraction: float, status: str) -> dict[str, Any]:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        side = int(round(min(width, height) * crop_fraction))
        x0 = max(0, (width - side) // 2)
        y0 = max(0, (height - side) // 2)
        output.parent.mkdir(parents=True, exist_ok=True)
        rgb.crop((x0, y0, x0 + side, y0 + side)).save(output)
    return {
        "source_image_path": str(source),
        "generation": "deterministic center crop",
        "sam3_status": status,
    }


def _fallback_box_crop(source: Path, output: Path, *, box: list[float], status: str) -> dict[str, Any]:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        cx, cy, bw, bh = box
        x0 = max(0, int(round((cx - bw / 2) * width)))
        y0 = max(0, int(round((cy - bh / 2) * height)))
        x1 = min(width, int(round((cx + bw / 2) * width)))
        y1 = min(height, int(round((cy + bh / 2) * height)))
        output.parent.mkdir(parents=True, exist_ok=True)
        rgb.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).save(output)
    return {
        "source_image_path": str(source),
        "generation": "deterministic context-box crop",
        "sam3_status": status,
        "box_cxcywh_norm": box,
    }
