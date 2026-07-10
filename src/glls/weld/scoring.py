from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from glls.eval.binary_ad import image_score_from_heatmap


@dataclass(frozen=True)
class WeldTileConfig:
    tile_size: int = 1024
    overlap: float = 0.25
    heatmap_max_side: int = 768
    score_tail_fraction: float = 0.01
    include_full_image: bool = True


class WeldAdaptCLIPScorer:
    """Multi-scale AdaptCLIP scoring that preserves small high-resolution defects."""

    def __init__(self, localizer: Any, config: WeldTileConfig | None = None):
        self.localizer = localizer
        self.config = config or WeldTileConfig()

    def score_path(self, image_path: str | Path) -> dict[str, Any]:
        image_path = str(image_path)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            full_score = None
            full_heatmap_tail = None
            if self.config.include_full_image:
                full_heatmap, _ = self.localizer.predict_anomaly_map(rgb)
                full_score = _finite_or_none(getattr(self.localizer, "last_image_score", None))
                full_heatmap_tail = image_score_from_heatmap(
                    np.asarray(full_heatmap), tail_fraction=self.config.score_tail_fraction
                )

            boxes = generate_tile_boxes(width, height, self.config.tile_size, self.config.overlap)
            coarse_width, coarse_height = _coarse_size(width, height, self.config.heatmap_max_side)
            heatmap_sum = np.zeros((coarse_height, coarse_width), dtype=np.float32)
            heatmap_count = np.zeros((coarse_height, coarse_width), dtype=np.float32)
            tile_rows = []
            for index, box in enumerate(boxes):
                x0, y0, x1, y1 = box
                crop = rgb.crop(box)
                tile_heatmap, _ = self.localizer.predict_anomaly_map(crop)
                image_score = _finite_or_none(getattr(self.localizer, "last_image_score", None))
                heatmap_tail = image_score_from_heatmap(
                    np.asarray(tile_heatmap), tail_fraction=self.config.score_tail_fraction
                )
                fused_score = max(image_score if image_score is not None else heatmap_tail, heatmap_tail)
                tile_rows.append(
                    {
                        "index": index,
                        "bbox_xyxy": [x0, y0, x1, y1],
                        "image_score": image_score,
                        "heatmap_tail_score": float(heatmap_tail),
                        "fused_score": float(fused_score),
                    }
                )
                _stitch_heatmap(
                    heatmap_sum,
                    heatmap_count,
                    np.asarray(tile_heatmap, dtype=np.float32),
                    box=box,
                    image_size=(width, height),
                )

        stitched = heatmap_sum / np.maximum(heatmap_count, 1.0)
        tile_rows.sort(key=lambda row: row["fused_score"], reverse=True)
        image_scores = [
            float(row["image_score"] if row["image_score"] is not None else row["heatmap_tail_score"])
            for row in tile_rows
        ]
        fused_scores = [float(row["fused_score"]) for row in tile_rows]
        policies = _aggregation_policies(
            full_score=full_score,
            full_heatmap_tail=full_heatmap_tail,
            image_scores=image_scores,
            fused_scores=fused_scores,
            stitched_heatmap=stitched,
            tail_fraction=self.config.score_tail_fraction,
        )
        return {
            "image_path": image_path,
            "image_size": [width, height],
            "coarse_heatmap_size": [coarse_width, coarse_height],
            "tile_config": {
                "tile_size": self.config.tile_size,
                "overlap": self.config.overlap,
                "tile_count": len(boxes),
                "include_full_image": self.config.include_full_image,
            },
            "full_image_score": full_score,
            "full_heatmap_tail_score": full_heatmap_tail,
            "scores": policies,
            "top_tiles": tile_rows[:8],
            "heatmap": stitched,
        }


def generate_tile_boxes(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    tile_size = max(64, int(tile_size))
    overlap = min(0.9, max(0.0, float(overlap)))
    tile_width = min(tile_size, width)
    tile_height = min(tile_size, height)
    stride_x = max(1, int(round(tile_width * (1.0 - overlap))))
    stride_y = max(1, int(round(tile_height * (1.0 - overlap))))
    xs = _axis_starts(width, tile_width, stride_x)
    ys = _axis_starts(height, tile_height, stride_y)
    return [(x, y, x + tile_width, y + tile_height) for y in ys for x in xs]


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    if tile >= length:
        return [0]
    starts = list(range(0, max(1, length - tile + 1), stride))
    final = length - tile
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def _coarse_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max(64, int(max_side)) / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _stitch_heatmap(
    heatmap_sum: np.ndarray,
    heatmap_count: np.ndarray,
    tile_heatmap: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> None:
    width, height = image_size
    coarse_height, coarse_width = heatmap_sum.shape
    x0, y0, x1, y1 = box
    cx0 = max(0, min(coarse_width - 1, int(math.floor(x0 * coarse_width / width))))
    cy0 = max(0, min(coarse_height - 1, int(math.floor(y0 * coarse_height / height))))
    cx1 = max(cx0 + 1, min(coarse_width, int(math.ceil(x1 * coarse_width / width))))
    cy1 = max(cy0 + 1, min(coarse_height, int(math.ceil(y1 * coarse_height / height))))
    resized = cv2.resize(tile_heatmap, (cx1 - cx0, cy1 - cy0), interpolation=cv2.INTER_LINEAR)
    heatmap_sum[cy0:cy1, cx0:cx1] += resized
    heatmap_count[cy0:cy1, cx0:cx1] += 1.0


def _aggregation_policies(
    *,
    full_score: float | None,
    full_heatmap_tail: float | None,
    image_scores: list[float],
    fused_scores: list[float],
    stitched_heatmap: np.ndarray,
    tail_fraction: float,
) -> dict[str, float]:
    def top_mean(values: list[float], count: int) -> float:
        selected = sorted(values, reverse=True)[: max(1, min(count, len(values)))]
        return float(np.mean(selected)) if selected else 0.0

    tile_max = max(image_scores) if image_scores else 0.0
    top2 = top_mean(image_scores, 2)
    top3 = top_mean(image_scores, 3)
    fused_top2 = top_mean(fused_scores, 2)
    full = full_score if full_score is not None else (full_heatmap_tail or 0.0)
    stitched_tail = image_score_from_heatmap(stitched_heatmap, tail_fraction=tail_fraction)
    return {
        "full_image": float(full),
        "tile_max": float(tile_max),
        "tile_top2_mean": float(top2),
        "tile_top3_mean": float(top3),
        "tile_q95": float(np.quantile(image_scores, 0.95)) if image_scores else 0.0,
        "tile_fused_top2_mean": float(fused_top2),
        "max_full_tile_top2": float(max(full, top2)),
        "mean_full_tile_top2": float((full + top2) / 2.0),
        "stitched_heatmap_tail": float(stitched_tail),
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def weld_heatmap_shape_policies(heatmap: np.ndarray) -> dict[str, float]:
    """Score compact hotspots after removing broad allowed structural responses."""
    values = np.asarray(heatmap, dtype=np.float32)
    policies = {}
    for sigma in (16, 32):
        background = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma)
        residual = np.maximum(values - background, 0.0)
        policies[f"compact_hotspot_sigma{sigma}_max"] = float(np.max(residual))
    return policies
