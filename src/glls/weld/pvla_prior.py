from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageStat


class WeldPVLANormalPrior:
    """Visual normal prior retrieved from source-backed station-2 references."""

    def __init__(
        self,
        localizer: Any,
        *,
        graph_path: Path,
        support_paths: list[Path] | None = None,
        patch_size: int = 512,
        max_reference_patches: int = 48,
    ) -> None:
        self.localizer = localizer
        self.graph_path = Path(graph_path).expanduser().resolve()
        self.patch_size = int(patch_size)
        self.max_reference_patches = int(max_reference_patches)
        self.reference_sources = self._graph_reference_paths()
        for path in support_paths or []:
            resolved = Path(path).expanduser().resolve()
            if resolved.exists() and resolved not in self.reference_sources:
                self.reference_sources.append(resolved)
        self.reference_embeddings, self.reference_audit = self._build_reference_bank()

    def score(self, image_path: Path, heatmap_path: Path) -> dict[str, Any]:
        heatmap = np.load(heatmap_path)["heatmap"].astype(np.float32)
        return self.score_heatmap(image_path, heatmap)

    def score_heatmap(self, image_path: Path, heatmap: np.ndarray) -> dict[str, Any]:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        response = np.maximum(heatmap - cv2.GaussianBlur(heatmap, (0, 0), sigmaX=32), 0.0)
        y, x = np.unravel_index(int(np.argmax(response)), response.shape)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            center_x = int(round((x + 0.5) * width / response.shape[1]))
            center_y = int(round((y + 0.5) * height / response.shape[0]))
            crop = _center_crop(rgb, center_x, center_y, self.patch_size)
        query = self.localizer.encode_image_embedding(crop)
        if self.reference_embeddings.size == 0:
            return {
                "normal_similarity": 0.0,
                "normal_distance": 1.0,
                "reference_count": 0,
                "hotspot_xy": [center_x, center_y],
            }
        similarities = self.reference_embeddings @ query
        best_index = int(np.argmax(similarities))
        similarity = float(similarities[best_index])
        return {
            "normal_similarity": similarity,
            "normal_distance": float(1.0 - similarity),
            "reference_count": int(len(self.reference_embeddings)),
            "best_reference": self.reference_audit[best_index],
            "hotspot_xy": [center_x, center_y],
            "crop_size": list(crop.size),
        }

    def _graph_reference_paths(self) -> list[Path]:
        if not self.graph_path.exists():
            return []
        with self.graph_path.open("rb") as handle:
            payload = pickle.load(handle)
        graph = payload.get("graph")
        if graph is None or not graph.has_node("station2_weld_arc"):
            return []
        paths = []
        for value in graph.nodes["station2_weld_arc"].get("image_paths", []):
            path = Path(value).expanduser().resolve()
            if path.exists() and path not in paths:
                paths.append(path)
        return paths

    def _build_reference_bank(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        embeddings = []
        audit = []
        for source in self.reference_sources:
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                patches = _reference_patches(rgb, self.patch_size)
            for box, patch in patches:
                if not _usable_reference_patch(patch):
                    continue
                embeddings.append(self.localizer.encode_image_embedding(patch))
                audit.append({"source_path": str(source), "crop_box": list(box)})
                if len(embeddings) >= self.max_reference_patches:
                    break
            if len(embeddings) >= self.max_reference_patches:
                break
        if not embeddings:
            return np.empty((0, 0), dtype=np.float32), []
        array = np.stack(embeddings).astype(np.float32)
        array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)
        return array, audit


def add_pvla_score_policies(scores: dict[str, float], prior: dict[str, Any]) -> None:
    distance = float(prior.get("normal_distance", 1.0))
    scores["pvla_normal_distance"] = distance
    for key in ("compact_hotspot_sigma16_max", "compact_hotspot_sigma32_max"):
        if key not in scores:
            continue
        base = float(scores[key])
        scores[f"{key}_pvla_x1"] = base * (1.0 + distance)
        scores[f"{key}_pvla_x2"] = base * (1.0 + 2.0 * distance)


def load_pvla_feature_cache(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_pvla_feature_cache(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reference_patches(image: Image.Image, patch_size: int) -> list[tuple[tuple[int, int, int, int], Image.Image]]:
    width, height = image.size
    if width <= patch_size and height <= patch_size:
        return [((0, 0, width, height), image.copy())]
    stride = max(1, patch_size // 2)
    xs = _axis_starts(width, min(width, patch_size), stride)
    ys = _axis_starts(height, min(height, patch_size), stride)
    patches = []
    for y in ys:
        for x in xs:
            box = (x, y, min(width, x + patch_size), min(height, y + patch_size))
            patches.append((box, image.crop(box)))
    return patches


def _axis_starts(length: int, patch: int, stride: int) -> list[int]:
    if patch >= length:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    final = length - patch
    if starts[-1] != final:
        starts.append(final)
    return starts


def _usable_reference_patch(image: Image.Image) -> bool:
    stat = ImageStat.Stat(image.convert("L"))
    if not stat.stddev or stat.stddev[0] < 12.0:
        return False
    values = np.asarray(image.convert("L"))
    return float(np.mean(values > 248)) < 0.85


def _center_crop(image: Image.Image, center_x: int, center_y: int, size: int) -> Image.Image:
    width, height = image.size
    crop_width = min(width, size)
    crop_height = min(height, size)
    x0 = max(0, min(width - crop_width, center_x - crop_width // 2))
    y0 = max(0, min(height - crop_height, center_y - crop_height // 2))
    return image.crop((x0, y0, x0 + crop_width, y0 + crop_height))
