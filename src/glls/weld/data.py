from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from glls.weld.knowledge import write_weld_knowledge


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CONTEXT_LABELS = {"ljtq"}
DEFAULT_REGION_LABELS = {"dx"}


def weld_annotation_reference(sample: dict[str, Any] | None) -> dict[str, Any]:
    """Map archive polygon codes to the station-2 inspection contract.

    The upgrade PDF says connection protrusions are detected and filtered at
    station 2, but are not anomalous by themselves. The compact ``dx`` marks
    are defect regions. This is a source-backed review label, not an official
    benchmark split supplied with the archive.
    """
    labels = {str(value).strip().lower() for value in (sample or {}).get("shape_labels") or [] if value}
    if "dx" in labels:
        return {
            "ground_truth": "NG",
            "binary_label": 1,
            "role": "annotated_anomaly",
            "display": "Annotated anomaly (dx)",
            "reason": "dx marks a compact weld-defect/spatter region",
        }
    if labels and labels <= DEFAULT_CONTEXT_LABELS:
        return {
            "ground_truth": "OK",
            "binary_label": 0,
            "role": "allowed_context",
            "display": "Allowed context (ljtq)",
            "reason": "station-2 connection protrusion is detected for filtering, not counted as anomaly",
        }
    if labels:
        return {
            "ground_truth": "UNKNOWN",
            "binary_label": None,
            "role": "unmapped_annotation",
            "display": "Unmapped annotation",
            "reason": "annotation code is not mapped by the official inspection contract",
        }
    return {
        "ground_truth": "UNKNOWN",
        "binary_label": None,
        "role": "unlabeled",
        "display": "Unlabelled review",
        "reason": "no region annotation is available",
    }


def prepare_weld_dataset(
    source_root: Path,
    output_root: Path,
    *,
    seed: int = 20260710,
    region_labels: set[str] | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    region_labels = set(region_labels or DEFAULT_REGION_LABELS)
    context_labels = set(DEFAULT_CONTEXT_LABELS)
    raw_zip = source_root / "原始数据.zip"
    annotation_zip = source_root / "annotation_data.zip"
    if not raw_zip.exists() or not annotation_zip.exists():
        raise FileNotFoundError(f"Expected 原始数据.zip and annotation_data.zip under {source_root}")

    station1_dir = output_root / "images" / "station1_unlabeled"
    station2_dir = output_root / "images" / "station2"
    region_mask_dir = output_root / "masks" / "station2_dx_region"
    context_mask_dir = output_root / "masks" / "station2_context_ljtq"
    compact_annotation_dir = output_root / "annotations" / "labelme_compact"
    for directory in (station1_dir, station2_dir, region_mask_dir, context_mask_dir, compact_annotation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    station1_rows = _extract_station1(raw_zip, output_root=output_root, image_dir=station1_dir)
    station2_rows, label_counts = _extract_station2(
        annotation_zip,
        output_root=output_root,
        image_dir=station2_dir,
        region_mask_dir=region_mask_dir,
        context_mask_dir=context_mask_dir,
        compact_annotation_dir=compact_annotation_dir,
        region_labels=region_labels,
        context_labels=context_labels,
    )
    knowledge_paths = write_weld_knowledge(output_root / "knowledge" / "weld")
    normal_reference = source_root / "reference" / "pdf_page4_normal.png"
    manifest = {
        "schema": "glls_weld_dataset_v2",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": int(seed),
        "data_policy": {
            "raw_archive": "45 independent unlabeled images; do not assume normal or use as one-shot support",
            "annotation_archive": (
                "50 independent images with LabelMe polygons; station-2 review labels are mapped from "
                "the official inspection rules: dx=defect and ljtq=allowed connection context"
            ),
            "archive_relationship": "no overlapping image filenames; the two archives are not image/annotation pairs",
            "dataset_split": "none supplied by the data owner",
            "metric_policy": "single-sample reference labels only; no benchmark metric without an official split",
        },
        "annotation_policy": {
            "region_labels": sorted(region_labels),
            "connection_context_labels": sorted(context_labels),
            "dx_rule": "compact defect/spatter region; map the reviewed image to NG",
            "ljtq_rule": "connection protrusion context; detect and filter it, but do not treat it as NG alone at station 2",
        },
        "counts": {
            "raw_unlabeled_images": len(station1_rows),
            "annotated_images": len(station2_rows),
            "annotated_polygons": int(sum(label_counts.values())),
            "shape_labels": dict(sorted(label_counts.items())),
        },
        "station1": station1_rows,
        "station2": station2_rows,
        "review_sets": {
            "unlabeled_ids": [row["id"] for row in station1_rows],
            "annotated_ids": [row["id"] for row in station2_rows],
        },
        "normal_reference": {
            "path": str(normal_reference.resolve()),
            "source_document": "焊缝缺陷检测升级.pdf",
            "page": 4,
            "embedded_image": "image_1.png",
            "caption": "正常图像",
            "trusted_for_one_shot": bool(normal_reference.exists()),
        },
        "knowledge": knowledge_paths,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_weld_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _extract_station1(raw_zip: Path, *, output_root: Path, image_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with zipfile.ZipFile(raw_zip) as archive:
        image_names = sorted(
            name for name in archive.namelist() if Path(name).suffix.lower() in IMAGE_SUFFIXES
        )
        for name in image_names:
            stem = Path(name).stem
            output_path = image_dir / f"{stem}.png"
            _save_zip_image(archive, name, output_path)
            with Image.open(output_path) as image:
                width, height = image.size
            rows.append(
                {
                    "id": stem,
                    "station": 1,
                    "image_path": _relative(output_root, output_path),
                    "label": None,
                    "quality_status": "unlabeled",
                    "defect_type": "unknown",
                    "width": width,
                    "height": height,
                    "source_member": name,
                }
            )
    return rows


def _extract_station2(
    annotation_zip: Path,
    *,
    output_root: Path,
    image_dir: Path,
    region_mask_dir: Path,
    context_mask_dir: Path,
    compact_annotation_dir: Path,
    region_labels: set[str],
    context_labels: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = []
    label_counts: Counter[str] = Counter()
    with zipfile.ZipFile(annotation_zip) as archive:
        json_names = sorted(name for name in archive.namelist() if name.lower().endswith(".json"))
        member_names = set(archive.namelist())
        for json_name in json_names:
            annotation = json.loads(archive.read(json_name))
            image_member = str(annotation.get("imagePath") or "")
            if image_member not in member_names:
                image_member = _find_image_member(member_names, Path(json_name).stem)
            if not image_member:
                raise FileNotFoundError(f"No image member for {json_name}")
            stem = Path(json_name).stem
            image_path = image_dir / f"{stem}.png"
            _save_zip_image(archive, image_member, image_path)
            with Image.open(image_path) as image:
                width, height = image.size

            shapes = list(annotation.get("shapes") or [])
            labels = [str(shape.get("label") or "").strip() for shape in shapes]
            label_counts.update(labels)
            annotation_reference = weld_annotation_reference({"shape_labels": labels})
            region_mask = _polygon_mask(width, height, shapes, included_labels=region_labels)
            context_mask = _polygon_mask(width, height, shapes, included_labels=context_labels)
            region_mask_path = region_mask_dir / f"{stem}.png"
            context_mask_path = context_mask_dir / f"{stem}.png"
            region_mask.save(region_mask_path)
            context_mask.save(context_mask_path)

            compact = {
                "version": annotation.get("version"),
                "flags": annotation.get("flags") or {},
                "shapes": shapes,
                "imagePath": image_path.name,
                "imageHeight": height,
                "imageWidth": width,
            }
            compact_path = compact_annotation_dir / f"{stem}.json"
            compact_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(
                {
                    "id": stem,
                    "station": 2,
                    "image_path": _relative(output_root, image_path),
                    "mask_path": _relative(output_root, region_mask_path),
                    "region_mask_path": _relative(output_root, region_mask_path),
                    "context_mask_path": _relative(output_root, context_mask_path),
                    "annotation_path": _relative(output_root, compact_path),
                    "label": annotation_reference["binary_label"],
                    "reference_ground_truth": annotation_reference["ground_truth"],
                    "quality_status": annotation_reference["role"],
                    "defect_type": (
                        "small_dot_weld_defect_or_spatter"
                        if annotation_reference["ground_truth"] == "NG"
                        else "allowed_connection_protrusion_context"
                        if annotation_reference["ground_truth"] == "OK"
                        else "unmapped_region_annotation"
                    ),
                    "shape_labels": labels,
                    "shape_count": len(shapes),
                    "has_dx_region": bool(set(labels) & region_labels),
                    "has_ljtq_context": bool(set(labels) & context_labels),
                    "width": width,
                    "height": height,
                    "source_member": image_member,
                }
            )
    return rows, label_counts


def _polygon_mask(
    width: int,
    height: int,
    shapes: list[dict[str, Any]],
    *,
    included_labels: set[str],
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        label = str(shape.get("label") or "").strip()
        if label not in included_labels:
            continue
        points = [tuple(float(value) for value in point) for point in shape.get("points") or []]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return mask


def _save_zip_image(archive: zipfile.ZipFile, member: str, output_path: Path) -> None:
    if output_path.exists():
        return
    with Image.open(io.BytesIO(archive.read(member))) as image:
        image.convert("RGB").save(output_path, format="PNG", optimize=False)


def _find_image_member(member_names: set[str], stem: str) -> str:
    for member in sorted(member_names):
        path = Path(member)
        if path.stem == stem and path.suffix.lower() in IMAGE_SUFFIXES:
            return member
    return ""


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))
