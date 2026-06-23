#!/usr/bin/env python3
"""Prepare local MPDD, DTD-Synthetic, and DAGM metadata for binary AD runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NORMAL_DIR_NAMES = {"good", "normal", "norm", "ok", "negative", "neg", "nondefective"}
LABEL_DIR_NAMES = {"ground_truth", "GroundTruth", "Label", "label", "labels", "Masks", "masks"}

DATASET_REFERENCES: dict[str, dict[str, str]] = {
    "mpdd": {
        "display_name": "MPDD",
        "local_root_name": "MPDD",
        "citation_key": "jezek2021deep",
        "title": "Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions",
        "authors": "Stepan Jezek; Martin Jonak; Radim Burget; Pavel Dvorak; Milos Skotak",
        "year": "2021",
        "doi": "10.1109/ICUMT54235.2021.9631567",
        "source_url": "https://github.com/stepanje/MPDD",
        "download_note": "Official GitHub points to an interactive SharePoint archive; this local copy uses the Kaggle mirror lephonghao/metal-parts-defect-detection-dataset-mpdd.",
    },
    "dtd": {
        "display_name": "DTD-Synthetic",
        "local_root_name": "DTD",
        "citation_key": "aota2023zero",
        "title": "Zero-Shot Versus Many-Shot: Unsupervised Texture Anomaly Detection",
        "authors": "Toshimichi Aota; Lloyd Teh Tzer Tong; Takayuki Okatani",
        "year": "2023",
        "doi": "10.1109/WACV56688.2023.00552",
        "source_url": "https://www.vision.is.tohoku.ac.jp/?p=383&lang=en",
        "download_note": "DTD in this evaluator is DTD-Synthetic, not the raw Oxford DTD texture dataset.",
    },
    "dagm": {
        "display_name": "DAGM 2007 Competition Dataset",
        "local_root_name": "DAGM_KaggleUpload",
        "citation_key": "matthias2007weakly",
        "title": "Weakly Supervised Learning for Industrial Optical Inspection",
        "authors": "Matthias Wieler; Tobias Hahn; Fred A. Hamprecht",
        "year": "2007",
        "doi": "10.5281/zenodo.8086136",
        "source_url": "https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection",
        "download_note": "Local copy uses the Kaggle mirror of the DAGM 2007 optical inspection dataset.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="", help="Defaults to GLLS_DATA_ROOT/GLLS_DATA_ROOT, then ~/data/GLLS or legacy ~/data/glls.")
    parser.add_argument("--mpdd-root", default="", help="MPDD root, default <data-root>/datasets/MPDD.")
    parser.add_argument("--dtd-root", default="", help="DTD-Synthetic root, default <data-root>/datasets/DTD.")
    parser.add_argument("--dagm-root", default="", help="DAGM root, default <data-root>/datasets/DAGM_KaggleUpload.")
    parser.add_argument("--manifest", default="", help="Output manifest path, default <data-root>/datasets/binary_ad_dataset_manifest.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing meta.json or manifest files.")
    parser.add_argument("--quiet", action="store_true", help="Write files without printing the manifest JSON.")
    args = parser.parse_args()

    data_root = Path(args.data_root or _default_data_root()).expanduser()
    roots = {
        "mpdd": Path(args.mpdd_root or _env_or_default("GLLS_MPDD_ROOT", "GLLS_MPDD_ROOT", data_root / "datasets" / "MPDD")).expanduser(),
        "dtd": Path(args.dtd_root or _env_or_default("GLLS_DTD_ROOT", "GLLS_DTD_ROOT", data_root / "datasets" / "DTD")).expanduser(),
        "dagm": Path(args.dagm_root or _env_or_default("GLLS_DAGM_ROOT", "GLLS_DAGM_ROOT", data_root / "datasets" / "DAGM_KaggleUpload")).expanduser(),
    }
    manifest_path = Path(args.manifest or data_root / "datasets" / "binary_ad_dataset_manifest.json").expanduser()

    summaries: dict[str, Any] = {}
    for dataset, root in roots.items():
        meta = _build_dagm_meta(root) if dataset == "dagm" else _build_mvtec_style_meta(root)
        summaries[dataset] = _summarize_meta(dataset, root, meta)
        if not args.dry_run and root.exists():
            _write_json(root / "meta.json", meta)

    manifest = {
        "schema": "glls_binary_ad_dataset_manifest_v1",
        "data_root": str(data_root),
        "datasets": {
            dataset: {
                **DATASET_REFERENCES[dataset],
                "root": str(roots[dataset]),
                "root_exists": roots[dataset].exists(),
                "meta_json": str(roots[dataset] / "meta.json"),
                "summary": summaries[dataset],
            }
            for dataset in ("mpdd", "dtd", "dagm")
        },
    }
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, manifest)
    if not args.quiet:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _default_data_root() -> str:
    env_value = os.environ.get("GLLS_DATA_ROOT") or os.environ.get("GLLS_DATA_ROOT")
    if env_value:
        return env_value
    default = Path.home() / "data" / "GLLS"
    legacy = Path.home() / "data" / "glls"
    if not default.exists() and legacy.exists():
        return str(legacy)
    return str(default)


def _env_or_default(primary: str, legacy: str, default: Path) -> str:
    return os.environ.get(primary) or os.environ.get(legacy) or str(default)


def _build_mvtec_style_meta(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    meta: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "test": {}}
    for category_dir in _category_dirs(root):
        category = category_dir.name
        train_root = category_dir / "train"
        test_root = category_dir / "test"
        meta["train"][category] = [
            _meta_item(root, category, path, label=0, defect_type="good")
            for path in _normal_train_images(train_root)
        ]
        meta["test"][category] = [
            item
            for defect_dir in sorted(test_root.iterdir() if test_root.exists() else [])
            if defect_dir.is_dir() and not _is_label_dir(defect_dir.name)
            for item in _test_items_from_defect_dir(root, category, defect_dir)
        ]
    return meta


def _build_dagm_meta(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    meta: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "test": {}}
    for category_dir in _category_dirs(root):
        category = category_dir.name
        train_items = _dagm_split_items(root, category, category_dir / "Train")
        test_items = _dagm_split_items(root, category, category_dir / "Test")
        meta["train"][category] = [item for item in train_items if int(item["anomaly"]) == 0]
        meta["test"][category] = test_items
    return meta


def _category_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    rows = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "train").is_dir() or (child / "Train").is_dir() or (child / "test").is_dir() or (child / "Test").is_dir():
            rows.append(child)
    return rows


def _normal_train_images(train_root: Path) -> list[Path]:
    if not train_root.exists():
        return []
    normal_dirs = [child for child in sorted(train_root.iterdir()) if child.is_dir() and _is_normal_name(child.name)]
    if normal_dirs:
        return [path for normal_dir in normal_dirs for path in _image_files(normal_dir)]
    return _direct_image_files(train_root)


def _test_items_from_defect_dir(root: Path, category: str, defect_dir: Path) -> list[dict[str, Any]]:
    label = 0 if _is_normal_name(defect_dir.name) else 1
    defect_type = "good" if label == 0 else defect_dir.name
    category_root = root / category
    mask_root = category_root / "ground_truth" / defect_dir.name
    rows = []
    for image_path in _image_files(defect_dir):
        item = _meta_item(root, category, image_path, label=label, defect_type=defect_type)
        if label:
            mask_path = _find_mask(mask_root, image_path)
            if mask_path:
                item["mask_path"] = _relative(root, mask_path)
        rows.append(item)
    return rows


def _dagm_split_items(root: Path, category: str, split_root: Path) -> list[dict[str, Any]]:
    if not split_root.exists():
        return []
    mask_index = _mask_index(split_root / "Label")
    rows = []
    for image_path in _direct_image_files(split_root):
        mask_path = mask_index.get(image_path.stem)
        label = int(mask_path is not None and _mask_has_foreground(mask_path))
        item = _meta_item(root, category, image_path, label=label, defect_type="defect" if label else "good")
        if mask_path is not None:
            item["mask_path"] = _relative(root, mask_path)
        rows.append(item)
    return rows


def _meta_item(root: Path, category: str, image_path: Path, *, label: int, defect_type: str) -> dict[str, Any]:
    return {
        "img_path": _relative(root, image_path),
        "cls_name": category,
        "specie_name": defect_type,
        "anomaly": int(label),
    }


def _summarize_meta(dataset: str, root: Path, meta: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    categories = []
    for category in sorted(set(meta.get("train", {})) | set(meta.get("test", {}))):
        train_rows = meta.get("train", {}).get(category, [])
        test_rows = meta.get("test", {}).get(category, [])
        categories.append({
            "category": category,
            "train_normal": sum(1 for row in train_rows if int(row.get("anomaly", 0)) == 0),
            "test_total": len(test_rows),
            "test_normal": sum(1 for row in test_rows if int(row.get("anomaly", 0)) == 0),
            "test_anomaly": sum(1 for row in test_rows if int(row.get("anomaly", 0)) == 1),
            "defect_types": sorted({str(row.get("specie_name") or "") for row in test_rows}),
        })
    return {
        "dataset": dataset,
        "root": str(root),
        "root_exists": root.exists(),
        "train_normal_total": sum(row["train_normal"] for row in categories),
        "test_total": sum(row["test_total"] for row in categories),
        "categories": categories,
    }


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and _is_image(path))


def _direct_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and _is_image(path))


def _mask_index(label_root: Path) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for mask_path in _image_files(label_root):
        stem = mask_path.stem
        for suffix in ("_label", "_mask"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        rows.setdefault(stem, mask_path)
    return rows


def _find_mask(mask_root: Path, image_path: Path) -> Path | None:
    if not mask_root.exists():
        return None
    for candidate in _image_files(mask_root):
        if candidate.stem == image_path.stem or candidate.stem.startswith(f"{image_path.stem}_"):
            return candidate
    return None


def _mask_has_foreground(mask_path: Path) -> bool:
    try:
        with Image.open(mask_path) as mask:
            return bool(np.any(np.asarray(mask.convert("L")) > 0))
    except Exception:
        return False


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_label_dir(name: str) -> bool:
    return name in LABEL_DIR_NAMES


def _is_normal_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in NORMAL_DIR_NAMES


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
