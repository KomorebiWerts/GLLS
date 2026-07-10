from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from glls.weld.data import load_weld_manifest
from glls.weld.scoring import WeldAdaptCLIPScorer, WeldTileConfig, weld_heatmap_shape_policies
from glls.weld.pvla_prior import (
    WeldPVLANormalPrior,
    add_pvla_score_policies,
    load_pvla_feature_cache,
    write_pvla_feature_cache,
)


WELD_NORMAL_STATES = [
    "{}",
    "complete continuous {}",
    "uniform annular {}",
    "{} without a missing segment",
    "{} without spatter",
]
WELD_ANOMALY_STATES = [
    "incomplete {}",
    "{} with a missing weld gap",
    "{} with weld spatter",
    "uneven damaged {}",
    "{} with a dark contamination defect",
]
SPATTER_NORMAL_STATES = [
    "{}",
    "smooth continuous {}",
    "{} without isolated dots",
    "{} without metal deposits",
    "clean uniform {}",
]
SPATTER_ANOMALY_STATES = [
    "{} with a small weld spatter dot",
    "{} with an isolated round metal deposit",
    "{} with a compact bead defect",
    "{} with pitting or a dark dot defect",
    "damaged {} with localized spatter",
]


def evaluate_weld_adaptclip(
    manifest_path: Path,
    *,
    localizer: Any,
    output_dir: Path,
    shot: int,
    tile_size: int = 1024,
    overlap: float = 0.25,
    heatmap_max_side: int = 768,
    target_recall: float = 0.98,
    threshold_safety_margin: float = 0.02,
    resume: bool = True,
    max_samples: int = 0,
    sam_engine: Any | None = None,
    online_sam_limit: int = 16,
    support_id: str = "",
    prompt_profile: str = "weld",
    include_full_image: bool = True,
    experiment_name: str = "",
    use_pvla_prior: bool = True,
    pvla_graph_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_weld_manifest(manifest_path)
    root = Path(manifest["output_root"])
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = {row["id"]: row for row in manifest["station2"]}
    if "split" not in manifest:
        return _evaluate_unlabeled_review(
            manifest_path=manifest_path,
            manifest=manifest,
            localizer=localizer,
            output_dir=output_dir,
            shot=shot,
            tile_size=tile_size,
            overlap=overlap,
            heatmap_max_side=heatmap_max_side,
            resume=resume,
            max_samples=max_samples,
            prompt_profile=prompt_profile,
            include_full_image=include_full_image,
            use_pvla_prior=use_pvla_prior,
            pvla_graph_path=pvla_graph_path,
            experiment_name=experiment_name,
        )
    split = manifest["split"]
    validation_ids = list(split["validation_ids"])
    test_ids = list(split["test_ids"])
    if max_samples > 0:
        validation_ids = validation_ids[:max_samples]
        test_ids = test_ids[:max_samples]

    configure_text = getattr(localizer, "configure_text_prompts", None)
    if callable(configure_text) and prompt_profile == "weld":
        configure_text(
            "gear weld seam",
            normal_states=WELD_NORMAL_STATES,
            anomaly_states=WELD_ANOMALY_STATES,
        )
    elif callable(configure_text) and prompt_profile == "spatter":
        configure_text(
            "gear weld seam",
            normal_states=SPATTER_NORMAL_STATES,
            anomaly_states=SPATTER_ANOMALY_STATES,
        )
    elif prompt_profile != "generic":
        raise ValueError(f"Unsupported weld prompt profile: {prompt_profile}")

    support_pool_ids = list(split.get("support_pool_ids") or [])
    if int(shot) > 0 and support_id:
        if support_id not in support_pool_ids:
            raise ValueError(f"support_id must be one of {support_pool_ids}, got {support_id}")
        support_ids = [support_id]
    else:
        support_ids = support_pool_ids[:1] if int(shot) > 0 else []
    configure_support = getattr(localizer, "configure_support", None)
    if callable(configure_support):
        configure_support(
            "gear_weld",
            [str(root / rows_by_id[sample_id]["image_path"]) for sample_id in support_ids],
        )

    scorer = WeldAdaptCLIPScorer(
        localizer,
        WeldTileConfig(
            tile_size=tile_size,
            overlap=overlap,
            heatmap_max_side=heatmap_max_side,
            include_full_image=include_full_image,
        ),
    )
    pvla_prior = None
    pvla_features_path = output_dir / "pvla_features.json"
    pvla_features = load_pvla_feature_cache(pvla_features_path) if resume else {}
    if use_pvla_prior:
        graph_path = Path(pvla_graph_path) if pvla_graph_path else root / "assets" / "graph" / "gear_weld_graph.pkl"
        pvla_prior = WeldPVLANormalPrior(
            localizer,
            graph_path=graph_path,
            support_paths=[root / rows_by_id[sample_id]["image_path"] for sample_id in support_ids],
        )
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scored_rows = []
    for sample_id in validation_ids + test_ids:
        sample = rows_by_id[sample_id]
        scored = _score_or_load(
            scorer,
            root=root,
            sample=sample,
            cache_dir=cache_dir,
            resume=resume,
        )
        if pvla_prior is not None:
            prior_row = pvla_features.get(sample_id)
            if not isinstance(prior_row, dict):
                prior_row = pvla_prior.score(
                    root / sample["image_path"],
                    Path(scored["heatmap_path"]),
                )
                pvla_features[sample_id] = prior_row
                write_pvla_feature_cache(pvla_features_path, pvla_features)
            scored["pvla_prior"] = prior_row
            add_pvla_score_policies(scored["scores"], prior_row)
        scored_rows.append(scored)

    validation_rows = [row for row in scored_rows if row["id"] in set(validation_ids)]
    test_rows = [row for row in scored_rows if row["id"] in set(test_ids)]
    selection = select_score_policy_and_threshold(
        validation_rows,
        target_recall=target_recall,
        safety_margin_ratio=threshold_safety_margin,
    )
    policy = selection["policy"]
    threshold = float(selection["threshold"])
    predictions = []
    for row in test_rows:
        score = float(row["scores"][policy])
        prediction = int(score > threshold)
        predictions.append(
            {
                **{key: row[key] for key in ("id", "image_path", "mask_path", "label", "defect_type", "shape_labels")},
                "score": score,
                "threshold": threshold,
                "prediction": prediction,
                "correct": bool(prediction == int(row["label"])),
                "policy": policy,
                "top_tiles": row["top_tiles"],
                "heatmap_path": row["heatmap_path"],
            }
        )
    image_metrics = binary_metrics(
        [int(row["label"]) for row in predictions],
        [int(row["prediction"]) for row in predictions],
        [float(row["score"]) for row in predictions],
    )
    pixel_selection = select_pixel_threshold(validation_rows, root=root)
    pixel_metrics = evaluate_pixel_metrics(test_rows, root=root, threshold=pixel_selection["threshold"])
    _write_prediction_visuals(predictions, root=root, output_dir=output_dir / "visuals")
    sam_rows = _write_online_sam_cutouts(
        predictions,
        root=root,
        output_dir=output_dir / "sam3_cutouts",
        sam_engine=sam_engine,
        limit=online_sam_limit,
    )

    result = {
        "schema": "glls_weld_adaptclip_evaluation_v1",
        "method": "GLLS weld text knowledge + tiled AdaptCLIP + SAM3 refinement",
        "experiment_name": str(experiment_name),
        "manifest_path": str(manifest_path),
        "shot": int(shot),
        "support_ids": support_ids,
        "prompt_profile": prompt_profile,
        "pvla_prior": {
            "enabled": bool(pvla_prior is not None),
            "graph_path": str(pvla_prior.graph_path) if pvla_prior is not None else "",
            "reference_source_count": len(pvla_prior.reference_sources) if pvla_prior is not None else 0,
            "reference_patch_count": len(pvla_prior.reference_embeddings) if pvla_prior is not None else 0,
            "feature_cache_path": str(pvla_features_path) if pvla_prior is not None else "",
        },
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "tile_config": {
            "tile_size": tile_size,
            "overlap": overlap,
            "heatmap_max_side": heatmap_max_side,
            "include_full_image": bool(include_full_image),
        },
        "threshold_selection": selection,
        "pixel_threshold_selection": pixel_selection,
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "sam3_refinements": sam_rows,
        "predictions": predictions,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def _evaluate_unlabeled_review(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    localizer: Any,
    output_dir: Path,
    shot: int,
    tile_size: int,
    overlap: float,
    heatmap_max_side: int,
    resume: bool,
    max_samples: int,
    prompt_profile: str,
    include_full_image: bool,
    use_pvla_prior: bool,
    pvla_graph_path: Path | None,
    experiment_name: str,
) -> dict[str, Any]:
    root = Path(manifest["output_root"])
    _configure_weld_prompts(localizer, prompt_profile)
    normal_reference = Path((manifest.get("normal_reference") or {}).get("path") or "")
    support_paths = [normal_reference] if int(shot) > 0 and normal_reference.exists() else []
    configure_support = getattr(localizer, "configure_support", None)
    if callable(configure_support):
        configure_support("gear_weld", [str(path) for path in support_paths])
    scorer = WeldAdaptCLIPScorer(
        localizer,
        WeldTileConfig(
            tile_size=tile_size,
            overlap=overlap,
            heatmap_max_side=heatmap_max_side,
            include_full_image=include_full_image,
        ),
    )
    graph_path = Path(pvla_graph_path) if pvla_graph_path else root / "assets" / "graph" / "gear_weld_graph.pkl"
    pvla_prior = (
        WeldPVLANormalPrior(localizer, graph_path=graph_path, support_paths=support_paths)
        if use_pvla_prior
        else None
    )
    rows = list(manifest.get("station2") or [])
    if max_samples > 0:
        rows = rows[:max_samples]
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for sample in rows:
        scored = _score_or_load(scorer, root=root, sample=sample, cache_dir=cache_dir, resume=resume)
        if pvla_prior is not None:
            prior = pvla_prior.score(root / sample["image_path"], Path(scored["heatmap_path"]))
            scored["pvla_prior"] = prior
            add_pvla_score_policies(scored["scores"], prior)
        compact = float(scored["scores"].get("compact_hotspot_sigma32_max", 0.0))
        scored["qualitative_local_score"] = compact
        review_rows.append(scored)
    review_rows.sort(key=lambda row: float(row["qualitative_local_score"]), reverse=True)
    result = {
        "schema": "glls_weld_qualitative_review_v1",
        "method": "GLLS global logic + tiled AdaptCLIP + one-image PVLA prior",
        "experiment_name": str(experiment_name),
        "manifest_path": str(manifest_path),
        "shot": int(shot),
        "support_paths": [str(path) for path in support_paths],
        "data_status": "unlabeled/region-annotated; no official image-level metric",
        "threshold_selection": None,
        "image_metrics": None,
        "pixel_metrics": None,
        "review_rows": review_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _configure_weld_prompts(localizer: Any, prompt_profile: str) -> None:
    configure_text = getattr(localizer, "configure_text_prompts", None)
    if not callable(configure_text):
        return
    if prompt_profile == "weld":
        configure_text("gear weld seam", normal_states=WELD_NORMAL_STATES, anomaly_states=WELD_ANOMALY_STATES)
    elif prompt_profile == "spatter":
        configure_text(
            "gear weld seam",
            normal_states=SPATTER_NORMAL_STATES,
            anomaly_states=SPATTER_ANOMALY_STATES,
        )
    elif prompt_profile != "generic":
        raise ValueError(f"Unsupported weld prompt profile: {prompt_profile}")


def select_score_policy_and_threshold(
    rows: list[dict[str, Any]],
    *,
    target_recall: float,
    safety_margin_ratio: float = 0.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Validation rows are required for threshold selection.")
    policies = sorted((rows[0].get("scores") or {}).keys())
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    candidates = []
    for policy in policies:
        scores = np.asarray([float(row["scores"][policy]) for row in rows], dtype=np.float64)
        for threshold in _threshold_candidates(scores):
            predictions = (scores > threshold).astype(np.int64)
            metrics = binary_metrics(labels.tolist(), predictions.tolist(), scores.tolist())
            meets_recall = float(metrics["recall"]) >= float(target_recall)
            rank = (
                int(meets_recall),
                float(metrics["recall"]),
                float(metrics["precision"]),
                float(metrics["balanced_accuracy"]),
                float(metrics["f1"]),
            )
            candidates.append((rank, policy, float(threshold), metrics))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, policy, raw_threshold, raw_metrics = candidates[0]
    safety_margin_ratio = max(0.0, float(safety_margin_ratio))
    threshold = float(raw_threshold - abs(raw_threshold) * safety_margin_ratio)
    scores = [float(row["scores"][policy]) for row in rows]
    predictions = [int(score > threshold) for score in scores]
    metrics = binary_metrics(labels.tolist(), predictions, scores)
    return {
        "policy": policy,
        "threshold": threshold,
        "raw_threshold": float(raw_threshold),
        "safety_margin_ratio": safety_margin_ratio,
        "target_recall": float(target_recall),
        "validation_metrics": metrics,
        "raw_validation_metrics": raw_metrics,
        "policy_candidates": policies,
        "selection_rule": "meet target recall, then maximize recall, precision, balanced accuracy, and F1",
    }


def binary_metrics(labels: list[int], predictions: list[int], scores: list[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    balanced = balanced_accuracy_score(y, p) if len(np.unique(y)) > 1 else accuracy
    auroc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else None
    average_precision = average_precision_score(y, s) if len(np.unique(y)) > 1 else None
    return {
        "total": int(len(y)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc) if auroc is not None else None,
        "average_precision": float(average_precision) if average_precision is not None else None,
    }


def select_pixel_threshold(rows: list[dict[str, Any]], *, root: Path) -> dict[str, Any]:
    values, labels = _pixel_arrays(rows, root=root)
    if values.size == 0 or not np.any(labels):
        return {"threshold": 0.5, "validation_f1": 0.0, "validation_iou": 0.0}
    quantiles = np.linspace(0.80, 0.9999, 160)
    thresholds = np.unique(np.quantile(values, quantiles))
    best = (0.0, 0.0, float(thresholds[0]))
    for threshold in thresholds:
        prediction = values > threshold
        tp = int(np.sum(prediction & labels))
        fp = int(np.sum(prediction & ~labels))
        fn = int(np.sum(~prediction & labels))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        iou = tp / max(1, tp + fp + fn)
        best = max(best, (float(f1), float(iou), float(threshold)))
    return {"threshold": best[2], "validation_f1": best[0], "validation_iou": best[1]}


def evaluate_pixel_metrics(rows: list[dict[str, Any]], *, root: Path, threshold: float) -> dict[str, Any]:
    values, labels = _pixel_arrays(rows, root=root)
    if values.size == 0:
        return {"pixel_count": 0}
    prediction = values > float(threshold)
    tp = int(np.sum(prediction & labels))
    fp = int(np.sum(prediction & ~labels))
    fn = int(np.sum(~prediction & labels))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    has_both = len(np.unique(labels)) > 1
    return {
        "pixel_count": int(values.size),
        "positive_pixel_count": int(labels.sum()),
        "threshold": float(threshold),
        "f1": float(f1),
        "iou": float(iou),
        "auroc": float(roc_auc_score(labels, values)) if has_both else None,
        "average_precision": float(average_precision_score(labels, values)) if has_both else None,
    }


def _score_or_load(
    scorer: WeldAdaptCLIPScorer,
    *,
    root: Path,
    sample: dict[str, Any],
    cache_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    json_path = cache_dir / f"{sample['id']}.json"
    heatmap_path = cache_dir / f"{sample['id']}.npz"
    if resume and json_path.exists() and heatmap_path.exists():
        row = json.loads(json_path.read_text(encoding="utf-8"))
        heatmap = np.load(heatmap_path)["heatmap"].astype(np.float32)
        shape_policies = weld_heatmap_shape_policies(heatmap)
        if any(row.get("scores", {}).get(key) != value for key, value in shape_policies.items()):
            row.setdefault("scores", {}).update(shape_policies)
            json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        return row
    scored = scorer.score_path(root / sample["image_path"])
    heatmap = np.asarray(scored.pop("heatmap"), dtype=np.float32)
    np.savez_compressed(heatmap_path, heatmap=heatmap)
    row = {
        **sample,
        "scores": scored["scores"],
        "full_image_score": scored["full_image_score"],
        "full_heatmap_tail_score": scored["full_heatmap_tail_score"],
        "tile_config": scored["tile_config"],
        "top_tiles": scored["top_tiles"],
        "heatmap_path": str(heatmap_path),
        "coarse_heatmap_size": scored["coarse_heatmap_size"],
    }
    row["scores"].update(weld_heatmap_shape_policies(heatmap))
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if unique.size == 1:
        return np.asarray([unique[0] - 1e-8, unique[0], unique[0] + 1e-8])
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate(([unique[0] - 1e-8], midpoints, [unique[-1] + 1e-8]))


def _pixel_arrays(rows: list[dict[str, Any]], *, root: Path) -> tuple[np.ndarray, np.ndarray]:
    values = []
    labels = []
    for row in rows:
        heatmap = np.load(row["heatmap_path"])["heatmap"].astype(np.float32)
        with Image.open(root / row["mask_path"]) as image:
            mask = np.asarray(image.convert("L")) > 0
        resized = cv2.resize(mask.astype(np.uint8), (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        values.append(heatmap.reshape(-1))
        labels.append(resized.reshape(-1))
    if not values:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=bool)
    return np.concatenate(values), np.concatenate(labels)


def _write_prediction_visuals(predictions: list[dict[str, Any]], *, root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in predictions:
        with Image.open(root / row["image_path"]) as image:
            rgb = image.convert("RGB")
            draw = ImageDraw.Draw(rgb)
            for rank, tile in enumerate(row.get("top_tiles") or []):
                x0, y0, x1, y1 = tile["bbox_xyxy"]
                color = (255, 0, 0) if rank == 0 else (255, 180, 0)
                width = 16 if rank == 0 else 8
                draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
            rgb.thumbnail((1600, 1200))
            rgb.save(output_dir / f"{row['id']}_pred{row['prediction']}_gt{row['label']}.jpg", quality=90)


def _write_online_sam_cutouts(
    predictions: list[dict[str, Any]],
    *,
    root: Path,
    output_dir: Path,
    sam_engine: Any | None,
    limit: int,
) -> list[dict[str, Any]]:
    if sam_engine is None or limit <= 0:
        return []
    candidates = sorted(
        predictions,
        key=lambda row: (int(row["prediction"]), int(row["label"]), float(row["score"])),
        reverse=True,
    )[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.png"):
        stale_path.unlink()
    audits = []
    for row in candidates:
        image_path = root / row["image_path"]
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = _hotspot_boxes(row, max_boxes=2)
        if not boxes:
            for tile in (row.get("top_tiles") or [])[:2]:
                x0, y0, x1, y1 = tile["bbox_xyxy"]
                boxes.append(
                    [
                        (x0 + x1) / (2 * width),
                        (y0 + y1) / (2 * height),
                        min(0.18, (x1 - x0) / width),
                        min(0.24, (y1 - y0) / height),
                    ]
                )
        try:
            sam_engine.set_image(str(image_path))
            mask, score = sam_engine.predict_mask_with_boxes(boxes, threshold=0.2)
            output_path = output_dir / f"{row['id']}.png"
            if mask is not None:
                sam_engine.save_masked_cutout(mask, str(output_path), padding=24)
            audits.append(
                {
                    "id": row["id"],
                    "path": str(output_path) if output_path.exists() else "",
                    "boxes": boxes,
                    "sam3_score": float(score),
                    "status": "saved" if output_path.exists() else "empty_mask",
                }
            )
        except Exception as exc:
            audits.append({"id": row["id"], "boxes": boxes, "status": "error", "error": str(exc)})
    return audits


def _hotspot_boxes(row: dict[str, Any], *, max_boxes: int) -> list[list[float]]:
    try:
        heatmap = np.load(row["heatmap_path"])["heatmap"].astype(np.float32)
    except Exception:
        return []
    policy = str(row.get("policy") or "")
    sigma = 32 if "sigma32" in policy else 16
    response = np.maximum(heatmap - cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma), 0.0)
    boxes = []
    radius = max(8, int(round(min(response.shape) * 0.10)))
    for _ in range(max(1, max_boxes)):
        flat_index = int(np.argmax(response))
        peak = float(response.flat[flat_index])
        if peak <= 0:
            break
        y, x = np.unravel_index(flat_index, response.shape)
        boxes.append(
            [
                float((x + 0.5) / response.shape[1]),
                float((y + 0.5) / response.shape[0]),
                0.12,
                0.18,
            ]
        )
        y0, y1 = max(0, y - radius), min(response.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(response.shape[1], x + radius + 1)
        response[y0:y1, x0:x1] = 0.0
    return boxes
