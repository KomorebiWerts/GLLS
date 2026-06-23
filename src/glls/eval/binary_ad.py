from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from glls.models.heatmap_regions import RegionProposal, adaptive_thresholds, detect_heatmap_regions
from glls.models.mcts_crop_selector import MCTSCropCandidateSelector


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NORMAL_DIR_NAMES = {"good", "normal", "norm", "ok", "negative", "neg", "nondefective"}
TRAIN_DIR_NAMES = ("train", "Train", "training", "Training")
TEST_DIR_NAMES = ("test", "Test", "testing", "Testing")
LABEL_DIR_NAMES = ("ground_truth", "GroundTruth", "Label", "label", "labels", "Masks", "masks")
BINARY_AD_DATASETS = ("mpdd", "dtd", "dagm")


@dataclass(frozen=True)
class BinaryADSample:
    dataset: str
    category: str
    split: str
    image_path: str
    label: int
    defect_type: str


@dataclass(frozen=True)
class BinaryADCalibration:
    threshold: float
    train_normal_count: int
    normal_score_median: float
    normal_score_mad: float
    normal_score_quantile: float
    normal_score_max: float
    score_policy: str
    threshold_policy: str
    calibration_shots: int


@dataclass(frozen=True)
class BinaryADResult:
    dataset: str
    category: str
    image_path: str
    defect_type: str
    label: int
    score: float
    threshold: float
    prediction: int
    correct: bool
    localizer_category: str
    score_policy: str
    trace: dict[str, Any]


def iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_category_dirs(root: Path, category: str | None = None) -> list[Path]:
    if category:
        requested = root / category
        return [requested if requested.exists() else root]

    categories = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir():
            continue
        if _first_existing_dir(child, TRAIN_DIR_NAMES) or _first_existing_dir(child, TEST_DIR_NAMES):
            categories.append(child)
    if categories:
        return categories
    return [root] if root.exists() else []


def build_binary_ad_samples(
    *,
    dataset: str,
    root: Path,
    category: str | None = None,
) -> tuple[list[BinaryADSample], list[BinaryADSample]]:
    meta_train, meta_test = _collect_from_meta_json(dataset=dataset, root=root, category=category)
    if meta_train or meta_test:
        return meta_train, meta_test

    train_samples: list[BinaryADSample] = []
    test_samples: list[BinaryADSample] = []
    for category_dir in discover_category_dirs(root, category=category):
        category_name = category or category_dir.name
        train_samples.extend(_collect_train_normals(dataset, category_name, category_dir))
        test_samples.extend(_collect_test_samples(dataset, category_name, category_dir))
    return train_samples, test_samples


def image_score_from_heatmap(heatmap: np.ndarray, *, tail_fraction: float = 0.01) -> float:
    values = np.asarray(heatmap, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    tail_fraction = min(1.0, max(0.0001, float(tail_fraction)))
    k = max(1, int(math.ceil(float(finite.size) * tail_fraction)))
    if k >= finite.size:
        return float(np.mean(finite))
    top_values = np.partition(finite, -k)[-k:]
    return float(np.mean(top_values))


def calibrate_threshold(
    normal_scores: list[float],
    *,
    normal_quantile: float = 0.995,
    mad_scale: float = 6.0,
    score_policy: str = "top_1pct_mean",
    threshold_policy: str = "normal_robust",
    fixed_threshold: float | None = None,
    table_threshold: float | None = None,
    table_source: str = "",
    calibration_shots: int = -1,
) -> BinaryADCalibration:
    if threshold_policy in {"fixed", "table"}:
        threshold = table_threshold if threshold_policy == "table" else fixed_threshold
        if threshold is None:
            threshold = 0.5
        scores = np.asarray(normal_scores, dtype=np.float64) if normal_scores else np.asarray([], dtype=np.float64)
        return BinaryADCalibration(
            threshold=float(threshold),
            train_normal_count=int(scores.size),
            normal_score_median=float(np.median(scores)) if scores.size else 0.0,
            normal_score_mad=float(np.median(np.abs(scores - np.median(scores)))) if scores.size else 0.0,
            normal_score_quantile=float(np.quantile(scores, min(1.0, max(0.0, normal_quantile)))) if scores.size else 0.0,
            normal_score_max=float(np.max(scores)) if scores.size else 0.0,
            score_policy=score_policy,
            threshold_policy=(
                f"table({table_source or 'binary_ad_threshold_table'}:{threshold:g})"
                if threshold_policy == "table"
                else f"fixed({threshold:g})"
            ),
            calibration_shots=int(calibration_shots),
        )
    if not normal_scores:
        raise ValueError("At least one normal training score is required for binary AD calibration.")

    scores = np.asarray(normal_scores, dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust_sigma = 1.4826 * mad
    q = float(np.quantile(scores, min(1.0, max(0.0, normal_quantile))))
    robust = median + float(mad_scale) * robust_sigma
    max_score = float(np.max(scores))
    if threshold_policy == "normal_quantile":
        threshold = q
        threshold_policy_text = f"q{normal_quantile:g}"
    elif threshold_policy == "normal_median_mad":
        threshold = robust
        threshold_policy_text = f"median+{mad_scale:g}*1.4826*MAD"
    elif threshold_policy == "normal_robust":
        threshold = max(q, robust)
        threshold_policy_text = f"max(q{normal_quantile:g}, median+{mad_scale:g}*1.4826*MAD)"
    else:
        raise ValueError(f"Unsupported threshold_policy: {threshold_policy}")
    if len(scores) < 5:
        threshold = max(threshold, max_score)
        threshold_policy_text += "; small_n_floor=max_train_score"

    return BinaryADCalibration(
        threshold=float(threshold),
        train_normal_count=int(len(scores)),
        normal_score_median=median,
        normal_score_mad=float(mad),
        normal_score_quantile=q,
        normal_score_max=max_score,
        score_policy=score_policy,
        threshold_policy=threshold_policy_text,
        calibration_shots=int(calibration_shots),
    )


def run_binary_ad_evaluation(
    *,
    dataset: str,
    root: Path,
    localizer: Any,
    output_dir: Path,
    category: str | None = None,
    normal_quantile: float = 0.995,
    mad_scale: float = 6.0,
    score_tail_fraction: float = 0.01,
    binary_score_source: str = "auto",
    threshold_policy: str = "normal_robust",
    fixed_threshold: float | None = None,
    threshold_table: dict[str, Any] | None = None,
    localizer_name: str = "",
    threshold_shot: int | None = None,
    calibration_shots: int = -1,
    sam_engine: Any | None = None,
    max_region_proposals: int = 3,
    max_pvla_refs: int = 3,
    save_evidence_images: bool = True,
    offline_assets: dict[str, Any] | None = None,
    final_verifier: Callable[[BinaryADSample, dict[str, Any]], dict[str, Any]] | None = None,
    final_verifier_policy: str = "anomaly_or",
    max_test_samples_per_category: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    train_samples, test_samples = build_binary_ad_samples(dataset=dataset, root=root, category=category)
    if not train_samples:
        raise FileNotFoundError(f"No train normal images found under {root}")
    if not test_samples:
        raise FileNotFoundError(f"No labeled test images found under {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_by_category = _group_by_category(train_samples)
    test_by_category = _group_by_category(test_samples)
    category_summaries = []
    all_results: list[BinaryADResult] = []
    calibration_rows: list[dict[str, Any]] = []
    score_policy = f"top_{score_tail_fraction:g}_mean"
    evidence_dir = output_dir / "evidence"
    predictions_path = output_dir / "predictions.jsonl"
    existing_results = _load_existing_results(predictions_path) if resume else {}

    for category_name in sorted(test_by_category):
        train_rows = train_by_category.get(category_name, [])
        if not train_rows:
            continue
        _configure_localizer_category(localizer, category_name, train_rows)
        calibration_source_rows = _calibration_source_rows(train_rows, calibration_shots=calibration_shots)
        train_score_rows = []
        for sample in calibration_source_rows:
            score_info = _score_image(
                localizer,
                sample.image_path,
                tail_fraction=score_tail_fraction,
                binary_score_source=binary_score_source,
            )
            train_score_rows.append({
                "dataset": dataset,
                "category": category_name,
                "image_path": sample.image_path,
                "score": float(score_info["score"]),
                "localizer_category": str(score_info.get("localizer_category") or ""),
            })
        train_scores = [float(row["score"]) for row in train_score_rows]
        table_threshold, table_source = _lookup_threshold_table(
            threshold_table,
            dataset=dataset,
            category=category_name,
            localizer=localizer_name,
            shot=threshold_shot if threshold_shot is not None else calibration_shots,
        )
        calibration = calibrate_threshold(
            train_scores,
            normal_quantile=normal_quantile,
            mad_scale=mad_scale,
            score_policy=score_policy,
            threshold_policy=threshold_policy,
            fixed_threshold=fixed_threshold,
            table_threshold=table_threshold,
            table_source=table_source,
            calibration_shots=calibration_shots,
        )
        for row in train_score_rows:
            row["threshold"] = float(calibration.threshold)
            row["score_policy"] = score_policy
            row["threshold_policy"] = calibration.threshold_policy
        calibration_rows.extend(train_score_rows)
        pvla_graph = _build_pvla_normal_graph(
            dataset=dataset,
            category=category_name,
            root=root,
            calibration=calibration,
            train_score_rows=train_score_rows,
            output_dir=output_dir,
            max_refs=max_pvla_refs,
            save_evidence_images=save_evidence_images,
            evidence_dir=evidence_dir,
            offline_assets=_offline_category_assets(offline_assets, dataset=dataset, category=category_name),
        )
        results = []
        category_test_rows = test_by_category[category_name]
        if int(max_test_samples_per_category or 0) > 0:
            category_test_rows = category_test_rows[: int(max_test_samples_per_category)]
        for sample in category_test_rows:
            existing_result = existing_results.get(sample.image_path)
            if existing_result is not None:
                results.append(existing_result)
                continue
            score_info = _score_image(
                localizer,
                sample.image_path,
                tail_fraction=score_tail_fraction,
                binary_score_source=binary_score_source,
                include_heatmap=True,
            )
            score = score_info["score"]
            threshold_prediction = int(score > calibration.threshold)
            heatmap = score_info.get("heatmap")
            trace = _build_binary_method_trace(
                sample=sample,
                score_info=score_info,
                score_policy=score_policy,
                calibration=calibration,
                prediction=threshold_prediction,
                pvla_graph=pvla_graph,
                sam_engine=sam_engine,
                output_dir=output_dir,
                evidence_dir=evidence_dir,
                max_region_proposals=max_region_proposals,
                save_evidence_images=save_evidence_images,
                heatmap=heatmap,
            )
            prediction = threshold_prediction
            if final_verifier is not None:
                verifier_audit = final_verifier(sample, trace)
                verifier_prediction = _verified_prediction(verifier_audit, fallback=threshold_prediction)
                prediction = _fuse_final_prediction(
                    threshold_prediction=threshold_prediction,
                    verifier_prediction=verifier_prediction,
                    policy=final_verifier_policy,
                )
                _apply_final_verifier_trace(
                    trace,
                    verifier_audit=verifier_audit,
                    threshold_prediction=threshold_prediction,
                    verifier_prediction=verifier_prediction,
                    prediction=prediction,
                    policy=final_verifier_policy,
                )
            result = BinaryADResult(
                dataset=dataset,
                category=category_name,
                image_path=sample.image_path,
                defect_type=sample.defect_type,
                label=int(sample.label),
                score=float(score),
                threshold=float(calibration.threshold),
                prediction=prediction,
                correct=bool(prediction == int(sample.label)),
                localizer_category=str(score_info.get("localizer_category") or ""),
                score_policy=score_policy,
                trace=trace,
            )
            results.append(result)
            if resume:
                _append_jsonl(predictions_path, asdict(result))
        all_results.extend(results)
        category_summaries.append(_summarize_category(category_name, calibration, results))

    if not all_results:
        raise RuntimeError("No categories had both train normal images and labeled test images.")

    summary = {
        "dataset": dataset,
        "root": str(root),
        "category_filter": category or "",
        "method": "localizer_mcts_sam3_pvla_normal_calibrated_binary_ad",
        "method_framework": binary_method_framework(
            max_region_proposals=max_region_proposals,
            max_pvla_refs=max_pvla_refs,
            final_verifier_enabled=final_verifier is not None,
        ),
        "dataset_layout": _describe_dataset_layout(root),
        "score_tail_fraction": score_tail_fraction,
        "binary_score_source": binary_score_source,
        "normal_quantile": normal_quantile,
        "mad_scale": mad_scale,
        "threshold_policy_name": threshold_policy,
        "fixed_threshold": fixed_threshold,
        "threshold_table_schema": str((threshold_table or {}).get("schema") or ""),
        "localizer_name": localizer_name,
        "threshold_shot": threshold_shot,
        "calibration_shots": int(calibration_shots),
        "score_policy": score_policy,
        "threshold_policy": f"per-category {threshold_policy}",
        "sam3_enabled": sam_engine is not None,
        "pvla_enabled": True,
        "offline_assets_prepared": bool(offline_assets),
        "offline_assets_schema": str((offline_assets or {}).get("schema") or ""),
        "max_region_proposals": int(max_region_proposals),
        "max_pvla_refs": int(max_pvla_refs),
        "save_evidence_images": bool(save_evidence_images),
        "final_verifier_enabled": final_verifier is not None,
        "final_verifier_policy": final_verifier_policy if final_verifier is not None else "",
        "max_test_samples_per_category": int(max_test_samples_per_category or 0),
        "resume_enabled": bool(resume),
        "train_normal_total": len(train_samples),
        "total": len(all_results),
        "correct": sum(1 for row in all_results if row.correct),
        "accuracy": _accuracy(all_results),
        "categories": category_summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(predictions_path, [asdict(row) for row in all_results])
    _write_jsonl(output_dir / "calibration_scores.jsonl", calibration_rows)
    return summary


def run_binary_ad_pipeline(
    *,
    dataset_roots: dict[str, Path],
    output_dir: Path,
    localizer_factory: Callable[[str], Any],
    sam_engine_factory: Callable[[str], Any | None] | None = None,
    datasets: Iterable[str] = BINARY_AD_DATASETS,
    normal_quantile: float = 0.995,
    mad_scale: float = 6.0,
    score_tail_fraction: float = 0.01,
    binary_score_source: str = "auto",
    threshold_policy: str = "normal_robust",
    fixed_threshold: float | None = None,
    threshold_table: dict[str, Any] | None = None,
    localizer_name: str = "",
    threshold_shot: int | None = None,
    calibration_shots: int = -1,
    max_region_proposals: int = 3,
    max_pvla_refs: int = 3,
    save_evidence_images: bool = True,
    offline_assets: dict[str, Any] | None = None,
    final_verifier_factory: Callable[[str], Callable[[BinaryADSample, dict[str, Any]], dict[str, Any]] | None] | None = None,
    final_verifier_policy: str = "anomaly_or",
    max_test_samples_per_category: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in datasets:
        key = str(dataset).strip().lower()
        if key not in BINARY_AD_DATASETS:
            raise ValueError(f"Unsupported binary AD dataset: {dataset}")
        summary = run_binary_ad_evaluation(
            dataset=key,
            root=Path(dataset_roots[key]).expanduser(),
            localizer=localizer_factory(key),
            output_dir=output_dir / key,
            normal_quantile=normal_quantile,
            mad_scale=mad_scale,
            score_tail_fraction=score_tail_fraction,
            binary_score_source=binary_score_source,
            threshold_policy=threshold_policy,
            fixed_threshold=fixed_threshold,
            threshold_table=threshold_table,
            localizer_name=localizer_name,
            threshold_shot=threshold_shot,
            calibration_shots=calibration_shots,
            sam_engine=sam_engine_factory(key) if sam_engine_factory else None,
            max_region_proposals=max_region_proposals,
            max_pvla_refs=max_pvla_refs,
            save_evidence_images=save_evidence_images,
            offline_assets=offline_assets,
            final_verifier=final_verifier_factory(key) if final_verifier_factory else None,
            final_verifier_policy=final_verifier_policy,
            max_test_samples_per_category=max_test_samples_per_category,
            resume=resume,
        )
        summary["output_dir"] = str(output_dir / key)
        summaries.append(summary)
    return aggregate_binary_ad_summaries(
        summaries,
        output_dir=output_dir,
        max_region_proposals=max_region_proposals,
        max_pvla_refs=max_pvla_refs,
        final_verifier_enabled=final_verifier_factory is not None,
        write_files=True,
    )


def aggregate_binary_ad_summaries(
    summaries: Iterable[dict[str, Any]],
    *,
    output_dir: Path | None = None,
    max_region_proposals: int = 3,
    max_pvla_refs: int = 3,
    final_verifier_enabled: bool | None = None,
    write_files: bool = False,
) -> dict[str, Any]:
    rows = list(summaries)
    has_accuracy = all("accuracy" in row for row in rows)
    total = sum(int(row.get("total", row.get("test_total", 0)) or 0) for row in rows)
    correct = sum(int(row.get("correct", 0) or 0) for row in rows)
    category_total = sum(len(row.get("categories") or []) for row in rows)
    aggregate = {
        "pipeline": "mpdd_dtd_dagm_binary_ad",
        "method": "localizer_mcts_sam3_pvla_normal_calibrated_binary_ad",
        "method_framework": binary_method_framework(
            max_region_proposals=max_region_proposals,
            max_pvla_refs=max_pvla_refs,
            final_verifier_enabled=(
                any(bool(row.get("final_verifier_enabled")) for row in rows)
                if final_verifier_enabled is None
                else bool(final_verifier_enabled)
            ),
        ),
        "dataset_count": len(rows),
        "category_count": category_total,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total and has_accuracy else None,
        "macro_dataset_accuracy": (
            sum(float(row.get("accuracy", 0.0) or 0.0) for row in rows) / len(rows)
            if rows and has_accuracy
            else None
        ),
        "datasets": [
            {
                "dataset": row.get("dataset"),
                "root": row.get("root"),
                "output_dir": row.get("output_dir", ""),
                "dataset_layout": row.get("dataset_layout"),
                "train_normal_total": row.get("train_normal_total", 0),
                "total": row.get("total", row.get("test_total", 0)),
                "correct": row.get("correct", 0),
                "accuracy": row.get("accuracy"),
                "category_count": len(row.get("categories") or []),
                "sam3_enabled": row.get("sam3_enabled", False),
                "pvla_enabled": row.get("pvla_enabled", False),
                "offline_assets_prepared": row.get("offline_assets_prepared", False),
                "final_verifier_enabled": row.get("final_verifier_enabled", False),
            }
            for row in rows
        ],
    }
    if write_files:
        if output_dir is None:
            raise ValueError("output_dir is required when write_files=True")
        _write_json(output_dir / "pipeline_summary.json", aggregate)
        _write_json(output_dir / "dataset_summaries.json", rows)
    return aggregate


def load_binary_ad_threshold_table(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {}
    threshold_path = Path(path).expanduser()
    if not threshold_path.exists():
        return {}
    with threshold_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def binary_method_framework(
    *,
    max_region_proposals: int = 3,
    max_pvla_refs: int = 3,
    final_verifier_enabled: bool = False,
) -> dict[str, Any]:
    return {
        "paper_alignment": "GLLS offline PVLA plus online dual-stream inspection, adapted to binary anomaly detection.",
        "offline_pvla": {
            "role": "Build source-backed normal reference graph per category from normal training images.",
            "source": "train_normal_images",
            "max_visual_references_per_category": int(max_pvla_refs),
            "max_visual_references_per_region": int(max_pvla_refs),
            "region_policy": "dataset_profiled_regions_mpdd_parts_dtd_dagm_texture_surfaces",
            "test_annotations_used": False,
        },
        "online_global_stream": {
            "role": "Score each image with a localizer heatmap and calibrate per-category thresholds from normal scores.",
            "decision_rule": "score_greater_than_category_normal_threshold",
        },
        "online_local_stream": {
            "role": "Use heatmap-guided MCTS-style region search to retain a fixed evidence crop budget.",
            "max_region_proposals": int(max_region_proposals),
            "selection_policy": "heatmap_score_then_sam3_validity",
        },
        "sam3_refinement": {
            "role": "Prompt SAM3 with selected region boxes and audit mask validity when SAM3 is configured.",
            "prompt_type": "box",
        },
        "final_decision": {
            "role": "Fuse global localizer score, MCTS/SAM3 local evidence, and PVLA normal references into the final binary decision.",
            "decision_rule": (
                "qwen3_vl_json_normal_anomaly_verifier"
                if final_verifier_enabled
                else "score_greater_than_category_normal_threshold"
            ),
        },
        "audit_outputs": [
            "summary.json",
            "predictions.jsonl",
            "calibration_scores.jsonl",
            "pvla_normal_graphs/*.json",
            "evidence/*",
        ],
    }


def _lookup_threshold_table(
    table: dict[str, Any] | None,
    *,
    dataset: str,
    category: str,
    localizer: str,
    shot: int | None,
) -> tuple[float | None, str]:
    if not table:
        return None, ""
    dataset = str(dataset).lower()
    localizer = str(localizer or "").lower()
    shot_value = None if shot is None else int(shot)
    rows = table.get("thresholds", [])

    if isinstance(rows, dict):
        row = ((rows.get(dataset) or {}).get(localizer) or {}).get(str(shot_value), {}).get(category)
        if isinstance(row, dict) and "threshold" in row:
            return float(row["threshold"]), str(row.get("source") or table.get("source") or "threshold_table")
        if row is not None:
            return float(row), str(table.get("source") or "threshold_table")
        return None, ""

    best_match = None
    for row in rows:
        if str(row.get("dataset") or "").lower() != dataset:
            continue
        if str(row.get("category") or "") != category:
            continue
        row_localizer = str(row.get("localizer") or "").lower()
        if row_localizer and localizer and row_localizer != localizer:
            continue
        row_shot = row.get("shot", None)
        if row_shot is not None and shot_value is not None and int(row_shot) != shot_value:
            continue
        best_match = row
        if row_localizer == localizer and (row_shot is None or shot_value is None or int(row_shot) == shot_value):
            break
    if best_match and "threshold" in best_match:
        return float(best_match["threshold"]), str(best_match.get("source") or table.get("source") or "threshold_table")
    return None, ""


def discovery_summary(*, dataset: str, root: Path, category: str | None = None) -> dict[str, Any]:
    train_samples, test_samples = build_binary_ad_samples(dataset=dataset, root=root, category=category)
    train_by_category = _group_by_category(train_samples)
    test_by_category = _group_by_category(test_samples)
    categories = []
    for category_name in sorted(set(train_by_category) | set(test_by_category)):
        test_rows = test_by_category.get(category_name, [])
        categories.append({
            "category": category_name,
            "train_normal": len(train_by_category.get(category_name, [])),
            "test_total": len(test_rows),
            "test_normal": sum(1 for row in test_rows if row.label == 0),
            "test_anomaly": sum(1 for row in test_rows if row.label == 1),
            "defect_types": sorted({row.defect_type for row in test_rows}),
        })
    return {
        "dataset": dataset,
        "root": str(root),
        "root_exists": root.exists(),
        "category_filter": category or "",
        "dataset_layout": _describe_dataset_layout(root),
        "train_normal_total": len(train_samples),
        "test_total": len(test_samples),
        "categories": categories,
    }


def _build_pvla_normal_graph(
    *,
    dataset: str,
    category: str,
    root: Path,
    calibration: BinaryADCalibration,
    train_score_rows: list[dict[str, Any]],
    output_dir: Path,
    max_refs: int,
    save_evidence_images: bool,
    evidence_dir: Path,
    offline_assets: dict[str, Any] | None,
) -> dict[str, Any]:
    offline_ref_items = _offline_reference_items(offline_assets, max_refs_per_region=max_refs)
    selected_refs = _select_normal_reference_rows(train_score_rows, max_refs=max_refs) if not offline_ref_items else []
    ref_nodes = []
    edges = []
    ref_source = "offline_pvla_normal_references" if offline_ref_items else "online_train_normal_score_references"
    region_names = list((offline_assets or {}).get("pvla_regions") or [])
    if not region_names and offline_ref_items:
        region_names = sorted({str(item.get("region") or "whole_object") for item in offline_ref_items})
    region_nodes = [
        {
            "node_id": f"region:{_safe_name(region_name)}",
            "node_type": "pvla_region",
            "region": region_name,
            "source_backed": True,
        }
        for region_name in region_names
    ]
    region_edges = [
        {
            "source": "category",
            "target": f"region:{_safe_name(region_name)}",
            "relation": "has_pvla_region",
        }
        for region_name in region_names
    ]
    for rank, ref_item in enumerate(offline_ref_items, start=1):
        region = str(ref_item.get("region") or "whole_object")
        source_path = str(ref_item.get("source_image_path") or ref_item.get("cutout_path") or "")
        cutout_path = str(ref_item.get("cutout_path") or source_path)
        node_id = f"normal_ref_{rank}_{_safe_name(region)}"
        ref_nodes.append({
            "node_id": node_id,
            "node_type": "visual_reference_cutout",
            "source_image_path": source_path,
            "cutout_path": cutout_path,
            "region": region,
            "score": None,
            "localizer_category": "",
            "reference_source": ref_source,
            "source_backed": True,
            "generation": str(ref_item.get("generation") or ""),
            "sam3_status": str(ref_item.get("sam3_status") or ""),
            "sam3_score": ref_item.get("sam3_score"),
        })
        region_node_id = f"region:{_safe_name(region)}"
        if region in region_names:
            edges.append({"source": region_node_id, "target": node_id, "relation": "has_normal_reference"})
        edges.append({"source": "normal_distribution", "target": node_id, "relation": "supported_by"})
    for rank, row in enumerate(selected_refs, start=1):
        source_path = str(row["image_path"])
        crop_path = ""
        if save_evidence_images:
            crop_path = _save_crop(
                image_path=source_path,
                bbox=None,
                output_dir=evidence_dir / category / "pvla_normal_refs",
                prefix=f"normal_ref_{rank}",
            )
        node_id = f"normal_ref_{rank}"
        ref_nodes.append({
            "node_id": node_id,
            "node_type": "visual_reference_cutout",
            "source_image_path": source_path,
            "cutout_path": crop_path,
            "region": "whole_object",
            "score": float(row["score"]),
            "localizer_category": str(row.get("localizer_category") or ""),
            "reference_source": ref_source,
            "source_backed": True,
            "generation": "online selected train-normal full-image reference",
            "sam3_status": "not_applicable",
            "sam3_score": None,
        })
        edges.append({"source": "normal_distribution", "target": node_id, "relation": "supported_by"})

    graph = {
        "graph_type": "binary_ad_pvla_normal_reference",
        "dataset": dataset,
        "category": category,
        "root": str(root),
        "source": ref_source,
        "offline_text_knowledge_path": str((offline_assets or {}).get("text_knowledge_path") or ""),
        "offline_graph_path": str((offline_assets or {}).get("graph_path") or ""),
        "offline_visual_reference_dir": str((offline_assets or {}).get("visual_reference_dir") or ""),
        "offline_prepared": bool(offline_assets),
        "pvla_regions": region_names,
        "nodes": [
            {
                "node_id": "category",
                "node_type": "object_category",
                "text": category,
                "source_backed": True,
            },
            {
                "node_id": "normal_distribution",
                "node_type": "score_distribution",
                "text": "Normal training heatmap scores calibrated into an image-level anomaly threshold.",
                "calibration": asdict(calibration),
                "source_backed": True,
            },
            *region_nodes,
            *ref_nodes,
        ],
        "edges": [
            {"source": "category", "target": "normal_distribution", "relation": "has_normal_score_model"},
            *region_edges,
            *edges,
        ],
    }
    graph_path = output_dir / "pvla_normal_graphs" / f"{_safe_name(category)}.json"
    _write_json(graph_path, graph)
    return {
        "graph_path": str(graph_path),
        "graph_type": graph["graph_type"],
        "source": graph["source"],
        "offline_prepared": bool(offline_assets),
        "offline_text_knowledge_path": graph["offline_text_knowledge_path"],
        "offline_graph_path": graph["offline_graph_path"],
        "offline_visual_reference_dir": graph["offline_visual_reference_dir"],
        "source_backed": True,
        "reference_count": len(ref_nodes),
        "pvla_regions": region_names,
        "references": ref_nodes,
    }


def _offline_reference_items(
    offline_assets: dict[str, Any] | None,
    *,
    max_refs_per_region: int,
) -> list[dict[str, Any]]:
    if not offline_assets:
        return []
    max_refs_per_region = max(1, int(max_refs_per_region or 1))
    audit_rows = [row for row in offline_assets.get("sam3_reference_audit", []) if row.get("path")]
    if audit_rows:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in audit_rows:
            grouped.setdefault(str(row.get("region") or "whole_object"), []).append(row)
        ordered_regions = list(offline_assets.get("pvla_regions") or grouped)
        items = []
        for region in ordered_regions:
            for row in grouped.get(str(region), [])[:max_refs_per_region]:
                items.append({
                    "source_image_path": str(row.get("source_image_path") or row.get("path") or ""),
                    "cutout_path": str(row.get("path") or ""),
                    "region": str(row.get("region") or region),
                    "generation": str(row.get("generation") or ""),
                    "sam3_status": str(row.get("sam3_status") or ""),
                    "sam3_score": row.get("sam3_score"),
                })
        return items

    return [
        {
            "source_image_path": str(path),
            "cutout_path": str(path),
            "region": "whole_object",
            "generation": "legacy offline normal reference",
            "sam3_status": "",
            "sam3_score": None,
        }
        for path in list(offline_assets.get("offline_reference_paths") or [])[:max_refs_per_region]
    ]


def _select_normal_reference_rows(rows: list[dict[str, Any]], *, max_refs: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    max_refs = max(1, int(max_refs or 1))
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    median = float(np.median(scores))
    ranked = sorted(
        rows,
        key=lambda row: (abs(float(row["score"]) - median), float(row["score"]), str(row["image_path"])),
    )
    return ranked[:max_refs]


def _build_binary_method_trace(
    *,
    sample: BinaryADSample,
    score_info: dict[str, Any],
    score_policy: str,
    calibration: BinaryADCalibration,
    prediction: int,
    pvla_graph: dict[str, Any],
    sam_engine: Any | None,
    output_dir: Path,
    evidence_dir: Path,
    max_region_proposals: int,
    save_evidence_images: bool,
    heatmap: Any,
) -> dict[str, Any]:
    image_path = sample.image_path
    with Image.open(image_path) as image:
        width, height = image.size

    proposals = _binary_region_proposals(heatmap, top_k=max_region_proposals)
    image_size = (width, height)
    heatmap_shape = _heatmap_shape(heatmap)
    mcts_action_trace = _binary_mcts_action_trace(proposals, heatmap_shape=heatmap_shape, image_size=image_size)
    crop_selection = MCTSCropCandidateSelector(max_candidates=max_region_proposals).select(
        _binary_crop_candidates(proposals, heatmap_shape=heatmap_shape, image_size=image_size)
    )
    selected_crop_paths = []
    if save_evidence_images:
        for idx, candidate in enumerate(crop_selection.selected_candidates, start=1):
            crop_path = _save_crop(
                image_path=image_path,
                bbox=tuple(candidate.get("bbox_xywh") or (0, 0, width, height)),
                output_dir=evidence_dir / sample.category / "selected_test_crops",
                prefix=f"{_safe_name(Path(image_path).stem)}_crop_{idx}",
            )
            candidate["crop_path"] = crop_path
            selected_crop_paths.append(crop_path)
    selected_bbox = (
        tuple(crop_selection.selected_candidates[0].get("bbox_xywh"))
        if crop_selection.selected_candidates
        else None
    )
    selected_crop_path = selected_crop_paths[0] if selected_crop_paths else ""
    selected_crop_audit = [
        {
            **item,
            "crop_path": crop_selection.selected_candidates[idx].get("crop_path", "")
            if idx < len(crop_selection.selected_candidates)
            else "",
        }
        for idx, item in enumerate(crop_selection.selected_audit)
    ]

    sam_audit = _run_sam3_region_audit(
        sam_engine=sam_engine,
        image_path=image_path,
        proposals=proposals,
        heatmap_shape=heatmap_shape,
        image_size=image_size,
    )
    if sam_audit.get("selected_mask_valid") and save_evidence_images:
        sam_audit["selected_crop_path"] = selected_crop_path

    return {
        "method": "localizer_mcts_sam3_pvla_normal_calibrated_binary_ad",
        "score": float(score_info["score"]),
        "score_source": str(score_info.get("score_source") or ""),
        "score_components": score_info.get("score_components", {}),
        "threshold": float(calibration.threshold),
        "prediction": int(prediction),
        "score_policy": score_policy,
        "threshold_policy": calibration.threshold_policy,
        "localizer_category": str(score_info.get("localizer_category") or ""),
        "mcts_participation_status": "searched" if proposals else "no_region_candidates",
        "mcts_budget_config": {
            "max_region_proposals": int(max_region_proposals),
            "search_depth": 1,
            "rollout_policy": "heatmap_score_then_sam3_validity",
        },
        "mcts_action_trace": mcts_action_trace,
        "mcts_search_summary": {
            "proposal_count": len(proposals),
            "selected_proposal_index": int(crop_selection.selected_candidates[0].get("proposal_index", 0))
            if crop_selection.selected_candidates
            else None,
            "selected_reason": (
                "ranked_budgeted_mcts_nms_score"
                if crop_selection.selected_candidates
                else "no_salient_heatmap_region; global score still calibrated"
            ),
        },
        "region_proposal_audit": [
            _proposal_to_dict(item, heatmap_shape=heatmap_shape, image_size=image_size)
            for item in proposals
        ],
        "crop_candidate_audit": [
            *crop_selection.candidate_audit,
            *selected_crop_audit,
            *crop_selection.skipped_audit,
        ],
        "crop_evidence_audit": selected_crop_audit,
        "crop_count": len(crop_selection.selected_candidates),
        "prompt_visible_crop_count": len(crop_selection.selected_candidates),
        "sam3_participation_status": sam_audit["status"],
        "sam_prompt_selection_audit": sam_audit["prompts"],
        "sam_mask_scores": sam_audit["mask_scores"],
        "sam_mask_audit": sam_audit,
        "pvla_participation_status": "source_backed_normal_graph",
        "pvla_graph_path": pvla_graph["graph_path"],
        "pvla_source": pvla_graph["source"],
        "pvla_offline_prepared": pvla_graph.get("offline_prepared", False),
        "pvla_offline_graph_path": pvla_graph.get("offline_graph_path", ""),
        "pvla_offline_text_knowledge_path": pvla_graph.get("offline_text_knowledge_path", ""),
        "pvla_reference_count": pvla_graph["reference_count"],
        "pvla_selected_blocks": [
            {
                "block_type": "normal_score_distribution",
                "graph_path": pvla_graph["graph_path"],
                "source_backed": True,
            },
            *[
                {
                    "block_type": "visual_reference_cutout",
                    "graph_path": pvla_graph["graph_path"],
                    "offline_graph_path": pvla_graph.get("offline_graph_path", ""),
                    "offline_text_knowledge_path": pvla_graph.get("offline_text_knowledge_path", ""),
                    "source_image_path": ref.get("source_image_path"),
                    "cutout_path": ref.get("cutout_path"),
                    "region": ref.get("region", ""),
                    "generation": ref.get("generation", ""),
                    "sam3_status": ref.get("sam3_status", ""),
                    "sam3_score": ref.get("sam3_score"),
                    "reference_source": ref.get("reference_source", ""),
                    "visual_reference_source_backed": bool(ref.get("source_backed")),
                }
                for ref in pvla_graph.get("references", [])
            ],
        ],
        "final_decision_audit": {
            "policy": "score_greater_than_category_normal_threshold",
            "score": float(score_info["score"]),
            "threshold": float(calibration.threshold),
            "prediction": int(prediction),
            "label_semantics": {"0": "normal", "1": "anomaly"},
        },
        "method_participation_summary": {
            "localizer": "heatmap_score",
            "mcts": "region_evidence_search" if proposals else "no_region_candidates",
            "sam3": sam_audit["status"],
            "pvla": "source_backed_normal_graph",
        },
    }


def _verified_prediction(verifier_audit: dict[str, Any], *, fallback: int) -> int:
    try:
        prediction = int(verifier_audit.get("prediction"))
    except Exception:
        return int(fallback)
    return 1 if prediction else 0


def _fuse_final_prediction(
    *,
    threshold_prediction: int,
    verifier_prediction: int,
    policy: str,
) -> int:
    policy = str(policy or "anomaly_or").strip().lower()
    if policy == "replace":
        return int(verifier_prediction)
    if policy == "audit":
        return int(threshold_prediction)
    if policy == "anomaly_or":
        return int(bool(threshold_prediction) or bool(verifier_prediction))
    raise ValueError(f"Unsupported final_verifier_policy: {policy}")


def _apply_final_verifier_trace(
    trace: dict[str, Any],
    *,
    verifier_audit: dict[str, Any],
    threshold_prediction: int,
    verifier_prediction: int,
    prediction: int,
    policy: str,
) -> None:
    threshold_audit = dict(trace.get("final_decision_audit") or {})
    trace["threshold_decision_audit"] = {
        **threshold_audit,
        "prediction": int(threshold_prediction),
    }
    trace["final_verifier_audit"] = verifier_audit
    verifier_prediction_text = str(verifier_audit.get("prediction_text") or ("anomaly" if verifier_prediction else "normal"))
    trace["final_decision_audit"] = {
        "policy": f"{policy}:{str(verifier_audit.get('policy') or 'external_binary_verifier')}",
        "verifier": str(verifier_audit.get("verifier") or ""),
        "parse_status": str(verifier_audit.get("parse_status") or ""),
        "threshold_prediction": int(threshold_prediction),
        "verifier_prediction": int(verifier_prediction),
        "prediction": int(prediction),
        "prediction_text": "anomaly" if prediction else "normal",
        "verifier_prediction_text": verifier_prediction_text,
        "confidence": verifier_audit.get("confidence"),
        "label_semantics": {"0": "normal", "1": "anomaly"},
    }
    trace["prediction"] = int(prediction)
    trace["method_participation_summary"] = {
        **(trace.get("method_participation_summary") or {}),
        "final_verifier": str(verifier_audit.get("verifier") or "external_binary_verifier"),
    }


def _offline_category_assets(offline_assets: dict[str, Any] | None, *, dataset: str, category: str) -> dict[str, Any] | None:
    for dataset_row in (offline_assets or {}).get("datasets", []):
        if str(dataset_row.get("dataset") or "").lower() != str(dataset).lower():
            continue
        for category_row in dataset_row.get("categories", []):
            if str(category_row.get("category") or "") == str(category):
                return category_row
    return None


def _binary_region_proposals(heatmap: Any, *, top_k: int) -> list[RegionProposal]:
    if heatmap is None:
        return []
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return []
    thresholds = adaptive_thresholds(
        values,
        default_image=image_score_from_heatmap(values, tail_fraction=0.01),
        default_pixel=image_score_from_heatmap(values, tail_fraction=0.05),
        source="binary_ad:adaptive_heatmap_region",
    )
    min_area = max(1, int(values.size * 0.001))
    group_kernel = max(3, min(15, int(round(min(values.shape) / 32)) * 2 + 1))
    return detect_heatmap_regions(
        values,
        thresholds,
        task_type="Defect Localization",
        top_k=max(1, int(top_k or 1)),
        min_area=min_area,
        group_kernel=group_kernel,
        fallback_min_area=min_area,
        fallback_min_score=0.0,
        tail_fraction=0.02,
    )


def _binary_mcts_action_trace(
    proposals: list[RegionProposal],
    *,
    heatmap_shape: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    if not proposals:
        return []
    max_score = max(float(item.score) for item in proposals) or 1.0
    rows = []
    for idx, proposal in enumerate(proposals):
        reward = float(proposal.score) / max_score
        image_bbox = _proposal_bbox_to_image_xywh(proposal, heatmap_shape=heatmap_shape, image_size=image_size)
        rows.append({
            "step": idx,
            "action": "inspect_region",
            "proposal_index": idx,
            "bbox_xywh": list(image_bbox),
            "heatmap_bbox_xywh": list(proposal.bbox),
            "prior_score": float(proposal.score),
            "reward": reward,
            "selected": idx == 0,
        })
    return rows


def _binary_crop_candidates(
    proposals: list[RegionProposal],
    *,
    heatmap_shape: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    rows = []
    for idx, proposal in enumerate(proposals):
        image_bbox = _proposal_bbox_to_image_xywh(proposal, heatmap_shape=heatmap_shape, image_size=image_size)
        image_xyxy = _xywh_to_xyxy(image_bbox)
        rows.append({
            "label": f"MCTS focus ROI {idx + 1}",
            "source": "mcts_region_proposal",
            "priority": float(proposal.selection_score),
            "bbox_xyxy": list(image_xyxy),
            "bbox_xywh": list(image_bbox),
            "heatmap_bbox_xyxy": list(proposal.xyxy),
            "heatmap_bbox_xywh": list(proposal.bbox),
            "heatmap_score": float(proposal.score),
            "search_score": float(proposal.selection_score),
            "selection_score": float(proposal.selection_score),
            "selection_reason": proposal.selection_reason,
            "proposal_source": proposal.source,
            "proposal_index": idx,
            "proposal_bbox_xyxy": list(image_xyxy),
            "proposal_heatmap_bbox_xyxy": list(proposal.xyxy),
        })
    return rows


def _heatmap_shape(heatmap: Any) -> tuple[int, int] | None:
    if heatmap is None:
        return None
    values = np.asarray(heatmap)
    if values.ndim != 2 or values.size == 0:
        return None
    return int(values.shape[0]), int(values.shape[1])


def _proposal_bbox_to_image_xywh(
    proposal: RegionProposal,
    *,
    heatmap_shape: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = proposal.bbox
    if not heatmap_shape:
        return _clamp_xywh((x, y, width, height), image_size=image_size)
    heatmap_height, heatmap_width = heatmap_shape
    scale_x = image_width / max(1, heatmap_width)
    scale_y = image_height / max(1, heatmap_height)
    scaled = (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(width * scale_x))),
        max(1, int(round(height * scale_y))),
    )
    return _clamp_xywh(scaled, image_size=image_size)


def _proposal_normalized_box(
    proposal: RegionProposal,
    *,
    heatmap_shape: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> list[float]:
    image_width, image_height = image_size
    x, y, width, height = _proposal_bbox_to_image_xywh(
        proposal,
        heatmap_shape=heatmap_shape,
        image_size=image_size,
    )
    return [
        (x + width / 2.0) / max(1, image_width),
        (y + height / 2.0) / max(1, image_height),
        width / max(1, image_width),
        height / max(1, image_height),
    ]


def _xywh_to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return x, y, x + width, y + height


def _clamp_xywh(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = bbox
    x = min(max(0, int(x)), max(0, image_width - 1))
    y = min(max(0, int(y)), max(0, image_height - 1))
    width = max(1, min(int(width), image_width - x))
    height = max(1, min(int(height), image_height - y))
    return x, y, width, height


def _run_sam3_region_audit(
    *,
    sam_engine: Any | None,
    image_path: str,
    proposals: list[RegionProposal],
    heatmap_shape: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    if sam_engine is None:
        return {
            "status": "not_configured",
            "prompts": [],
            "mask_scores": [],
            "selected_mask_valid": False,
            "reason": "sam3_engine_not_provided",
        }
    if not proposals:
        return {
            "status": "no_region_proposal",
            "prompts": [],
            "mask_scores": [],
            "selected_mask_valid": False,
            "reason": "no_heatmap_region_to_prompt",
        }

    width, height = image_size
    boxes = [
        _proposal_normalized_box(proposal, heatmap_shape=heatmap_shape, image_size=image_size)
        for proposal in proposals
    ]
    prompts = [
        {
            "role": "region_box",
            "proposal_index": idx,
            "box_cxcywh_norm": box,
            "source": "mcts_region_proposal",
        }
        for idx, box in enumerate(boxes)
    ]
    try:
        sam_engine.set_image(image_path)
        mask, score = sam_engine.predict_mask_with_boxes(boxes, threshold=0.4)
    except Exception as exc:
        return {
            "status": "error",
            "prompts": prompts,
            "mask_scores": [],
            "selected_mask_valid": False,
            "error": str(exc),
        }

    stats = _mask_stats(mask, image_size=image_size)
    valid = bool(stats["area_pixels"] > 0 and stats["area_ratio"] <= 0.85)
    status = "box_prompt_refinement" if valid else "invalid_mask"
    if stats["area_ratio"] > 0.85:
        status = "overlarge_mask_rejected"
    return {
        "status": status,
        "prompts": prompts,
        "mask_scores": [float(score)],
        "selected_mask_valid": valid,
        "mask_stats": stats,
    }


def _mask_stats(mask: Any, *, image_size: tuple[int, int]) -> dict[str, Any]:
    width, height = image_size
    if mask is None:
        return {"area_pixels": 0, "area_ratio": 0.0, "bbox_xywh": []}
    arr = np.asarray(mask).astype(bool)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return {"area_pixels": 0, "area_ratio": 0.0, "bbox_xywh": []}
    ys, xs = np.where(arr)
    area = int(arr.sum())
    if area == 0:
        return {"area_pixels": 0, "area_ratio": 0.0, "bbox_xywh": []}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "area_pixels": area,
        "area_ratio": float(area / max(1, width * height)),
        "bbox_xywh": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
    }


def _proposal_to_dict(
    proposal: RegionProposal,
    *,
    heatmap_shape: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    image_bbox = _proposal_bbox_to_image_xywh(proposal, heatmap_shape=heatmap_shape, image_size=image_size)
    image_xyxy = _xywh_to_xyxy(image_bbox)
    return {
        "bbox_xywh": list(image_bbox),
        "bbox_xyxy": list(image_xyxy),
        "heatmap_bbox_xywh": list(proposal.bbox),
        "heatmap_bbox_xyxy": list(proposal.xyxy),
        "box_cxcywh_norm": _proposal_normalized_box(proposal, heatmap_shape=heatmap_shape, image_size=image_size),
        "score": float(proposal.score),
        "source": proposal.source,
        "selection_score": proposal.selection_score,
        "selection_reason": proposal.selection_reason,
        "pixel_threshold": proposal.pixel_threshold,
        "selection_features": proposal.selection_features,
    }


def _collect_from_meta_json(
    *,
    dataset: str,
    root: Path,
    category: str | None,
) -> tuple[list[BinaryADSample], list[BinaryADSample]]:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return [], []
    try:
        meta = _load_json(meta_path)
    except Exception:
        return [], []

    train_rows = _samples_from_meta_split(
        dataset=dataset,
        root=root,
        split_name="train",
        split_data=meta.get("train", {}),
        category=category,
        train_only_normals=True,
    )
    test_rows = _samples_from_meta_split(
        dataset=dataset,
        root=root,
        split_name="test",
        split_data=meta.get("test", {}),
        category=category,
        train_only_normals=False,
    )
    return train_rows, test_rows


def _samples_from_meta_split(
    *,
    dataset: str,
    root: Path,
    split_name: str,
    split_data: Any,
    category: str | None,
    train_only_normals: bool,
) -> list[BinaryADSample]:
    rows: list[BinaryADSample] = []
    for category_name, item in _iter_meta_items(split_data):
        if category and category_name != category:
            continue
        image_path = root / str(item.get("img_path") or item.get("image_path") or item.get("path") or "")
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = int(item.get("anomaly", item.get("label", 0)) or 0)
        if train_only_normals and label != 0:
            continue
        defect_type = str(item.get("specie_name") or item.get("defect_type") or ("defect" if label else "good"))
        if _is_normal_name(defect_type):
            defect_type = "good"
        rows.append(
            BinaryADSample(
                dataset=dataset,
                category=str(item.get("cls_name") or category_name),
                split=split_name,
                image_path=str(image_path),
                label=label,
                defect_type=defect_type,
            )
        )
    return rows


def _iter_meta_items(split_data: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(split_data, dict):
        for category_name, rows in split_data.items():
            if isinstance(rows, list):
                for item in rows:
                    if isinstance(item, dict):
                        yield str(item.get("cls_name") or category_name), item
            elif isinstance(rows, dict):
                yield str(rows.get("cls_name") or category_name), rows
    elif isinstance(split_data, list):
        for item in split_data:
            if isinstance(item, dict):
                yield str(item.get("cls_name") or "default"), item


def _collect_train_normals(dataset: str, category: str, category_dir: Path) -> list[BinaryADSample]:
    train_root = _first_existing_dir(category_dir, TRAIN_DIR_NAMES)
    if train_root is None:
        return []

    normal_dirs = [
        child for child in sorted(train_root.iterdir())
        if child.is_dir() and _is_normal_name(child.name)
    ]
    image_paths = []
    if normal_dirs:
        for normal_dir in normal_dirs:
            image_paths.extend(iter_image_files(normal_dir))
    elif label_root := _first_existing_dir(train_root, LABEL_DIR_NAMES):
        image_paths = [
            path for path in _direct_image_files(train_root)
            if not _matching_mask_has_foreground(label_root, path)
        ]
    else:
        image_paths = iter_image_files(train_root)

    return [
        BinaryADSample(
            dataset=dataset,
            category=category,
            split="train",
            image_path=str(path),
            label=0,
            defect_type="good",
        )
        for path in image_paths
    ]


def _collect_test_samples(dataset: str, category: str, category_dir: Path) -> list[BinaryADSample]:
    test_root = _first_existing_dir(category_dir, TEST_DIR_NAMES)
    if test_root is None:
        return []

    label_root = _first_existing_dir(test_root, LABEL_DIR_NAMES) or _first_existing_dir(category_dir, LABEL_DIR_NAMES)
    child_dirs = [
        child for child in sorted(test_root.iterdir())
        if child.is_dir() and not _is_label_dir_name(child.name)
    ]
    if child_dirs:
        rows = []
        for child in child_dirs:
            label = 0 if _is_normal_name(child.name) else 1
            defect_type = "good" if label == 0 else child.name
            for path in iter_image_files(child):
                rows.append(
                    BinaryADSample(
                        dataset=dataset,
                        category=category,
                        split="test",
                        image_path=str(path),
                        label=label,
                        defect_type=defect_type,
                    )
                )
        return rows

    rows = []
    for path in _direct_image_files(test_root):
        has_defect = _matching_mask_has_foreground(label_root, path) if label_root else False
        rows.append(
            BinaryADSample(
                dataset=dataset,
                category=category,
                split="test",
                image_path=str(path),
                label=int(has_defect),
                defect_type="defect" if has_defect else "good",
            )
        )
    return rows


def _score_image(
    localizer: Any,
    image_path: str,
    *,
    tail_fraction: float,
    binary_score_source: str = "auto",
    include_heatmap: bool = False,
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        heatmap, localizer_category = localizer.predict_anomaly_map(image.convert("RGB"))
    image_level_score = getattr(localizer, "last_image_score", None)
    heatmap_tail_score = image_score_from_heatmap(heatmap, tail_fraction=tail_fraction)
    score, score_source = _select_binary_score(
        image_level_score=image_level_score,
        heatmap_tail_score=heatmap_tail_score,
        binary_score_source=binary_score_source,
    )
    result = {
        "score": score,
        "localizer_category": localizer_category,
        "score_source": score_source,
        "score_components": {
            "localizer_image_score": float(image_level_score) if image_level_score is not None else None,
            "heatmap_tail_score": float(heatmap_tail_score),
        },
    }
    if include_heatmap:
        result["heatmap"] = heatmap
    return result


def _select_binary_score(
    *,
    image_level_score: Any,
    heatmap_tail_score: float,
    binary_score_source: str,
) -> tuple[float, str]:
    source = str(binary_score_source or "auto")
    image_score = None
    if image_level_score is not None and np.isfinite(float(image_level_score)):
        image_score = float(image_level_score)
    heatmap_score = float(heatmap_tail_score)
    if source == "auto":
        return (image_score, "localizer_image_score") if image_score is not None else (heatmap_score, "heatmap_tail_score")
    if source == "localizer_image":
        return (image_score, "localizer_image_score") if image_score is not None else (heatmap_score, "heatmap_tail_score_fallback")
    if source == "heatmap_tail":
        return heatmap_score, "heatmap_tail_score"
    if source == "max_image_heatmap":
        return max(image_score if image_score is not None else heatmap_score, heatmap_score), "max_image_heatmap"
    if source == "mean_image_heatmap":
        return (
            (image_score + heatmap_score) / 2.0 if image_score is not None else heatmap_score,
            "mean_image_heatmap" if image_score is not None else "heatmap_tail_score_fallback",
        )
    raise ValueError(f"Unsupported binary_score_source: {binary_score_source}")


def _configure_localizer_category(
    localizer: Any,
    category: str,
    train_rows: list[BinaryADSample],
) -> None:
    configure = getattr(localizer, "configure_support", None)
    if callable(configure):
        configure(category, [row.image_path for row in train_rows])
        return
    set_category = getattr(localizer, "set_active_category", None)
    if callable(set_category):
        set_category(category)


def _calibration_source_rows(
    train_rows: list[BinaryADSample],
    *,
    calibration_shots: int,
) -> list[BinaryADSample]:
    if int(calibration_shots) < 0:
        return train_rows
    if int(calibration_shots) == 0:
        return []
    return train_rows[: int(calibration_shots)]


def _group_by_category(samples: Iterable[BinaryADSample]) -> dict[str, list[BinaryADSample]]:
    grouped: dict[str, list[BinaryADSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    return grouped


def _summarize_category(
    category: str,
    calibration: BinaryADCalibration,
    results: list[BinaryADResult],
) -> dict[str, Any]:
    normal_rows = [row for row in results if row.label == 0]
    anomaly_rows = [row for row in results if row.label == 1]
    return {
        "category": category,
        "total": len(results),
        "normal_total": len(normal_rows),
        "anomaly_total": len(anomaly_rows),
        "correct": sum(1 for row in results if row.correct),
        "accuracy": _accuracy(results),
        "normal_accuracy": _accuracy(normal_rows),
        "anomaly_accuracy": _accuracy(anomaly_rows),
        "calibration": asdict(calibration),
    }


def _accuracy(rows: list[BinaryADResult]) -> float:
    return sum(1 for row in rows if row.correct) / len(rows) if rows else 0.0


def _first_existing_dir(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _is_normal_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in NORMAL_DIR_NAMES


def _is_label_dir_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {item.lower().replace("-", "_").replace(" ", "_") for item in LABEL_DIR_NAMES}


def _direct_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _matching_mask_has_foreground(label_root: Path | None, image_path: Path) -> bool:
    if label_root is None or not label_root.exists():
        return False
    candidates = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(label_root.rglob(f"{image_path.stem}{ext}"))
    candidates.extend(label_root.rglob(f"{image_path.stem}*"))
    for mask_path in candidates:
        if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(mask_path) as mask:
                arr = np.asarray(mask.convert("L"))
            if bool(np.any(arr > 0)):
                return True
        except Exception:
            continue
    return False


def _describe_dataset_layout(root: Path) -> str:
    if (root / "meta.json").exists():
        return "meta_json"
    categories = discover_category_dirs(root)
    if not categories:
        return "missing"
    first = categories[0]
    if _first_existing_dir(first, LABEL_DIR_NAMES):
        return "flat_test_with_label_masks"
    if _first_existing_dir(first, TRAIN_DIR_NAMES) and _first_existing_dir(first, TEST_DIR_NAMES):
        return "category_train_test_dirs"
    return "unknown"


def _save_crop(
    *,
    image_path: str,
    bbox: tuple[int, int, int, int] | None,
    output_dir: Path,
    prefix: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(image_path)
    out_path = output_dir / f"{_safe_name(prefix)}_{_safe_name(source.stem)}.png"
    with Image.open(source) as image:
        image = image.convert("RGB")
        if bbox is not None:
            x, y, w, h = bbox
            crop = image.crop((max(0, x), max(0, y), max(0, x + w), max(0, y + h)))
        else:
            crop = image
        crop.save(out_path)
    return str(out_path)


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "item"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_existing_results(path: Path) -> dict[str, BinaryADResult]:
    if not path.exists():
        return {}
    rows: dict[str, BinaryADResult] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                result = _binary_result_from_row(row)
            except Exception:
                continue
            rows[result.image_path] = result
    return rows


def _binary_result_from_row(row: dict[str, Any]) -> BinaryADResult:
    prediction = int(row.get("prediction", 0) or 0)
    label = int(row.get("label", 0) or 0)
    return BinaryADResult(
        dataset=str(row.get("dataset") or ""),
        category=str(row.get("category") or ""),
        image_path=str(row.get("image_path") or ""),
        defect_type=str(row.get("defect_type") or ""),
        label=label,
        score=float(row.get("score", 0.0) or 0.0),
        threshold=float(row.get("threshold", 0.0) or 0.0),
        prediction=prediction,
        correct=bool(row.get("correct", prediction == label)),
        localizer_category=str(row.get("localizer_category") or ""),
        score_policy=str(row.get("score_policy") or ""),
        trace=row.get("trace") if isinstance(row.get("trace"), dict) else {},
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
