from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glls.eval.binary_ad import BinaryADSample


@dataclass(frozen=True)
class BinaryADQwen3VerifierConfig:
    max_crops: int = 2
    max_refs: int = 4
    max_tokens: int = 512
    include_score_hint: bool = False
    fallback_to_threshold: bool = True


class BinaryADQwen3Verifier:
    """Final binary verifier for GLLS evidence packages.

    The verifier is intentionally thin: the binary AD evaluator owns evidence
    generation, while this class only formats that evidence for a VLM and parses
    a strict normal/anomaly answer.
    """

    def __init__(
        self,
        inference_engine: Any,
        *,
        config: BinaryADQwen3VerifierConfig | None = None,
        model_name: str = "qwen3",
    ) -> None:
        self.inference_engine = inference_engine
        self.config = config or BinaryADQwen3VerifierConfig()
        self.model_name = model_name

    def verify(self, sample: BinaryADSample, trace: dict[str, Any]) -> dict[str, Any]:
        content, content_audit = build_binary_ad_qwen3_content(
            sample,
            trace,
            max_crops=self.config.max_crops,
            max_refs=self.config.max_refs,
            include_score_hint=self.config.include_score_hint,
        )
        raw_response = self.inference_engine.generate(content, max_tokens=self.config.max_tokens)
        parsed = parse_binary_ad_verifier_response(
            raw_response,
            fallback_prediction=int(trace.get("prediction", 0)),
            fallback_to_threshold=self.config.fallback_to_threshold,
        )
        return {
            "verifier": self.model_name,
            "policy": "qwen3_vl_binary_normal_anomaly_verifier",
            "raw_response": raw_response,
            "content_audit": content_audit,
            **parsed,
        }


def build_binary_ad_qwen3_content(
    sample: BinaryADSample,
    trace: dict[str, Any],
    *,
    max_crops: int = 2,
    max_refs: int = 4,
    include_score_hint: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    crop_paths = _selected_crop_paths(trace, max_items=max_crops)
    ref_paths = _selected_reference_paths(trace, max_items=max_refs)
    regions = _selected_regions(trace)
    score_hint = _score_hint(trace) if include_score_hint else ""
    prompt = _build_prompt(
        sample=sample,
        trace=trace,
        regions=regions,
        crop_count=len(crop_paths),
        reference_count=len(ref_paths),
        score_hint=score_hint,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": "\n[Image 1] Test image to classify."})
    content.append({"type": "image", "image": sample.image_path})

    for idx, crop_path in enumerate(crop_paths, start=1):
        content.append({"type": "text", "text": f"\n[Crop {idx}] Heatmap/MCTS-selected local evidence from the test image."})
        content.append({"type": "image", "image": crop_path})

    for idx, ref in enumerate(ref_paths, start=1):
        content.append({
            "type": "text",
            "text": (
                f"\n[Normal reference {idx}] Region={ref['region']}. "
                "Source-backed PVLA normal reference for this category."
            ),
        })
        content.append({"type": "image", "image": ref["path"]})

    return content, {
        "test_image_path": sample.image_path,
        "test_crop_paths": crop_paths,
        "normal_reference_paths": ref_paths,
        "pvla_regions": regions,
        "score_hint_included": bool(score_hint),
        "image_count": 1 + len(crop_paths) + len(ref_paths),
    }


def parse_binary_ad_verifier_response(
    raw_response: Any,
    *,
    fallback_prediction: int = 0,
    fallback_to_threshold: bool = True,
) -> dict[str, Any]:
    text = str(raw_response or "").strip()
    parsed = _parse_json_object(text)
    status = "json"
    if parsed is None:
        parsed = {}
        status = "keyword"

    prediction_text = _normalize_prediction(parsed.get("prediction") or parsed.get("label") or parsed.get("answer"))
    if prediction_text is None and status == "keyword":
        prediction_text = _keyword_prediction(text)

    if prediction_text is None:
        if fallback_to_threshold:
            prediction = int(fallback_prediction)
            prediction_text = "anomaly" if prediction else "normal"
            status = "fallback_threshold"
        else:
            prediction = int(fallback_prediction)
            prediction_text = "unknown"
            status = "unparsed"
    else:
        prediction = 1 if prediction_text == "anomaly" else 0

    confidence = _safe_float(parsed.get("confidence"), default=None)
    return {
        "parse_status": status,
        "prediction_text": prediction_text,
        "prediction": int(prediction),
        "confidence": confidence,
        "evidence": str(parsed.get("evidence") or parsed.get("rationale") or "")[:1000],
    }


def _build_prompt(
    *,
    sample: BinaryADSample,
    trace: dict[str, Any],
    regions: list[str],
    crop_count: int,
    reference_count: int,
    score_hint: str,
) -> str:
    region_text = ", ".join(regions[:12]) if regions else "whole object / visible surface"
    score_block = f"\nLocalizer hint: {score_hint}" if score_hint else ""
    return (
        "You are the final verifier in a GLLS industrial visual inspection pipeline.\n"
        "Classify the test image as normal or anomaly for binary anomaly detection.\n"
        "Use the full test image first, then inspect the supplied local crops and PVLA normal references.\n"
        "Treat normal references as non-exhaustive examples; normal pose, alignment, illumination, texture scale, and "
        "category-specific shape variation can differ from the references.\n"
        "A normal image should match the category's normal appearance; any visible defect, damage, stain, scratch, "
        "structural irregularity, or abnormal texture should be anomaly.\n"
        "Do not classify anomaly only because a normal reference has a different viewpoint, crop, pose, or benign "
        "geometric variant. Require a visible defect cue in the test image or selected crop.\n"
        "Do not invent defect types that are not visually supported. If evidence is ambiguous, choose the more likely "
        "binary class from the images and evidence.\n\n"
        f"Dataset: {sample.dataset}\n"
        f"Category: {sample.category}\n"
        f"PVLA regions to compare: {region_text}\n"
        f"Visible local crop count: {crop_count}\n"
        f"PVLA normal reference count: {reference_count}"
        f"{score_block}\n\n"
        "Return exactly one JSON object and no markdown:\n"
        '{"prediction":"normal|anomaly","confidence":0.0,"evidence":"one short visual reason"}'
    )


def _score_hint(trace: dict[str, Any]) -> str:
    score = _safe_float(trace.get("score"), default=None)
    threshold = _safe_float(trace.get("threshold"), default=None)
    score_source = str(trace.get("score_source") or "")
    threshold_prediction = int(trace.get("prediction", 0) or 0)
    label = "anomaly" if threshold_prediction else "normal"
    if score is None or threshold is None:
        return f"threshold branch predicted {label}."
    return f"{score_source} score={score:.6g}, threshold={threshold:.6g}, threshold branch predicted {label}."


def _selected_crop_paths(trace: dict[str, Any], *, max_items: int) -> list[str]:
    rows = []
    for item in trace.get("crop_evidence_audit") or []:
        path = str(item.get("crop_path") or "")
        if path and Path(path).is_file():
            rows.append(path)
    return _dedupe(rows)[: max(0, int(max_items))]


def _selected_reference_paths(trace: dict[str, Any], *, max_items: int) -> list[dict[str, str]]:
    refs = []
    for block in trace.get("pvla_selected_blocks") or []:
        if block.get("block_type") != "visual_reference_cutout":
            continue
        path = str(block.get("cutout_path") or block.get("source_image_path") or "")
        if path and Path(path).is_file():
            refs.append({"path": path, "region": str(block.get("region") or "whole_object")})
    deduped = []
    seen = set()
    for ref in refs:
        key = (ref["path"], ref["region"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped[: max(0, int(max_items))]


def _selected_regions(trace: dict[str, Any]) -> list[str]:
    regions = []
    for block in trace.get("pvla_selected_blocks") or []:
        region = str(block.get("region") or "").strip()
        if region:
            regions.append(region)
    return _dedupe(regions)


def _dedupe(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalize_prediction(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9_-]+", " ", text).strip()
    if text in {"normal", "good", "ok", "negative", "nondefective", "non defective", "no defect"}:
        return "normal"
    if text in {"anomaly", "abnormal", "defect", "defective", "positive", "bad", "damaged"}:
        return "anomaly"
    if "anomal" in text or "defect" in text or "abnormal" in text:
        return "anomaly"
    if "normal" in text or "good" in text or "no defect" in text:
        return "normal"
    return None


def _keyword_prediction(text: str) -> str | None:
    lowered = text.lower()
    anomaly_hits = len(re.findall(r"\b(anomaly|abnormal|defect|defective|damaged|scratch|stain)\b", lowered))
    normal_hits = len(re.findall(r"\b(normal|good|nondefective|non-defective|no defect)\b", lowered))
    if anomaly_hits > normal_hits:
        return "anomaly"
    if normal_hits > anomaly_hits:
        return "normal"
    return None


def _safe_float(value: Any, *, default: float | None) -> float | None:
    try:
        number = float(value)
    except Exception:
        return default
    if not (number == number):
        return default
    return number
