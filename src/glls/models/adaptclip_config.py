from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ADAPTCLIP_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parent / "AdaptCLIP" / "adaptcliplib" / "model_config.json"
)


def load_adaptclip_model_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else ADAPTCLIP_MODEL_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def load_adaptclip_binary_ad_decision_table(path: Path | str | None = None) -> dict[str, Any]:
    config = load_adaptclip_model_config(path)
    decision_parameters = config.get("decision_parameters", {})
    section = decision_parameters.get("binary_ad") if isinstance(decision_parameters, dict) else None
    source_name = "adaptclip_model_config:decision_parameters.binary_ad"
    if not isinstance(section, dict):
        section = config.get("binary_ad_thresholds_by_shot")
        source_name = "adaptclip_model_config:binary_ad_thresholds_by_shot"
    if not isinstance(section, dict):
        return {}

    shots = section.get("shots")
    if not isinstance(shots, dict):
        return {}

    localizer = str(section.get("localizer") or "adaptclip").lower()
    source = str(section.get("source") or source_name)
    table: dict[str, Any] = {
        "schema": "glls_adaptclip_binary_ad_threshold_table_v1",
        "source": source_name,
        "config_source": source,
        "score_source": str(section.get("score_source") or ""),
        "threshold_policy": str(section.get("cutoff_policy") or section.get("threshold_policy") or ""),
        "thresholds": {},
    }

    thresholds = table["thresholds"]
    for shot_text, datasets in shots.items():
        if not isinstance(datasets, dict):
            continue
        try:
            shot = str(int(str(shot_text).replace("-shot", "")))
        except ValueError:
            continue
        for dataset, categories in datasets.items():
            if not isinstance(categories, dict):
                continue
            dataset_table = thresholds.setdefault(str(dataset).lower(), {}).setdefault(localizer, {}).setdefault(shot, {})
            for category, value in categories.items():
                try:
                    dataset_table[str(category)] = {
                        "threshold": float(value),
                        "source": table["source"],
                    }
                except (TypeError, ValueError):
                    continue

    return table if table["thresholds"] else {}


def load_adaptclip_binary_ad_threshold_table(path: Path | str | None = None) -> dict[str, Any]:
    return load_adaptclip_binary_ad_decision_table(path)
