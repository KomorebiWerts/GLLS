"""Published runtime localizer and checkpoint selection rules."""

from __future__ import annotations

import json
import os
from typing import Any

from glls import paths as glls_paths


ABOUND_ONE_SHOT_DATASETS = {"mvtec", "visa"}


def normalize_dataset_key(dataset_name: Any) -> str:
    key = str(dataset_name or "").strip().lower().replace("_", "-")
    aliases = {
        "ds-mvtec": "mvtec",
        "ds-mvtec-ad": "mvtec",
        "mvtec-ad": "mvtec",
        "vis-a": "visa",
        "dagm-kaggleupload": "dagm",
    }
    return aliases.get(key, key)


def parse_k_shot(k_shot: Any) -> int:
    try:
        return int(k_shot or 0)
    except (TypeError, ValueError):
        return 1


def published_localizer_name(dataset_name: Any, k_shot: Any) -> str:
    dataset_key = normalize_dataset_key(dataset_name)
    shot = parse_k_shot(k_shot)
    if dataset_key in ABOUND_ONE_SHOT_DATASETS and shot == 1:
        return "abound"
    return "adaptclip"


def resolve_localizer_name(localizer_choice: Any, dataset_name: Any, k_shot: Any) -> str:
    choice = str(localizer_choice or "auto").strip().lower().replace("-", "").replace("_", "")
    if choice in {"auto", ""}:
        return published_localizer_name(dataset_name, k_shot)
    if choice == "abound":
        return "abound"
    if choice == "adaptclip":
        return "adaptclip"
    return published_localizer_name(dataset_name, k_shot)


def default_adaptclip_checkpoint(dataset_name: Any, adaptclip_root: str | None = None) -> str:
    root = adaptclip_root or glls_paths.adaptclip_root()
    domain = "visa" if normalize_dataset_key(dataset_name) == "visa" else "mvtec"
    return os.path.join(root, "checkpoints", f"{domain}_epoch_15.pth")


def resolve_adaptclip_checkpoint(path_value: str, dataset_name: Any, adaptclip_root: str | None = None) -> str:
    if not path_value:
        return default_adaptclip_checkpoint(dataset_name, adaptclip_root)
    if os.path.isdir(path_value):
        domain = "visa" if normalize_dataset_key(dataset_name) == "visa" else "mvtec"
        candidates = [
            os.path.join(path_value, "checkpoints", f"{domain}_epoch_15.pth"),
            os.path.join(path_value, f"{domain}_epoch_15.pth"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return path_value


def abound_dataset_weight_paths(save_path: str, dataset_name: Any) -> dict[str, str]:
    dataset_key = normalize_dataset_key(dataset_name)
    return {
        "lora": os.path.join(save_path, dataset_key, f"final_vvclip_model_state_{dataset_key}.pth"),
        "soft_prompt": os.path.join(save_path, dataset_key, f"final_soft_prompt_state_{dataset_key}.pth"),
        "memory_bank": os.path.join(save_path, dataset_key, f"final_memory_bank_{dataset_key}.pt"),
    }


def runtime_weight_config(
    dataset_name: Any,
    localizer_choice: Any,
    k_shot: Any,
    adaptclip_ckpt_path: str = "",
    *,
    adaptclip_root: str | None = None,
    abound_model_path: str | None = None,
    abound_save_path: str | None = None,
) -> dict[str, Any]:
    dataset_key = normalize_dataset_key(dataset_name)
    shot = parse_k_shot(k_shot)
    localizer_name = resolve_localizer_name(localizer_choice, dataset_key, shot)

    if localizer_name == "abound":
        if dataset_key not in ABOUND_ONE_SHOT_DATASETS:
            raise ValueError("ABounD release artifacts are available only for MVTec and VisA 1-shot runs.")
        if shot != 1:
            raise ValueError("ABounD is available only for 1-shot runtime. Use AdaptCLIP for 0-shot.")
        checkpoint_path = abound_model_path or glls_paths.abound_model_path()
        save_path = abound_save_path or glls_paths.abound_save_path()
        image_size = 336
        dataset_weight_paths = abound_dataset_weight_paths(save_path, dataset_key)
    else:
        checkpoint_path = resolve_adaptclip_checkpoint(adaptclip_ckpt_path, dataset_key, adaptclip_root)
        save_path = ""
        image_size = 518
        dataset_weight_paths = {"checkpoint": checkpoint_path}

    signature_payload = {
        "dataset": dataset_key,
        "localizer": localizer_name,
        "k_shot": shot,
        "checkpoint_path": checkpoint_path,
        "save_path": save_path,
        "dataset_weight_paths": dataset_weight_paths,
    }
    return {
        "dataset_name": dataset_key,
        "localizer_name": localizer_name,
        "k_shot": shot,
        "checkpoint_path": checkpoint_path,
        "save_path": save_path,
        "image_size": image_size,
        "dataset_weight_paths": dataset_weight_paths,
        "signature": json.dumps(signature_payload, sort_keys=True),
    }


def runtime_weight_summary(config: dict[str, Any]) -> str:
    localizer_label = "ABounD" if config["localizer_name"] == "abound" else "AdaptCLIP"
    lines = [
        f"Dataset: {config['dataset_name']}",
        f"Localizer: {localizer_label} | Shot: {config['k_shot']}",
    ]
    if config["localizer_name"] == "abound":
        weights = config["dataset_weight_paths"]
        lines.extend([
            f"Backbone: {config['checkpoint_path']}",
            f"LoRA: {weights['lora']}",
            f"Soft prompt: {weights['soft_prompt']}",
            f"Memory bank: {weights['memory_bank']}",
        ])
    else:
        lines.append(f"Checkpoint: {config['checkpoint_path']}")
    return "\n".join(lines)
