"""Centralized default paths for local KRagAD resources."""

from pathlib import Path
import os


def _as_str(path: Path) -> str:
    return str(path.expanduser())


def project_root() -> str:
    return os.environ.get("KRAGAD_PROJECT_ROOT", _as_str(Path(__file__).resolve().parents[2]))


def data_root() -> str:
    return os.environ.get("KRAGAD_DATA_ROOT", _as_str(Path.home() / "data" / "kragad"))


def dataset_root() -> str:
    return os.environ.get("KRAGAD_DATASET_ROOT", _as_str(Path(data_root()) / "datasets" / "MMAD"))


def qa_root() -> str:
    return os.environ.get("KRAGAD_QA_ROOT", _as_str(Path(data_root()) / "datasets" / "QA_collection"))


def database_root() -> str:
    return os.environ.get(
        "KRAGAD_DATABASE_ROOT",
        _as_str(Path(data_root()) / "datasets" / "KRagAD" / "databases"),
    )


def graph_cache_root() -> str:
    return os.environ.get("KRAGAD_GRAPH_CACHE_ROOT", _as_str(Path(database_root()) / "graph_index"))


def sam3_path() -> str:
    return os.environ.get("KRAGAD_SAM3_PATH", _as_str(Path(data_root()) / "models" / "sam3" / "sam3.pt"))


def adaptclip_root() -> str:
    return os.environ.get("KRAGAD_ADAPTCLIP_ROOT", _as_str(Path(data_root()) / "models" / "AdaptCLIP"))


def abound_model_path() -> str:
    return os.environ.get("KRAGAD_ABOUND_MODEL_PATH", _as_str(Path(data_root()) / "models" / "ABounD" / "model"))


def abound_save_path() -> str:
    return os.environ.get(
        "KRAGAD_ABOUND_SAVE_PATH",
        _as_str(Path(data_root()) / "models" / "ABounD" / "vit336" / "336" / "shot4_CL"),
    )


def vlm_model_path(default_name: str = "qwen3-vl-8B") -> str:
    return os.environ.get("KRAGAD_VLM_MODEL_PATH", _as_str(Path(data_root()) / "models" / default_name))


def embedding_model_path() -> str:
    return os.environ.get(
        "KRAGAD_EMBEDDING_MODEL_PATH",
        _as_str(Path(data_root()) / "models" / "bge-base-en-v1.5"),
    )


def mpdd_root() -> str:
    return os.environ.get("KRAGAD_MPDD_ROOT", _as_str(Path(data_root()) / "datasets" / "MPDD"))
