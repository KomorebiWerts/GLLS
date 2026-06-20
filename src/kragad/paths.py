"""Centralized default paths for local GLLS resources."""

from pathlib import Path
import os


def _as_str(path: Path) -> str:
    return str(path.expanduser())


def _getenv(primary: str, legacy: str, default: str) -> str:
    return os.environ.get(primary) or os.environ.get(legacy) or default


def project_root() -> str:
    return _getenv("GLLS_PROJECT_ROOT", "KRAGAD_PROJECT_ROOT", _as_str(Path(__file__).resolve().parents[2]))


def data_root() -> str:
    return _getenv("GLLS_DATA_ROOT", "KRAGAD_DATA_ROOT", _as_str(Path.home() / "data" / "GLLS"))


def dataset_root() -> str:
    return _getenv("GLLS_DATASET_ROOT", "KRAGAD_DATASET_ROOT", _as_str(Path(data_root()) / "datasets" / "MMAD"))


def qa_root() -> str:
    return _getenv("GLLS_QA_ROOT", "KRAGAD_QA_ROOT", _as_str(Path(data_root()) / "datasets" / "QA_collection"))


def database_root() -> str:
    return _getenv(
        "GLLS_DATABASE_ROOT",
        "KRAGAD_DATABASE_ROOT",
        _as_str(Path(data_root()) / "datasets" / "GLLS" / "databases"),
    )


def graph_cache_root() -> str:
    return _getenv("GLLS_GRAPH_CACHE_ROOT", "KRAGAD_GRAPH_CACHE_ROOT", _as_str(Path(database_root()) / "graph_index"))


def sam3_path() -> str:
    return _getenv("GLLS_SAM3_PATH", "KRAGAD_SAM3_PATH", _as_str(Path(data_root()) / "models" / "sam3" / "sam3.pt"))


def adaptclip_root() -> str:
    return _getenv("GLLS_ADAPTCLIP_ROOT", "KRAGAD_ADAPTCLIP_ROOT", _as_str(Path(data_root()) / "models" / "AdaptCLIP"))


def abound_model_path() -> str:
    return _getenv("GLLS_ABOUND_MODEL_PATH", "KRAGAD_ABOUND_MODEL_PATH", _as_str(Path(data_root()) / "models" / "ABounD" / "model"))


def abound_save_path() -> str:
    return _getenv(
        "GLLS_ABOUND_SAVE_PATH",
        "KRAGAD_ABOUND_SAVE_PATH",
        _as_str(Path(data_root()) / "models" / "ABounD" / "vit336" / "336" / "shot4_CL"),
    )


def vlm_model_path(default_name: str = "qwen3-vl-8B") -> str:
    return _getenv("GLLS_VLM_MODEL_PATH", "KRAGAD_VLM_MODEL_PATH", _as_str(Path(data_root()) / "models" / default_name))


def embedding_model_path() -> str:
    return _getenv(
        "GLLS_EMBEDDING_MODEL_PATH",
        "KRAGAD_EMBEDDING_MODEL_PATH",
        _as_str(Path(data_root()) / "models" / "bge-base-en-v1.5"),
    )


def mpdd_root() -> str:
    return _getenv("GLLS_MPDD_ROOT", "KRAGAD_MPDD_ROOT", _as_str(Path(data_root()) / "datasets" / "MPDD"))
