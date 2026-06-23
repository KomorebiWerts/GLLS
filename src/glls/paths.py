"""Centralized default paths for local GLLS resources."""

from pathlib import Path
import os


def _as_str(path: Path) -> str:
    return str(path.expanduser())


def _getenv(key: str, default: str) -> str:
    return os.environ.get(key) or default


def project_root() -> str:
    return _getenv("GLLS_PROJECT_ROOT", _as_str(Path(__file__).resolve().parents[2]))


def data_root() -> str:
    return _getenv("GLLS_DATA_ROOT", _as_str(Path.home() / "data" / "GLLS"))


def dataset_root() -> str:
    return _getenv("GLLS_DATASET_ROOT", _as_str(Path(data_root()) / "datasets" / "MMAD"))


def qa_root() -> str:
    configured = os.environ.get("GLLS_QA_ROOT")
    if configured:
        return configured
    bundled_qa = Path(project_root()) / "qa_collection"
    if bundled_qa.exists():
        return _as_str(bundled_qa)
    return _as_str(Path(data_root()) / "datasets" / "QA_collection")


def database_root() -> str:
    return _getenv("GLLS_DATABASE_ROOT", _as_str(Path(data_root()) / "databases"))


def graph_cache_root() -> str:
    return _getenv("GLLS_GRAPH_CACHE_ROOT", _as_str(Path(database_root()) / "graph_index"))


def sam3_path() -> str:
    return _getenv("GLLS_SAM3_PATH", _as_str(Path(data_root()) / "models" / "sam3" / "sam3.pt"))


def adaptclip_root() -> str:
    return _getenv("GLLS_ADAPTCLIP_ROOT", _as_str(Path(data_root()) / "models" / "AdaptCLIP"))


def abound_model_path() -> str:
    return _getenv("GLLS_ABOUND_MODEL_PATH", _as_str(Path(data_root()) / "models" / "ABounD" / "model"))


def abound_save_path() -> str:
    return _getenv(
        "GLLS_ABOUND_SAVE_PATH",
        _as_str(Path(data_root()) / "models" / "ABounD" / "vit336" / "336" / "shot4_CL"),
    )


def vlm_model_path(default_name: str = "qwen3-vl-8B") -> str:
    return _getenv("GLLS_VLM_MODEL_PATH", _as_str(Path(data_root()) / "models" / default_name))


def embedding_model_path() -> str:
    return _getenv(
        "GLLS_EMBEDDING_MODEL_PATH",
        _as_str(Path(data_root()) / "models" / "bge-base-en-v1.5"),
    )


def mpdd_root() -> str:
    return _getenv("GLLS_MPDD_ROOT", _as_str(Path(data_root()) / "datasets" / "MPDD"))


def dtd_root() -> str:
    return _getenv("GLLS_DTD_ROOT", _as_str(Path(data_root()) / "datasets" / "DTD"))


def dagm_root() -> str:
    return _getenv("GLLS_DAGM_ROOT", _as_str(Path(data_root()) / "datasets" / "DAGM_KaggleUpload"))


def binary_ad_root(dataset: str) -> str:
    key = str(dataset or "").strip().lower()
    if key == "mpdd":
        return mpdd_root()
    if key == "dtd":
        return dtd_root()
    if key in {"dagm", "dagm_kaggleupload", "dagm-kaggleupload"}:
        return dagm_root()
    return _as_str(Path(data_root()) / "datasets" / dataset)
