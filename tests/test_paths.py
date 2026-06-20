from pathlib import Path

from kragad import paths


PATH_ENV_VARS = [
    "GLLS_PROJECT_ROOT",
    "GLLS_DATA_ROOT",
    "GLLS_DATASET_ROOT",
    "GLLS_QA_ROOT",
    "GLLS_DATABASE_ROOT",
    "GLLS_GRAPH_CACHE_ROOT",
    "GLLS_SAM3_PATH",
    "GLLS_ADAPTCLIP_ROOT",
    "GLLS_ABOUND_MODEL_PATH",
    "GLLS_ABOUND_SAVE_PATH",
    "GLLS_VLM_MODEL_PATH",
    "GLLS_EMBEDDING_MODEL_PATH",
    "GLLS_MPDD_ROOT",
    "KRAGAD_PROJECT_ROOT",
    "KRAGAD_DATA_ROOT",
    "KRAGAD_DATASET_ROOT",
    "KRAGAD_QA_ROOT",
    "KRAGAD_DATABASE_ROOT",
    "KRAGAD_GRAPH_CACHE_ROOT",
    "KRAGAD_SAM3_PATH",
    "KRAGAD_ADAPTCLIP_ROOT",
    "KRAGAD_ABOUND_MODEL_PATH",
    "KRAGAD_ABOUND_SAVE_PATH",
    "KRAGAD_VLM_MODEL_PATH",
    "KRAGAD_EMBEDDING_MODEL_PATH",
    "KRAGAD_MPDD_ROOT",
]


def test_default_paths_are_home_relative(monkeypatch):
    for env_var in PATH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    root = Path.home() / "data" / "GLLS"

    assert paths.data_root() == str(root)
    assert paths.dataset_root() == str(root / "datasets" / "MMAD")
    assert paths.database_root() == str(root / "datasets" / "GLLS" / "databases")
    assert paths.sam3_path() == str(root / "models" / "sam3" / "sam3.pt")


def test_environment_overrides_are_respected(monkeypatch, tmp_path):
    for env_var in PATH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    data_root = tmp_path / "glls-data"
    sam3 = tmp_path / "models" / "sam3.pt"

    monkeypatch.setenv("GLLS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("GLLS_SAM3_PATH", str(sam3))

    assert paths.data_root() == str(data_root)
    assert paths.dataset_root() == str(data_root / "datasets" / "MMAD")
    assert paths.sam3_path() == str(sam3)


def test_legacy_environment_overrides_still_work(monkeypatch, tmp_path):
    for env_var in PATH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    data_root = tmp_path / "legacy-kragad-data"
    monkeypatch.setenv("KRAGAD_DATA_ROOT", str(data_root))

    assert paths.data_root() == str(data_root)
    assert paths.dataset_root() == str(data_root / "datasets" / "MMAD")
