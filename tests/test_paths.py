from pathlib import Path

from kragad import paths


PATH_ENV_VARS = [
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

    root = Path.home() / "data" / "kragad"

    assert paths.data_root() == str(root)
    assert paths.dataset_root() == str(root / "datasets" / "MMAD")
    assert paths.database_root() == str(root / "datasets" / "KRagAD" / "databases")
    assert paths.sam3_path() == str(root / "models" / "sam3" / "sam3.pt")


def test_environment_overrides_are_respected(monkeypatch, tmp_path):
    for env_var in PATH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    data_root = tmp_path / "kragad-data"
    sam3 = tmp_path / "models" / "sam3.pt"

    monkeypatch.setenv("KRAGAD_DATA_ROOT", str(data_root))
    monkeypatch.setenv("KRAGAD_SAM3_PATH", str(sam3))

    assert paths.data_root() == str(data_root)
    assert paths.dataset_root() == str(data_root / "datasets" / "MMAD")
    assert paths.sam3_path() == str(sam3)
