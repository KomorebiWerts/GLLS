from pathlib import Path

import pytest

from glls.eval.binary_ad import _lookup_threshold_table, load_binary_ad_threshold_table
from glls.models.adaptclip_config import load_adaptclip_model_config
from glls.models.localizer import _abound_model_config, _threshold_tables


def test_adaptclip_model_config_supplies_binary_ad_thresholds_by_default():
    table = load_binary_ad_threshold_table("")

    assert table["schema"] == "glls_adaptclip_binary_ad_threshold_table_v1"
    assert table["source"] == "adaptclip_model_config:decision_parameters.binary_ad"

    mpdd_threshold, mpdd_source = _lookup_threshold_table(
        table,
        dataset="mpdd",
        category="bracket_black",
        localizer="adaptclip",
        shot=0,
    )
    assert mpdd_threshold == pytest.approx(0.8194127655029297)
    assert mpdd_source == "adaptclip_model_config:decision_parameters.binary_ad"

    dtd_threshold, _ = _lookup_threshold_table(
        table,
        dataset="dtd",
        category="Woven_125",
        localizer="adaptclip",
        shot=1,
    )
    assert dtd_threshold == pytest.approx(0.24601564537882806)


def test_adaptclip_model_config_names_thresholds_as_decision_parameters():
    config = load_adaptclip_model_config()

    assert "thresholds_by_shot" not in config
    assert "binary_ad_thresholds_by_shot" not in config
    decision_parameters = config["decision_parameters"]
    assert "heatmap_by_shot" in decision_parameters
    assert "binary_ad" in decision_parameters
    assert decision_parameters["binary_ad"]["cutoff_policy"] == "normal_robust_snapshot"
    assert decision_parameters["binary_ad"]["normal_source_compliance"] == {
        "source_split": "train/good normal images",
        "test_annotations_used": False,
        "defect_labels_used": False,
    }


def test_abound_model_config_uses_normal_sourced_one_shot_decision_parameters():
    config = _abound_model_config("")

    assert "thresholds" not in config
    heatmap = config["decision_parameters"]["heatmap"]
    assert heatmap["shot"] == 1
    assert heatmap["normal_source_compliance"] == {
        "source_split": "train/good normal images",
        "test_annotations_used": False,
        "defect_labels_used": False,
    }

    image_thresholds, pixel_thresholds = _threshold_tables(config)
    assert image_thresholds["bottle"] == pytest.approx(0.9793)
    assert image_thresholds["pipe_fryum"] == pytest.approx(0.8505)
    assert pixel_thresholds["leather"] == pytest.approx(1.09)
    assert pixel_thresholds["pipe_fryum"] == pytest.approx(0.7478)


def test_run_binary_ad_wrapper_defaults_to_full_method_guardrails():
    script = Path("scripts/run/run_binary_ad.sh").read_text(encoding="utf-8")

    assert "GLLS_BINARY_AD_LIGHTWEIGHT" in script
    assert "AdaptCLIP model_config.json" in script
    assert "adaptclip_${shot}shot_mvtec_thresholds.json" not in script
    assert "--threshold_policy table" in script
    assert "--binary_score_source localizer_image" in script
    assert "--final_verifier qwen3" in script
    assert "--final_verifier_policy anomaly_or" in script
    assert "--qwen3_max_crops 3" in script


def test_qwen3_sweep_uses_paper_comparable_binary_ad_arguments():
    script = Path("scripts/eval/run_binary_ad_qwen3_sweep.py").read_text(encoding="utf-8")

    assert "glls.cli.binary_ad" in script
    assert 'parser.add_argument("--threshold_table", default="")' in script
    assert '"--threshold_policy"' in script
    assert '"table"' in script
    assert '"--binary_score_source"' in script
    assert '"localizer_image"' in script
    assert '"--final_verifier"' in script
    assert '"qwen3"' in script
    assert '"--final_verifier_policy"' in script
    assert "anomaly_or" in script
