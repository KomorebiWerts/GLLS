"""Adapter that exposes the weld method through the shared GLLS demo UI."""

from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from glls.weld.data import load_weld_manifest, weld_annotation_reference
from glls.weld.demo import DEFAULT_LOCAL_ALERT_THRESHOLD, DEFAULT_TILE_SIZE, run_weld_orchestra
from glls.weld.pvla_prior import WeldPVLANormalPrior


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "焊缝缺陷检测数据" / "prepared_glls"
DEFAULT_MANIFEST = Path(
    os.environ.get("GLLS_WELD_MANIFEST", DEFAULT_PREPARED_ROOT / "manifest.json")
).expanduser()
WELD_DATASET = "weld"
WELD_CATEGORY = "gear_weld"
WELD_TASK_PREFIX = "weld::"
WELD_QUESTION = (
    "Is this gear weld anomalous? Inspect global weld continuity and local defect evidence."
)
WELD_OPTIONS = {
    "A": "OK — no global or local anomaly trigger",
    "B": "NG — global logic or local anomaly trigger",
}
WELD_DEMO_SAMPLES = {
    "Image_20250623145203392": "Demo A · annotated local dx anomaly",
    "Image_20250731104322556": "Demo B · logic-detected fragmented ring (unscored)",
    "Image_20250731132219446": "Demo C · logic-detected ring gap (unscored)",
}


class WeldFrontendRuntime:
    """Own the one-category weld runtime and its demo sample catalogue."""

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST) -> None:
        self.manifest_path = Path(manifest_path).expanduser()
        self._manifest: dict[str, Any] | None = None
        self.localizer: Any | None = None
        self.sam_engine: Any | None = None
        self.pvla_prior: WeldPVLANormalPrior | None = None
        self.device = ""
        self.runtime_signature: tuple[str, str, str] | None = None
        self.lock = threading.Lock()

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(
                    f"Weld manifest not found: {self.manifest_path}. Run weld_ad --stage prepare first."
                )
            self._manifest = load_weld_manifest(self.manifest_path)
        return self._manifest

    @property
    def root(self) -> Path:
        return Path(self.manifest["output_root"]).expanduser().resolve()

    @property
    def is_ready(self) -> bool:
        return self.localizer is not None and self.pvla_prior is not None

    def sample_choices(self, sample_group: str = "All", search_query: str = "") -> list[tuple[str, str]]:
        group = str(sample_group or "All").lower()
        query = str(search_query or "").strip().lower()
        ranked_choices: list[tuple[int, str, str]] = []
        groups = [
            ("annotated", self.manifest.get("station2") or []),
            ("unlabeled", self.manifest.get("station1") or []),
        ]
        for group_name, rows in groups:
            for row in rows:
                reference = weld_annotation_reference(row)
                role = reference["role"]
                if group not in {"all", group_name, role}:
                    continue
                labels = ",".join(row.get("shape_labels") or [])
                search_text = f"{row.get('id', '')} {labels} {group_name} {role} {reference['display']}".lower()
                if query and query not in search_text:
                    continue
                key = f"{group_name}::{row['id']}::{labels or 'no-label'}"
                demo_label = WELD_DEMO_SAMPLES.get(row["id"], "")
                label = (
                    f"{reference['display']} · {row['id']} · {labels or 'no region label'}"
                    if group_name == "annotated"
                    else f"Unlabelled review · {row['id']}"
                )
                if demo_label:
                    label = f"{demo_label} · {label}"
                rank = list(WELD_DEMO_SAMPLES).index(row["id"]) if demo_label else len(WELD_DEMO_SAMPLES)
                ranked_choices.append((rank, label, WELD_TASK_PREFIX + key))
        ranked_choices.sort(key=lambda item: (item[0], item[1]))
        return [(label, token) for _, label, token in ranked_choices]

    def sample_from_token(self, token: str) -> dict[str, Any] | None:
        key = str(token or "")
        if key.startswith(WELD_TASK_PREFIX):
            key = key[len(WELD_TASK_PREFIX):]
        parts = key.split("::")
        if len(parts) < 2 or parts[0] not in {"annotated", "unlabeled"}:
            return None
        rows = self.manifest.get("station2") if parts[0] == "annotated" else self.manifest.get("station1")
        for row in rows or []:
            if row.get("id") == parts[1]:
                return {**row, "_root": str(self.root)}
        return None

    def sample_payload(self, token: str) -> dict[str, Any]:
        sample = self.sample_from_token(token)
        if not sample:
            return {"image": None, "question": "", "task": "", "ground_truth": "", "options": {}}
        image_path = self.root / sample["image_path"]
        with Image.open(image_path) as image:
            preview = image.convert("RGB")
        station = int(sample.get("station") or 0)
        labels = ", ".join(sample.get("shape_labels") or []) or "none"
        reference = weld_annotation_reference(sample)
        return {
            "image": preview,
            "question": WELD_QUESTION,
            "task": str(token),
            "ground_truth": reference["ground_truth"],
            "options": dict(WELD_OPTIONS),
            "summary": (
                f"station{station} | {reference['display']} | region labels: {labels} | "
                f"{reference['reason']}"
            ),
        }

    def normal_reference_paths(self) -> list[str]:
        paths: list[Path] = []
        assets_path = self.root / "assets" / "assets_manifest.json"
        if assets_path.exists():
            assets = json.loads(assets_path.read_text(encoding="utf-8"))
            values = assets.get("references_by_region", {}).get("station1_full_weld_ring", [])
            paths.extend(self._resolve_path(value) for value in values)
        normal_reference = (self.manifest.get("normal_reference") or {}).get("path")
        if normal_reference:
            paths.append(self._resolve_path(normal_reference))
        unique: list[str] = []
        for path in paths:
            if path.exists() and str(path) not in unique:
                unique.append(str(path))
        return unique

    def initialize(self, gpu_id: str, checkpoint_path: str, sam_path: str = "") -> str:
        with self.lock:
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(f"AdaptCLIP checkpoint not found: {checkpoint}")
            resolved_sam = Path(sam_path).expanduser().resolve() if sam_path else None
            references = self.normal_reference_paths()
            if not references:
                raise FileNotFoundError(
                    "The trusted PDF normal reference is missing. Run weld_ad --stage assets first."
                )
            support = Path(references[0])
            graph_path = self.root / "assets" / "graph" / "gear_weld_graph.pkl"
            if not graph_path.exists():
                raise FileNotFoundError(
                    f"Weld PVLA graph not found: {graph_path}. Run weld_ad --stage assets first."
                )

            import torch

            from glls.models.localizer import AdaptCLIP_Localizer

            gpu = int(str(gpu_id).strip())
            if torch.cuda.is_available():
                if gpu < 0 or gpu >= torch.cuda.device_count():
                    raise ValueError(
                        f"GPU {gpu} is not visible; PyTorch sees {torch.cuda.device_count()} CUDA device(s)."
                    )
                torch.cuda.set_device(gpu)
                device = f"cuda:{gpu}"
            else:
                device = "cpu"

            signature = (device, str(checkpoint), str(resolved_sam or ""))
            if self.is_ready and self.runtime_signature == signature:
                return self._ready_status()

            self._clear_components()
            try:
                args = SimpleNamespace(
                    dataset=WELD_DATASET,
                    image_size=518,
                    k_shot=1,
                    checkpoint_path=str(checkpoint),
                    save_path="",
                )
                self.localizer = AdaptCLIP_Localizer(
                    args,
                    device=device,
                    pretrained_model="ViT-L/14@336px",
                )
                self.localizer.configure_text_prompts(
                    "gear weld seam",
                    normal_states=[
                        "{}",
                        "complete continuous {}",
                        "uniform annular {}",
                        "{} without a missing segment",
                        "{} without spatter",
                    ],
                    anomaly_states=[
                        "incomplete {}",
                        "{} with a missing weld gap",
                        "{} with weld spatter",
                        "uneven damaged {}",
                        "{} with a dark contamination defect",
                    ],
                )
                self.localizer.configure_support(WELD_CATEGORY, [str(support)])
                self.pvla_prior = WeldPVLANormalPrior(
                    self.localizer,
                    graph_path=graph_path,
                    support_paths=[support],
                    patch_size=384,
                    max_reference_patches=48,
                )
                self.sam_engine = None
                if resolved_sam and resolved_sam.exists():
                    from glls.seg.sam3_engine import Sam3Engine

                    self.sam_engine = Sam3Engine(str(resolved_sam), device=device)
                self.device = device
                self.runtime_signature = signature
                return self._ready_status()
            except Exception:
                self._clear_components()
                raise

    def inspect(self, sample_token: str) -> dict[str, Any]:
        with self.lock:
            if not self.is_ready:
                raise RuntimeError("Initialize the weld runtime before running GLLS.")
            sample = self.sample_from_token(sample_token)
            if not sample:
                raise ValueError("Select a weld sample before running GLLS.")
            image_path = self.root / sample["image_path"]
            output_dir = PROJECT_ROOT / "outputs" / "weld_demo_runs" / image_path.stem
            return run_weld_orchestra(
                image_path,
                localizer=self.localizer,
                sam_engine=self.sam_engine,
                pvla_prior=self.pvla_prior,
                output_dir=output_dir,
                sample=sample,
                tile_size=DEFAULT_TILE_SIZE,
                overlap=0.25,
                local_threshold=DEFAULT_LOCAL_ALERT_THRESHOLD,
            )

    def clear(self) -> None:
        with self.lock:
            self._clear_components()

    def _ready_status(self) -> str:
        patch_count = len(self.pvla_prior.reference_embeddings) if self.pvla_prior is not None else 0
        return (
            f"System ready | GPU:{self.device} | Dataset:weld | Category:gear_weld | "
            f"AdaptCLIP 1-shot | PVLA:{patch_count} patches | "
            f"SAM3:{'ON' if self.sam_engine is not None else 'fallback'} | QA:qualitative"
        )

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def _clear_components(self) -> None:
        self.localizer = None
        self.sam_engine = None
        self.pvla_prior = None
        self.device = ""
        self.runtime_signature = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
