from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def is_valid_content(value: Any) -> bool:
    if not value:
        return False
    if not isinstance(value, str):
        return True
    return value.strip().upper() != "N/A"


@dataclass(frozen=True)
class PVLARegionContext:
    region: str
    text: str
    image_paths: list[str]
    topology: dict[str, Any]


class PVLARegionContextBuilder:
    """Build prompt/audit context for one PVLA graph region."""

    def __init__(self, *, k_shot: int = 1):
        self.k_shot = max(0, int(k_shot or 0))

    def build(self, rag: Any, region: str) -> PVLARegionContext:
        info = rag.get_inspection_checklist(region)
        image_paths = self._select_existing_images(rag.get_image_paths(region))
        return PVLARegionContext(
            region=region,
            text=self.format_region_text(region, info),
            image_paths=image_paths,
            topology={
                "target_object": rag.root_name,
                "region": region,
                "defects": rag.iter_region_defects([region]),
            },
        )

    def _select_existing_images(self, image_paths: list[str] | tuple[str, ...] | None) -> list[str]:
        selected = []
        for path in list(image_paths or [])[: self.k_shot]:
            if os.path.exists(str(path)):
                selected.append(str(path))
        return selected

    @staticmethod
    def format_region_text(region: str, info: dict[str, Any]) -> str:
        lines = [f"--- Reference Knowledge for Region: '{region}' ---"]

        if is_valid_content(info.get("definition")):
            lines.append(f"Definition: {info['definition']}")

        if is_valid_content(info.get("normal_standard")):
            lines.append(f"Normal Standard: {info['normal_standard']}")

        if is_valid_content(info.get("critical_check")):
            lines.append(f"Critical Check: {info['critical_check']}")

        raw_defects = info.get("defects", [])
        if raw_defects:
            lines.append("\n[Potential Defects Details]:")
            for defect in raw_defects:
                if isinstance(defect, dict):
                    lines.extend(PVLARegionContextBuilder._format_defect_dict(defect))
                else:
                    lines.append(f"- {str(defect)}")

        if "anti_hallucination_rules" in info:
            lines.append("\n[Anti-Hallucination Rules]:")
            for rule in info["anti_hallucination_rules"]:
                if is_valid_content(rule):
                    lines.append(f"!!! {rule}")

        return "\n".join(lines)

    @staticmethod
    def _format_defect_dict(defect: dict[str, Any]) -> list[str]:
        lines = []
        defect_name = defect.get("type", defect.get("name", "Unknown Defect"))
        lines.append(f"\n>>> Defect: {defect_name}")

        if is_valid_content(defect.get("visual_signature")):
            lines.append(f"    Visual Signature: {defect['visual_signature']}")

        if is_valid_content(defect.get("visual_appearance")):
            lines.append(f"    Visual Appearance: {defect['visual_appearance']}")

        if is_valid_content(defect.get("contrast_vs_normal")):
            lines.append(f"    Contrast vs Normal: {defect['contrast_vs_normal']}")

        distinctions = defect.get("distinctions")
        if isinstance(distinctions, list):
            for distinction in distinctions:
                if not isinstance(distinction, dict):
                    continue
                diff = distinction.get("difference", "N/A")
                if is_valid_content(diff):
                    lines.append(f"    * Distinction vs {distinction.get('target_defect')}: {diff}")
        return lines
