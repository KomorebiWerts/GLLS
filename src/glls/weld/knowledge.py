from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SOURCE_DOCUMENTS = {
    "discussion": "焊缝缺陷检测讨论.pdf",
    "upgrade": "焊缝缺陷检测升级.pdf",
}


def build_weld_knowledge() -> dict[str, Any]:
    """Return the source-backed weld inspection knowledge in GLLS graph schema."""
    return {
        "target_object": "gear_weld",
        "regions": {
            "station1_full_weld_ring": {
                "definition": (
                    "The complete annular black weld region in the station-1 full-face image, "
                    "including its inner and outer boundaries."
                ),
                "normal_standard": (
                    "The weld is a complete closed ring with continuous boundaries and broadly "
                    "uniform radial thickness. Transparent bright oil reflections are normal and "
                    "must not be confused with the opaque dark weld region."
                ),
                "critical_check": (
                    "Check ring closure, boundary continuity, radial thickness, outward bulges, "
                    "and whether weld material exceeds the permitted mirror boundary."
                ),
                "defects": [
                    {
                        "type": "incomplete_weld",
                        "visual_signature": (
                            "A gap, break, or missing segment interrupts the otherwise continuous "
                            "dark annular weld."
                        ),
                        "contrast_vs_normal": "A normal weld forms a closed uninterrupted ring.",
                        "visual_attributes": ["gap", "broken_boundary", "missing_material", "ring_discontinuity"],
                        "distinctions": [
                            {
                                "target_defect": "uneven_weld",
                                "difference": (
                                    "Incomplete weld removes a segment and breaks continuity; uneven weld "
                                    "retains continuity but changes thickness."
                                ),
                            }
                        ],
                    },
                    {
                        "type": "uneven_weld",
                        "visual_signature": (
                            "The annular weld remains present but alternates between abnormally thin "
                            "and thick portions."
                        ),
                        "contrast_vs_normal": "A normal weld has stable radial thickness around the ring.",
                        "visual_attributes": ["thickness_change", "thin_section", "thick_section"],
                        "distinctions": [
                            {
                                "target_defect": "incomplete_weld",
                                "difference": "Uneven weld remains connected; incomplete weld contains a true gap.",
                            }
                        ],
                    },
                    {
                        "type": "excess_weld_or_large_bulge",
                        "visual_signature": (
                            "A large local mass protrudes outward from the expected annular boundary, "
                            "or weld material extends beyond the permitted mirror radius."
                        ),
                        "contrast_vs_normal": "A normal outer weld boundary follows a smooth circular envelope.",
                        "visual_attributes": ["outward_protrusion", "extra_material", "boundary_deformation"],
                        "distinctions": [],
                    },
                    {
                        "type": "multiple_connection_protrusions",
                        "visual_signature": (
                            "More than one weld connection protrusion is visible in a station-1 image. "
                            "The upgraded station-1 rule treats a count greater than one as abnormal."
                        ),
                        "contrast_vs_normal": "At most one expected connection protrusion is allowed at station 1.",
                        "visual_attributes": ["component_count", "connection_protrusion", "extra_structure"],
                        "distinctions": [],
                    },
                ],
                "anti_hallucination_rules": [
                    "Do not classify transparent bright oil reflection as an opaque dark weld defect.",
                    "Do not use gear center offset as the primary visual anomaly when the station sensor owns alignment checking.",
                    "Confirm a suspected gap or bulge on the weld ring itself, not on surrounding gear teeth or fixtures.",
                ],
            },
            "station2_weld_arc": {
                "definition": (
                    "The real and mirrored curved weld areas visible in one station-2 quarter-camera image."
                ),
                "normal_standard": (
                    "The visible weld arcs are continuous and locally consistent. A labeled connection "
                    "protrusion (ljtq) is recorded for filtering and is not by itself an abnormal station-2 result."
                ),
                "critical_check": (
                    "Inspect only the weld and its allowed surrounding zone for missing weld, spatter, "
                    "and black-water-like contamination; report defect class and location."
                ),
                "defects": [
                    {
                        "type": "missing_weld",
                        "visual_signature": (
                            "A compact gap or absent weld patch occurs inside the expected weld arc. "
                            "Arc endpoints and manually scraped areas are known false-positive risks."
                        ),
                        "contrast_vs_normal": "Normal weld texture continues through the expected arc.",
                        "visual_attributes": ["gap", "missing_material", "local_discontinuity"],
                        "distinctions": [
                            {
                                "target_defect": "normal_connection_protrusion_context",
                                "difference": (
                                    "Missing weld removes expected material; the connection protrusion adds a broad "
                                    "continuous mass and is a filtering context rather than a station-2 defect."
                                ),
                            }
                        ],
                    },
                    {
                        "type": "spatter",
                        "visual_signature": (
                            "One or more small localized bead-like or dot-like deposits appear on or near the weld. "
                            "Area, width, and height thresholds should reject insignificant marks."
                        ),
                        "contrast_vs_normal": "Normal surrounding surfaces do not contain isolated weld deposits.",
                        "visual_attributes": ["small_blob", "dot", "extra_material", "localized_deposit"],
                        "distinctions": [
                            {
                                "target_defect": "black_water",
                                "difference": "Spatter is material with a compact bead shape; black water is a stain-like dark region.",
                            }
                        ],
                    },
                    {
                        "type": "black_water",
                        "visual_signature": (
                            "A dark stain-like region resembles black liquid or residue. Size filtering is required "
                            "because similar process appearance can be acceptable."
                        ),
                        "contrast_vs_normal": "Normal variation lacks a stable dark stain exceeding the configured size threshold.",
                        "visual_attributes": ["dark_stain", "residue", "color_change", "irregular_patch"],
                        "distinctions": [
                            {
                                "target_defect": "spatter",
                                "difference": "Black water is stain-like and spread; spatter is a raised compact deposit.",
                            }
                        ],
                    },
                    {
                        "type": "normal_connection_protrusion_context",
                        "visual_signature": (
                            "A broad elongated weld connection protrusion marked as ljtq. It must be detected or masked "
                            "for context, but the upgraded station-2 rule says it is not an anomaly by itself."
                        ),
                        "contrast_vs_normal": (
                            "This is an allowed structural context at station 2, unlike a compact missing-weld gap or spatter dot."
                        ),
                        "visual_attributes": ["elongated_region", "connection", "allowed_context"],
                        "distinctions": [],
                    },
                ],
                "anti_hallucination_rules": [
                    "Do not label an ljtq-only image abnormal under the upgraded station-2 rule.",
                    "Do not treat arc endpoints as missing weld without evidence inside the valid weld span.",
                    "Do not treat manually scraped spatter marks as missing weld without confirming absent weld material.",
                    "Apply size filtering to very small spatter and black-water-like marks.",
                ],
            },
            "mirror_and_surrounding_surface": {
                "definition": "The mirror boundary and nearby metal surface around the weld.",
                "normal_standard": (
                    "Reflections and mild process discoloration may occur, but no weld mass should exceed the configured "
                    "mirror limit and no significant spatter should remain in the allowed inspection zone."
                ),
                "critical_check": "Separate reflection, acceptable discoloration, and fixtures from physical weld deposits.",
                "defects": [
                    {
                        "type": "out_of_limit_weld_bulge",
                        "visual_signature": "Weld material crosses the configured mirror-radius boundary.",
                        "contrast_vs_normal": "The normal weld remains inside the allowed circular boundary.",
                        "visual_attributes": ["boundary_crossing", "protrusion", "extra_material"],
                        "distinctions": [],
                    },
                    {
                        "type": "external_spatter",
                        "visual_signature": "A compact weld deposit lies outside the main weld arc but inside the inspected zone.",
                        "contrast_vs_normal": "The surrounding metal surface is free of significant weld deposits.",
                        "visual_attributes": ["external_blob", "deposit", "extra_material"],
                        "distinctions": [],
                    },
                ],
                "anti_hallucination_rules": [
                    "Reflection alone is not evidence of extra weld material.",
                    "Use geometric size and boundary checks after semantic detection.",
                ],
            },
        },
    }


def build_weld_provenance() -> dict[str, Any]:
    return {
        "schema": "glls_weld_knowledge_provenance_v1",
        "documents": [
            {
                "file": SOURCE_DOCUMENTS["discussion"],
                "pages": [1, 2, 3, 4],
                "supports": [
                    "station layout",
                    "complete ring and uniformity rules",
                    "incomplete weld and weld scar definitions",
                    "98 percent defect-workpiece detection target",
                ],
            },
            {
                "file": SOURCE_DOCUMENTS["upgrade"],
                "pages": [1, 5, 6, 7, 8, 9, 11, 12],
                "supports": [
                    "connection protrusion handling",
                    "missing weld, spatter, and black water classes",
                    "confidence and geometric thresholding",
                    "known false-positive and false-negative risks",
                ],
            },
        ],
        "annotation_mapping": {
            "ljtq": {
                "meaning": "connection protrusion",
                "binary_role": "normal_context_for_station2",
                "confidence": "high",
            },
            "dx": {
                "meaning": "small dot-like weld defect/spatter annotation",
                "binary_role": "anomaly",
                "confidence": "medium",
                "note": "The source archive does not include a label dictionary; visual geometry and upgraded defect rules support this mapping.",
            },
        },
    }


def write_weld_knowledge(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    knowledge_path = output_dir / "gear_weld.json"
    provenance_path = output_dir / "gear_weld_provenance.json"
    knowledge_path.write_text(json.dumps(build_weld_knowledge(), ensure_ascii=False, indent=2), encoding="utf-8")
    provenance_path.write_text(json.dumps(build_weld_provenance(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"knowledge_path": str(knowledge_path), "provenance_path": str(provenance_path)}
