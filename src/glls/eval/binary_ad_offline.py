from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from PIL import Image

from glls import paths as glls_paths
from glls.eval.binary_ad import BINARY_AD_DATASETS, BinaryADSample, build_binary_ad_samples
from glls.rag.source_chain import build_graph_source_metadata


PVLA_ROOT_REGION_NAME = "whole_object"


@dataclass(frozen=True)
class BinaryADRegionSpec:
    name: str
    definition: str
    normal_standard: str
    critical_check: str
    visual_attributes: tuple[str, ...]
    prompt: str = ""
    crop_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class BinaryADOfflineRoots:
    text_knowledge_root: Path
    visual_reference_root: Path
    graph_output_root: Path
    manifest_path: Path


def default_binary_ad_offline_roots(database_root: Path | None = None) -> BinaryADOfflineRoots:
    root = Path(database_root or glls_paths.database_root()).expanduser()
    return BinaryADOfflineRoots(
        text_knowledge_root=root / "text_knowledge",
        visual_reference_root=root / "img",
        graph_output_root=root / "graph_index_binary_ad",
        manifest_path=root / "binary_ad_offline_manifest.json",
    )


def prepare_binary_ad_offline_assets(
    *,
    dataset_roots: dict[str, Path],
    database_root: Path | None = None,
    datasets: Iterable[str] = BINARY_AD_DATASETS,
    max_refs: int = 4,
    sam_engine: Any | None = None,
) -> dict[str, Any]:
    roots = default_binary_ad_offline_roots(database_root)
    dataset_rows = []
    for dataset in datasets:
        key = str(dataset).strip().lower()
        if key not in BINARY_AD_DATASETS:
            raise ValueError(f"Unsupported binary AD dataset: {dataset}")
        dataset_root = Path(dataset_roots[key]).expanduser()
        train_samples, test_samples = build_binary_ad_samples(dataset=key, root=dataset_root)
        dataset_rows.append(
            _prepare_dataset_assets(
                dataset=key,
                dataset_root=dataset_root,
                train_samples=train_samples,
                test_samples=test_samples,
                roots=roots,
                max_refs=max_refs,
                sam_engine=sam_engine,
            )
        )

    manifest = {
        "schema": "glls_binary_ad_offline_assets_v2",
        "method_alignment": "Offline PVLA/SAM3 normal-reference assets prepared before online dual-stream binary AD evaluation.",
        "test_annotations_used_for_offline_assets": False,
        "pvla_region_policy": "dataset_profiled_regions_mpdd_parts_dtd_dagm_texture_surfaces",
        "text_knowledge_root": str(roots.text_knowledge_root),
        "visual_reference_root": str(roots.visual_reference_root),
        "graph_output_root": str(roots.graph_output_root),
        "datasets": dataset_rows,
    }
    _write_json(roots.manifest_path, manifest)
    return manifest


def load_binary_ad_offline_manifest(manifest_path: Path | None = None, *, database_root: Path | None = None) -> dict[str, Any]:
    roots = default_binary_ad_offline_roots(database_root)
    path = Path(manifest_path or roots.manifest_path).expanduser()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def offline_category_assets(manifest: dict[str, Any] | None, *, dataset: str, category: str) -> dict[str, Any] | None:
    for dataset_row in (manifest or {}).get("datasets", []):
        if str(dataset_row.get("dataset") or "").lower() != str(dataset).lower():
            continue
        for category_row in dataset_row.get("categories", []):
            if str(category_row.get("category") or "") == str(category):
                return category_row
    return None


def _prepare_dataset_assets(
    *,
    dataset: str,
    dataset_root: Path,
    train_samples: list[BinaryADSample],
    test_samples: list[BinaryADSample],
    roots: BinaryADOfflineRoots,
    max_refs: int,
    sam_engine: Any | None,
) -> dict[str, Any]:
    train_by_category = _group_by_category(train_samples)
    test_by_category = _group_by_category(test_samples)
    category_rows = []
    for category in sorted(set(train_by_category) | set(test_by_category)):
        normal_refs = train_by_category.get(category, [])[: max(1, int(max_refs))]
        region_specs = _build_category_region_profile(dataset=dataset, category=category)
        category_root = roots.visual_reference_root / dataset / category
        reference_rows = _write_visual_references(
            normal_refs,
            output_root=category_root,
            region_specs=region_specs,
            sam_engine=sam_engine,
        )
        text_json = roots.text_knowledge_root / dataset / f"{category}.json"
        knowledge = _build_text_knowledge(
            dataset=dataset,
            category=category,
            train_normal_count=len(train_by_category.get(category, [])),
            region_specs=region_specs,
        )
        _write_json(text_json, knowledge)
        graph_path = roots.graph_output_root / dataset / f"{category}_graph.pkl"
        reference_paths_by_region = _reference_paths_by_region(reference_rows)
        _write_graph_pickle(
            knowledge=knowledge,
            graph_path=graph_path,
            source_metadata=build_graph_source_metadata(
                dataset=dataset,
                category=category,
                source_json_path=str(text_json),
                text_knowledge_root=str(roots.text_knowledge_root / dataset),
                visual_reference_root=str(roots.visual_reference_root / dataset),
                graph_output_root=str(roots.graph_output_root / dataset),
                visual_reference_builder="glls.eval.binary_ad_offline",
                visual_reference_max_k_shot=max_refs,
            ),
            reference_paths_by_region=reference_paths_by_region,
        )
        category_rows.append({
            "category": category,
            "text_knowledge_path": str(text_json),
            "visual_reference_dir": str(category_root),
            "graph_path": str(graph_path),
            "pvla_root_region": PVLA_ROOT_REGION_NAME,
            "pvla_regions": [region.name for region in region_specs],
            "pvla_region_count": len(region_specs),
            "train_normal_total": len(train_by_category.get(category, [])),
            "offline_reference_count": len(reference_rows),
            "offline_reference_paths": [row["path"] for row in reference_rows],
            "offline_reference_paths_by_region": reference_paths_by_region,
            "sam3_reference_audit": reference_rows,
        })

    return {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "category_count": len(category_rows),
        "categories": category_rows,
    }


def _write_visual_references(
    samples: list[BinaryADSample],
    *,
    output_root: Path,
    region_specs: list[BinaryADRegionSpec],
    sam_engine: Any | None,
) -> list[dict[str, Any]]:
    rows = []
    for idx, sample in enumerate(samples):
        shot_dir = output_root / f"{idx:03d}"
        for region in region_specs:
            out_path = shot_dir / f"{region.name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            audit = {
                "source_image_path": sample.image_path,
                "path": str(out_path),
                "region": region.name,
                "region_definition": region.definition,
                "source_backed": True,
            }
            audit.update(_write_region_reference(sample.image_path, out_path, region=region, sam_engine=sam_engine))
            rows.append(audit)
    return rows


def _write_region_reference(
    image_path: str,
    out_path: Path,
    *,
    region: BinaryADRegionSpec,
    sam_engine: Any | None,
) -> dict[str, Any]:
    if sam_engine is not None and region.name == PVLA_ROOT_REGION_NAME:
        return _write_sam3_whole_object_cutout(image_path, out_path, sam_engine=sam_engine)
    if sam_engine is not None and region.prompt:
        return _write_sam3_text_cutout(
            image_path,
            out_path,
            sam_engine=sam_engine,
            prompt=region.prompt,
            fallback_crop_box=region.crop_box,
        )
    if region.crop_box is not None:
        status = _write_normalized_crop(image_path, out_path, crop_box=region.crop_box)
        status["sam3_status"] = "not_configured_deterministic_crop" if sam_engine is None else "deterministic_crop"
        return status

    _copy_rgb(image_path, out_path)
    return {
        "generation": "direct-copy normal reference whole image",
        "sam3_status": "not_configured" if sam_engine is None else "direct_copy_no_region_prompt",
    }


def _write_sam3_whole_object_cutout(image_path: str, out_path: Path, *, sam_engine: Any) -> dict[str, Any]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        sam_engine.set_image(image_path)
        mask, score = sam_engine.predict_mask_with_boxes([[0.5, 0.5, 1.0, 1.0]], threshold=0.4)
        if mask is None or not np.any(np.asarray(mask).astype(bool)):
            _copy_rgb(image_path, out_path)
            return {
                "generation": "direct-copy normal reference whole image after empty SAM3 mask",
                "sam3_status": "empty_mask_fallback",
                "sam3_score": 0.0,
                "image_size": [width, height],
            }
        sam_engine.save_masked_cutout(mask, str(out_path), padding=5)
        return {
            "generation": "SAM3 box-prompt normal reference cutout",
            "sam3_status": "box_prompt_cutout",
            "sam3_score": float(score),
            "image_size": [width, height],
        }
    except Exception as exc:
        _copy_rgb(image_path, out_path)
        return {
            "generation": "direct-copy normal reference whole image after SAM3 error",
            "sam3_status": "error_fallback",
            "sam3_error": str(exc),
        }


def _write_sam3_text_cutout(
    image_path: str,
    out_path: Path,
    *,
    sam_engine: Any,
    prompt: str,
    fallback_crop_box: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        sam_engine.set_image(image_path)
        mask, score = sam_engine.predict_mask(prompt, threshold=0.4)
        if mask is None or not np.any(np.asarray(mask).astype(bool)):
            status = _write_crop_or_copy_fallback(image_path, out_path, fallback_crop_box=fallback_crop_box)
            status.update({
                "sam3_status": "empty_text_prompt_mask_fallback",
                "sam3_score": 0.0,
                "sam3_prompt": prompt,
                "image_size": [width, height],
            })
            return status
        sam_engine.save_masked_cutout(mask, str(out_path), padding=5)
        return {
            "generation": "SAM3 text-prompt normal reference cutout",
            "sam3_status": "text_prompt_cutout",
            "sam3_score": float(score),
            "sam3_prompt": prompt,
            "image_size": [width, height],
        }
    except Exception as exc:
        status = _write_crop_or_copy_fallback(image_path, out_path, fallback_crop_box=fallback_crop_box)
        status.update({
            "sam3_status": "error_text_prompt_fallback",
            "sam3_prompt": prompt,
            "sam3_error": str(exc),
        })
        return status


def _write_crop_or_copy_fallback(
    image_path: str,
    out_path: Path,
    *,
    fallback_crop_box: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    if fallback_crop_box is not None:
        return _write_normalized_crop(image_path, out_path, crop_box=fallback_crop_box)
    _copy_rgb(image_path, out_path)
    return {"generation": "direct-copy normal reference whole image after SAM3 fallback"}


def _write_normalized_crop(
    image_path: str,
    out_path: Path,
    *,
    crop_box: tuple[float, float, float, float],
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        left, top, right, bottom = crop_box
        x0 = max(0, min(width - 1, int(round(left * width))))
        y0 = max(0, min(height - 1, int(round(top * height))))
        x1 = max(x0 + 1, min(width, int(round(right * width))))
        y1 = max(y0 + 1, min(height, int(round(bottom * height))))
        rgb.crop((x0, y0, x1, y1)).save(out_path)
    return {
        "generation": "deterministic normal-reference region crop",
        "crop_box_xyxy_norm": [float(item) for item in crop_box],
        "image_size": [width, height],
    }


def _build_text_knowledge(
    *,
    dataset: str,
    category: str,
    train_normal_count: int,
    region_specs: list[BinaryADRegionSpec],
) -> dict[str, Any]:
    regions = {}
    for region in region_specs:
        regions[region.name] = {
            "definition": region.definition,
            "normal_standard": (
                f"{region.normal_standard} The normal standard is grounded only in "
                f"{train_normal_count} train-normal image(s) for this category and the offline visual references generated from them."
            ),
            "critical_check": region.critical_check,
            "defects": [
                {
                    "type": "visible_anomaly",
                    "visual_signature": (
                        "Any localized or global visual deviation from the source-backed train-normal references for this region. "
                        "The concrete abnormal evidence must come from online heatmap, MCTS crop, and SAM3 audits."
                    ),
                    "contrast_vs_normal": "Normal evidence is defined by offline train-normal references, not by test labels or masks.",
                    "visual_attributes": list(region.visual_attributes),
                    "distinctions": [],
                }
            ],
            "anti_hallucination_rules": [
                "Do not infer fine-grained part names unless visible evidence supports them.",
                "Do not use test masks, test labels, or defect folder names when constructing the offline normal atlas.",
                "Use the offline references only as normal evidence; local anomaly evidence must come from online inspection.",
            ],
        }
    return {
        "target_object": category,
        "dataset": dataset,
        "pvla_region_policy": "dataset_profiled_regions_mpdd_parts_dtd_dagm_texture_surfaces",
        "regions": regions,
    }


def _write_graph_pickle(
    *,
    knowledge: dict[str, Any],
    graph_path: Path,
    source_metadata: dict[str, Any],
    reference_paths_by_region: dict[str, list[str]],
) -> None:
    target = str(knowledge["target_object"])
    graph = nx.DiGraph()
    graph.add_node(target, type="root", label=target)
    node_names = []
    for region_name, region_data in knowledge["regions"].items():
        node_names.append(region_name)
        graph.add_node(
            region_name,
            type="region",
            definition=region_data["definition"],
            normal_standard=region_data["normal_standard"],
            critical_check=region_data["critical_check"],
            rules=region_data.get("anti_hallucination_rules", []),
            image_paths=reference_paths_by_region.get(region_name, []),
        )
        graph.add_edge(target, region_name, relation="has_region")
        for defect in region_data.get("defects", []):
            defect_node = f"{region_name}_{defect['type']}"
            graph.add_node(
                defect_node,
                type="defect_pattern",
                short_name=defect["type"],
                visual_signature=defect["visual_signature"],
                contrast=defect["contrast_vs_normal"],
                visual_attributes=defect.get("visual_attributes", []),
            )
            graph.add_edge(region_name, defect_node, relation="possible_anomaly")

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("wb") as handle:
        pickle.dump({
            "root_name": target,
            "graph": graph,
            "node_embeddings": None,
            "node_names": node_names,
            "source_metadata": source_metadata,
        }, handle)


def _build_category_region_profile(*, dataset: str, category: str) -> list[BinaryADRegionSpec]:
    if dataset == "mpdd":
        return _mpdd_region_profile(category)
    if dataset == "dtd":
        return _texture_region_profile(
            dataset=dataset,
            category=category,
            surface_name="texture sample",
            field_region="texture_field",
            patch_region="central_texture_patch",
            corner_region="corner_texture_patch",
        )
    if dataset == "dagm":
        return _texture_region_profile(
            dataset=dataset,
            category=category,
            surface_name="industrial surface",
            field_region="surface_field",
            patch_region="central_surface_patch",
            corner_region="corner_surface_patch",
        )
    raise ValueError(f"Unsupported binary AD dataset: {dataset}")


def _mpdd_region_profile(category: str) -> list[BinaryADRegionSpec]:
    label = _humanize(category)
    base = [
        BinaryADRegionSpec(
            name=PVLA_ROOT_REGION_NAME,
            definition=f"The complete {label} part in MPDD.",
            normal_standard="The object silhouette, material, color, and component layout should match the normal references.",
            critical_check="Check global shape, object completeness, material continuity, and part arrangement before local inspection.",
            visual_attributes=("shape_change", "missing_or_extra_material", "color_change", "surface_discontinuity"),
            crop_box=(0.0, 0.0, 1.0, 1.0),
        )
    ]
    lower = category.lower()
    if lower.startswith("bracket"):
        return [
            *base,
            BinaryADRegionSpec(
                name="mounting_holes",
                definition=f"The visible mounting holes or circular cutouts on the {label}.",
                normal_standard="Hole boundaries should be clean, consistently shaped, and unobstructed in the normal references.",
                critical_check="Inspect for blocked holes, deformation around holes, missing material, or abnormal residue near openings.",
                visual_attributes=("hole_shape_change", "edge_damage", "blocked_opening", "surface_discontinuity"),
                prompt=f"the circular mounting holes on the {label}",
                crop_box=(0.18, 0.18, 0.82, 0.82),
            ),
            BinaryADRegionSpec(
                name="outer_edges",
                definition=f"The external contour, corners, and rim of the {label}.",
                normal_standard="Outer edges should preserve the normal bracket contour without chips, bends, or added material.",
                critical_check="Inspect the contour for dents, broken corners, scratches crossing the boundary, or unexpected protrusions.",
                visual_attributes=("edge_damage", "shape_change", "missing_or_extra_material", "scratch_or_crack"),
                prompt=f"the outer edges and corners of the {label}",
                crop_box=(0.0, 0.0, 1.0, 1.0),
            ),
            BinaryADRegionSpec(
                name="coated_surface",
                definition=f"The main coated or painted surface area of the {label}.",
                normal_standard="The coating should keep normal color, texture, and surface continuity across reference images.",
                critical_check="Inspect for scratches, stains, discoloration, peeling, cracks, or unexpected surface texture changes.",
                visual_attributes=("texture_change", "color_change", "scratch_or_crack", "surface_discontinuity"),
                prompt=f"the flat coated surface of the {label}",
                crop_box=(0.12, 0.12, 0.88, 0.88),
            ),
        ]
    if "connector" in lower:
        return [
            *base,
            BinaryADRegionSpec(
                name="connector_body",
                definition=f"The molded body or housing of the {label}.",
                normal_standard="The connector body should preserve normal molded shape, color, and surface continuity.",
                critical_check="Inspect for broken housing, abnormal deformation, missing plastic, cracks, or staining.",
                visual_attributes=("shape_change", "color_change", "missing_or_extra_material", "scratch_or_crack"),
                prompt=f"the molded body of the {label}",
                crop_box=(0.08, 0.08, 0.92, 0.92),
            ),
            BinaryADRegionSpec(
                name="contact_slots",
                definition=f"The contact slots, pin openings, or terminal-facing structures of the {label}.",
                normal_standard="Contact slots should keep their regular spacing, clean boundaries, and normal visibility.",
                critical_check="Inspect for blocked slots, missing contacts, abnormal dark gaps, deformation, or extra material.",
                visual_attributes=("component_misalignment", "blocked_opening", "missing_or_extra_material", "color_change"),
                prompt=f"the contact slots and pin openings of the {label}",
                crop_box=(0.2, 0.2, 0.8, 0.8),
            ),
            BinaryADRegionSpec(
                name="outer_edges",
                definition=f"The external outline and edge structures of the {label}.",
                normal_standard="The outline should match normal references without broken corners or unexpected protrusions.",
                critical_check="Inspect edge continuity, corner integrity, and boundary color or material changes.",
                visual_attributes=("edge_damage", "shape_change", "missing_or_extra_material"),
                prompt=f"the outer edges of the {label}",
                crop_box=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
    if "metal_plate" in lower:
        return [
            *base,
            BinaryADRegionSpec(
                name="plate_surface",
                definition=f"The broad flat metal surface of the {label}.",
                normal_standard="The surface should maintain normal reflectance, texture, and material continuity.",
                critical_check="Inspect for scratches, pits, stains, dents, cracks, or surface texture discontinuities.",
                visual_attributes=("texture_change", "scratch_or_crack", "surface_discontinuity", "color_change"),
                prompt=f"the flat metal surface of the {label}",
                crop_box=(0.08, 0.08, 0.92, 0.92),
            ),
            BinaryADRegionSpec(
                name="plate_edges",
                definition=f"The perimeter edges and corners of the {label}.",
                normal_standard="Edges should stay straight or smoothly curved as in normal references.",
                critical_check="Inspect for chipped edges, deformed corners, burrs, or missing material.",
                visual_attributes=("edge_damage", "shape_change", "missing_or_extra_material"),
                prompt=f"the perimeter edges of the {label}",
                crop_box=(0.0, 0.0, 1.0, 1.0),
            ),
            BinaryADRegionSpec(
                name="holes_or_cutouts",
                definition=f"The holes, slots, or cutout structures visible on the {label}.",
                normal_standard="Cutout boundaries should be clean and have the expected relative placement.",
                critical_check="Inspect for deformation, blocked openings, missing cutouts, burrs, or abnormal material near holes.",
                visual_attributes=("hole_shape_change", "blocked_opening", "component_misalignment", "edge_damage"),
                prompt=f"the holes and cutouts in the {label}",
                crop_box=(0.15, 0.15, 0.85, 0.85),
            ),
        ]
    if "tube" in lower:
        return [
            *base,
            BinaryADRegionSpec(
                name="tube_bodies",
                definition=f"The cylindrical bodies or sidewalls of the {label}.",
                normal_standard="Tube bodies should keep normal cylindrical shape, surface continuity, and color.",
                critical_check="Inspect for dents, cracks, discoloration, residue, scratches, or missing material.",
                visual_attributes=("shape_change", "texture_change", "color_change", "scratch_or_crack"),
                prompt=f"the cylindrical tube bodies of the {label}",
                crop_box=(0.08, 0.08, 0.92, 0.92),
            ),
            BinaryADRegionSpec(
                name="tube_ends",
                definition=f"The visible tube ends, rims, or caps of the {label}.",
                normal_standard="Tube ends should preserve normal rim shape, opening visibility, and edge cleanliness.",
                critical_check="Inspect for blocked openings, damaged rims, deformation, or abnormal material around the ends.",
                visual_attributes=("edge_damage", "blocked_opening", "shape_change", "missing_or_extra_material"),
                prompt=f"the tube ends and rims of the {label}",
                crop_box=(0.0, 0.0, 1.0, 1.0),
            ),
            BinaryADRegionSpec(
                name="inner_openings",
                definition=f"The inner visible openings or hollow areas of the {label}.",
                normal_standard="Openings should remain clean, consistent, and visually separated from the tube wall.",
                critical_check="Inspect for obstruction, abnormal darkness, deformation, extra material, or missing inner boundary.",
                visual_attributes=("blocked_opening", "hole_shape_change", "color_change", "missing_or_extra_material"),
                prompt=f"the inner openings of the {label}",
                crop_box=(0.18, 0.18, 0.82, 0.82),
            ),
        ]
    return [
        *base,
        BinaryADRegionSpec(
            name="part_surface",
            definition=f"The main visible surface area of the {label}.",
            normal_standard="The visible material surface should match normal references in color, texture, and continuity.",
            critical_check="Inspect for scratches, stains, cracks, discoloration, dents, or missing material.",
            visual_attributes=("texture_change", "color_change", "scratch_or_crack", "surface_discontinuity"),
            prompt=f"the main visible surface of the {label}",
            crop_box=(0.1, 0.1, 0.9, 0.9),
        ),
        BinaryADRegionSpec(
            name="outer_contour",
            definition=f"The outline and boundary of the {label}.",
            normal_standard="The boundary should preserve the normal shape and edge continuity.",
            critical_check="Inspect for deformed edges, missing parts, unexpected protrusions, or chipped corners.",
            visual_attributes=("edge_damage", "shape_change", "missing_or_extra_material"),
            prompt=f"the outer contour of the {label}",
            crop_box=(0.0, 0.0, 1.0, 1.0),
        ),
    ]


def _texture_region_profile(
    *,
    dataset: str,
    category: str,
    surface_name: str,
    field_region: str,
    patch_region: str,
    corner_region: str,
) -> list[BinaryADRegionSpec]:
    label = _humanize(category)
    dataset_label = dataset.upper()
    return [
        BinaryADRegionSpec(
            name=PVLA_ROOT_REGION_NAME,
            definition=f"The complete {label} {surface_name} in the {dataset_label} binary anomaly-detection benchmark.",
            normal_standard="The full sample should preserve the source-backed normal texture, color distribution, and surface continuity.",
            critical_check="Check the global texture field for abnormal color, contrast, periodicity, or material discontinuity.",
            visual_attributes=("texture_change", "color_change", "surface_discontinuity", "pattern_break"),
            crop_box=(0.0, 0.0, 1.0, 1.0),
        ),
        BinaryADRegionSpec(
            name=field_region,
            definition=f"The main visible {surface_name} field for {label}.",
            normal_standard="The region should preserve the dominant normal texture statistics and pattern continuity.",
            critical_check="Inspect for distribution shifts, unexpected structures, holes, stains, scratches, or broken pattern continuity.",
            visual_attributes=("texture_change", "pattern_break", "surface_discontinuity", "color_change"),
            crop_box=(0.05, 0.05, 0.95, 0.95),
        ),
        BinaryADRegionSpec(
            name=patch_region,
            definition=f"A central local patch from the {label} {surface_name}.",
            normal_standard="The patch should match local normal granularity, edge density, and color variation from train-normal references.",
            critical_check="Inspect for small local anomalies that may be diluted in whole-image comparison.",
            visual_attributes=("local_texture_change", "small_spot", "scratch_or_crack", "surface_discontinuity"),
            crop_box=(0.25, 0.25, 0.75, 0.75),
        ),
        BinaryADRegionSpec(
            name=corner_region,
            definition=f"A corner local patch from the {label} {surface_name}.",
            normal_standard="The patch should provide a second normal local context away from the central crop.",
            critical_check="Inspect whether anomalies are position-local rather than global texture shifts.",
            visual_attributes=("local_texture_change", "small_spot", "pattern_break", "color_change"),
            crop_box=(0.0, 0.0, 0.5, 0.5),
        ),
    ]


def _reference_paths_by_region(reference_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for row in reference_rows:
        rows.setdefault(str(row["region"]), []).append(str(row["path"]))
    return rows


def _group_by_category(samples: Iterable[BinaryADSample]) -> dict[str, list[BinaryADSample]]:
    rows: dict[str, list[BinaryADSample]] = {}
    for sample in samples:
        rows.setdefault(sample.category, []).append(sample)
    return rows


def _copy_rgb(source: str, target: Path) -> None:
    with Image.open(source) as image:
        image.convert("RGB").save(target)


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
