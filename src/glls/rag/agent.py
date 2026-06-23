"""PVLA/RAG context provider used by the MVTec and VisA QA pipeline."""

from __future__ import annotations

import os
from typing import Any

from glls import paths as glls_paths
from glls.rag.GraphRag import SimInspecGraphEngine
from glls.rag.pvla_context import PVLARegionContextBuilder
from glls.rag.pvla_retrieval import PVLARetrievalPlanner
from glls.rag.source_chain import (
    KNOWLEDGE_SOURCE_TYPE,
    UNKNOWN_KNOWLEDGE_SOURCE_TYPE,
    UNKNOWN_VISUAL_REFERENCE_GENERATION,
    UNKNOWN_VISUAL_REFERENCE_SOURCE,
    visual_reference_trust_audit,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimInspecAgent:
    """Build source-backed PVLA context blocks for the main QA pipeline.

    This class intentionally does not own a VLM or run inference. The main QA
    worker asks it for prompt-ready knowledge blocks, while MCTS/SAM3 and final
    answering remain in their own modules.
    """

    def __init__(self, vlm_engine: Any, cache_root: str, k_shot: int = 1):
        self.vlm = vlm_engine
        self.cache_root = cache_root
        self.rag_engines: dict[str, SimInspecGraphEngine] = {}
        self.k_shot = k_shot

    def get_rag_context(
        self,
        subclass: str,
        query: str | None = None,
        top_k: int | None = None,
        include_global: bool = True,
        retrieval_mode: str = "full",
        options: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return prompt-ready PVLA blocks plus source provenance.

        Default retrieval is fail-open over all graph regions. Query-conditioned
        top-k is opt-in because MMAD QA text is noisy and can otherwise drop the
        region needed for image-level decisions.
        """

        rag = self._get_rag_engine(subclass)
        if not rag.G.number_of_nodes():
            return []

        all_regions = [n for n, d in rag.G.nodes(data=True) if d.get("type") == "region"]
        retrieval_plan = PVLARetrievalPlanner(rag).plan(
            all_regions=all_regions,
            query=query,
            top_k=top_k,
            include_global=include_global,
            retrieval_mode=retrieval_mode,
            options=options,
        )
        selected_regions = retrieval_plan["selected_regions"]
        selection_by_region = {
            item["region"]: item
            for item in retrieval_plan.get("selection_audit", [])
            if isinstance(item, dict) and item.get("region")
        }

        rag_blocks: list[dict[str, Any]] = []
        option_grounding_audit = self._build_option_grounding_audit(
            rag,
            all_regions=all_regions,
            query=query,
            options=options,
        )
        option_matches = option_grounding_audit.get("matches") or []
        if option_matches:
            rag_blocks.append({
                "region": "pvla_option_grounding",
                "text": "",
                "images": [],
                "prompt_visible": False,
                "retrieval": {
                    "query": query or "",
                    "top_k": top_k,
                    "mode": retrieval_mode,
                    "selected_from": len(all_regions),
                    "provenance": self._build_provenance(rag, subclass),
                    "pvla_plan": retrieval_plan,
                },
                "topology": {
                    "target_object": rag.root_name,
                    "option_matches": option_matches,
                    "option_grounding_audit": option_grounding_audit,
                    "prompt_visible": False,
                    "note": "Internal option grounding audit; not injected into the VLM prompt.",
                },
            })

        region_context_builder = PVLARegionContextBuilder(k_shot=self.k_shot)
        for region in selected_regions:
            region_context = region_context_builder.build(rag, region)
            image_paths = region_context.image_paths
            rag_blocks.append({
                "region": region,
                "text": region_context.text,
                "images": image_paths,
                "prompt_visible": True,
                "retrieval": {
                    "query": query or "",
                    "top_k": top_k,
                    "mode": retrieval_mode,
                    "selected_from": len(all_regions),
                    "provenance": self._build_provenance(
                        rag,
                        subclass,
                        region=region,
                        image_paths=image_paths,
                    ),
                    "pvla_selection": selection_by_region.get(region, {}),
                },
                "topology": region_context.topology,
            })

        return rag_blocks

    def _build_option_grounding_audit(
        self,
        rag: SimInspecGraphEngine,
        *,
        all_regions: list[str],
        query: str | None,
        options: dict[str, str] | None,
    ) -> dict[str, Any]:
        if not self._should_build_option_grounding(query):
            return {
                "prompt_visible": False,
                "discriminative": False,
                "matches": [],
                "raw_match_count": 0,
                "match_count": 0,
            }
        return rag.build_option_grounding_audit(options, region_names=all_regions)

    def _candidate_text_knowledge_paths(self, rag: SimInspecGraphEngine, subclass: str) -> list[str]:
        database_root = glls_paths.database_root()
        candidate_roots = []
        graph_path = None
        if hasattr(rag, "source_metadata"):
            graph_path = rag.source_metadata.get("graph_cache_path")
            source_json = rag.source_metadata.get("source_json_path") or rag.source_metadata.get("text_knowledge_path")
            if source_json and os.path.exists(source_json):
                return [source_json]

        if graph_path:
            graph_dir = os.path.basename(os.path.dirname(graph_path))
            if graph_dir.startswith("graph_index"):
                candidate_roots.append(os.path.join(database_root, graph_dir.replace("graph_index", "text_knowledge", 1)))

        candidate_roots.extend([
            os.path.join(database_root, "text_knowledge"),
            os.path.join(database_root, "text_knowledge_weak_agent"),
        ])

        text_candidates = []
        for root in candidate_roots:
            for dataset_name in ("mvtec", "visa"):
                candidate = os.path.join(root, dataset_name, f"{subclass}.json")
                if os.path.exists(candidate) and candidate not in text_candidates:
                    text_candidates.append(candidate)
        return text_candidates

    def _build_provenance(
        self,
        rag: SimInspecGraphEngine,
        subclass: str,
        region: str | None = None,
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        source_metadata = rag.source_metadata if hasattr(rag, "source_metadata") else {}
        graph_path = source_metadata.get("graph_cache_path")
        source_json_path = source_metadata.get("source_json_path") or source_metadata.get("text_knowledge_path")
        knowledge_source_type = (
            source_metadata.get("knowledge_source_type")
            or source_metadata.get("source_type")
            or UNKNOWN_KNOWLEDGE_SOURCE_TYPE
        )
        visual_source = source_metadata.get("visual_reference_source") or UNKNOWN_VISUAL_REFERENCE_SOURCE
        visual_generation = source_metadata.get("visual_reference_generation") or UNKNOWN_VISUAL_REFERENCE_GENERATION
        visual_paths = list(image_paths or [])
        visual_trust = visual_reference_trust_audit(
            source=visual_source,
            generation=visual_generation,
            image_paths=visual_paths,
        )
        provenance = {
            "graph_cache_path": graph_path,
            "source_json_path": source_json_path or "",
            "knowledge_source_type": knowledge_source_type,
            "knowledge_source_type_trusted": knowledge_source_type == KNOWLEDGE_SOURCE_TYPE,
            "knowledge_dataset": source_metadata.get("dataset", ""),
            "knowledge_category": source_metadata.get("category", subclass),
            "text_knowledge_paths": self._candidate_text_knowledge_paths(rag, subclass),
            "visual_reference_source": visual_source,
            "visual_reference_generation": visual_generation,
            "visual_reference_root": source_metadata.get("visual_reference_root", ""),
            "visual_reference_builder": source_metadata.get("visual_reference_builder", ""),
            "visual_reference_paths": visual_paths,
            "visual_reference_role": "region_cutout" if visual_paths else "text_only_reference",
            "visual_reference_trust": visual_trust,
            "visual_reference_trusted_source": visual_trust["trusted_source"],
            "visual_reference_path_count": visual_trust["path_count"],
            "visual_reference_existing_path_count": visual_trust["existing_path_count"],
            "visual_reference_missing_path_count": visual_trust["missing_path_count"],
            "visual_reference_source_backed": visual_trust["source_backed"],
            "graph_source_metadata_present": bool(source_metadata),
            "graph_source_metadata_visual_reference_recorded": bool(
                source_metadata.get("visual_reference_source")
                and source_metadata.get("visual_reference_generation")
            ),
        }
        if region:
            provenance["region"] = region
        return provenance

    @staticmethod
    def _should_build_option_grounding(query: str | None) -> bool:
        query_l = str(query or "").lower()
        return any(
            marker in query_l
            for marker in (
                "defect classification",
                "defect description",
                "defect analysis",
                "type of the defect",
                "appearance of the defect",
                "effect does the defect",
                "how does the defect affect",
            )
        )

    def _get_rag_engine(self, subclass: str) -> SimInspecGraphEngine:
        if subclass in self.rag_engines:
            return self.rag_engines[subclass]
        pkl_path = os.path.join(self.cache_root, f"{subclass}_graph.pkl")
        engine = SimInspecGraphEngine()
        if os.path.exists(pkl_path):
            engine.load_from_disk(pkl_path)
        self.rag_engines[subclass] = engine
        return engine
