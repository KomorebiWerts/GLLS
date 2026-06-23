import json
import pickle
import os
import networkx as nx
import torch
import numpy as np
import re
from typing import Any, List, Dict, Union, Optional
from pydantic import BaseModel, Field

# 引入 SentenceTransformer
from sentence_transformers import SentenceTransformer, util
from glls import paths as glls_paths
from glls.rag.semantic_grounding import SemanticOptionGrounder
from glls.rag.source_chain import KNOWLEDGE_SOURCE_TYPE

# ==========================================
# 1. 数据 Schema (升级版：支持缺陷对比)
# ==========================================

class Distinction(BaseModel):
    """描述当前缺陷与另一个易混淆缺陷的区别"""
    target_defect: str  # 易混淆的目标缺陷名 (e.g., "poke_sheath")
    difference: str     # 区分逻辑 (e.g., "Cut is linear, Poke is round.")

class DefectPattern(BaseModel):
    """描述某个部位可能出现的具体缺陷模式"""
    type: str                     # e.g., "cut_outer_insulation"
    visual_signature: str         # e.g., "Jagged slice or tear..."
    contrast_vs_normal: str       # e.g., "Normal is smooth..."
    visual_attributes: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    # [新增] 易混淆对比列表
    distinctions: Optional[List[Distinction]] = [] 

class RegionNode(BaseModel):
    definition: str
    normal_standard: Optional[str] = "N/A" 
    critical_check: Optional[str] = "N/A"  
    defects: List[DefectPattern]
    anti_hallucination_rules: Optional[List[str]] = []

class IndustrialKnowledgeBase(BaseModel):
    target_object: str
    regions: Dict[str, RegionNode] 

# ==========================================
# 2. 语义图谱引擎
# ==========================================

class SimInspecGraphEngine:
    def __init__(self, embedding_model_name=None):
        if embedding_model_name is None:
            embedding_model_name = glls_paths.embedding_model_path()
        self.G = nx.DiGraph()
        self.root_name = None
        
        print(f"[RAG] Loading Embedding Model: {embedding_model_name}...")
        try:
            self.encoder = SentenceTransformer(embedding_model_name, local_files_only=True)
        except TypeError:
            # Older sentence-transformers versions do not support local_files_only.
            # Passing a local path still keeps loading local-only for our BGE path.
            self.encoder = SentenceTransformer(embedding_model_name)
        except Exception as e:
            print(f"[Warning] local load failed, downloading: {e}")
            self.encoder = SentenceTransformer(embedding_model_name)
        
        self.node_embeddings = None 
        self.node_names = []
        self.source_metadata = {}

    @staticmethod
    def _is_valid_text(text) -> bool:
        if not text:
            return False
        if not isinstance(text, str):
            return True
        return text.strip().upper() != "N/A"

    @staticmethod
    def _tokenize_for_match(text: str) -> set[str]:
        return SemanticOptionGrounder.tokenize(text)

    @staticmethod
    def _weak_grounding_tokens() -> set[str]:
        return SemanticOptionGrounder.weak_tokens()

    def _option_grounder(self) -> SemanticOptionGrounder:
        encoder = getattr(self, "encoder", None)
        grounder = getattr(self, "_semantic_option_grounder", None)
        if grounder is None or getattr(grounder, "encoder", None) is not encoder:
            grounder = SemanticOptionGrounder(encoder=encoder)
            self._semantic_option_grounder = grounder
        return grounder

    def _compute_region_text(self, name: str, region: RegionNode) -> str:
        text = f"Region: {name}. Definition: {region.definition}. Standard: {region.normal_standard}. Defects: "
        for defect in region.defects:
            text += f"{defect.type} ({defect.visual_signature}); "
        return text

    def load_json(self, json_data: Union[str, Dict]):
        """加载 JSON 并构建 Region-Based 图谱 (包含对比关系)"""
        if isinstance(json_data, str):
            data_dict = json.loads(json_data)
        else:
            data_dict = json_data
            
        kb_data = IndustrialKnowledgeBase(**data_dict)
        self.root_name = kb_data.target_object
        self.source_metadata = {
            "target_object": kb_data.target_object,
            "source_type": KNOWLEDGE_SOURCE_TYPE,
            "knowledge_source_type": KNOWLEDGE_SOURCE_TYPE,
        }
        
        self.G.clear()
        self.G.add_node(self.root_name, type="root", label=self.root_name)
        
        print(f"正在构建 [{self.root_name}] 的部位感知图谱 (含对比逻辑)...")

        corpus_texts = []
        self.node_names = []

        for region_key, region_data in kb_data.regions.items():
            # 1. Region Node
            self.G.add_node(
                region_key, 
                type="region", 
                definition=region_data.definition,
                normal_standard=region_data.normal_standard,
                critical_check=region_data.critical_check,
                rules=region_data.anti_hallucination_rules
            )
            self.G.add_edge(self.root_name, region_key, relation="has_region")

            # 2. Defect Nodes & Distinctions
            for defect in region_data.defects:
                defect_node_id = f"{region_key}_{defect.type}" # 唯一ID
                
                self.G.add_node(
                    defect_node_id,
                    type="defect_pattern",
                    short_name=defect.type,
                    visual_signature=defect.visual_signature,
                    contrast=defect.contrast_vs_normal,
                    visual_attributes=defect.visual_attributes or [],
                )
                self.G.add_edge(region_key, defect_node_id, relation="possible_anomaly")

                # [新增] 处理对比逻辑 (Distinctions)
                if defect.distinctions:
                    for dist in defect.distinctions:
                        # 假设易混淆对象也在同一部位下
                        target_id = f"{region_key}_{dist.target_defect}"
                        # 添加一条红色的“对比边”
                        self.G.add_edge(
                            defect_node_id, 
                            target_id, 
                            relation="distinct_from", 
                            logic=dist.difference
                        )

            # 3. Embedding Prep
            rich_text = self._compute_region_text(region_key, region_data)
            corpus_texts.append(rich_text)
            self.node_names.append(region_key)

        if corpus_texts:
            print(f"[RAG] Encoding {len(corpus_texts)} region nodes...")
            self.node_embeddings = self.encoder.encode(corpus_texts, convert_to_tensor=True)
        
        print(f"✅ 图谱构建完成! 包含 {self.G.number_of_nodes()} 个节点。")

    # ==========================================
    # 3. 检索与查询接口
    # ==========================================

    def search_regions(self, query: str, top_k: int = 3) -> List[str]:
        if self.node_embeddings is None: return []
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.node_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.node_names)))
        return [self.node_names[idx] for idx in top_results.indices]

    def get_inspection_checklist(self, region_name: str) -> Dict[str, Union[str, List[str]]]:
        """
        [增强版] 获取检查清单，会自动包含 "VS" 对比信息。
        """
        if not self.G.has_node(region_name):
            return {"error": "Region not found"}
            
        node_data = self.G.nodes[region_name]
        
        info = {
            "region": region_name,
            "definition": node_data.get("definition", ""),
            "normal_standard": node_data.get("normal_standard", ""),
            "critical_check": node_data.get("critical_check", ""),
            "rules": node_data.get("rules", []),
            "anti_hallucination_rules": node_data.get("rules", []),
            "defects": []
        }
        
        for neighbor in self.G.neighbors(region_name):
            edge = self.G.get_edge_data(region_name, neighbor)
            if edge.get("relation") == "possible_anomaly":
                defect_data = self.G.nodes[neighbor]
                short_name = defect_data.get('short_name')
                
                distinctions = []
                for _, target, d_data in self.G.out_edges(neighbor, data=True):
                    if d_data.get("relation") == "distinct_from":
                        target_short_name = self.G.nodes[target].get('short_name', target)
                        distinctions.append({
                            "target_defect": target_short_name,
                            "difference": d_data.get('logic', "")
                        })

                info["defects"].append({
                    "type": short_name,
                    "visual_signature": defect_data.get("visual_signature", ""),
                    "contrast_vs_normal": defect_data.get("contrast", ""),
                    "visual_attributes": defect_data.get("visual_attributes", []),
                    "distinctions": distinctions,
                })
                
        return info

    def get_image_paths(self, region_name: str) -> List[str]:
        if self.G.has_node(region_name):
            paths = self.G.nodes[region_name].get("image_paths", [])
            resolved_paths = []
            for path in paths:
                resolved = self._resolve_reference_image_path(str(path))
                if resolved and resolved not in resolved_paths:
                    resolved_paths.append(resolved)
            return resolved_paths
        return []

    def _resolve_reference_image_path(self, path: str) -> Optional[str]:
        if os.path.exists(path):
            return path

        marker = f"{os.sep}databases{os.sep}img{os.sep}"
        if marker in path:
            suffix = path.split(marker, 1)[1]
            candidate = os.path.join(glls_paths.database_root(), "img", suffix)
            if os.path.exists(candidate):
                return candidate

        return None

    def iter_region_defects(self, region_names: Optional[List[str]] = None) -> List[Dict[str, str]]:
        regions = region_names or [n for n, d in self.G.nodes(data=True) if d.get("type") == "region"]
        defects = []
        for region in regions:
            if not self.G.has_node(region):
                continue
            region_data = self.G.nodes[region]
            for neighbor in self.G.neighbors(region):
                edge = self.G.get_edge_data(region, neighbor) or {}
                if edge.get("relation") != "possible_anomaly":
                    continue
                defect_data = self.G.nodes[neighbor]
                distinctions = []
                for _, target, d_data in self.G.out_edges(neighbor, data=True):
                    if d_data.get("relation") == "distinct_from":
                        distinctions.append({
                            "target_defect": self.G.nodes[target].get("short_name", target),
                            "difference": d_data.get("logic", ""),
                        })
                defects.append({
                    "region": region,
                    "region_definition": region_data.get("definition", ""),
                    "critical_check": region_data.get("critical_check", ""),
                    "defect_node": neighbor,
                    "defect": defect_data.get("short_name", neighbor),
                    "visual_signature": defect_data.get("visual_signature", ""),
                    "contrast_vs_normal": defect_data.get("contrast", ""),
                    "visual_attributes": defect_data.get("visual_attributes", []),
                    "graph_path": f"{self.root_name} -> {region} -> {neighbor}",
                    "distinctions": distinctions,
                })
        return defects

    def _score_option_against_defect(
        self,
        option_text: str,
        option_tokens: set[str],
        defect: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._option_grounder().score_option_against_defect(option_text, defect)

    @staticmethod
    def _option_has_compound_scope(option_text: str) -> bool:
        normalized = SemanticOptionGrounder.normalize_text(option_text)
        return bool(re.search(r"\b(and|plus|both|multiple|combined|together)\b|[,;/]", normalized))

    def _rank_defects_for_option(
        self,
        option_text: str,
        defects: List[Dict[str, Any]],
        min_score: float = 1.6,
        relative_floor: float = 0.75,
        max_matches: int = 3,
    ) -> List[Dict[str, Any]]:
        option_tokens = self._tokenize_for_match(option_text)
        if not option_tokens:
            return []

        ranked = []
        for defect in defects:
            score_info = self._score_option_against_defect(str(option_text), option_tokens, defect)
            # Region-only matches are too weak for option grounding. They can be
            # useful for region retrieval, but here they become false label priors.
            if score_info["signal_score"] <= 0:
                continue
            if not score_info.get("evidence_consensus"):
                continue
            score = float(score_info["score"])
            if score < min_score:
                continue
            ranked.append({**defect, **score_info, "score": score})

        if not ranked:
            return []

        ranked.sort(
            key=lambda item: (
                item["score"],
                item["signal_score"],
                len(item.get("evidence_tokens", [])),
            ),
            reverse=True,
        )

        best_by_defect = {}
        for item in ranked:
            defect_name = item["defect"]
            if defect_name not in best_by_defect:
                best_by_defect[defect_name] = item

        deduped = sorted(
            best_by_defect.values(),
            key=lambda item: (item["score"], item["signal_score"]),
            reverse=True,
        )
        best_score = float(deduped[0]["score"])
        floor = 0.45 if self._option_has_compound_scope(str(option_text)) else relative_floor
        selected = [
            item for item in deduped
            if item["score"] >= max(min_score, best_score * floor)
        ]
        return selected[:max_matches]

    @staticmethod
    def _match_defect_set(match: Dict[str, Any]) -> tuple[str, ...]:
        defects = match.get("matched_defects") or []
        if not defects:
            defect = match.get("defect")
            return (str(defect),) if defect else ()
        return tuple(sorted(str(item.get("defect", "")) for item in defects if item.get("defect")))

    def _option_grounding_is_discriminative(
        self,
        matches: List[Dict[str, Any]],
        options: Optional[Dict[str, str]],
    ) -> bool:
        if not matches:
            return False
        option_count = len(options or {})
        if option_count > 1 and len(matches) < 2:
            only_match = matches[0]
            matched_defects = only_match.get("matched_defects") or []
            return (
                bool(matched_defects)
                and bool(only_match.get("evidence_consensus", False))
                and any(item.get("source_evidence_channels") for item in matched_defects)
                and float(only_match.get("signal_score", 0.0)) >= 3.2
            )

        defect_sets = [self._match_defect_set(match) for match in matches]
        non_empty_sets = [item for item in defect_sets if item]
        if not non_empty_sets:
            return False
        if option_count > 1 and len(set(non_empty_sets)) < 2:
            return False
        return True

    def match_options_to_defects(
        self,
        options: Optional[Dict[str, str]],
        region_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Union[str, float, List[Dict[str, str]]]]]:
        if not options:
            return []

        defects = self.iter_region_defects(region_names)
        matches = []
        for key, option_text in options.items():
            ranked_defects = self._rank_defects_for_option(str(option_text), defects)
            if not ranked_defects:
                continue

            primary = ranked_defects[0]
            matched_defects = []
            for item in ranked_defects:
                matched_defects.append({
                    "region": item["region"],
                    "defect": item["defect"],
                    "defect_node": item["defect_node"],
                    "graph_path": item["graph_path"],
                    "visual_signature": item["visual_signature"],
                    "contrast_vs_normal": item["contrast_vs_normal"],
                    "visual_attributes": item.get("visual_attributes", []),
                    "critical_check": item["critical_check"],
                    "distinctions": item["distinctions"],
                    "score": round(float(item["score"]), 3),
                    "signal_score": round(float(item.get("signal_score", 0.0)), 3),
                    "evidence_tokens": item.get("evidence_tokens", []),
                    "attribute_overlap": item.get("attribute_overlap", {}),
                    "signature_attribute_overlap": item.get("signature_attribute_overlap", {}),
                    "graph_attribute_overlap": item.get("graph_attribute_overlap", {}),
                    "attribute_term_count": item.get("attribute_term_count", 0),
                    "source_attribute_term_count": item.get("source_attribute_term_count", 0),
                    "support_channels": item.get("support_channels", []),
                    "source_evidence_channels": item.get("source_evidence_channels", []),
                    "label_only_match": bool(item.get("label_only_match", False)),
                    "evidence_consensus": bool(item.get("evidence_consensus", False)),
                    "semantic_similarity": item.get("semantic_similarity", 0.0),
                })

            matches.append({
                "option_key": str(key),
                "option_text": str(option_text),
                "region": primary["region"],
                "defect": primary["defect"],
                "defect_node": primary["defect_node"],
                "graph_path": primary["graph_path"],
                "visual_signature": primary["visual_signature"],
                "contrast_vs_normal": primary["contrast_vs_normal"],
                "visual_attributes": primary.get("visual_attributes", []),
                "critical_check": primary["critical_check"],
                "distinctions": primary["distinctions"],
                "score": round(float(primary["score"]), 3),
                "signal_score": round(float(primary.get("signal_score", 0.0)), 3),
                "attribute_overlap": primary.get("attribute_overlap", {}),
                "signature_attribute_overlap": primary.get("signature_attribute_overlap", {}),
                "graph_attribute_overlap": primary.get("graph_attribute_overlap", {}),
                "attribute_term_count": primary.get("attribute_term_count", 0),
                "source_attribute_term_count": primary.get("source_attribute_term_count", 0),
                "support_channels": primary.get("support_channels", []),
                "source_evidence_channels": primary.get("source_evidence_channels", []),
                "label_only_match": bool(primary.get("label_only_match", False)),
                "evidence_consensus": bool(primary.get("evidence_consensus", False)),
                "semantic_similarity": primary.get("semantic_similarity", 0.0),
                "matched_defects": matched_defects,
            })
        return matches

    def build_option_grounding_text(
        self,
        options: Optional[Dict[str, str]],
        region_names: Optional[List[str]] = None,
    ) -> str:
        audit = self.build_option_grounding_audit(options, region_names=region_names)
        matches = audit["matches"]
        if not audit["discriminative"]:
            return ""

        lines = [
            "=== PVLA Topological Option Grounding ===",
            f"Target object node: {self.root_name}",
            "Use this graph traversal as a decision guide, not as a label prior.",
            "Only the options with discriminative graph evidence are listed below.",
        ]
        for match in matches:
            matched_defects = match.get("matched_defects") or [match]
            defect_labels = ", ".join(
                f"{item['graph_path']} (score {item.get('score')})"
                for item in matched_defects
            )
            lines.append(
                f"- Option {match['option_key']} ({match['option_text']}) maps to "
                f"{defect_labels}."
            )
            attr_text = SemanticOptionGrounder.format_attribute_overlap(match.get("attribute_overlap", {}))
            if attr_text:
                lines.append(f"  Shared visual attributes: {attr_text}")
            graph_attr_text = SemanticOptionGrounder.format_attribute_overlap(match.get("graph_attribute_overlap", {}))
            if graph_attr_text:
                lines.append(f"  Graph visual attributes: {graph_attr_text}")
            channels = ", ".join(match.get("support_channels") or [])
            if channels:
                lines.append(f"  Evidence consensus channels: {channels}")
            source_channels = ", ".join(match.get("source_evidence_channels") or [])
            if source_channels:
                lines.append(f"  Source-grounded channels: {source_channels}")
            if "only" in str(match["option_text"]).lower():
                lines.append("  Scope note: this option excludes additional visible graph defects.")

            for item in matched_defects:
                item_attr_text = SemanticOptionGrounder.format_attribute_overlap(item.get("attribute_overlap", {}))
                if item_attr_text:
                    lines.append(f"  Evidence attributes for {item['defect']}: {item_attr_text}")
                item_graph_attr_text = SemanticOptionGrounder.format_attribute_overlap(item.get("graph_attribute_overlap", {}))
                if item_graph_attr_text:
                    lines.append(f"  Graph attributes for {item['defect']}: {item_graph_attr_text}")
                item_channels = ", ".join(item.get("support_channels") or [])
                if item_channels:
                    lines.append(f"  Support channels for {item['defect']}: {item_channels}")
                item_source_channels = ", ".join(item.get("source_evidence_channels") or [])
                if item_source_channels:
                    lines.append(f"  Source-grounded channels for {item['defect']}: {item_source_channels}")
                if self._is_valid_text(item.get("critical_check", "")):
                    lines.append(f"  Region check for {item['defect']}: {item['critical_check']}")
                if self._is_valid_text(item.get("visual_signature", "")):
                    lines.append(f"  Defect signature for {item['defect']}: {item['visual_signature']}")
                if self._is_valid_text(item.get("contrast_vs_normal", "")):
                    lines.append(f"  Contrast for {item['defect']}: {item['contrast_vs_normal']}")
                distinctions = item.get("distinctions") or []
                for dist in distinctions:
                    diff = dist.get("difference", "")
                    if self._is_valid_text(diff):
                        lines.append(f"  Distinction vs {dist.get('target_defect')}: {diff}")

        lines.extend([
            "Decision rule: first identify every physical region that contains highlighted anomaly evidence, then choose the option whose mapped graph defect set best covers those visible defects.",
            "Reject an option whose physical region is not supported by the red-box or focus-view evidence.",
        ])
        return "\n".join(lines)

    def build_option_grounding_audit(
        self,
        options: Optional[Dict[str, str]],
        region_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return source-grounded option topology as audit data only.

        This is the interface used by prompt-building code. It intentionally
        does not format option mappings as prompt text, so option grounding can
        remain inspectable without becoming a prompt-visible answer prior.
        """

        matches = self.match_options_to_defects(options, region_names=region_names)
        discriminative = self._option_grounding_is_discriminative(matches, options)
        return {
            "target_object": self.root_name,
            "prompt_visible": False,
            "discriminative": bool(discriminative),
            "raw_match_count": len(matches),
            "match_count": len(matches) if discriminative else 0,
            "matches": matches if discriminative else [],
            "note": "Internal PVLA topology audit only; do not inject option mappings into the final VLM prompt.",
        }

    # ==========================================
    # 4. 持久化
    # ==========================================

    def save_to_disk(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data_to_save = {
            "root_name": self.root_name,
            "graph": self.G,
            "node_embeddings": self.node_embeddings,
            "node_names": self.node_names,
            "source_metadata": self.source_metadata,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"💾 Graph saved to: {save_path}")

    def load_from_disk(self, load_path: str):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Graph file not found: {load_path}")
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.root_name = data["root_name"]
        self.G = data["graph"]
        self.node_embeddings = data.get("node_embeddings")
        self.node_names = data.get("node_names", [])
        self.source_metadata = data.get("source_metadata", {})
        self.source_metadata["graph_cache_path"] = load_path
        print(f"🚀 Graph loaded: {load_path} ({self.G.number_of_nodes()} nodes)")
