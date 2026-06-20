import json
import pickle
import os
import networkx as nx
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field

# 引入 SentenceTransformer
from sentence_transformers import SentenceTransformer, util
from kragad import paths as kragad_paths

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
            embedding_model_name = kragad_paths.embedding_model_path()
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
                    contrast=defect.contrast_vs_normal
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
            "normal_standard": node_data.get("normal_standard", ""),
            "critical_check": node_data.get("critical_check", ""),
            "rules": node_data.get("rules", []),
            "defects": []
        }
        
        for neighbor in self.G.neighbors(region_name):
            edge = self.G.get_edge_data(region_name, neighbor)
            if edge.get("relation") == "possible_anomaly":
                defect_data = self.G.nodes[neighbor]
                short_name = defect_data.get('short_name')
                
                # 基础描述
                desc = f"Type: {short_name} | Visual: {defect_data.get('visual_signature')}"
                
                # [新增] 查找该缺陷是否有 distinct_from 边
                # 我们在图中查找从当前缺陷出发的 'distinct_from' 边
                distinctions = []
                for _, target, d_data in self.G.out_edges(neighbor, data=True):
                    if d_data.get("relation") == "distinct_from":
                        # 获取目标缺陷的短名 (从ID里解析，或者去查节点)
                        target_short_name = self.G.nodes[target].get('short_name', target)
                        distinctions.append(f"vs {target_short_name}: {d_data.get('logic')}")
                
                if distinctions:
                    desc += f" | [⚠️ Distinction]: {'; '.join(distinctions)}"
                
                info["defects"].append(desc)
                
        return info

    def get_image_paths(self, region_name: str) -> List[str]:
        if self.G.has_node(region_name):
            return self.G.nodes[region_name].get("image_paths", [])
        return []

    # ==========================================
    # 4. 持久化
    # ==========================================

    def save_to_disk(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data_to_save = {
            "root_name": self.root_name,
            "graph": self.G,
            "node_embeddings": self.node_embeddings,
            "node_names": self.node_names
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
        print(f"🚀 Graph loaded: {load_path} ({self.G.number_of_nodes()} nodes)")
