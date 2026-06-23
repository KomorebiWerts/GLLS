import numpy as np
import random
import traceback
import cv2
import PIL.Image as Image
from PIL import ImageDraw
import os
import re
import torch  # Added for OOM catching
from glls.models.heatmap_regions import (
    HeatmapThresholds,
    boxes_iou,
    detect_heatmap_regions,
    resolve_heatmap_thresholds,
    salient_heatmap_mask,
)
from glls.models.evidence_policy import EvidencePromptPolicy
from glls.models.mcts_action_policy import MCTSActionPolicy
from glls.models.mcts_candidate_selector import MCTSFinalCandidateSelector
from glls.models.mcts_crop_selector import (
    MCTSCropCandidateSelector,
    crop_candidate_audit_item,
    is_duplicate_bbox,
)
from glls.models.task_policy import policy_for_task
from glls.models.logical import VisualLogicProcessor
from glls.rag.semantic_grounding import (
    PVLAEvidenceHypothesizer,
    SemanticOptionGrounder,
    is_normal_or_no_defect_report,
    should_prefer_normal_without_local_evidence,
)
from glls.rag.prompt_context import select_prompt_rag_blocks
from glls.seg.sam3_profiles import (
    OBJECT_PROMPT_ROLES,
    SAM3_PROMPT_SELECTION_POLICY,
    describe_profile,
    normal_part_veto_config,
    plan_refinement_prompts,
    prompt_role_family,
    should_use_sam3_local_refinement,
)

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEFAULT_THRESHOLD = 0.9

# ==========================================
# 1. MCTS Node
# ==========================================
class MCTSNode:
    def __init__(self, state, parent=None, available_actions=None, shared_processor=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.leaf_reward = 0.0
        # 如果 Node 内部真的需要调用它，就保存引用；否则可以直接删掉这个属性
        self.logic_preprocessor = shared_processor
        self.untried_actions = list(available_actions) if available_actions else []
        self.heatmap_score = state.get('heatmap_score', 0.0)
        
    # === 新增：手动断开引用的方法 ===
    def destroy(self):
        """手动断开循环引用，加速 GC"""
        self.parent = None
        self.state = None
        self.logic_preprocessor = None # 只是断开引用，不销毁对象（因为是共享的）
        for child in self.children.values():
            child.destroy()
        self.children.clear()

# ==========================================
# 2. Agent 类
# ==========================================
class MCTSQuestionSample:
    DEFAULT_MCTS_SIMULATIONS = 50
    DEFAULT_MCTS_MAX_DEPTH = 4
    DEFAULT_MCTS_C_PUCT = 2.0

    def __init__(self, row, args, inference_engine, localizer, rag_agent=None, rag_cache=None, rag_blocks=None, sam_engine=None):
        self.row = row
        self.args = args
        self.localizer = localizer
        self.inference_engine = inference_engine
        
        self.rag_agent = rag_agent
        self.rag_cache = rag_cache
        self.rag_blocks = rag_blocks if rag_blocks else []
        
        self.sam_engine = sam_engine
        self.logic_preprocessor = VisualLogicProcessor()
        
        self.debug_logic_view_img = None 
        self.debug_phase1_prompt = ""
        self.image = row['image'] 
        self.image_width, self.image_height = self.image.size 
        self.question = row['question']
        self.options = row.get('options', {})
        self.task_type = row.get('type', '').lower()
        self.task_policy = policy_for_task(self.task_type, dataset=getattr(args, "dataset", None))
        self.mcts_budget_config = self._read_mcts_budget_config(args)
        self.max_depth = self.mcts_budget_config["max_depth"]
        self.c_puct = self.mcts_budget_config["c_puct"]
        self.action_policy = MCTSActionPolicy(c_puct=self.c_puct)
        self.n_simulations = self.mcts_budget_config["n_simulations"]
        self.DISCRETE_ACTIONS = ["move_left", "move_right", "move_up", "move_down", "zoom_in", "zoom_out"]
        self.global_conclusion = "Pending"
        self.category = None
        self.image_threshold = DEFAULT_THRESHOLD
        self.pixel_threshold = DEFAULT_THRESHOLD
        self.threshold_config = HeatmapThresholds(DEFAULT_THRESHOLD, DEFAULT_THRESHOLD, source="default")
        self.region_proposals = []
        self.mcts_action_trace = []
        self.mcts_search_summary = {}
        self.mcts_final_candidate_audit = []
        self.has_global_standard = True
        self.used_logic_engine = False
        self.logic_report_for_verification = ""
        self.atlas_context_for_verification = ""

    @classmethod
    def _read_mcts_budget_config(cls, args):
        def read_int(name, default, min_value=1):
            value = getattr(args, name, default) if args is not None else default
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = default
            return max(min_value, value)

        def read_float(name, default, min_value=0.0):
            value = getattr(args, name, default) if args is not None else default
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = default
            return max(min_value, value)

        # Defaults intentionally preserve the original runtime behavior.
        return {
            "n_simulations": read_int("mcts_simulations", cls.DEFAULT_MCTS_SIMULATIONS),
            "max_depth": read_int("mcts_max_depth", cls.DEFAULT_MCTS_MAX_DEPTH),
            "c_puct": read_float("mcts_c_puct", cls.DEFAULT_MCTS_C_PUCT),
        }
    
    def cleanup(self):
        """显式清理 MCTS 树和重对象"""
        # 1. 清理搜索树
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
            self.root = None

        # 2. 清理 SAM 和 推理引擎的引用 (不是销毁引擎本身，是断开引用)
        self.sam_engine = None
        self.inference_engine = None
        self.localizer = None

        # 3. 清理图像缓存
        self.image = None
        self.debug_logic_view_img = None

        # 4. 清理 RAG 缓存引用
        self.rag_agent = None
        self.rag_cache = None
        self.rag_blocks = None

    @staticmethod
    def clear_sam_cache(sam_engine):
        """
        静态方法：清理SAM引擎的图像缓存
        SAM在set_image后会在GPU上缓存特征图，需要定期清理
        """
        if sam_engine is None:
            return
        try:
            if hasattr(sam_engine, 'predictor'):
                predictor = sam_engine.predictor
                # 尝试清理各种可能的缓存属性
                if hasattr(predictor, 'reset_image'):
                    predictor.reset_image()
                if hasattr(predictor, 'features'):
                    predictor.features = None
                if hasattr(predictor, 'original_size'):
                    predictor.original_size = None
                if hasattr(predictor, 'input_size'):
                    predictor.input_size = None
                if hasattr(predictor, 'is_image_set'):
                    predictor.is_image_set = False
        except Exception:
            pass  # SAM清理失败不影响主流程

    def _extract_atlas_context_text(self, include_global=False):
        lines = []
        for i, block in enumerate(self.rag_blocks or []):
            if isinstance(block, dict) and block.get("prompt_visible") is False:
                continue
            region_name = block.get('region', f'Region {i+1}')
            region_name_l = region_name.lower()
            if (not include_global) and ("whole" in region_name_l or "global" in region_name_l):
                continue
            txt = block.get('text', '').strip()
            if txt:
                lines.append(f"[{region_name}] {txt}")
        return "\n".join(lines).strip()

    def _pvla_option_matches(self):
        matches = []
        for block in self.rag_blocks or []:
            topology = block.get("topology") if isinstance(block, dict) else None
            if not topology:
                continue
            for match in topology.get("option_matches", []) or []:
                matches.append(match)
        return matches

    def _pvla_region_defects(self):
        defects = []
        seen = set()
        for block in self.rag_blocks or []:
            if not isinstance(block, dict):
                continue
            topology = block.get("topology") if isinstance(block.get("topology"), dict) else {}
            for item in topology.get("defects", []) or []:
                if not isinstance(item, dict):
                    continue
                key = item.get("defect_node") or item.get("graph_path") or item.get("defect")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                defects.append(item)
        return defects

    def _pvla_source_backing_by_region(self):
        backing = {}
        for block in self.rag_blocks or []:
            if not isinstance(block, dict):
                continue
            region = str(block.get("region", "")).strip()
            if not region:
                continue
            retrieval = block.get("retrieval") if isinstance(block.get("retrieval"), dict) else {}
            provenance = retrieval.get("provenance") if isinstance(retrieval.get("provenance"), dict) else {}
            graph_cache_path = provenance.get("graph_cache_path", "")
            source_json_path = provenance.get("source_json_path", "")
            text_knowledge_paths = provenance.get("text_knowledge_paths", [])
            has_graph = bool(graph_cache_path)
            has_text = bool(source_json_path or text_knowledge_paths)
            backing[region] = {
                "graph_cache_path": graph_cache_path,
                "source_json_path": source_json_path,
                "text_knowledge_paths": text_knowledge_paths,
                "visual_reference_source_backed": bool(provenance.get("visual_reference_source_backed", False)),
                "source_backed": bool(has_graph and has_text),
            }
        return backing

    def _pvla_option_supported_graph_paths(self):
        supported = set()
        for match in self._pvla_option_matches():
            for item in match.get("matched_defects") or [match]:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("label_only_match", False)):
                    continue
                if "source_evidence_channels" in item and not item.get("source_evidence_channels"):
                    continue
                graph_path = str(item.get("graph_path", "")).strip()
                if graph_path:
                    supported.add(graph_path)
        return supported

    @staticmethod
    def _has_graph_visual_evidence(item):
        return bool(
            str(item.get("graph_path", "")).strip()
            and (
                str(item.get("visual_signature", "")).strip()
                or str(item.get("contrast_vs_normal", "")).strip()
                or item.get("visual_attributes")
                or item.get("distinctions")
            )
        )

    def _format_pvla_evidence_hypothesis(self, defects=None):
        if not self.task_policy.use_pvla_defect_hypothesis_prompt:
            return "", []

        phase_text = self._normalize_text(self.global_conclusion)
        if not phase_text or is_normal_or_no_defect_report(phase_text):
            return "", []

        candidates = PVLAEvidenceHypothesizer().rank_phase_report(
            phase_text,
            defects if defects is not None else self._pvla_region_defects(),
            max_candidates=3,
        )
        source_backing_by_region = self._pvla_source_backing_by_region()
        option_supported_graph_paths = self._pvla_option_supported_graph_paths()
        candidates = [
            {
                **item,
                "source_backing": source_backing_by_region.get(item.get("region", ""), {}),
                "option_scope_supported": str(item.get("graph_path", "")).strip() in option_supported_graph_paths,
            }
            for item in candidates
            if self._has_graph_visual_evidence(item)
            and source_backing_by_region.get(item.get("region", ""), {}).get("source_backed", False)
            and str(item.get("graph_path", "")).strip() in option_supported_graph_paths
        ]
        if not candidates:
            return "", []

        lines = [
            "PVLA graph-backed visual hypotheses from the Phase-1 report:",
            "These are defect/region hypotheses, not answer-option mappings.",
        ]
        for idx, item in enumerate(candidates, start=1):
            attr_text = SemanticOptionGrounder.format_attribute_overlap(item.get("attribute_overlap", {}))
            graph_attr_text = SemanticOptionGrounder.format_attribute_overlap(item.get("graph_attribute_overlap", {}))
            channels = ", ".join(item.get("support_channels") or [])
            path = item.get("graph_path", "")
            signature = item.get("visual_signature", "")
            line = f"- Hypothesis {idx}: {item.get('defect', 'unknown_defect')}"
            if path:
                line += f" via {path}"
            if signature:
                line += f"; expected visual signature: {signature}"
            if attr_text:
                line += f"; shared visual attributes: {attr_text}"
            if graph_attr_text:
                line += f"; graph visual attributes: {graph_attr_text}"
            if channels:
                line += f"; support channels: {channels}"
            lines.append(line)
            distinctions = item.get("distinctions") or []
            for dist in distinctions:
                if isinstance(dist, dict) and dist.get("difference"):
                    lines.append(
                        f"  Distinguish from {dist.get('target_defect', 'other defect')}: {dist.get('difference')}"
                    )
        lines.append(
            "Use these hypotheses only to decide what visual evidence to verify in the red-box and focus views; select an answer option only after that visual verification."
        )
        return "\n".join(lines), candidates

    @staticmethod
    def _is_global_or_whole_region(region_name):
        region_l = str(region_name or "").lower()
        return "whole" in region_l or "global" in region_l

    def _should_keep_global_rag_block(self, block):
        if not isinstance(block, dict):
            return False
        if self.task_policy.include_global_reference_in_final:
            return True
        if self.task_type in {"defect classification", "defect description", "defect analysis"}:
            return True
        return False

    @staticmethod
    def _global_rag_anchor_text(text):
        text = str(text or "").strip()
        if not text:
            return text
        marker_positions = [
            idx for marker in (
                "\n[Potential Defects Details]:",
                "\nPotential Defects Details:",
                "\n>>> Defect:",
            )
            if (idx := text.find(marker)) >= 0
        ]
        if not marker_positions:
            return text
        anchor_text = text[:min(marker_positions)].rstrip()
        return anchor_text or text

    @staticmethod
    def _defect_pattern_count(text):
        return len(re.findall(r"(?m)^\s*>>>\s*Defect\s*:", str(text or "")))

    @classmethod
    def _select_prompt_rag_blocks(cls, rag_blocks, pvla_evidence_hypothesis_audit):
        return select_prompt_rag_blocks(rag_blocks, pvla_evidence_hypothesis_audit)

    def _rag_block_provenance(self):
        provenance_items = []
        for block in self.rag_blocks or []:
            if not isinstance(block, dict):
                continue
            retrieval = block.get("retrieval") if isinstance(block.get("retrieval"), dict) else {}
            provenance = retrieval.get("provenance") if isinstance(retrieval.get("provenance"), dict) else {}
            if not provenance:
                continue
            topology = block.get("topology") if isinstance(block.get("topology"), dict) else {}
            provenance_items.append({
                "region": block.get("region", ""),
                "prompt_visible": block.get("prompt_visible", True),
                "retrieval_mode": retrieval.get("mode", ""),
                "query": retrieval.get("query", ""),
                "graph_cache_path": provenance.get("graph_cache_path", ""),
                "source_json_path": provenance.get("source_json_path", ""),
                "knowledge_source_type": provenance.get("knowledge_source_type", ""),
                "knowledge_dataset": provenance.get("knowledge_dataset", ""),
                "knowledge_category": provenance.get("knowledge_category", ""),
                "text_knowledge_paths": provenance.get("text_knowledge_paths", []),
                "visual_reference_source": provenance.get("visual_reference_source", ""),
                "visual_reference_generation": provenance.get("visual_reference_generation", ""),
                "visual_reference_paths": provenance.get("visual_reference_paths", []),
                "visual_reference_fallback_region": provenance.get("visual_reference_fallback_region", ""),
                "visual_reference_trust": provenance.get("visual_reference_trust", {}),
                "visual_reference_trusted_source": provenance.get("visual_reference_trusted_source", False),
                "visual_reference_path_count": provenance.get("visual_reference_path_count", 0),
                "visual_reference_existing_path_count": provenance.get("visual_reference_existing_path_count", 0),
                "visual_reference_missing_path_count": provenance.get("visual_reference_missing_path_count", 0),
                "visual_reference_source_backed": provenance.get("visual_reference_source_backed", False),
                "pvla_selection": retrieval.get("pvla_selection", {}),
                "pvla_plan_summary": {
                    "selected_from": (retrieval.get("pvla_plan") or {}).get("selected_from", retrieval.get("selected_from", "")),
                    "selected_regions": (retrieval.get("pvla_plan") or {}).get("selected_regions", []),
                    "selection_reason_counts": (retrieval.get("pvla_plan") or {}).get("selection_reason_counts", {}),
                } if isinstance(retrieval.get("pvla_plan"), dict) else {},
                "option_grounding_audit": topology.get("option_grounding_audit", {}),
            })
        return provenance_items

    def _format_pvla_option_grounding_summary(self, matches):
        if not matches:
            return ""

        lines = ["PVLA option grounding from graph paths:"]
        for match in matches:
            defects = match.get("matched_defects") or [match]
            labels = []
            for item in defects:
                defect = item.get("defect", "")
                path = item.get("graph_path", "")
                signature = item.get("visual_signature", "")
                if defect and path:
                    label = f"{defect} via {path}"
                    if signature:
                        label += f" [signature: {signature}]"
                    attr_text = SemanticOptionGrounder.format_attribute_overlap(item.get("attribute_overlap", {}))
                    if attr_text:
                        label += f" [shared attributes: {attr_text}]"
                    graph_attr_text = SemanticOptionGrounder.format_attribute_overlap(item.get("graph_attribute_overlap", {}))
                    if graph_attr_text:
                        label += f" [graph attributes: {graph_attr_text}]"
                    channels = ", ".join(item.get("support_channels") or [])
                    if channels:
                        label += f" [consensus: {channels}]"
                    distinctions = item.get("distinctions") or []
                    dist_text = "; ".join(
                        f"vs {dist.get('target_defect')}: {dist.get('difference')}"
                        for dist in distinctions
                        if isinstance(dist, dict) and dist.get("difference")
                    )
                    if dist_text:
                        label += f" [graph distinctions: {dist_text}]"
                    labels.append(label)
            if labels:
                lines.append(f"- Option {match.get('option_key')} ({match.get('option_text')}): " + "; ".join(labels))
        return "\n".join(lines)

    def _format_phase1_option_prior(self, matches):
        if self.task_type != "defect classification" or not matches:
            return ""

        phase_text = self._normalize_text(self.global_conclusion)
        if not phase_text or is_normal_or_no_defect_report(phase_text):
            return ""

        grounder = SemanticOptionGrounder()
        scored = []
        for match in matches:
            best_item_score = None
            for item in match.get("matched_defects") or [match]:
                if bool(item.get("label_only_match", False)):
                    continue
                if "source_evidence_channels" in item and not item.get("source_evidence_channels"):
                    continue
                option_support_channels = set(item.get("support_channels") or [])
                if option_support_channels and not (
                    option_support_channels & {"defect_name", "visual_signature", "graph_visual_attribute", "semantic_embedding"}
                ):
                    continue
                evidence_text = SemanticOptionGrounder.evidence_text_for_defect(
                    item,
                    include_region=False,
                    include_contrast=False,
                    include_distinction_targets=False,
                )
                score_info = grounder.score_text_pair(phase_text, evidence_text)
                if not score_info.get("evidence_consensus"):
                    continue
                phase_support_channels = set(score_info.get("support_channels") or [])
                if phase_support_channels and not (
                    phase_support_channels & {"lexical_overlap", "semantic_embedding"}
                ):
                    continue
                if best_item_score is None or score_info["score"] > best_item_score["score"]:
                    best_item_score = {**score_info, "defect": item.get("defect", "")}
            if best_item_score and best_item_score["signal_score"] > 0:
                scored.append({
                    "score": float(best_item_score["score"]),
                    "signal_score": float(best_item_score["signal_score"]),
                    "option_key": str(match.get("option_key")),
                    "option_text": str(match.get("option_text")),
                    "defect": str(best_item_score.get("defect", "")),
                    "attribute_overlap": best_item_score.get("attribute_overlap", {}),
                    "support_channels": best_item_score.get("support_channels", []),
                    "source_evidence_channels": best_item_score.get("source_evidence_channels", []),
                    "evidence_consensus": bool(best_item_score.get("evidence_consensus", False)),
                    "token_overlap": best_item_score.get("token_overlap", []),
                })

        if not scored:
            return ""
        scored.sort(reverse=True, key=lambda item: item["score"])
        best = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        if not SemanticOptionGrounder.is_decisive_pair(best, runner_up, margin=0.8):
            return ""
        attr_text = SemanticOptionGrounder.format_attribute_overlap(best.get("attribute_overlap", {}))
        evidence_text = attr_text or ", ".join(best.get("token_overlap", []))

        return (
            f"Phase-1 graph-backed semantic hint: the Phase 1 report shares evidence "
            f"{evidence_text} with graph defect hypothesis {best['defect']}. "
            "Treat this as supporting context only; compare answer options only after verifying the local focus views or red contours."
        )

    def _format_phase1_analysis_prior(self):
        if self.task_type != "defect analysis" or not isinstance(self.options, dict):
            return ""

        phase_text = self._normalize_text(self.global_conclusion)
        if not phase_text or is_normal_or_no_defect_report(phase_text):
            return ""

        grounder = SemanticOptionGrounder()
        scored = []
        for key, text in self.options.items():
            score_info = grounder.score_text_pair(phase_text, text)
            if score_info["signal_score"] > 0 and score_info.get("evidence_consensus"):
                scored.append({
                    "score": float(score_info["score"]),
                    "signal_score": float(score_info["signal_score"]),
                    "option_key": str(key),
                    "option_text": str(text),
                    "attribute_overlap": score_info.get("attribute_overlap", {}),
                    "support_channels": score_info.get("support_channels", []),
                    "evidence_consensus": bool(score_info.get("evidence_consensus", False)),
                    "token_overlap": score_info.get("token_overlap", []),
                })
        if not scored:
            return ""
        scored.sort(reverse=True, key=lambda item: item["score"])
        best = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        if not SemanticOptionGrounder.is_decisive_pair(best, runner_up, margin=0.6):
            return ""
        attr_text = SemanticOptionGrounder.format_attribute_overlap(best.get("attribute_overlap", {}))
        evidence_text = attr_text or ", ".join(best.get("token_overlap", []))
        return (
            f"Phase-1 analysis evidence hint: the Phase 1 report shares visual/semantic evidence "
            f"{evidence_text} with one candidate analysis statement. "
            "Use this only as consistency context; select an option only after visible image verification."
        )

    def _source_backed_graph_attribute_support(self):
        support = {}
        source_backing_by_region = self._pvla_source_backing_by_region()
        for defect in self._pvla_region_defects():
            if not isinstance(defect, dict):
                continue
            region = defect.get("region", "")
            if not source_backing_by_region.get(region, {}).get("source_backed", False):
                continue
            if not self._has_graph_visual_evidence(defect):
                continue
            evidence_text = SemanticOptionGrounder.evidence_text_for_defect(
                defect,
                include_region=False,
                include_contrast=False,
                include_distinction_targets=True,
                include_visual_attributes=True,
            )
            profile = SemanticOptionGrounder.profile(evidence_text)
            for attr in profile.attribute_names:
                support.setdefault(attr, []).append({
                    "region": region,
                    "defect": defect.get("defect", ""),
                    "graph_path": defect.get("graph_path", ""),
                    "hits": sorted(profile.attributes.get(attr, [])),
                })
        return support

    def _format_phase1_visual_primitive_hint(self):
        if self.task_type not in {"defect classification", "defect description", "defect analysis"}:
            return "", []
        if not isinstance(self.options, dict) or not self.options:
            return "", []

        phase_text = self._normalize_text(self.global_conclusion)
        if not phase_text or is_normal_or_no_defect_report(phase_text):
            return "", []

        graph_attribute_support = self._source_backed_graph_attribute_support()
        if not graph_attribute_support:
            return "", []

        grounder = SemanticOptionGrounder()
        candidates = []
        for key, text in self.options.items():
            score_info = grounder.score_text_pair(phase_text, text)
            if not score_info.get("evidence_consensus"):
                continue
            if float(score_info.get("signal_score", 0.0)) < 2.0:
                continue
            supported_attrs = {
                attr: hits
                for attr, hits in (score_info.get("attribute_overlap") or {}).items()
                if attr in graph_attribute_support
            }
            if not supported_attrs:
                continue
            candidates.append({
                "score": float(score_info.get("score", 0.0)),
                "signal_score": float(score_info.get("signal_score", 0.0)),
                "option_key": str(key),
                "option_text": str(text),
                "attribute_overlap": supported_attrs,
                "support_channels": score_info.get("support_channels", []),
                "graph_support": {
                    attr: graph_attribute_support.get(attr, [])[:3]
                    for attr in supported_attrs
                },
            })

        if not candidates:
            return "", []
        candidates.sort(reverse=True, key=lambda item: item["score"])
        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        if not SemanticOptionGrounder.is_decisive_pair(best, runner_up, margin=0.55):
            return "", candidates

        attr_text = SemanticOptionGrounder.format_attribute_overlap(best.get("attribute_overlap", {}))
        graph_paths = []
        for items in best.get("graph_support", {}).values():
            for item in items:
                path = item.get("graph_path", "")
                if path and path not in graph_paths:
                    graph_paths.append(path)
        path_text = "; ".join(graph_paths[:3])
        hint = (
            "Source-backed visual primitive cue: the Phase-1 report, current answer options, "
            f"and PVLA graph evidence share {attr_text}. "
            "Treat this as a visual hypothesis to verify in the Global View/red boxes and Focus Views; "
            "if visible, prefer the option describing this physical primitive over alternatives describing a different primitive."
        )
        if path_text:
            hint += f" Graph support: {path_text}."
        return hint, candidates

    def _is_binary_anomaly_detection_question(self):
        if self.task_type != "anomaly detection" or not isinstance(self.options, dict):
            return False
        option_text = " ".join(str(v).lower() for v in self.options.values())
        return "yes" in option_text and "no" in option_text

    def _should_apply_no_local_anomaly_gate(self, heatmap_peak_score):
        if not self.task_policy.use_no_local_anomaly_gate:
            return False
        return should_prefer_normal_without_local_evidence(
            task_type=self.task_type,
            options=self.options,
            phase_report=self.global_conclusion,
            heatmap_peak_score=heatmap_peak_score,
            image_threshold=self.image_threshold,
            local_evidence_available=bool(getattr(self, "region_proposals", None)),
        )

    async def get_anomaly_heatmap(self, image):
        heatmap_small, obj_name = self.localizer.predict_anomaly_map(image)
        return cv2.resize(heatmap_small, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR), obj_name
    
    def _detect_heatmap_regions(self, heatmap):
        if not self.task_policy.use_local_anomaly_stream:
            self.region_proposals = []
            return []
        self.region_proposals = detect_heatmap_regions(
            heatmap,
            self.threshold_config,
            task_type=self.task_type,
            top_k=3,
        )
        return [
            {
                "bbox": proposal.bbox,
                "score": proposal.score,
                "source": proposal.source,
                "proposal": proposal,
            }
            for proposal in self.region_proposals
        ]
    
    def execute_discrete_action(self, node, action_type):
        gx1, gy1, gx2, gy2 = node.state['region_coords']
        w = gx2 - gx1
        h = gy2 - gy1
        W, H = self.image_width, self.image_height

        step_x = max(1, int(w * 0.1))
        step_y = max(1, int(h * 0.1))
        
        nx1, ny1, nx2, ny2 = gx1, gy1, gx2, gy2

        if action_type == "move_left":
            nx1 -= step_x; nx2 -= step_x
        elif action_type == "move_right":
            nx1 += step_x; nx2 += step_x
        elif action_type == "move_up":
            ny1 -= step_y; ny2 -= step_y
        elif action_type == "move_down":
            ny1 += step_y; ny2 += step_y
        elif action_type == "zoom_in":
            nx1 += int(w * 0.05); ny1 += int(h * 0.05)
            nx2 -= int(w * 0.05); ny2 -= int(h * 0.05)
        elif action_type == "zoom_out":
            nx1 -= int(w * 0.05); ny1 -= int(h * 0.05)
            nx2 += int(w * 0.05); ny2 += int(h * 0.05)

        nx1 = max(0, nx1); ny1 = max(0, ny1)
        nx2 = min(W, nx2); ny2 = min(H, ny2)

        MIN_WINDOW_SIZE = 96 # Increased from 32
        if (nx2 - nx1) < MIN_WINDOW_SIZE or (ny2 - ny1) < MIN_WINDOW_SIZE:
            return None 
        if abs(nx1 - gx1) < 2 and abs(ny1 - gy1) < 2 and abs(nx2 - gx2) < 2:
            return None
        
        current_score = self._score_region_box((int(nx1), int(ny1), int(nx2), int(ny2)))

        return self._create_child_node(node, (int(nx1), int(ny1), int(nx2), int(ny2)), heatmap_score=current_score)

    async def execute_inspect_region(self, node, region_idx):
        regions = node.state.get('global_regions', [])
        if region_idx >= len(regions): return None
        
        x, y, w, h = regions[region_idx]['bbox']
        
        # Context Expansion
        pad_w = int(w * 0.25)
        pad_h = int(h * 0.25)
        target_w = w + 2 * pad_w
        target_h = h + 2 * pad_h
        
        # Size Cap: 不超过全图 50%
        max_allowed_w = self.image_width // 2
        max_allowed_h = self.image_height // 2
        target_w = min(target_w, max_allowed_w)
        target_h = min(target_h, max_allowed_h)
        
        # Min Size
        target_w = max(target_w, 224)
        target_h = max(target_h, 224)
        
        cx = x + w // 2
        cy = y + h // 2
        x1 = max(0, cx - target_w // 2)
        y1 = max(0, cy - target_h // 2)
        x2 = min(self.image_width, x1 + target_w)
        y2 = min(self.image_height, y1 + target_h)
        
        return self._create_child_node(
            node, (int(x1), int(y1), int(x2), int(y2)), 
            heatmap_score=regions[region_idx]['score'], 
        )

    def _region_score_components(self, coords):
        if not hasattr(self, "root") or self.root is None:
            return {
                "score": 0.0,
                "peak": 0.0,
                "focus_mean": 0.0,
                "coverage": 0.0,
                "proposal_alignment": 0.0,
                "roi_area_fraction": 0.0,
            }

        gx1, gy1, gx2, gy2 = coords
        global_heatmap = self.root.state["heatmap_array"]
        h, w = global_heatmap.shape
        y1, y2 = max(0, int(gy1)), min(h, int(gy2))
        x1, x2 = max(0, int(gx1)), min(w, int(gx2))
        roi = global_heatmap[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                "score": 0.0,
                "peak": 0.0,
                "focus_mean": 0.0,
                "coverage": 0.0,
                "proposal_alignment": 0.0,
                "roi_area_fraction": 0.0,
            }

        peak = float(np.max(roi))
        salient_mask, salient_cutoff = salient_heatmap_mask(
            roi,
            tail_fraction=0.05,
            min_score=0.0,
            min_tail_pixels=max(1, int(float(roi.size) * 0.01)),
        )
        if np.any(salient_mask):
            salient_values = roi[salient_mask]
            focus_mean = float(np.mean(salient_values))
            coverage = min(1.0, float(salient_values.size) / float(roi.size) * 8.0)
        else:
            flat = roi.reshape(-1)
            top_n = max(1, int(flat.size * 0.05))
            focus_mean = float(np.mean(np.partition(flat, -top_n)[-top_n:]))
            coverage = 0.0

        proposal_alignment = boxes_iou((x1, y1, x2, y2), self.region_proposals)
        score = 0.60 * peak + 0.25 * focus_mean + 0.10 * coverage + 0.05 * proposal_alignment
        roi_area_fraction = float(max(0, x2 - x1) * max(0, y2 - y1)) / float(max(1, w * h))
        return {
            "score": round(float(score), 6),
            "peak": round(float(peak), 6),
            "focus_mean": round(float(focus_mean), 6),
            "coverage": round(float(coverage), 6),
            "proposal_alignment": round(float(proposal_alignment), 6),
            "roi_area_fraction": round(float(roi_area_fraction), 6),
            "saliency_cutoff": round(float(salient_cutoff), 6) if salient_cutoff is not None else 0.0,
            "score_source": "rank_mass_saliency",
        }

    def _score_region_box(self, coords):
        return float(self._region_score_components(coords).get("score", 0.0))

    def _create_child_node(self, parent_node, coords, heatmap_score=0.0):
        gx1, gy1, gx2, gy2 = coords
        
        crop_pil = self.image.crop((gx1, gy1, gx2, gy2))
        
        global_heatmap = self.root.state['heatmap_array'] 
        
        h, w = global_heatmap.shape
        cy1, cy2 = max(0, int(gy1)), min(h, int(gy2))
        cx1, cx2 = max(0, int(gx1)), min(w, int(gx2))
        
        child_heatmap = global_heatmap[cy1:cy2, cx1:cx2]
        if child_heatmap.size == 0: 
            child_heatmap = np.zeros((int(gy2-gy1), int(gx2-gx1)))
        
        current_peak = float(np.max(child_heatmap)) if child_heatmap.size > 0 else 0.0
        reward_components = self._region_score_components(coords)
        search_score = heatmap_score if heatmap_score > 0 else float(reward_components.get("score", 0.0))
        if heatmap_score > 0: current_peak = heatmap_score
        reward_components["search_score"] = round(float(search_score), 6)
        reward_components["score_source"] = "proposal_heatmap_prior" if heatmap_score > 0 else "local_region_score"

        new_state = {
            'depth': parent_node.state['depth'] + 1,
            'image': crop_pil,
            'image_width': gx2 - gx1, 
            'image_height': gy2 - gy1,
            'region_coords': (gx1, gy1, gx2, gy2),
            'heatmap_score': current_peak, 
            'search_score': search_score,
            'reward_components': reward_components,
            'heatmap_array': child_heatmap,
            'global_regions': parent_node.state.get('global_regions', [])
        }

        return MCTSNode(
            new_state, 
            parent=parent_node, 
            available_actions=self.action_policy.local_actions(new_state, pixel_threshold=self.pixel_threshold),
            shared_processor=self.logic_preprocessor # 传递 Agent 持有的那个唯一实例
        )

    def selection(self, node):
        if node.untried_actions: return node
        if not node.children: return node
        total_visits = sum(c.visits for c in node.children.values())
        return self.selection(max(node.children.values(), key=lambda child: self.action_policy.ucb_score(child, total_visits)))

    async def expansion(self, node):
        if node.state['depth'] >= self.max_depth or not node.untried_actions: 
            return node, None, "terminal_or_no_action"
        
        while node.untried_actions:
            action = node.untried_actions.pop(0)
            child = None

            if action.startswith("inspect_region_"):
                child = await self.execute_inspect_region(node, int(action.split("_")[-1]))
            elif action in self.DISCRETE_ACTIONS:
                child = self.execute_discrete_action(node, action)

            if child:
                child.state["incoming_action"] = action
                node.children[action] = child
                return child, action, "expanded"
        
        return node, None, "no_valid_action"

    async def simulation(self, node):
        score = node.state.get('search_score', node.heatmap_score)
        if score < 0.05: return 0.0
        return score

    async def simulation_with_audit(self, node):
        reward = await self.simulation(node)
        components = dict((node.state or {}).get("reward_components") or {})
        if not components:
            components = {
                "score": round(float(reward), 6),
                "search_score": round(float(reward), 6),
                "score_source": "node_search_score",
            }
        return reward, {
            "reward_source": components.get("score_source", "node_search_score"),
            "reward_components": components,
        }

    def backpropagation(self, node, reward):
        while node:
            node.visits += 1; node.value += reward; node = node.parent

    def _build_rag_query(self):
        # RAG retrieval is driven by task/question evidence. Answer options are
        # retained as audit-only PVLA grounding in RagAgent, but they should not
        # pull prompt-visible graph regions by lexical match.
        return f"{self.task_type}\n{self.question}".strip()

    def _build_sam3_prompt_query(self):
        # SAM3 prompt selection must not be driven by answer-option wording.
        # Options can contain defect labels that would make text prompts look
        # relevant even when the image evidence does not support that part.
        return f"{self.task_type}\n{self.question}\n{self.global_conclusion}".strip()

    async def single_run(self, root):
        selected = self.selection(root)
        node = selected
        expanded_action = None
        expansion_status = "not_expanded"
        if node.state['depth'] < self.max_depth:
            node, expanded_action, expansion_status = await self.expansion(node)
        reward, simulation_audit = await self.simulation_with_audit(node)
        node.leaf_reward = reward
        backprop_path = self._node_path_actions(node)
        self.backpropagation(node, reward)
        self.mcts_action_trace.append({
            "iteration": len(self.mcts_action_trace) + 1,
            "selection": self.action_policy.node_snapshot(selected),
            "expanded_action": expanded_action,
            "expansion_status": expansion_status,
            "result": self.action_policy.node_snapshot(node),
            "reward": round(float(reward), 6),
            "simulation": simulation_audit,
            "backprop_path_length": len(backprop_path) + 1,
            "terminal": expansion_status != "expanded",
            "path_actions": backprop_path,
        })

    @staticmethod
    def _node_path_actions(node):
        actions = []
        current = node
        while current is not None:
            state = getattr(current, "state", {}) or {}
            action = state.get("incoming_action")
            if action:
                actions.append(str(action))
            current = getattr(current, "parent", None)
        return list(reversed(actions))

    async def process(self):
        setup_seed(42) 
        return await self._process_logic()

    async def _process_logic(self):
        if self.task_policy.use_local_anomaly_stream:
            global_heatmap, detected_category = await self.get_anomaly_heatmap(self.image)
        else:
            global_heatmap = np.zeros((self.image_height, self.image_width), dtype=np.float32)
            detected_category = None
        self.logic_report_for_verification = ""
        self.atlas_context_for_verification = ""

        if self.sam_engine:
            self.sam_engine.set_image(self.image)
        self.dataset_category = self.row.get("category")
        self.localizer_detected_category = detected_category
        if self.dataset_category:
            # The QA row category is the source-backed key for graph/RAG/SAM3
            # profiles. The localizer may emit a visual nearest-neighbor class,
            # but using it to choose knowledge can silently cross-contaminate
            # category provenance.
            self.category = self.dataset_category
            self.category_source = "row_category"
        elif detected_category is not None:
            self.category = detected_category
            self.category_source = "localizer_detected_category_fallback"
        else:
            self.category = self.row["category"]
            self.category_source = "row_category_fallback"

        self.threshold_config = resolve_heatmap_thresholds(
            self.localizer,
            self.category,
            global_heatmap,
            default_image=DEFAULT_THRESHOLD,
            default_pixel=DEFAULT_THRESHOLD,
        )
        self.image_threshold = self.threshold_config.image
        self.pixel_threshold = self.threshold_config.pixel

        if not self.rag_blocks and self.rag_agent is not None:
            rag_query = self._build_rag_query()
            rag_top_k = getattr(self.args, "rag_top_k", 4)
            rag_retrieval_mode = getattr(self.args, "rag_retrieval_mode", "full")
            options_key = tuple(
                sorted((str(key), str(value)) for key, value in (self.options or {}).items())
            )
            cache_key = (self.category, rag_query, rag_top_k, rag_retrieval_mode, options_key)
            if self.rag_cache is not None and cache_key in self.rag_cache:
                self.rag_blocks = self.rag_cache[cache_key]
            else:
                try:
                    blocks = self.rag_agent.get_rag_context(
                        self.category,
                        query=rag_query,
                        top_k=rag_top_k,
                        retrieval_mode=rag_retrieval_mode,
                        options=self.options,
                    )
                except TypeError:
                    blocks = self.rag_agent.get_rag_context(self.category)
                self.rag_blocks = blocks
                if self.rag_cache is not None:
                    self.rag_cache[cache_key] = blocks

        self._perform_global_check()

        regions = self._detect_heatmap_regions(global_heatmap)
        root_actions = self.action_policy.root_actions(regions) if self.task_policy.use_mcts_search else []
        root_salient_mask, root_salient_cutoff = salient_heatmap_mask(
            global_heatmap,
            tail_fraction=0.05,
            min_score=0.0,
            min_tail_pixels=max(1, int(float(global_heatmap.size) * 0.01)),
        )
        root_coverage = 0.0
        if np.any(root_salient_mask):
            root_coverage = min(1.0, float(np.count_nonzero(root_salient_mask)) / float(global_heatmap.size) * 8.0)
        
        root_state = {
            'depth': 0, 'image': self.image, 
            'image_width': self.image_width, 'image_height': self.image_height,
            'region_coords': (0, 0, self.image_width, self.image_height),
            'heatmap_score': float(np.max(global_heatmap)), 'heatmap_array': global_heatmap,
            'search_score': float(np.max(global_heatmap)),
            'reward_components': {
                "score": round(float(np.max(global_heatmap)), 6),
                "search_score": round(float(np.max(global_heatmap)), 6),
                "peak": round(float(np.max(global_heatmap)), 6),
                "focus_mean": round(float(np.max(global_heatmap)), 6),
                "coverage": round(float(root_coverage), 6),
                "proposal_alignment": 1.0,
                "roi_area_fraction": 1.0,
                "saliency_cutoff": round(float(root_salient_cutoff), 6) if root_salient_cutoff is not None else 0.0,
                "score_source": "global_rank_mass_saliency",
            },
            'global_regions': regions
        }
        # [修改] 创建根节点时也一样
        self.root = MCTSNode(
            root_state, 
            available_actions=root_actions,
            shared_processor=self.logic_preprocessor
        )
        
        self.mcts_action_trace = []
        mcts_skipped_reason = ""
        if self.task_policy.use_mcts_search and root_actions:
            for _ in range(self.n_simulations):
                await self.single_run(self.root)
            budget = self.n_simulations
        else:
            budget = 0
            if self.task_policy.use_mcts_search and not root_actions:
                mcts_skipped_reason = "no_region_proposals"
        self.mcts_search_summary = self.action_policy.summarize_trace(
            self.mcts_action_trace,
            budget=budget,
        )
        if mcts_skipped_reason:
            self.mcts_search_summary["skipped_reason"] = mcts_skipped_reason

        all_nodes = []
        queue = [self.root]
        while queue:
            n = queue.pop(0)
            if n != self.root: all_nodes.append(n)
            queue.extend(n.children.values())
        
        final_candidates = self._select_final_mcts_nodes(all_nodes, iou_threshold=0.1)
        if not final_candidates:
            final_candidates = [self.root]
        
        return await self._generate_final_answer_multi(final_candidates)
 
    def _perform_global_check(self):
        rag_content_list = []
        self.has_global_standard = False
        self.used_logic_engine = False
        for i, block in enumerate(self.rag_blocks):
            if isinstance(block, dict) and block.get("prompt_visible") is False:
                continue
            region_name = block.get('region', f'Region {i+1}')
            if "whole" in region_name.lower() or "global" in region_name.lower() or self.category == "cable":
                self.has_global_standard = True
                txt = block.get('text', '').strip()
                rag_content_list.append({"type": "text", "text": f"\n--- [REFERENCE STANDARD] for {region_name} ---\n"})
                
                include_ref_images = getattr(self.args, "include_rag_reference_images", False)
                if include_ref_images and block.get('images'):
                    rag_content_list.append({"type": "text", "text": "(Visual Reference of a NORMAL object):\n"})
                    for img_item in block['images']:
                        pil_img = None
                        if isinstance(img_item, Image.Image): pil_img = img_item
                        else:
                            if os.path.exists(str(img_item)): pil_img = Image.open(str(img_item)).convert("RGB")
                        if pil_img: rag_content_list.append({"type": "image", "image": pil_img})
                
                if txt:
                    rag_content_list.append({"type": "text", "text": f"(Text Definitions): {txt}\n"})

        if not self.has_global_standard and self.category != "cable" and not self.task_policy.is_object_task:
            self.global_conclusion = "Skipped (No global standard defined)."
            self.logic_report_for_verification = self.global_conclusion
            return
        
        logic_report = ""
        debug_img = None
        hard_verdict = None
        
        if self.sam_engine and hasattr(self, 'logic_preprocessor'):
            try:
                logic_report, debug_img, hard_verdict = self.logic_preprocessor.get_logic_analysis(
                    self.category, self.image, self.sam_engine
                )
                self.debug_logic_view_img = debug_img
            except Exception as e:
                print(f"[Logic Check Error] {e}")
                logic_report = f"Logic check error: {str(e)}"

        if hard_verdict:
            self.used_logic_engine = True
            self.global_conclusion = hard_verdict
            self.logic_report_for_verification = logic_report.strip() if isinstance(logic_report, str) and logic_report.strip() else hard_verdict
            
            self.debug_phase1_prompt = (
                "###  LOGIC ENGINE VERDICT \n\n"
                "Computer Vision detected a definitive structural error.\n"
                "**DETECTED REPORT:**\n" + logic_report + "\n\n"
                "**FINAL CONCLUSION:**\n" + hard_verdict
            )
            return

        content_list = []
        content_list.append({"type": "text", "text": "=== PHASE 1: GLOBAL INSPECTION ===\n"})
        content_list.append({"type": "text", "text": "You are a strict industrial QA inspector.\n"})
        
        # 如果有 logic_report（结构性检查结果），先告知 VLM
        if logic_report and "[Logic Engine]" in logic_report:
            content_list.append({"type": "text", "text": f"{logic_report}\n\n"})
        
        # 关键修改 1: 明确分离“参考标准”和“待测图片”
        if rag_content_list:
            content_list.append({"type": "text", "text": "Below is the REFERENCE KNOWLEDGE (The 'Rulebook'). Do NOT assume the target image follows these rules.\n"})
            content_list.extend(rag_content_list)
            content_list.append({"type": "text", "text": "\n=============================================\n"})

        # 添加待测图
        content_list.append({"type": "text", "text": "(TARGET IMAGE - The object you must inspect):\n"})
        content_list.append({"type": "image", "image": self.image.convert("RGB")})
        
        # 关键修改 2: 强制分步思维链 (Chain of Thought)
        # 强迫模型先输出视觉事实，再进行比对
        if self.task_policy.is_object_task:
            prompt = (
                "INSTRUCTION: Perform global object QA analysis strictly following these steps. Do not skip steps.\n\n"
                "STEP 1: TARGET OBJECT OBSERVATION\n"
                "Look at the TARGET IMAGE and describe the object category, visible parts, structure, material, color, and printed/functional details relevant to the question.\n\n"
                "STEP 2: REFERENCE STRUCTURE CHECK\n"
                "Read the REFERENCE KNOWLEDGE if provided. Use it as the normal object/part structure standard, not as proof that the target image is normal.\n\n"
                "STEP 3: QUESTION-FOCUSED SUMMARY\n"
                "Summarize only the visual facts needed for the question and answer options. Do not infer an anomaly from the absence of local heatmap crops.\n\n"
                "Output your final conclusion strictly in one sentence starting with 'Global Status:'."
            )
        else:
            prompt = (
                "INSTRUCTION: Perform the inspection strictly following these steps. Do not skip steps.\n\n"
                "STEP 1: BLIND OBSERVATION (Ignore the standard for a moment)\n"
                "Look at the 'TARGET IMAGE' only. Describe the key visual attributes you see.\n"
                "- Do not try to match the standard yet. Just report what you see.\n\n"

                "STEP 2: READ STANDARD\n"
                "Now read the [REFERENCE STANDARD] provided above. What should the attributes be for a 'Normal' object?\n\n"

                "STEP 3: COMPARISON & VERIFICATION\n"
                "Compare your observation from Step 1 with the rule from Step 2.\n"
                "- Does it match the 'Normal' description perfectly?\n"
                "- OR does it clearly match one of the 'Defect' visual signatures?\n"

                "STEP 4: CONCLUSION\n"
                "Output the final result. If a mismatch is found, name the defect.\n\n"

                "Output your final conclusion strictly in one sentence starting with 'Global Status:'."
            )
        content_list.append({"type": "text", "text": prompt})

        response = self.inference_engine.generate(content_list, max_tokens=1024)
        self.global_conclusion = response.strip()
        self.logic_report_for_verification = logic_report.strip() if isinstance(logic_report, str) and logic_report.strip() else self.global_conclusion

    def _resize_mask_to_image(self, mask_bool):
        if mask_bool is None:
            return None
        mask_arr = np.asarray(mask_bool)
        while mask_arr.ndim > 2:
            if mask_arr.shape[0] == 1:
                mask_arr = mask_arr[0]
            elif mask_arr.shape[-1] == 1:
                mask_arr = mask_arr[..., 0]
            else:
                mask_arr = np.max(mask_arr, axis=0)
        mask_arr = mask_arr.astype(bool)
        if mask_arr.shape != (self.image_height, self.image_width):
            mask_arr = cv2.resize(
                mask_arr.astype(np.uint8),
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask_arr

    def _sam_mask_quality(self, mask_bool):
        mask_arr = self._resize_mask_to_image(mask_bool)
        if mask_arr is None or mask_arr.size == 0:
            return {
                "accepted": False,
                "reasons": ["empty_mask"],
                "area_ratio": 0.0,
                "bbox_area_ratio": 0.0,
                "edge_touch_count": 0,
                "bbox_xyxy": None,
            }
        mask_area = int(np.count_nonzero(mask_arr))
        image_area = max(1, int(mask_arr.size))
        if mask_area <= 0:
            return {
                "accepted": False,
                "reasons": ["empty_mask"],
                "area_ratio": 0.0,
                "bbox_area_ratio": 0.0,
                "edge_touch_count": 0,
                "bbox_xyxy": None,
            }

        ys, xs = np.where(mask_arr)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        edge_touch_count = int(x1 <= 0) + int(y1 <= 0) + int(x2 >= self.image_width) + int(y2 >= self.image_height)
        area_ratio = float(mask_area) / float(image_area)
        bbox_area_ratio = float(bbox_area) / float(max(1, self.image_width * self.image_height))

        reasons = []
        if area_ratio < 0.0005:
            reasons.append("mask_too_small")
        if area_ratio > 0.85:
            reasons.append("mask_too_large")
        if bbox_area_ratio > 0.92:
            reasons.append("bbox_near_whole_image")
        if edge_touch_count >= 3 and area_ratio > 0.25:
            reasons.append("mask_hugs_image_boundary")

        return {
            "accepted": not reasons,
            "reasons": reasons,
            "area_ratio": round(float(area_ratio), 6),
            "bbox_area_ratio": round(float(bbox_area_ratio), 6),
            "edge_touch_count": edge_touch_count,
            "bbox_xyxy": [x1, y1, x2, y2],
        }

    def _is_usable_sam_mask(self, mask_bool):
        return bool(self._sam_mask_quality(mask_bool).get("accepted", False))

    def _sam3_normal_part_veto_audit(self, dataset_name):
        config = normal_part_veto_config(self.category, dataset_name)
        audit = {
            "enabled": bool(config),
            "policy": (config or {}).get("policy", ""),
            "status": "not_configured",
            "applies_to": "binary_anomaly_detection_no_local_evidence",
            "normal_role": "",
            "normal_text": "",
            "normal_score": 0.0,
            "normal_quality": {},
            "normal_passed": False,
            "competing_defect_attempts": [],
            "best_defect_score": 0.0,
            "best_defect_role": "",
            "veto_yes_to_no": False,
            "reason": "",
        }
        if not config:
            audit["reason"] = "no_normal_part_veto_profile"
            return audit
        if self.sam_engine is None:
            audit["status"] = "not_attempted"
            audit["reason"] = "sam_engine_unavailable"
            return audit

        normal_prompt = config["normal_prompt"]
        audit["status"] = "attempted"
        audit["normal_role"] = normal_prompt.role
        audit["normal_text"] = normal_prompt.text
        try:
            normal_mask, normal_score = self.sam_engine.predict_mask(
                normal_prompt.text,
                threshold=normal_prompt.threshold,
            )
            normal_quality = self._sam_mask_quality(normal_mask)
        except Exception as exc:
            audit["status"] = "error"
            audit["reason"] = f"normal_prompt_error:{exc}"
            return audit

        normal_area = float(normal_quality.get("area_ratio", 0.0) or 0.0)
        normal_score = float(normal_score or 0.0)
        audit["normal_score"] = round(normal_score, 6)
        audit["normal_quality"] = normal_quality
        normal_passed = (
            bool(normal_quality.get("accepted"))
            and normal_score >= float(config.get("normal_min_score", 0.0))
            and normal_area >= float(config.get("normal_min_area_ratio", 0.0))
            and normal_area <= float(config.get("normal_max_area_ratio", 1.0))
        )
        audit["normal_passed"] = normal_passed

        best_defect_score = 0.0
        best_defect_role = ""
        for defect_prompt in config.get("defect_prompts", ()):
            attempt = {
                "role": defect_prompt.role,
                "text": defect_prompt.text,
                "threshold": defect_prompt.threshold,
                "score": 0.0,
                "mask_quality": {},
                "status": "attempted",
            }
            try:
                defect_mask, defect_score = self.sam_engine.predict_mask(
                    defect_prompt.text,
                    threshold=defect_prompt.threshold,
                )
                defect_quality = self._sam_mask_quality(defect_mask)
                defect_score = float(defect_score or 0.0)
                attempt["score"] = round(defect_score, 6)
                attempt["mask_quality"] = defect_quality
                attempt["status"] = "accepted" if defect_quality.get("accepted") else ";".join(
                    defect_quality.get("reasons") or ["unusable_sam_mask"]
                )
                if defect_quality.get("accepted") and defect_score > best_defect_score:
                    best_defect_score = defect_score
                    best_defect_role = defect_prompt.role
            except Exception as exc:
                attempt["status"] = "error"
                attempt["error"] = str(exc)
            audit["competing_defect_attempts"].append(attempt)

        audit["best_defect_score"] = round(float(best_defect_score), 6)
        audit["best_defect_role"] = best_defect_role
        score_advantage = float(config.get("score_advantage", 0.0))
        veto = normal_passed and normal_score >= (best_defect_score + score_advantage)
        audit["veto_yes_to_no"] = bool(veto)
        if veto:
            audit["reason"] = "normal_part_mask_beats_competing_defect_masks"
        elif not normal_passed:
            audit["reason"] = "normal_part_mask_not_confident"
        else:
            audit["reason"] = "competing_defect_mask_not_sufficiently_lower"
        return audit

    def _mask_bbox_xyxy(self, mask_bool):
        mask_arr = self._resize_mask_to_image(mask_bool)
        if mask_arr is None or not np.any(mask_arr):
            return None
        ys, xs = np.where(mask_arr)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def _proposal_roi_mask(self, bbox, padding_factor=0.2):
        x, y, w, h = bbox
        pad_w = int(w * padding_factor)
        pad_h = int(h * padding_factor)
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(self.image_width, int(x + w + pad_w))
        y2 = min(self.image_height, int(y + h + pad_h))
        roi = np.zeros((self.image_height, self.image_width), dtype=bool)
        if x2 > x1 and y2 > y1:
            roi[y1:y2, x1:x2] = True
        return roi

    @staticmethod
    def _is_duplicate_bbox(bbox_xyxy, existing_boxes, iou_threshold=0.78):
        return is_duplicate_bbox(bbox_xyxy, existing_boxes, iou_threshold=iou_threshold)

    def _proposal_padded_bbox_xyxy(self, bbox, padding_factor=0.4):
        x, y, w, h = bbox
        pad_w = int(w * padding_factor)
        pad_h = int(h * padding_factor)
        cx1 = max(0, int(x - pad_w))
        cy1 = max(0, int(y - pad_h))
        cx2 = min(self.image_width, int(x + w + pad_w))
        cy2 = min(self.image_height, int(y + h + pad_h))
        return (cx1, cy1, cx2, cy2)

    @staticmethod
    def _track_crop_box(bbox_xyxy, selected_crop_boxes):
        if bbox_xyxy is None:
            return
        selected_crop_boxes.append(tuple(int(v) for v in bbox_xyxy))

    @staticmethod
    def _skipped_bbox_audit_item(*, source, label, bbox_xyxy, reason, **extra):
        item = {
            "status": "skipped",
            "reason": reason,
            "label": label,
            "source": source,
            "bbox_xyxy": [int(v) for v in bbox_xyxy] if bbox_xyxy is not None else None,
        }
        item.update(extra)
        return item

    def _crop_proposal_with_padding(self, bbox, padding_factor=0.4):
        cx1, cy1, cx2, cy2 = self._proposal_padded_bbox_xyxy(bbox, padding_factor=padding_factor)
        return self.image.crop((cx1, cy1, cx2, cy2))

    def _crop_bbox_xyxy_with_padding(self, bbox_xyxy, padding_factor=0.25):
        if bbox_xyxy is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad_w = int(w * padding_factor)
        pad_h = int(h * padding_factor)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(self.image_width, x2 + pad_w)
        cy2 = min(self.image_height, y2 + pad_h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return self.image.crop((cx1, cy1, cx2, cy2))

    def _proposal_audit(self, proposals):
        items = []
        image_area = max(1, int(self.image_width) * int(self.image_height))
        for idx, proposal in enumerate(proposals, start=1):
            x, y, w, h = proposal.bbox
            items.append({
                "index": idx,
                "bbox": [int(x), int(y), int(w), int(h)],
                "xyxy": [int(v) for v in proposal.xyxy],
                "normalized_box": [round(float(v), 4) for v in proposal.normalized_box(self.image_width, self.image_height)],
                "score": round(float(proposal.score), 6),
                "source": proposal.source,
                "component_id": proposal.component_id,
                "selection_score": round(float(proposal.selection_score), 6) if proposal.selection_score is not None else None,
                "selection_reason": proposal.selection_reason,
                "selection_features": proposal.selection_features or {},
                "proposal_pixel_threshold": round(float(proposal.pixel_threshold), 6) if proposal.pixel_threshold is not None else None,
                "area_fraction": round(float(w * h) / float(image_area), 6),
                "threshold_source": self.threshold_config.source,
                "image_threshold": round(float(self.image_threshold), 6),
                "pixel_threshold": round(float(self.pixel_threshold), 6),
            })
        return items

    def _select_final_mcts_nodes(self, nodes, *, iou_threshold=0.1):
        score_threshold = float(getattr(self, "pixel_threshold", DEFAULT_THRESHOLD) or DEFAULT_THRESHOLD)
        selection = MCTSFinalCandidateSelector(self.action_policy, max_candidates=3).select(
            nodes or [],
            root=getattr(self, "root", None),
            image_width=int(getattr(self, "image_width", 0) or 0),
            image_height=int(getattr(self, "image_height", 0) or 0),
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        self.mcts_final_candidate_audit = selection.audit
        return selection.selected_nodes

    @staticmethod
    def _crop_candidate_audit_item(candidate, *, status, rank=None, selected_index=None, reason=""):
        return crop_candidate_audit_item(
            candidate,
            status=status,
            rank=rank,
            selected_index=selected_index,
            reason=reason,
        )

    @staticmethod
    def _sam3_crop_artifact_rule(crop_evidence_audit):
        has_sam3_crop = any(
            isinstance(item, dict) and str(item.get("source", "")).startswith("sam3")
            for item in crop_evidence_audit or []
        )
        if not has_sam3_crop:
            return ""
        return (
            "**SAM3 Mask-Cutout Rule**: Some Focus Views are segmentation-mask cutouts. "
            "Treat white/blank background, mask edges, alpha boundaries, and crop borders "
            "as cutout artifacts, not as cracks, holes, missing material, or object boundaries by themselves. "
            "Confirm any suspected structural defect against the Global View/red-box context or a non-mask focus crop before choosing a defect type.\n"
        )

    def _raw_proposal_fill_reasons(self, *, use_sam3_local_refinement, proposals, has_crop_images):
        if not self.task_policy.use_raw_heatmap_fill:
            return []
        reasons = []
        if use_sam3_local_refinement:
            reasons.append("sam3_local_refinement_enabled")
        if any(proposal.source == "strict" for proposal in proposals):
            reasons.append("strict_heatmap_proposal_available")
        if any(proposal.source == "ranked_tail" for proposal in proposals):
            reasons.append("ranked_tail_heatmap_proposal_available")
        if any(proposal.source == "fallback" for proposal in proposals):
            reasons.append("fallback_heatmap_proposal_available")
        if not has_crop_images:
            reasons.append("no_selected_crop_candidates")
        return reasons

    @staticmethod
    def _normalize_text(text):
        return re.sub(r"\s+", " ", str(text or "").lower().replace("_", " ")).strip()

    def _spatial_terms_from_center(self, cx, cy):
        horizontal = "center"
        if cx < 0.34:
            horizontal = "left"
        elif cx > 0.66:
            horizontal = "right"

        vertical = "center"
        if cy < 0.34:
            vertical = "top"
        elif cy > 0.66:
            vertical = "bottom"

        fine_horizontal = horizontal
        fine_vertical = vertical
        fine_modifier = ""
        if horizontal == "center":
            if 0.40 <= cx < 0.50:
                fine_horizontal = "left"
                fine_modifier = "slightly"
            elif 0.50 < cx <= 0.60:
                fine_horizontal = "right"
                fine_modifier = "slightly"
        if vertical == "center":
            if 0.40 <= cy < 0.50 and fine_horizontal == "center":
                fine_vertical = "top"
                fine_modifier = "slightly"
            elif 0.50 < cy <= 0.60 and fine_horizontal == "center":
                fine_vertical = "bottom"
                fine_modifier = "slightly"

        terms = {horizontal, vertical}
        if fine_horizontal != "center":
            terms.add(fine_horizontal)
        if fine_vertical != "center":
            terms.add(fine_vertical)
        if fine_modifier:
            terms.add(fine_modifier)
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "fine_horizontal": fine_horizontal,
            "fine_vertical": fine_vertical,
            "modifier": fine_modifier,
            "terms": {term for term in terms if term},
        }

    @staticmethod
    def _phrase_from_terms(terms):
        horizontal = terms.get("horizontal", "center")
        vertical = terms.get("vertical", "center")
        if horizontal == "center" and vertical == "center":
            return "center"
        if vertical == "center":
            return f"center {horizontal}"
        if horizontal == "center":
            return f"{vertical} center"
        return f"{vertical} {horizontal}"

    @staticmethod
    def _fine_phrase_from_terms(terms):
        modifier = terms.get("modifier", "")
        horizontal = terms.get("fine_horizontal", terms.get("horizontal", "center"))
        vertical = terms.get("fine_vertical", terms.get("vertical", "center"))
        if modifier and horizontal in {"left", "right"} and vertical == "center":
            return f"center slightly {horizontal}"
        if modifier and vertical in {"top", "bottom"} and horizontal == "center":
            return f"slightly {vertical} center"
        return MCTSQuestionSample._phrase_from_terms({"horizontal": horizontal, "vertical": vertical})

    def _spatial_phrase_for_bbox(self, bbox):
        x, y, w, h = bbox
        cx = (float(x) + float(w) / 2.0) / max(1.0, float(self.image_width))
        cy = (float(y) + float(h) / 2.0) / max(1.0, float(self.image_height))
        terms = self._spatial_terms_from_center(cx, cy)
        return self._phrase_from_terms(terms), cx, cy

    def _spatial_detail_for_bbox_xyxy(self, bbox_xyxy):
        if bbox_xyxy is None:
            return {}
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        except Exception:
            return {}
        if x2 <= x1 or y2 <= y1:
            return {}
        cx = ((x1 + x2) / 2.0) / max(1.0, float(self.image_width))
        cy = ((y1 + y2) / 2.0) / max(1.0, float(self.image_height))
        terms = self._spatial_terms_from_center(cx, cy)
        coarse_phrase = self._phrase_from_terms(terms)
        fine_phrase = self._fine_phrase_from_terms(terms)
        return {
            "center": [round(float(cx), 3), round(float(cy), 3)],
            "coarse_phrase": coarse_phrase,
            "fine_phrase": fine_phrase,
            "terms": sorted(terms.get("terms") or []),
        }

    def _with_spatial_crop_label(self, label, spatial_detail):
        if self.task_type != "defect localization":
            return label
        if not isinstance(spatial_detail, dict) or not spatial_detail:
            return label
        return label

    @staticmethod
    def _union_bbox(proposals):
        if not proposals:
            return None
        x1 = min(p.bbox[0] for p in proposals)
        y1 = min(p.bbox[1] for p in proposals)
        x2 = max(p.bbox[0] + p.bbox[2] for p in proposals)
        y2 = max(p.bbox[1] + p.bbox[3] for p in proposals)
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    @staticmethod
    def _option_location_terms(option_text):
        text = MCTSQuestionSample._normalize_text(option_text)
        terms = set()
        patterns = {
            "left": [r"\bleft\b"],
            "right": [r"\bright\b"],
            "top": [r"\btop\b", r"\bupper\b"],
            "bottom": [r"\bbottom\b", r"\blower\b"],
            "center": [r"\bcenter\b", r"\bcentre\b", r"\bcentral\b", r"\bmiddle\b"],
            "slightly": [r"\bslight(?:ly)?\b", r"\boff center\b", r"\boff-center\b"],
            "corner": [r"\bcorner\b"],
        }
        for term, term_patterns in patterns.items():
            if any(re.search(pattern, text) for pattern in term_patterns):
                terms.add(term)
        return terms

    @staticmethod
    def _score_location_option(primary_terms, option_terms):
        if not option_terms:
            return None

        score = 0.0
        for term in ("left", "right", "top", "bottom"):
            if term in primary_terms and term in option_terms:
                score += 2.0
        if "center" in primary_terms and "center" in option_terms:
            score += 1.5
        if "slightly" in primary_terms and "slightly" in option_terms:
            score += 0.5
        if "corner" in option_terms and not (
            ({"top", "bottom"} & primary_terms) and ({"left", "right"} & primary_terms)
        ):
            score -= 1.0

        for a, b in (("left", "right"), ("top", "bottom")):
            if a in primary_terms and b in option_terms:
                score -= 2.0
            if b in primary_terms and a in option_terms:
                score -= 2.0
        return score

    def _format_spatial_option_prior(self, primary_terms):
        if not isinstance(self.options, dict) or not self.options:
            return ""

        scored = []
        for key, text in self.options.items():
            option_terms = self._option_location_terms(text)
            score = self._score_location_option(primary_terms, option_terms)
            if score is None:
                continue
            scored.append((float(score), str(key), str(text), sorted(option_terms)))

        if not scored:
            return ""
        scored.sort(reverse=True, key=lambda item: item[0])
        best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -999.0
        if best[0] < 1.5 or best[0] - runner_up < 0.5:
            return ""

        return (
            f"Coordinate-to-option audit: Option {best[1]} ({best[2]}) best matches "
            f"the aggregated red-contour location terms {sorted(primary_terms)}. "
            "This audit is not prompt-visible; final selection must be based on visible localization evidence."
        )

    @staticmethod
    def _spatial_candidate_instruction(option_prior):
        return EvidencePromptPolicy.spatial_localization_instruction()

    @staticmethod
    def _is_structural_sam3_role(prompt_role):
        role = str(prompt_role or "").lower()
        if not role.startswith("defect_"):
            return False
        weak_terms = {
            "scratch", "contamination", "stain", "color", "oil", "glue",
            "gray", "liquid", "rough",
        }
        if any(term in role for term in weak_terms):
            return False
        structural_terms = {
            "broken", "missing", "crack", "hole", "split", "bent", "damaged",
            "cut", "bite", "chip", "deform", "misplaced", "pin", "wick",
            "component", "spoke", "lead", "teeth", "bristle", "short",
        }
        return any(term in role for term in structural_terms)

    def _sam3_structural_location_lines(self, crop_evidence_audit):
        candidates = []
        for item in crop_evidence_audit or []:
            if not isinstance(item, dict) or item.get("source") != "sam3_text":
                continue
            if not self._is_structural_sam3_role(item.get("prompt_role", "")):
                continue
            try:
                sam_score = float(item.get("sam_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                sam_score = 0.0
            quality = item.get("sam_mask_quality") if isinstance(item.get("sam_mask_quality"), dict) else {}
            if sam_score < 0.85 or not quality.get("accepted", False):
                continue
            if float(quality.get("bbox_area_ratio", 1.0) or 1.0) > 0.35:
                continue
            spatial_detail = item.get("spatial_detail") if isinstance(item.get("spatial_detail"), dict) else {}
            phrase = spatial_detail.get("fine_phrase") or spatial_detail.get("coarse_phrase")
            if not phrase:
                continue
            candidates.append((sam_score, item.get("prompt_role", ""), phrase))

        candidates.sort(reverse=True, key=lambda row: row[0])
        lines = []
        for sam_score, role, phrase in candidates[:1]:
            lines.append(
                f"- SAM3 structural cue: {role} mask is high-confidence "
                f"(score {sam_score:.2f}) at approximate location: {phrase}."
            )
        return lines

    def _build_spatial_candidate_evidence(self, proposals, crop_evidence_audit=None):
        lines = []
        active = list(proposals[:3])
        for idx, proposal in enumerate(active, start=1):
            phrase, cx, cy = self._spatial_phrase_for_bbox(proposal.bbox)
            fine_terms = self._spatial_terms_from_center(cx, cy)
            fine_phrase = self._fine_phrase_from_terms(fine_terms)
            detail = phrase if fine_phrase == phrase else f"{phrase}; fine-grained: {fine_phrase}"
            lines.append(
                f"- Red contour {idx}: center=({cx:.2f} from left, {cy:.2f} from top), approximate location: {detail}."
            )

        primary_terms = set()
        primary_phrase = ""
        fine_primary_phrase = ""
        union_bbox = self._union_bbox(active)
        if union_bbox is not None:
            primary_phrase, ux, uy = self._spatial_phrase_for_bbox(union_bbox)
            union_terms = self._spatial_terms_from_center(ux, uy)
            fine_primary_phrase = self._fine_phrase_from_terms(union_terms)
            primary_terms = set(union_terms["terms"])
            if len(active) > 1:
                detail = primary_phrase if fine_primary_phrase == primary_phrase else f"{primary_phrase}; fine-grained: {fine_primary_phrase}"
                lines.append(
                    f"- Aggregated red-contour union: center=({ux:.2f} from left, {uy:.2f} from top), primary location: {detail}."
                )

        structural_sam3_lines = self._sam3_structural_location_lines(crop_evidence_audit)
        lines.extend(structural_sam3_lines)

        option_prior = self._format_spatial_option_prior(primary_terms)
        option_visibility = EvidencePromptPolicy.spatial_option_audit_visibility(bool(option_prior))

        return {
            "report": "\n".join(lines),
            "sam3_structural_location_lines": structural_sam3_lines,
            "option_prior": option_prior,
            "option_prior_prompt_visible": option_visibility.prompt_visible,
            "option_prior_prompt_reason": option_visibility.reason,
            "primary_phrase": fine_primary_phrase or primary_phrase,
            "primary_terms": sorted(primary_terms),
        }

    def _format_spatial_candidate_report(self, proposals):
        return self._build_spatial_candidate_evidence(proposals).get("report", "")

    def _is_prompt_visible_structural_sam3_crop(self, audit_item):
        if not isinstance(audit_item, dict) or audit_item.get("source") != "sam3_text":
            return False
        if not self._is_structural_sam3_role(audit_item.get("prompt_role", "")):
            return False
        try:
            sam_score = float(audit_item.get("sam_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            sam_score = 0.0
        quality = audit_item.get("sam_mask_quality") if isinstance(audit_item.get("sam_mask_quality"), dict) else {}
        if sam_score < 0.90 or not quality.get("accepted", False):
            return False
        try:
            bbox_area_ratio = float(quality.get("bbox_area_ratio", 1.0) or 1.0)
        except (TypeError, ValueError):
            bbox_area_ratio = 1.0
        return bbox_area_ratio <= 0.35

    def _select_prompt_visible_crops(self, crop_images, crop_labels, crop_evidence_audit, is_loc_question):
        if not crop_images:
            return [], [], []

        records = []
        for idx, image in enumerate(crop_images):
            audit = {}
            if idx < len(crop_evidence_audit) and isinstance(crop_evidence_audit[idx], dict):
                audit = dict(crop_evidence_audit[idx])
            label = crop_labels[idx] if idx < len(crop_labels) else ""
            records.append({
                "candidate_index": idx,
                "image": image,
                "label": label,
                "audit": audit,
                "source": str(audit.get("source", "")),
            })

        selected = []
        selected_indices = set()

        def add_matching(predicate, prompt_reason, limit):
            for record in records:
                if len(selected) >= limit:
                    break
                if record["candidate_index"] in selected_indices:
                    continue
                if not predicate(record):
                    continue
                item = dict(record["audit"])
                item["prompt_visible"] = True
                item["prompt_selection_reason"] = prompt_reason
                selected.append((record, item))
                selected_indices.add(record["candidate_index"])

        # The old effective prompt used an either/or local evidence path:
        # strict global contours were followed by MCTS focus nodes, while weak
        # heatmap fallback crops were shown only when no strict contour existed.
        # Keep generic SAM3 crops audit-visible unless SAM3 produced a
        # high-confidence structural text mask.
        max_visible = 2 if self.task_type in {
            "defect localization",
            "defect classification",
            "defect description",
            "defect analysis",
            "anomaly detection",
        } else 3
        has_strict_heatmap = any(
            record["source"] == "heatmap_proposal"
            and record["audit"].get("proposal_source") == "strict"
            for record in records
        )
        weak_heatmap_sources = {
            "weak_pixel_threshold_fallback",
            "weak_ranked_tail_fallback",
        }
        has_weak_heatmap = any(
            record["source"] == "heatmap_proposal"
            and record["audit"].get("proposal_source") in weak_heatmap_sources
            for record in records
        )
        if has_weak_heatmap and not has_strict_heatmap:
            add_matching(
                lambda record: (
                    record["source"] == "heatmap_proposal"
                    and record["audit"].get("proposal_source") in weak_heatmap_sources
                ),
                "pixel_threshold_heatmap_fallback_crop",
                1,
            )
        else:
            add_matching(
                lambda record: record["source"] == "mcts_node",
                "mcts_focus_after_strict_contour",
                max_visible,
            )
            if not selected:
                add_matching(
                    lambda record: record["source"] == "heatmap_proposal",
                    "strict_heatmap_proposal_fallback_when_mcts_missing",
                    1,
                )
        add_matching(
            lambda record: self._is_prompt_visible_structural_sam3_crop(record["audit"]),
            "high_confidence_structural_sam3_text_mask",
            max_visible,
        )

        if not selected:
            record = records[0]
            item = dict(record["audit"])
            item["prompt_visible"] = True
            item["prompt_selection_reason"] = "fallback_first_selected_candidate"
            selected.append((record, item))

        return (
            [record["image"] for record, _ in selected],
            [record["label"] for record, _ in selected],
            [item for _, item in selected],
        )

    async def _generate_final_answer_multi(self, nodes):
        img_np = np.array(self.image.convert("RGB"))
        global_heatmap = self.root.state['heatmap_array']
        W, H = self.image.size

        # === [MEMORY FIX] 预先记录用于返回的debug信息，避免保留整个树引用 ===
        heatmap_peak_score = float(self.root.state['heatmap_score']) if hasattr(self, 'root') else 0.0

        if self.task_policy.use_local_anomaly_stream:
            proposals = self.region_proposals or detect_heatmap_regions(
                global_heatmap,
                self.threshold_config,
                task_type=self.task_type,
                top_k=3,
            )
        else:
            proposals = []
        region_proposal_audit = self._proposal_audit(proposals)
        box_prompts = [proposal.normalized_box(W, H) for proposal in proposals[:3]]

        crop_candidates = []
        crop_images = []
        crop_labels = []
        crop_evidence_audit = []
        skipped_crop_candidates = []
        sam_mask_scores = []
        sam_text_prompt_attempts = []
        sam_text_prompt_hits = []
        sam_prompt_selection_audit = []
        valid_box_idx = 0
        sam_profile = describe_profile(self.category)
        dataset_name = getattr(self.args, "dataset", None)
        use_sam3_local_refinement = (
            self.task_policy.use_sam3_local_refinement
            and should_use_sam3_local_refinement(
                self.category,
                dataset_name,
                self.task_type,
            )
        )

        for proposal in proposals[:3]:
            x, y, w, h = proposal.bbox
            valid_box_idx += 1
            if proposal.contour is not None:
                cv2.drawContours(img_np, [proposal.contour], -1, (255, 0, 0), 3)
            else:
                cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(img_np, str(valid_box_idx), (x, max(0, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        annotated_global = Image.fromarray(img_np)

        if self.sam_engine and box_prompts and use_sam3_local_refinement:
            for idx, (proposal, box) in enumerate(zip(proposals[:3], box_prompts), start=1):
                try:
                    mask_bool, mask_score = self.sam_engine.predict_mask_with_boxes(
                        [box],
                        threshold=getattr(self.args, "sam_mask_threshold", 0.4),
                    )
                    sam_mask_scores.append(float(mask_score))
                    mask_quality = self._sam_mask_quality(mask_bool)
                    if mask_quality.get("accepted"):
                        mask_bbox = tuple(mask_quality.get("bbox_xyxy") or self._mask_bbox_xyxy(mask_bool))
                        context_crop = self._crop_bbox_xyxy_with_padding(mask_bbox, padding_factor=0.35)
                        if context_crop is not None:
                            spatial_detail = self._spatial_detail_for_bbox_xyxy(mask_bbox)
                            label = self._with_spatial_crop_label(
                                f"SAM3 box ROI {idx} (score {float(mask_score):.2f})",
                                spatial_detail,
                            )
                            crop_candidates.append({
                                "image": context_crop,
                                "label": label,
                                "source": "sam3_box",
                                "priority": 1.0 + float(mask_score),
                                "bbox_xyxy": mask_bbox,
                                "spatial_detail": spatial_detail,
                                "sam_score": float(mask_score),
                                "sam_mask_quality": mask_quality,
                                "prompt_role": "box",
                                "prompt_text": "",
                            })
                    else:
                        skipped_crop_candidates.append(
                            self._crop_candidate_audit_item(
                                {
                                    "label": f"SAM3 box ROI {idx} (score {float(mask_score):.2f})",
                                    "source": "sam3_box",
                                    "priority": 1.0 + float(mask_score),
                                    "bbox_xyxy": mask_quality.get("bbox_xyxy"),
                                    "sam_score": float(mask_score),
                                    "sam_mask_quality": mask_quality,
                                    "prompt_role": "box",
                                    "prompt_text": "",
                                },
                                status="skipped",
                                rank=idx,
                                reason=";".join(mask_quality.get("reasons") or ["unusable_sam_mask"]),
                            )
                        )
                except Exception as e:
                    print(f"[SAM3 Refinement Error] {e}")

        if self.sam_engine and proposals and use_sam3_local_refinement:
            prompt_plan = plan_refinement_prompts(
                self.category,
                dataset=dataset_name,
                query=self._build_sam3_prompt_query(),
                max_prompts=6,
                include_parts=True,
                task_type=self.task_type,
            )
            profile_prompts = list(prompt_plan.selected)
            sam_prompt_selection_audit = list(prompt_plan.audit)
            for scored_prompt in profile_prompts:
                prompt = scored_prompt.prompt
                attempt = {
                    "role": prompt.role,
                    "text": prompt.text,
                    "threshold": prompt.threshold,
                    "prompt_family": prompt_role_family(prompt),
                    "selection_policy": SAM3_PROMPT_SELECTION_POLICY,
                    "relevance": scored_prompt.relevance,
                    "selection_reason": scored_prompt.selection_reason,
                    "source": prompt.source,
                    "status": "attempted",
                }
                if (
                    prompt.role in OBJECT_PROMPT_ROLES
                    and any(candidate.get("source") == "sam3_box" for candidate in crop_candidates)
                ):
                    attempt["status"] = "skipped_redundant_object_prompt"
                    sam_text_prompt_attempts.append(attempt)
                    continue
                try:
                    mask_bool, mask_score = self.sam_engine.predict_mask(
                        prompt.text,
                        threshold=prompt.threshold,
                    )
                    attempt["sam_score"] = float(mask_score)
                    mask_bool = self._resize_mask_to_image(mask_bool)
                    mask_quality = self._sam_mask_quality(mask_bool)
                    attempt["mask_quality"] = mask_quality
                    if not mask_quality.get("accepted"):
                        attempt["status"] = ";".join(mask_quality.get("reasons") or ["unusable_sam_mask"])
                        sam_text_prompt_attempts.append(attempt)
                        continue

                    best_constrained = None
                    best_score = -1.0
                    best_proposal_idx = None
                    best_roi_coverage = 0.0
                    for proposal_idx, proposal in enumerate(proposals[:3], start=1):
                        roi_mask = self._proposal_roi_mask(proposal.bbox, padding_factor=0.25)
                        constrained_mask = np.logical_and(mask_bool, roi_mask)
                        constrained_quality = self._sam_mask_quality(constrained_mask)
                        if not constrained_quality.get("accepted"):
                            continue
                        constrained_area = float(np.count_nonzero(constrained_mask))
                        roi_area = float(np.count_nonzero(roi_mask))
                        roi_coverage = constrained_area / max(1.0, roi_area)
                        selection_score = (
                            constrained_area
                            + roi_coverage * 1000.0
                            + scored_prompt.relevance * 100.0
                        )
                        if selection_score > best_score:
                            best_score = selection_score
                            best_constrained = constrained_mask
                            best_proposal_idx = proposal_idx
                            best_roi_coverage = roi_coverage

                    if best_constrained is None:
                        attempt["status"] = "no_roi_overlap"
                        sam_text_prompt_attempts.append(attempt)
                        continue

                    constrained_bbox = self._mask_bbox_xyxy(best_constrained)
                    context_crop = self._crop_bbox_xyxy_with_padding(constrained_bbox, padding_factor=0.35)
                    if context_crop is None:
                        attempt["status"] = "context_crop_failed"
                        sam_text_prompt_attempts.append(attempt)
                        continue

                    part_bonus = 0.20 if prompt.role not in OBJECT_PROMPT_ROLES else 0.0
                    semantic_defect_bonus = 0.55 if (
                        "localization" in str(self.task_type or "").lower()
                        and prompt.role.startswith("defect_")
                    ) else 0.0
                    spatial_detail = self._spatial_detail_for_bbox_xyxy(constrained_bbox)
                    label = self._with_spatial_crop_label(
                        f"SAM3 {prompt.role} ROI {best_proposal_idx} (score {float(mask_score):.2f})",
                        spatial_detail,
                    )
                    crop_candidates.append({
                        "image": context_crop,
                        "label": label,
                        "source": "sam3_text",
                        "priority": (
                            0.90
                            + (0.12 * scored_prompt.relevance)
                            + part_bonus
                            + semantic_defect_bonus
                            + (0.05 * float(mask_score))
                        ),
                        "bbox_xyxy": constrained_bbox,
                        "spatial_detail": spatial_detail,
                        "sam_score": float(mask_score),
                        "proposal_index": best_proposal_idx,
                        "sam_mask_quality": self._sam_mask_quality(best_constrained),
                        "prompt_role": prompt.role,
                        "prompt_text": prompt.text,
                        "semantic_defect_priority_bonus": round(float(semantic_defect_bonus), 6),
                    })
                    hit = dict(attempt)
                    hit["status"] = "used_candidate"
                    hit["proposal_index"] = best_proposal_idx
                    sam_text_prompt_hits.append(hit)
                    sam_text_prompt_attempts.append(hit)
                except Exception as e:
                    attempt["status"] = "error"
                    attempt["error"] = str(e)
                    sam_text_prompt_attempts.append(attempt)
                    print(f"[SAM3 Text Refinement Error] {prompt.role}: {e}")

        sam3_normal_part_veto = {}
        if (
            self.sam_engine
            and use_sam3_local_refinement
            and self.task_type == "anomaly detection"
            and not proposals
        ):
            sam3_normal_part_veto = self._sam3_normal_part_veto_audit(dataset_name)

        crop_selection = MCTSCropCandidateSelector(
            max_candidates=3,
            iou_threshold=0.78,
            source_limits={"sam3_box": 1, "sam3_text": 1},
        ).select(crop_candidates)
        selected_candidates = crop_selection.selected_candidates
        crop_candidate_audit = crop_selection.candidate_audit
        skipped_crop_candidates.extend(crop_selection.skipped_audit)

        crop_images = [candidate["image"] for candidate in selected_candidates]
        crop_labels = [candidate["label"] for candidate in selected_candidates]
        crop_evidence_audit = list(crop_selection.selected_audit)
        selected_crop_boxes = list(crop_selection.selected_boxes)
        reserve_heatmap_proposal = bool(
            proposals
            and self.task_policy.use_raw_heatmap_fill
            and self.task_policy.use_local_evidence_crops
            and len(crop_images) < 3
        )
        mcts_crop_limit = 2 if reserve_heatmap_proposal else 3

        for n in (nodes if self.task_policy.use_local_evidence_crops else []):
            if len(crop_images) >= mcts_crop_limit:
                break
            gx1, gy1, gx2, gy2 = n.state['region_coords']
            is_whole_image = (gx1 == 0 and gy1 == 0 and gx2 == W and gy2 == H)
            if not is_whole_image:
                node_bbox = (int(gx1), int(gy1), int(gx2), int(gy2))
                if self._is_duplicate_bbox(node_bbox, selected_crop_boxes):
                    spatial_detail = self._spatial_detail_for_bbox_xyxy(node_bbox)
                    label = self._with_spatial_crop_label("MCTS focus ROI", spatial_detail)
                    skipped_crop_candidates.append(
                        self._skipped_bbox_audit_item(
                            source="mcts_node",
                            label=label,
                            bbox_xyxy=node_bbox,
                            reason="duplicate_bbox_iou",
                            spatial_detail=spatial_detail,
                            heatmap_score=round(float(n.state.get("heatmap_score", 0.0)), 6),
                            search_score=round(float(n.state.get("search_score", n.state.get("heatmap_score", 0.0))), 6),
                        )
                    )
                    continue
                crop_images.append(n.state['image'])
                spatial_detail = self._spatial_detail_for_bbox_xyxy(node_bbox)
                label = self._with_spatial_crop_label(f"MCTS focus ROI {len(crop_images)}", spatial_detail)
                crop_labels.append(label)
                crop_evidence_audit.append({
                    "selected_index": len(crop_images),
                    "status": "selected",
                    "reason": "mcts_nms_focus_fill",
                    "label": label,
                    "source": "mcts_node",
                    "bbox_xyxy": [int(gx1), int(gy1), int(gx2), int(gy2)],
                    "spatial_detail": spatial_detail,
                    "heatmap_score": round(float(n.state.get("heatmap_score", 0.0)), 6),
                    "search_score": round(float(n.state.get("search_score", n.state.get("heatmap_score", 0.0))), 6),
                })
                self._track_crop_box(node_bbox, selected_crop_boxes)

        raw_proposal_fill_reasons = self._raw_proposal_fill_reasons(
            use_sam3_local_refinement=use_sam3_local_refinement,
            proposals=proposals,
            has_crop_images=bool(crop_images),
        )
        if reserve_heatmap_proposal:
            raw_proposal_fill_reasons.append("reserved_heatmap_proposal_after_mcts_search")
        allow_raw_proposal_fill = bool(raw_proposal_fill_reasons) and self.task_policy.use_raw_heatmap_fill
        if allow_raw_proposal_fill:
            for proposal in proposals:
                if len(crop_images) >= 3:
                    break
                padded_bbox = self._proposal_padded_bbox_xyxy(proposal.bbox, padding_factor=0.4)
                if self._is_duplicate_bbox(padded_bbox, selected_crop_boxes):
                    spatial_detail = self._spatial_detail_for_bbox_xyxy(padded_bbox)
                    label = self._with_spatial_crop_label("Heatmap proposal ROI", spatial_detail)
                    skipped_crop_candidates.append(
                        self._skipped_bbox_audit_item(
                            source="heatmap_proposal",
                            label=label,
                            bbox_xyxy=padded_bbox,
                            reason="duplicate_bbox_iou",
                            spatial_detail=spatial_detail,
                            proposal_source=proposal.source,
                            bbox=[int(v) for v in proposal.bbox],
                            proposal_bbox_xyxy=[int(v) for v in proposal.xyxy],
                            score=round(float(proposal.score), 6),
                            selection_score=round(float(proposal.selection_score), 6) if proposal.selection_score is not None else None,
                            selection_reason=proposal.selection_reason,
                        )
                    )
                    continue
                crop_images.append(self._crop_proposal_with_padding(proposal.bbox, padding_factor=0.4))
                spatial_detail = self._spatial_detail_for_bbox_xyxy(padded_bbox)
                label = self._with_spatial_crop_label(f"Heatmap proposal ROI {len(crop_images)}", spatial_detail)
                crop_labels.append(label)
                crop_evidence_audit.append({
                    "selected_index": len(crop_images),
                    "status": "selected",
                    "reason": "raw_heatmap_proposal_fill",
                    "fill_reasons": raw_proposal_fill_reasons,
                    "label": label,
                    "source": "heatmap_proposal",
                    "proposal_source": proposal.source,
                    "bbox": [int(v) for v in proposal.bbox],
                    "bbox_xyxy": [int(v) for v in padded_bbox],
                    "spatial_detail": spatial_detail,
                    "proposal_bbox_xyxy": [int(v) for v in proposal.xyxy],
                    "score": round(float(proposal.score), 6),
                    "selection_score": round(float(proposal.selection_score), 6) if proposal.selection_score is not None else None,
                    "selection_reason": proposal.selection_reason,
                })
                self._track_crop_box(padded_bbox, selected_crop_boxes)

        is_loc_question = (self.task_type == "defect localization")
        logic_says_defect = bool(
            self.used_logic_engine and not is_normal_or_no_defect_report(self.global_conclusion)
        )
        phase1_text = str(self.global_conclusion or "").strip()
        weak_localization_phase1 = (
            not phase1_text
            or phase1_text == "None"
            or phase1_text.lower().startswith("skipped")
            or is_normal_or_no_defect_report(phase1_text)
        )
        logic_primary_without_local_crops = bool(logic_says_defect and not is_loc_question)
        show_crop_images_in_final_prompt = bool(
            crop_images
            and self.task_policy.show_local_crops_in_final_prompt
            and not logic_primary_without_local_crops
        )
        if show_crop_images_in_final_prompt:
            prompt_crop_images, prompt_crop_labels, prompt_crop_selection_audit = self._select_prompt_visible_crops(
                crop_images,
                crop_labels,
                crop_evidence_audit,
                is_loc_question,
            )
        else:
            prompt_crop_images = []
            prompt_crop_labels = []
            prompt_crop_selection_audit = []
        local_evidence_prompt_policy = (
            "prompt_visible_strict_mcts_or_weak_heatmap_localization"
            if show_crop_images_in_final_prompt and is_loc_question
            else "prompt_visible_strict_mcts_or_weak_heatmap"
            if show_crop_images_in_final_prompt
            else (
                "phase1_logic_primary_local_crops_audit_only"
                if logic_primary_without_local_crops
                else
                "audit_only_global_spatial_localization"
                if is_loc_question and crop_images
                else "audit_only_for_global_semantic_defect_task"
                if crop_images else "no_selected_local_crops"
            )
        )

        sam_refined_crop_count = sum(
            1 for candidate in selected_candidates
            if str(candidate.get("source", "")).startswith("sam3")
        )
        sam_text_refined_crop_count = sum(
            1 for candidate in selected_candidates
            if candidate.get("source") == "sam3_text"
        )
        spatial_candidate_evidence = self._build_spatial_candidate_evidence(
            proposals,
            crop_evidence_audit=crop_evidence_audit,
        )
        spatial_candidate_report = spatial_candidate_evidence.get("report", "")
        spatial_option_prior = spatial_candidate_evidence.get("option_prior", "")
        sam3_crop_artifact_rule = self._sam3_crop_artifact_rule(crop_evidence_audit)
        pvla_option_matches = self._pvla_option_matches()
        pvla_option_grounding_summary = self._format_pvla_option_grounding_summary(pvla_option_matches)
        pvla_evidence_hypothesis, pvla_evidence_hypothesis_audit = self._format_pvla_evidence_hypothesis()
        pvla_hypothesis_prompt_decision = EvidencePromptPolicy.pvla_hypothesis_visibility(
            has_hypotheses=bool(pvla_evidence_hypothesis_audit),
            local_crop_count=len(crop_images),
            used_logic_engine=bool(self.used_logic_engine),
            allow_with_local_evidence=bool(self.task_policy.use_pvla_defect_hypothesis_prompt),
        )
        pvla_hypothesis_prompt_visible = pvla_hypothesis_prompt_decision.prompt_visible
        pvla_hypothesis_prompt_reason = pvla_hypothesis_prompt_decision.reason
        show_pvla_hypothesis = bool(pvla_evidence_hypothesis and pvla_hypothesis_prompt_visible)
        phase1_visual_primitive_hint, phase1_visual_primitive_hint_audit = self._format_phase1_visual_primitive_hint()
        phase1_option_prior = ""
        phase1_analysis_prior = ""
        prompt_rag_blocks, prompt_rag_selection_audit = self._select_prompt_rag_blocks(
            self.rag_blocks,
            pvla_evidence_hypothesis_audit,
        )
        show_rag_blocks_in_final_prompt = bool(self.task_policy.show_rag_blocks_in_final_prompt)
        show_phase1_report_in_final_prompt = bool(self.task_policy.show_phase1_report_in_final_prompt)
        self.atlas_context_for_verification = self._extract_atlas_context_text(include_global=False)

        # RAG Logic (Kept Identical)
        rag_content_list = []
        prompt_rag_text_blocks = []
        if prompt_rag_blocks and show_rag_blocks_in_final_prompt:
            rag_content_list.append({"type": "text", "text": "=== 📚 REFERENCE STANDARDS (VISUAL & TEXT) ===\n"})
            has_local_rag_block = any(
                not self._is_global_or_whole_region(block.get("region", ""))
                for block in prompt_rag_blocks
                if isinstance(block, dict)
            )
            for i, block in enumerate(prompt_rag_blocks):
                region_name = block.get('region', f'Region {i+1}')
                if (
                    self._is_global_or_whole_region(region_name)
                    and not self._should_keep_global_rag_block(block)
                    and has_local_rag_block
                ):
                    continue
                txt = block.get('text', '').strip()
                global_anchor_trimmed = False
                if (
                    txt
                    and self._is_global_or_whole_region(region_name)
                    and has_local_rag_block
                    and not self.task_policy.include_global_reference_in_final
                    and self._defect_pattern_count(txt) <= 1
                ):
                    trimmed_txt = self._global_rag_anchor_text(txt)
                    global_anchor_trimmed = trimmed_txt != txt
                    txt = trimmed_txt
                rag_content_list.append({"type": "text", "text": f"\n--- Standard for {region_name} ---\n"})
                include_ref_images = getattr(self.args, "include_rag_reference_images", False)
                visual_reference_prompt_image_count = 0
                if include_ref_images and block.get('images') and not is_loc_question:
                    rag_content_list.append({"type": "text", "text": "(Visual Reference - Normal Examples):\n"})
                    for img_item in block['images']:
                        pil_img = None
                        if isinstance(img_item, Image.Image):
                            pil_img = img_item
                        else:
                            img_path_str = str(img_item)
                            if os.path.exists(img_path_str):
                                pil_img = Image.open(img_path_str).convert("RGB")
                        if pil_img:
                            rag_content_list.append({"type": "image", "image": pil_img})
                            visual_reference_prompt_image_count += 1
                if txt:
                    retrieval = block.get("retrieval") or {}
                    provenance = retrieval.get("provenance") or {}
                    rag_content_list.append({"type": "text", "text": f"(Definition & Defects): {txt}\n"})
                    prompt_rag_text_blocks.append({
                        "region": region_name,
                        "text": txt,
                        "source_json_path": provenance.get("source_json_path", ""),
                        "graph_cache_path": provenance.get("graph_cache_path", ""),
                        "knowledge_source_type": provenance.get("knowledge_source_type", ""),
                        "visual_reference_paths": provenance.get("visual_reference_paths", []),
                        "visual_reference_prompt_visible": bool(visual_reference_prompt_image_count),
                        "visual_reference_prompt_image_count": visual_reference_prompt_image_count,
                        "visual_reference_source": provenance.get("visual_reference_source", ""),
                        "visual_reference_generation": provenance.get("visual_reference_generation", ""),
                        "visual_reference_source_backed": provenance.get("visual_reference_source_backed", False),
                        "global_anchor_trimmed": global_anchor_trimmed,
                    })
            rag_content_list.append({"type": "text", "text": "\n=============================================\n"})
        
        # 3. Construct Base Prompt
        options_str = ""
        if self.options:
            options_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in self.options.items()])
        
        prompt = f"Question: {self.question}\n\n"
        prompt += f"{options_str}\n\n"
        if show_phase1_report_in_final_prompt:
            prompt += "You are a professional industrial QA inspector conducting a multi-stage inspection. Below is the report from the previous step.\n\n"
            prompt += "=== PHASE 1: GLOBAL CHECK REPORT ===\n"
            prompt += f"Automatic Global Inspection Result: {self.global_conclusion}\n\n"
        else:
            if is_loc_question:
                prompt += (
                    "You are a professional industrial QA inspector. Inspect IMAGE 1 as the global red-contour localization view.\n\n"
                    "=== AUDIT-ONLY METHOD CONTEXT ===\n"
                    "The system has run Phase-1 screening, MCTS local search, SAM3 segmentation, and source-backed graph/RAG retrieval. "
                    "Because Phase-1 produced only a normal/no-defect or skipped report for this localization question, its text is recorded in the trace only. "
                    "Answer from the visible Global View, red contours, spatial candidate report, and option descriptions.\n\n"
                )
            else:
                prompt += (
                    "You are a professional industrial QA inspector. Inspect IMAGE 1 directly as the global target image.\n\n"
                    "=== AUDIT-ONLY METHOD CONTEXT ===\n"
                    "The system has run Phase-1 screening, MCTS local search, SAM3 segmentation, and source-backed graph/RAG retrieval. "
                    "For this global semantic defect question, those streams are recorded in the trace for provenance and are not shown as answer priors. "
                    "Answer from IMAGE 1 and the option descriptions.\n\n"
                )
        if show_pvla_hypothesis and show_rag_blocks_in_final_prompt:
            prompt += "=== PVLA GRAPH VISUAL HYPOTHESES ===\n"
            prompt += f"{pvla_evidence_hypothesis}\n\n"
        prompt += "=== PHASE 2: FINAL DIAGNOSIS ===\n"
        # Keep crop/SAM3 artifact policy in trace rather than the answer prompt.
        # The original local-evidence prompt was tuned for this benchmark; extra
        # guard prose has empirically reduced QA accuracy on semantic defect
        # questions.
        # Keep primitive overlap in trace for provenance, but do not inject it as
        # an answer prior. In practice this cue is too coarse for options that
        # share the same visual primitive and can override stronger whole-image
        # evidence.
        defect_specific_task = self.task_type in {
            "defect classification",
            "defect description",
            "defect analysis",
            "defect localization",
        }
        defect_specific_question = defect_specific_task or (
            "defect" in self.question.lower() and self.task_type != "anomaly detection"
        )
        if self.task_policy.is_object_task:
            prompt += (
                f"**Global Object QA Task**:\n"
                f"1. Use the target Global View as the primary evidence for object category, structure, details, and normal functional parts.\n"
                f"2. Use the Reference Standards as normal object/part knowledge when available; do not treat them as the answer by themselves.\n"
                f"3. This task type is evaluated without local anomaly MCTS/SAM3 crops. Do not infer an option from missing local crops or from anomaly heatmap absence.\n"
                f"4. Select the answer option that best matches the visible whole-object evidence and the Phase 1 global observation.\n"
            )
        elif is_loc_question:
            # Check if we have a valid Stage 1 conclusion
            has_stage1_info = self.global_conclusion and self.global_conclusion != "None" and len(self.global_conclusion) > 0

            if has_stage1_info and logic_says_defect :
                # Case A: 使用了 logical_mvtec 的确定性分析，优先采信
                prompt += (
                    f"**Integrated Diagnosis Task (Logic Engine Mode)**:\n"
                    f"1. **Input Analysis**: Refer to the **Stage 1 Logic Report** (provided in context above) and the **Visual View** with a Red Contour.\n"
                    f"2. **Synthesis Strategy**: **Prioritize the Stage 1 Logic Report** as the primary truth. The logic report contains specific findings from computer vision analysis about the defect type.\n"
                    f"3. **Role of Image**: Use the Red Contour primarily to verify the location described. If the visual shape is ambiguous, trust the definitions in the Logic Report.\n"
                    f"4. **Decision**: Select the option that aligns best with the Stage 1 Report.\n"
                )
            elif has_stage1_info:
                # Case B: 有 Stage 1 信息但来自 VLM 推理，需要综合判断
                prompt += (
                    f"**Integrated Diagnosis Task (Synthesis Mode)**:\n"
                    f"1. **Input Analysis**: You have TWO sources of information:\n"
                    f"   - **Stage 1 VLM Report**: A preliminary analysis from the previous step (may contain errors).\n"
                    f"   - **Visual View**: The image with Red Contour highlighting suspected anomaly regions.\n"
                    f"2. **Synthesis Strategy**: **Do NOT blindly trust the Stage 1 Report**. It is a preliminary reference only.\n"
                    f"   - Carefully examine the Red Contour region in the Visual View.\n"
                    f"   - Cross-check with the Reference Standards if available.\n"
                    f"   - Use your own visual judgment as the primary basis.\n"
                    f"3. **Conflict Resolution**: If Stage 1 Report conflicts with your visual observation, **trust what you see in the image**.\n"
                    f"4. **Decision**: Select the option that best matches your integrated analysis, prioritizing visual evidence.\n"
                )
            else:
                # Case C: 没有 Stage 1 信息，完全依赖视觉
                prompt += (
                    f"**Visual Localization Task**:\n"
                    f"1. **Input Analysis**: No prior textual logic report is available. Focus entirely on the **Visual View** with the Red Contour.\n"
                    f"2. **Strategy**: The Red Contour explicitly highlights the suspected anomaly location.\n"
                    f"3. **Decision**: Analyze the region inside the Red Contour. Select the answer that best matches the location and appearance of the highlighted area.\n"
                )
        else:
            if show_crop_images_in_final_prompt:
                if logic_says_defect:
                    prompt += (
                        f"**Situation**: The PHASE 1 Logic Engine has identified a **definitive structural anomaly**.\n"
                        f"You are provided with a 'Global View' (with Red Boxes) and **one or more zoomed-in 'Focus Views'**.\n"
                        f"**Instruction**: Use the PHASE 1 conclusion as the primary basis. Then, map each Focus View to its Red Box and check whether **additional visible anomalies** exist.\n"
                        f"**Compound Option Rule**: If one option describes only the Phase 1 defect and another option describes the Phase 1 defect plus an additional visible local defect, choose the compound option when the Focus Views support that additional defect.\n"
                        f"**Final Decision**: Determine the final defect based on the **combined result** of (1) the Logic Engine anomaly and (2) any visually confirmed anomalies.\n"
                    )
                else:    
                    prompt += (
                        f"**Situation**: Potential anomalies have been localized. Since the system detected anomalies, this object is **DEFECTIVE**.\n"
                        f" You are provided with a 'Global View' (with Red Boxes) and **one or more zoomed-in 'Focus Views'**.\n\n"
                        f"**Step-by-step Reasoning Task**:\n"
                        f"1. **Localization & Mapping**: For **EACH** 'Focus View', match it to its corresponding **Red Box** in the 'Global View'. Identify exactly which part is shown in each crop.\n"
                        f"2. **Knowledge Retrieval**: Consult the Knowledge Base for the identified parts to recall their 'Normal' vs 'Defect' standards.\n"
                        f"3. **Defect Diagnosis**: Inspect **EVERY** Focus View sequentially. Check if *any* view contains a confirmed defect based on the visual evidence. (Note: If multiple defects exist, prioritize the most severe structural failure).\n"
                        f"4. **Final Decision**: Synthesize findings from ALL views. Match the primary defect with the provided 'Options' list and select the most accurate one.\n"
                    )
            else:
                if not self.has_global_standard:
                    prompt += (
                        f"**STATUS: NO ANOMALY DETECTED**\n"
                        f"1. No local anomalies were found by the scanner.\n"
                        f"2. No global inspection standard is defined for this object type.\n\n"
                        f"**INSTRUCTION**:\n"
                        f"Therefore, the object is considered **NORMAL**.\n"
                        f"Please directly select the option corresponding to 'Good' or 'Normal'.\n"
                    )
                else:
                    # [修复] 非定位问题也需要区分是否使用了 logic_engine
                    if logic_says_defect:
                        prompt += (
                            f"**Situation**: The PHASE 1 Logic Engine has provided a definitive structural analysis.\n"
                            f"**Instruction**: Use the PHASE 1: GLOBAL CHECK REPORT above as the primary basis to determine the defect type.\n"
                        )
                    elif defect_specific_question:
                        if crop_images:
                            prompt += (
                                f"**Situation**: This is a defect-specific semantic question. The local search selected candidate regions, but they are audit-only for this task and are not shown as extra answer images.\n"
                                f"**Instruction**: Inspect IMAGE 1 directly and select the option that best matches the visible whole-object defect. Do not infer the answer from hidden local candidates or from a normal Phase-1 screening report.\n"
                            )
                        else:
                            prompt += (
                                f"**Situation**: This is a defect-specific question, but the local scanner did not produce confident crops.\n"
                                f"**Instruction**: Inspect the global image directly and select the visible defect option. Do not treat the absence of crops or a normal global report as proof of no defect for this question type.\n"
                            )
                    else:
                        if self._should_apply_no_local_anomaly_gate(heatmap_peak_score):
                            prompt += (
                                f"**Local Verification Gate**:\n"
                                f"The local anomaly stream did not select a stable saliency-ranked local evidence region "
                                f"(heatmap peak {heatmap_peak_score:.3f}; compatibility image threshold {self.image_threshold:.3f}). "
                                f"For binary anomaly detection, a shallow surface-only Phase 1 wording is not enough to answer Yes. "
                                f"Cross-check the Reference Standards above before trusting the Phase 1 report: choose Yes only when the reported cue matches a listed defect signature and is not described as allowed normal variation. "
                                f"Choose Yes only if the target image itself clearly shows a localized structural/foreign/printed defect; "
                                f"choose No for natural texture or color variations without local support.\n"
                            )
                        prompt += (
                            f"No stable local anomaly crop was selected by the scanner.\n"
                            f"Use the Global View and PHASE 1 conclusion directly; absence of a local crop is not proof of normality. Choose the normal answer only when the target image itself looks normal.\n"
                        )

        prompt += (
            f"\n**REQUIRED OUTPUT FORMAT**:\n"
            f"The correct answer is (X)"
        )

        verification_strategy = "staged"

        # === [MEMORY FIX] 构建返回字典 ===
        result = {
            "status": "ready",
            "red_box_image": annotated_global if proposals else self.image.convert("RGB"),
            "crop_images": crop_images,
            "crop_labels": crop_labels,
            "prompt_crop_images": prompt_crop_images,
            "show_crop_images_in_final_prompt": show_crop_images_in_final_prompt,
            "prompt_visible_crop_count": len(prompt_crop_images),
            "prompt_crop_labels": prompt_crop_labels,
            "prompt_crop_selection_audit": prompt_crop_selection_audit,
            "suppress_prompt_crop_labels": bool(show_crop_images_in_final_prompt),
            "show_rag_blocks_in_final_prompt": show_rag_blocks_in_final_prompt,
            "show_phase1_report_in_final_prompt": show_phase1_report_in_final_prompt,
            "sam3_debug_items": [
                {
                    "label": candidate.get("label", ""),
                    "source": candidate.get("source", ""),
                    "sam_score": candidate.get("sam_score", 0.0),
                    "spatial_detail": candidate.get("spatial_detail", {}),
                    "prompt_role": candidate.get("prompt_role", ""),
                    "prompt_text": candidate.get("prompt_text", ""),
                }
                for candidate in selected_candidates
            ],
            "rag_content_list": rag_content_list,
            "prompt": prompt,
            "verification_strategy": verification_strategy,
            "logic_report_text": self.logic_report_for_verification,
            "atlas_context_text": self.atlas_context_for_verification,
            "logic_debug_info": {
                "view_image": self.debug_logic_view_img,
                "phase1_prompt": self.debug_phase1_prompt
            },
            "debug_metadata": {
            "category": self.category,
            "dataset_category": self.dataset_category,
            "localizer_detected_category": self.localizer_detected_category,
            "category_source": self.category_source,
            "heatmap_peak_score": heatmap_peak_score,
            "anomaly_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "threshold_source": self.threshold_config.source,
            "region_proposal_count": len(proposals),
            "region_proposal_sources": [p.source for p in proposals],
            "region_proposal_audit": region_proposal_audit,
            "crop_candidate_audit": crop_candidate_audit,
            "skipped_crop_candidates": skipped_crop_candidates,
            "crop_evidence_audit": crop_evidence_audit,
            "raw_proposal_fill_reasons": raw_proposal_fill_reasons,
            "mcts_budget_config": {
                "n_simulations": int(self.n_simulations),
                "max_depth": int(self.max_depth),
                "c_puct": float(self.c_puct),
                "action_count": len(self.DISCRETE_ACTIONS),
            },
            "mcts_action_trace": self.mcts_action_trace,
            "mcts_search_summary": self.mcts_search_summary,
            "mcts_final_candidate_audit": self.mcts_final_candidate_audit,
            "sam_refined_crop_count": sam_refined_crop_count,
            "sam_text_refined_crop_count": sam_text_refined_crop_count,
            "sam_local_refinement_enabled": use_sam3_local_refinement,
            "sam_mask_scores": sam_mask_scores,
            "sam_profile": sam_profile,
            "sam_prompt_selection_audit": sam_prompt_selection_audit,
            "sam_text_prompt_attempts": sam_text_prompt_attempts,
            "sam_text_prompt_hits": sam_text_prompt_hits,
            "sam3_normal_part_veto": sam3_normal_part_veto,
            "spatial_candidate_report": spatial_candidate_report,
            "spatial_option_prior": spatial_option_prior,
            "spatial_option_prior_prompt_visible": spatial_candidate_evidence.get("option_prior_prompt_visible", False),
            "spatial_option_prior_prompt_reason": spatial_candidate_evidence.get("option_prior_prompt_reason", ""),
            "spatial_primary_phrase": spatial_candidate_evidence.get("primary_phrase", ""),
            "spatial_primary_terms": spatial_candidate_evidence.get("primary_terms", []),
            "sam3_structural_location_lines": spatial_candidate_evidence.get("sam3_structural_location_lines", []),
            "sam3_crop_artifact_rule": sam3_crop_artifact_rule,
            "show_crop_images_in_final_prompt": show_crop_images_in_final_prompt,
            "prompt_visible_crop_count": len(prompt_crop_images),
            "prompt_crop_selection_audit": prompt_crop_selection_audit,
            "prompt_crop_labels": prompt_crop_labels,
            "suppress_prompt_crop_labels": bool(show_crop_images_in_final_prompt),
            "local_evidence_prompt_policy": local_evidence_prompt_policy,
            "show_rag_blocks_in_final_prompt": show_rag_blocks_in_final_prompt,
            "show_phase1_report_in_final_prompt": show_phase1_report_in_final_prompt,
            "weak_localization_phase1": bool(weak_localization_phase1) if is_loc_question else False,
            "pvla_option_grounding_summary": pvla_option_grounding_summary,
            "pvla_evidence_hypothesis": pvla_evidence_hypothesis,
            "pvla_evidence_hypothesis_audit": pvla_evidence_hypothesis_audit,
            "pvla_hypothesis_prompt_visible": pvla_hypothesis_prompt_visible,
            "pvla_hypothesis_prompt_reason": pvla_hypothesis_prompt_reason,
            "show_pvla_hypothesis": show_pvla_hypothesis,
            "phase1_option_prior": phase1_option_prior,
            "phase1_analysis_prior": phase1_analysis_prior,
            "phase1_visual_primitive_hint": phase1_visual_primitive_hint,
            "phase1_visual_primitive_hint_audit": phase1_visual_primitive_hint_audit,
            "prompt_rag_selection_audit": prompt_rag_selection_audit,
            "prompt_rag_text_blocks": prompt_rag_text_blocks,
            "crop_labels": crop_labels,
            "is_crop_triggered": len(crop_images) > 0,
            "crop_count": len(crop_images),
            "phase_1_conclusion": self.global_conclusion,
            "rag_retrieved_blocks": len(self.rag_blocks) if self.rag_blocks else 0,
            "rag_block_provenance": self._rag_block_provenance(),
            "rag_retrieval_mode": getattr(self.args, "rag_retrieval_mode", "full"),
            "used_logic_engine": bool(self.used_logic_engine),
            "task_policy": {
                "task_group": self.task_policy.task_group,
                "use_local_anomaly_stream": self.task_policy.use_local_anomaly_stream,
                "use_mcts_search": self.task_policy.use_mcts_search,
                "use_local_evidence_crops": self.task_policy.use_local_evidence_crops,
                "show_local_crops_in_final_prompt": self.task_policy.show_local_crops_in_final_prompt,
                "use_sam3_local_refinement": self.task_policy.use_sam3_local_refinement,
                "use_raw_heatmap_fill": self.task_policy.use_raw_heatmap_fill,
                "include_global_reference_in_final": self.task_policy.include_global_reference_in_final,
                "show_rag_blocks_in_final_prompt": self.task_policy.show_rag_blocks_in_final_prompt,
                "show_phase1_report_in_final_prompt": self.task_policy.show_phase1_report_in_final_prompt,
                "use_pvla_defect_hypothesis_prompt": self.task_policy.use_pvla_defect_hypothesis_prompt,
                "phase1_logic_role": self.task_policy.phase1_logic_role,
            },
            }
        }

        # === [CRITICAL MEMORY FIX] 立即清理MCTS树，防止显存泄露 ===
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
            self.root = None

        del img_np, global_heatmap

        return result
