import os
import json
import abc
from typing import Dict, Generator, List, Optional

class BaseMMADDatasetLoader(abc.ABC):
    def __init__(self, root_path: str):
        self.root_path = root_path

    @abc.abstractmethod
    def get_subclasses(self) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def parse_samples(self, subclass: str) -> Generator[Dict, None, None]:
        raise NotImplementedError

    def _load_qa_json(self, subclass: str) -> Optional[Dict]:
        """Helper to load QA.json for a subclass"""
        json_path = os.path.join(self.root_path, subclass, "QA.json")
        if not os.path.exists(json_path):
            print(f"[Dataset Error] QA.json not found for subclass: {subclass} at {json_path}")
            return None
        try:
            with open(json_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Dataset Error] Failed to load json: {e}")
            return None

    def _resolve_image_path(self, subclass: str, rel_path: str) -> Optional[str]:
        """Helper to resolve image paths trying multiple common patterns"""
        # Try direct path inside subclass folder
        p1 = os.path.join(self.root_path, subclass, rel_path)
        if os.path.exists(p1): return p1
        
        # Try path relative to root
        p2 = os.path.join(self.root_path, rel_path)
        if os.path.exists(p2): return p2
        
        # MVTec-LOCO sometimes has paths starting with the subclass name itself in the JSON
        # e.g., json says "breakfast_box/test/...", and we are already in root/breakfast_box
        # So we check root/rel_path directly again (covered by p2), but also check stripping subclass
        if rel_path.startswith(subclass + "/"):
            stripped_path = rel_path.replace(subclass + "/", "", 1)
            p3 = os.path.join(self.root_path, subclass, stripped_path)
            if os.path.exists(p3): return p3

        return None

# ============================================================
#                       DS-MVTec Loader
# ============================================================
class DSMVTecDatasetLoader(BaseMMADDatasetLoader):
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self.subclasses = [
            "bottle", "cable", "capsule", "carpet", "grid",
            "hazelnut", "leather", "metal_nut", "pill", "screw",
            "tile", "toothbrush", "transistor", "wood", "zipper"
        ]

    def get_subclasses(self) -> List[str]:
        return self.subclasses

    def parse_samples(self, subclass: str) -> Generator[Dict, None, None]:
        data = self._load_qa_json(subclass)
        if data is None: return

        for rel_img_key, content in data.items():
            # Support both structure types if image_path key is missing
            rel_path = content.get("image_path", rel_img_key)
            img_path = self._resolve_image_path(subclass, rel_path)
            
            if img_path is None: 
                # print(f"Warning: Image not found {rel_path}")
                continue

            conversations = content.get("conversation", [])
            for turn in conversations:
                # MMAD benchmark standard: only use annotated questions
                if not turn.get("annotation", False): continue
                
                yield {
                    "image_path": img_path,
                    "subclass": subclass,
                    "question": turn["Question"],
                    "gt_answer": turn["Answer"],
                    "options": turn["Options"],
                    "task_type": turn.get("type", "Unknown"),
                    "annotation": True,
                }

# ============================================================
#                       VisA Loader
# ============================================================
class VisaDatasetLoader(BaseMMADDatasetLoader):
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self.subclasses = [
            "candle", "capsules", "cashew", "chewinggum", "fryum", 
            "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
        ]

    def get_subclasses(self) -> List[str]:
        return self.subclasses

    def parse_samples(self, subclass: str) -> Generator[Dict, None, None]:
        # Re-using the same logic as DSMVTec since MMAD standardizes the JSON format
        data = self._load_qa_json(subclass)
        if data is None: return

        for rel_img_key, content in data.items():
            rel_path = content.get("image_path", rel_img_key)
            img_path = self._resolve_image_path(subclass, rel_path)
            
            if img_path is None: continue

            conversations = content.get("conversation", [])
            for turn in conversations:
                if not turn.get("annotation", False): continue
                yield {
                    "image_path": img_path,
                    "subclass": subclass,
                    "question": turn["Question"],
                    "gt_answer": turn["Answer"],
                    "options": turn["Options"],
                    "task_type": turn.get("type", "Unknown"),
                    "annotation": True,
                }

# ============================================================
#                  MVTec-LOCO Loader (NEW)
# ============================================================
class MVTecLocoDatasetLoader(BaseMMADDatasetLoader):
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self.subclasses = [
            "breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"
        ]

    def get_subclasses(self) -> List[str]:
        return self.subclasses

    def parse_samples(self, subclass: str) -> Generator[Dict, None, None]:
        data = self._load_qa_json(subclass)
        if data is None: return

        for rel_img_key, content in data.items():
            rel_path = content.get("image_path", rel_img_key)
            img_path = self._resolve_image_path(subclass, rel_path)
            
            if img_path is None: continue

            conversations = content.get("conversation", [])
            for turn in conversations:
                if not turn.get("annotation", False): continue
                yield {
                    "image_path": img_path,
                    "subclass": subclass,
                    "question": turn["Question"],
                    "gt_answer": turn["Answer"],
                    "options": turn["Options"],
                    "task_type": turn.get("type", "Unknown"),
                    "annotation": True,
                }

# ============================================================
#                    GoodsAD Loader (NEW)
# ============================================================
class GoodsADDatasetLoader(BaseMMADDatasetLoader):
    def __init__(self, root_path: str):
        super().__init__(root_path)
        # GoodsAD common categories
        self.subclasses = [
             "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill"
        ]
        # Note: GoodsAD class names sometimes overlap with MVTec, 
        # but the loader is instantiated with a different root path.

    def get_subclasses(self) -> List[str]:
        return self.subclasses

    def parse_samples(self, subclass: str) -> Generator[Dict, None, None]:
        data = self._load_qa_json(subclass)
        if data is None: return

        for rel_img_key, content in data.items():
            rel_path = content.get("image_path", rel_img_key)
            img_path = self._resolve_image_path(subclass, rel_path)
            
            if img_path is None: continue

            conversations = content.get("conversation", [])
            for turn in conversations:
                if not turn.get("annotation", False): continue
                yield {
                    "image_path": img_path,
                    "subclass": subclass,
                    "question": turn["Question"],
                    "gt_answer": turn["Answer"],
                    "options": turn["Options"],
                    "task_type": turn.get("type", "Unknown"),
                    "annotation": True,
                }