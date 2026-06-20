import numpy as np
import random
import math
import traceback
import cv2
import PIL.Image as Image
from PIL import ImageDraw
import os
import torch  # Added for OOM catching
from kragad.models.logical import VisualLogicProcessor

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    xi_min = max(x1_min, x2_min); yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max); yi_max = min(y1_max, y2_max)
    inter_width = max(0, xi_max - xi_min); inter_height = max(0, yi_max - yi_min)
    inter_area = inter_width * inter_height
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms_filter(nodes, iou_threshold=0.1, score_threshold=0.6):
    unique_map = {}
    for n in nodes:
        coords = tuple(n.state['region_coords'])
        if coords not in unique_map or n.leaf_reward > unique_map[coords].leaf_reward:
            unique_map[coords] = n
    unique_nodes = list(unique_map.values())
    
    candidates = [n for n in unique_nodes if n.leaf_reward >= score_threshold]
    if not candidates:
        if not unique_nodes: return []
        return [max(unique_nodes, key=lambda x: x.leaf_reward)]

    candidates.sort(key=lambda x: x.leaf_reward, reverse=True)
    keep = []
    while candidates:
        current = candidates.pop(0)
        keep.append(current)
        candidates = [n for n in candidates if calculate_iou(current.state['region_coords'], n.state['region_coords']) < iou_threshold]
    return keep[:3]

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
        self.untried_actions = sorted(available_actions.copy()) if available_actions else []
        if self.untried_actions:
            random.shuffle(self.untried_actions)
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
        self.max_depth = 4
        self.c_puct = 1.0
        # === [MEMORY FIX] 减少模拟次数以降低内存压力 ===
        # 原值100可能产生600+节点，改为50减少一半内存占用
        self.n_simulations = 50  # 原值: 100
        self.DISCRETE_ACTIONS = ["move_left", "move_right", "move_up", "move_down", "zoom_in", "zoom_out"]
        self.global_conclusion = "Pending"
        self.category = None
        self.image_threshold = DEFAULT_THRESHOLD
        self.pixel_threshold = DEFAULT_THRESHOLD
        self.has_global_standard = True
        self.used_logic_engine = False
        self.logic_report_for_verification = ""
        self.atlas_context_for_verification = ""
    
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
            region_name = block.get('region', f'Region {i+1}')
            region_name_l = region_name.lower()
            if (not include_global) and ("whole" in region_name_l or "global" in region_name_l):
                continue
            txt = block.get('text', '').strip()
            if txt:
                lines.append(f"[{region_name}] {txt}")
        return "\n".join(lines).strip()

    def _build_single_prompt_verification(self, options_str, logic_report, crop_count):
        logic_text = logic_report.strip() if isinstance(logic_report, str) else ""
        if not logic_text:
            logic_text = self.global_conclusion.strip() if isinstance(self.global_conclusion, str) else "N/A"

        prompt = f"Question: {self.question}\n\n"
        prompt += f"{options_str}\n\n"
        prompt += "You are a professional industrial QA inspector. Run one-step final verification from concatenated evidence only.\n\n"
        prompt += "=== LOGIC REPORT ===\n"
        prompt += f"{logic_text}\n\n"
        prompt += "=== LOCAL EVIDENCE CROPS ===\n"
        prompt += f"{crop_count} local evidence crop image(s) are attached below in order.\n"
        prompt += "Use all evidence jointly and select the best option.\n\n"
        prompt += "**REQUIRED OUTPUT FORMAT**:\nThe correct answer is (X)"
        return prompt


    async def get_anomaly_heatmap(self, image):
        heatmap_small, obj_name = self.localizer.predict_anomaly_map(image)
        return cv2.resize(heatmap_small, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR), obj_name
    
    def _load_global_reference_image(self):
        """
        Helper to load the 'whole_object.png' specific to the current category.
        You should adjust 'base_path' to match your actual file structure.
        """
        # TODO: Adjust this path to where your reference images are actually stored
        # Example: ./data/mvtec/bottle/whole_object.png
        possible_paths = [
            f"./data/mvtec/{self.category}/whole_object.png",
            f"./references/{self.category}/whole_object.png",
            f"whole_object.png" # If in current directory
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    return Image.open(p).convert("RGB")
                except Exception as e:
                    print(f"Error loading reference image at {p}: {e}")
        return None
        
    def _detect_heatmap_regions(self, heatmap):
        """
        Modified to strictly match 'annotated_global' generation logic AND its fallback.
        Corrected to support ALL visual task types, not just localization.
        """
        # 1. Threshold
        if heatmap.max() < self.pixel_threshold:
            return []
            
        mask = ((heatmap > self.pixel_threshold) * 255).astype(np.uint8)
        
        # 2. Connected Components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        clean_mask = np.zeros_like(mask)
        component_candidates = []
        
        # Global fallback trackers
        global_max_score = -1.0
        global_best_box = None
        global_best_mask_idx = -1 

        # 3. Collect Candidates
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            # Get peak score within this connected component
            peak_val = np.max(heatmap[y:y+h, x:x+w][labels[y:y+h, x:x+w] == i])
            
            # Track best index for fallback consistency
            if peak_val > global_max_score:
                global_max_score = peak_val
                global_best_box = (x, y, w, h)
                global_best_mask_idx = i 
            
            # Use image_threshold to filter valid high-confidence regions
            if peak_val >= self.image_threshold:
                component_candidates.append({
                    'id': i, 
                    'score': peak_val
                })

        # Sort by score descending and keep ONLY TOP 3
        component_candidates.sort(key=lambda item: item['score'], reverse=True)
        top_candidates = component_candidates[:3]

        # Draw only the Top 3 components to the mask
        for cand in top_candidates:
            clean_mask[labels == cand['id']] = 255
            
        # 4. Grouping (Main Logic: 15x15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask_grouped = cv2.dilate(clean_mask, kernel)
        
        # 5. Find Contours
        contours, _ = cv2.findContours(mask_grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue 
            x, y, w, h = cv2.boundingRect(cnt)
            score = np.max(heatmap[y:y+h, x:x+w])
            candidates.append({'bbox': (x, y, w, h), 'score': float(score)})

        # --- [UPDATED FALLBACK LOGIC] ---
        # Define the relevant visual tasks
        visual_task_types = [
            "defect localization", 
            "defect classification", 
            "defect description", 
            "defect analysis"
        ]
        
        # Check if fallback is needed AND allowed for this task type
        if not candidates and self.task_type in visual_task_types:
            if global_best_box is not None and global_max_score > 0.15:
                # Reconstruct mask for the single best component
                single_best_mask = np.zeros_like(mask)
                single_best_mask[labels == global_best_mask_idx] = 255
                
                # Apply morphology (Consistent with Step 1.5 fallback logic: 5x5 kernel)
                kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                single_best_mask = cv2.morphologyEx(single_best_mask, cv2.MORPH_CLOSE, kernel_fix)
                single_best_mask = cv2.dilate(single_best_mask, kernel_fix)
                
                sb_contours, _ = cv2.findContours(single_best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in sb_contours:
                    if cv2.contourArea(cnt) < 20: continue 
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Recalculate score for the new smoothed region
                    score = np.max(heatmap[y:y+h, x:x+w])
                    candidates.append({'bbox': (x, y, w, h), 'score': float(score)})
                    # Only take the largest contour from the best mask
                    break 

        candidates.sort(key=lambda r: r['score'], reverse=True)
        return candidates
    
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
        
        global_heatmap = self.root.state['heatmap_array']
        hm_y1, hm_y2 = int(ny1), int(ny2)
        hm_x1, hm_x2 = int(nx1), int(nx2)
        roi = global_heatmap[hm_y1:hm_y2, hm_x1:hm_x2]
        
        current_score = 0.0
        if roi.size > 0:
            avg_val = np.mean(roi)
            max_val = np.max(roi)
            current_score = 0.8 * max_val + 0.2 * avg_val

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
        if heatmap_score > 0: current_peak = heatmap_score

        new_state = {
            'depth': parent_node.state['depth'] + 1,
            'image': crop_pil,
            'image_width': gx2 - gx1, 
            'image_height': gy2 - gy1,
            'region_coords': (gx1, gy1, gx2, gy2),
            'heatmap_score': current_peak, 
            'heatmap_array': child_heatmap,
            'global_regions': parent_node.state.get('global_regions', [])
        }

        return MCTSNode(
            new_state, 
            parent=parent_node, 
            available_actions=self.DISCRETE_ACTIONS.copy(),
            shared_processor=self.logic_preprocessor # 传递 Agent 持有的那个唯一实例
        )

    def selection(self, node):
        if node.untried_actions: return node
        if not node.children: return node
        total_visits = sum(c.visits for c in node.children.values())
        def ucb(child):
            if child.visits == 0: return float('inf')
            return child.value/child.visits + self.c_puct * math.sqrt(2*math.log(total_visits)/child.visits) + child.heatmap_score*0.5
        return self.selection(max(node.children.values(), key=ucb))

    async def expansion(self, node):
        if node.state['depth'] >= self.max_depth or not node.untried_actions: 
            return node
        
        action = node.untried_actions.pop(0) 
        child = None

        if action.startswith("inspect_region_"):
            child = await self.execute_inspect_region(node, int(action.split("_")[-1]))
            
        elif action in self.DISCRETE_ACTIONS:
            child = self.execute_discrete_action(node, action)

        if child: 
            node.children[action] = child
            return child
        
        return await self.expansion(node)

    async def simulation(self, node):
        if node.heatmap_score < 0.2: return 0.0
        return node.heatmap_score

    def backpropagation(self, node, reward):
        while node:
            node.visits += 1; node.value += reward; node = node.parent

    async def single_run(self, root):
        node = self.selection(root)
        if node.state['depth'] < self.max_depth: node = await self.expansion(node)
        reward = await self.simulation(node)
        node.leaf_reward = reward
        self.backpropagation(node, reward)

    async def process(self):
        setup_seed(42) 
        return await self._process_logic()

    async def _process_logic(self):
        global_heatmap, detected_category = await self.get_anomaly_heatmap(self.image)
        self.logic_report_for_verification = ""
        self.atlas_context_for_verification = ""

        if self.sam_engine:
            self.sam_engine.set_image(self.image)
        if detected_category is not None:
            self.category = detected_category
        else:
            self.category = self.row["category"]

        # === DYNAMIC THRESHOLD SELECTION (Unified) ===
        # All localizers now support get_thresholds(category) -> (image_thresh, pixel_thresh)
        if hasattr(self.localizer, 'get_thresholds'):
            self.image_threshold, self.pixel_threshold = self.localizer.get_thresholds(self.category)
        else:
            # Fallback (Should not happen with current localizers)
            self.image_threshold = DEFAULT_THRESHOLD
            self.pixel_threshold = DEFAULT_THRESHOLD

        if not self.rag_blocks and self.rag_agent is not None:
            if self.rag_cache is not None and self.category in self.rag_cache:
                self.rag_blocks = self.rag_cache[self.category]
            else:
                blocks = self.rag_agent.get_rag_context(self.category)
                self.rag_blocks = blocks
                if self.rag_cache is not None:
                    self.rag_cache[self.category] = blocks

        self._perform_global_check()

        regions = self._detect_heatmap_regions(global_heatmap)
        root_actions = [f"inspect_region_{i}" for i in range(len(regions))] or ["inspect_region_0"]
        
        root_state = {
            'depth': 0, 'image': self.image, 
            'image_width': self.image_width, 'image_height': self.image_height,
            'region_coords': (0, 0, self.image_width, self.image_height),
            'heatmap_score': float(np.max(global_heatmap)), 'heatmap_array': global_heatmap,
            'global_regions': regions
        }
        # [修改] 创建根节点时也一样
        self.root = MCTSNode(
            root_state, 
            available_actions=root_actions,
            shared_processor=self.logic_preprocessor
        )
        
        for _ in range(self.n_simulations): 
            await self.single_run(self.root)

        all_nodes = []
        queue = [self.root]
        while queue:
            n = queue.pop(0)
            if n != self.root: all_nodes.append(n)
            queue.extend(n.children.values())
        
        final_candidates = nms_filter(all_nodes, iou_threshold=0.1, score_threshold=self.pixel_threshold)
        if not final_candidates: final_candidates = [self.root]
        
        return await self._generate_final_answer_multi(final_candidates)
 
    def _perform_global_check(self):
        rag_content_list = []
        self.has_global_standard = False
        self.used_logic_engine = False
        for i, block in enumerate(self.rag_blocks):
            region_name = block.get('region', f'Region {i+1}')
            if "whole" in region_name.lower() or "global" in region_name.lower() or self.category == "cable":
                self.has_global_standard = True
                txt = block.get('text', '').strip()
                rag_content_list.append({"type": "text", "text": f"\n--- [REFERENCE STANDARD] for {region_name} ---\n"})
                
                if block.get('images'):
                    rag_content_list.append({"type": "text", "text": "(Visual Reference of a NORMAL object):\n"})
                    for img_item in block['images']:
                        pil_img = None
                        if isinstance(img_item, Image.Image): pil_img = img_item
                        else:
                            if os.path.exists(str(img_item)): pil_img = Image.open(str(img_item)).convert("RGB")
                        if pil_img: rag_content_list.append({"type": "image", "image": pil_img})
                
                if txt:
                    rag_content_list.append({"type": "text", "text": f"(Text Definitions): {txt}\n"})

        if not self.has_global_standard and self.category != "cable" :
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
   
    async def _generate_final_answer_multi(self, nodes):
        img_np = np.array(self.image.convert("RGB"))
        global_heatmap = self.root.state['heatmap_array']
        W, H = self.image.size

        # === [MEMORY FIX] 预先记录用于返回的debug信息，避免保留整个树引用 ===
        heatmap_peak_score = float(self.root.state['heatmap_score']) if hasattr(self, 'root') else 0.0
        
        # --- Step 1: Strict Segmentation (Original Logic) ---
        # Try to find regions using the strict/optimal thresholds first
        mask = ((global_heatmap > self.pixel_threshold) * 255).astype(np.uint8)
        
        # Connected Components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        
        global_max_score = -1.0
        global_best_box = None # (x, y, w, h)
        global_best_mask_idx = -1

        # [MODIFIED START] Collect candidates first, then filter Top 3
        component_candidates = []

        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            # Get peak score within this connected component
            peak_val = np.max(global_heatmap[y:y+h, x:x+w][labels[y:y+h, x:x+w] == i])
            
            # Track the absolute best region found so far (For Step 1.5 Fallback)
            if peak_val > global_max_score:
                global_max_score = peak_val
                global_best_box = (x, y, w, h)
                global_best_mask_idx = i
            
            # Collect valid high-confidence regions
            if peak_val >= self.image_threshold:
                component_candidates.append({
                    'id': i, 
                    'score': peak_val
                })

        # Sort by score descending and keep ONLY TOP 3
        component_candidates.sort(key=lambda item: item['score'], reverse=True)
        top_candidates = component_candidates[:3]

        # Draw only the Top 3 components to the mask
        for cand in top_candidates:
            clean_mask[labels == cand['id']] = 255
        # [MODIFIED END]
                    
        # Grouping for strict mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask_grouped = cv2.dilate(clean_mask, kernel)
        contours, _ = cv2.findContours(mask_grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box_prompts = []
        raw_contours = [] 
        
        # [MODIFIED] Sort contours by area or score if needed, but the mask is already Top-3 driven.
        # However, dilate might merge them, or split them. 
        # We will strictly limit crop output in Step 2.
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue 
            x, y, w, h = cv2.boundingRect(cnt)
            raw_contours.append(cnt) 
            box_prompts.append([(x + w/2.0)/W, (y + h/2.0)/H, w/W, h/H])

        # --- Step 1.5: [Single Best Region Injection] ---
        # (代码保持不变，省略...)
        visual_task_types = [
            "defect localization", 
            "defect classification", 
            "defect description", 
            "defect analysis"
        ]
        
        should_trigger_fallback = (not box_prompts) and (self.task_type in visual_task_types) and (not self.used_logic_engine)

        fallback_triggered = False # 标记变量

        if should_trigger_fallback:
             if global_max_score > 0.15 and global_best_box is not None:
                single_best_mask = np.zeros_like(mask)
                single_best_mask[labels == global_best_mask_idx] = 255
                kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                single_best_mask = cv2.morphologyEx(single_best_mask, cv2.MORPH_CLOSE, kernel_fix)
                single_best_mask = cv2.dilate(single_best_mask, kernel_fix)
                sb_contours, _ = cv2.findContours(single_best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in sb_contours:
                    if cv2.contourArea(cnt) < 20: continue 
                    x, y, w, h = cv2.boundingRect(cnt)
                    raw_contours.append(cnt)
                    box_prompts.append([(x + w/2.0)/W, (y + h/2.0)/H, w/W, h/H])
                    fallback_triggered = True # 标记兜底已触发
                    break 

        # --- Step 2: Visualization & Cropping ---
        crop_images = []
        valid_box_idx = 0 
        
        for idx, cnt in enumerate(raw_contours):
            # [MODIFIED] Strict safety break: Ensure we never exceed 3 crops
            if len(crop_images) >= 3: 
                break

            x, y, w, h = cv2.boundingRect(cnt)
            valid_box_idx += 1
            cv2.drawContours(img_np, [cnt], -1, (255, 0, 0), 3)
            # Add label
            cv2.putText(img_np, str(valid_box_idx), (x, max(0, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        annotated_global = Image.fromarray(img_np)
        # B. [Local Crops]: Hybrid Decision
        if fallback_triggered:
            # Case 1: Fallback Triggered -> Use Old Logic (Crop from raw_contours with padding)
            # 因为 MCTS 可能没找到东西，或者我们信任兜底逻辑找到的微弱信号
            for idx, cnt in enumerate(raw_contours):
                if len(crop_images) >= 3: break
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Old Logic Padding
                pad_factor = 0.4
                pad_w = int(w * pad_factor)
                pad_h = int(h * pad_factor)
                cx1 = max(0, x - pad_w)
                cy1 = max(0, y - pad_h)
                cx2 = min(W, x + w + pad_w)
                cy2 = min(H, y + h + pad_h)
                
                crop_images.append(self.image.crop((cx1, cy1, cx2, cy2)))
        else:
            # Case 2: Normal -> Use MCTS Found Nodes
            # 使用 MCTS 搜索到的节点图片，这通常比 OpenCV 的框更准或视野更好
            for n in nodes:
                # 过滤全图节点，避免重复
                gx1, gy1, gx2, gy2 = n.state['region_coords']
                is_whole_image = (gx1 == 0 and gy1 == 0 and gx2 == W and gy2 == H)
                if not is_whole_image:
                    crop_images.append(n.state['image'])
        
        
        is_loc_question = (self.task_type == "defect localization")
        self.atlas_context_for_verification = self._extract_atlas_context_text(include_global=False)

        # RAG Logic (Kept Identical)
        rag_content_list = []
        if self.rag_blocks:
            rag_content_list.append({"type": "text", "text": "=== 📚 REFERENCE STANDARDS (VISUAL & TEXT) ===\n"})
            for i, block in enumerate(self.rag_blocks):
                region_name = block.get('region', f'Region {i+1}')
                if "whole" in region_name.lower() or "global" in region_name.lower():
                    continue
                txt = block.get('text', '').strip()
                rag_content_list.append({"type": "text", "text": f"\n--- Standard for {region_name} ---\n"})
                if block.get('images') and not is_loc_question:  
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
                if txt:
                    rag_content_list.append({"type": "text", "text": f"(Definition & Defects): {txt}\n"})
            rag_content_list.append({"type": "text", "text": "\n=============================================\n"})
        
        # 3. Construct Base Prompt
        options_str = ""
        if self.options:
            options_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in self.options.items()])
        
        prompt = f"Question: {self.question}\n\n"
        prompt += f"{options_str}\n\n"
        prompt += "You are a professional industrial QA inspector conducting a multi-stage inspection. Below is the report from the previous step.\n\n"
        prompt += "=== PHASE 1: GLOBAL CHECK REPORT ===\n"
        prompt += f"Automatic Global Inspection Result: {self.global_conclusion}\n\n"       
        prompt += "=== PHASE 2: FINAL DIAGNOSIS ===\n"
        logic_says_defect = False
        if self.used_logic_engine:
             # logical.py returns "Global Status: Normal..." or "Global Status: No logical defects..." for normal.
             # We only treat it as "Definitive Anomaly" if it's NOT one of these clean states.
             if "Normal" not in self.global_conclusion and "No logical defects" not in self.global_conclusion:
                 logic_says_defect = True
        if is_loc_question:
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
            if crop_images:
                if logic_says_defect:
                    prompt += (
                        f"**Situation**: The PHASE 1 Logic Engine has identified a **definitive anomaly**.\n"
                        f"You are provided with a 'Global View' (with Red Boxes) and **one or more zoomed-in 'Focus Views'**.\n"
                        f"**Instruction**: Use the PHASE 1 conclusion as the primary basis. Then, map each Focus View to its Red Box and check whether **additional visible anomalies** exist.\n"
                        f"**Final Decision**: Determine the final defect based on the **combined result** of (1) the Logic Engine anomaly and (2) any visually confirmed anomalies.\n"
                    )
                else:    
                    prompt += (
                        f"**Situation**: Potential anomalies have been localized. Since the system detected anomalies, this object is **DEFECTIVE**.\n You are provided with a 'Global View' (with Red Boxes) and **one or more zoomed-in 'Focus Views'**.\n\n"
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
                    else:
                        prompt += (
                            f"No local anomalies were found by the scanner.\n"
                            f"check PHASE 1 conclusion. If it is not a visible anomaly, you should prefer to choose normal.\n"
                        )

        prompt += (
            f"\n**REQUIRED OUTPUT FORMAT**:\n"
            f"The correct answer is (X)"
        )

        verification_strategy = getattr(self.args, "verification_strategy", "staged")
        if verification_strategy == "single_prompt_concat":
            prompt = self._build_single_prompt_verification(
                options_str=options_str,
                logic_report=self.logic_report_for_verification,
                crop_count=len(crop_images)
            )

        # === [MEMORY FIX] 构建返回字典 ===
        result = {
            "status": "ready",
            "red_box_image": annotated_global,
            "crop_images": crop_images,
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
            "heatmap_peak_score": heatmap_peak_score,
            "anomaly_threshold": self.image_threshold,
            "is_crop_triggered": len(crop_images) > 0,
            "crop_count": len(crop_images),
            "phase_1_conclusion": self.global_conclusion,
            "rag_retrieved_blocks": len(self.rag_blocks) if self.rag_blocks else 0
            }
        }

        # === [CRITICAL MEMORY FIX] 立即清理MCTS树，防止显存泄露 ===
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
            self.root = None

        # 清理中间变量
        del img_np, global_heatmap, mask, clean_mask
        if 'labels' in locals():
            del labels
        if 'mask_grouped' in locals():
            del mask_grouped
        if 'contours' in locals():
            del contours

        return result

