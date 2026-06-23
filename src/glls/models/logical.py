import math
import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks, medfilt
from glls.seg.sam3_profiles import get_object_prompt


class VisualLogicProcessor:
    def __init__(self):
        # --- MVTec Cable 颜色规则配置 ---
        self.STANDARD_RULES = {
            "Center Top":   "YELLOW", 
            "Center Left":  "BLUE",
            "Center Right": "BROWN"   
        }
    def get_heatmap_intersection(self, category, image_pil, sam_engine, heatmap, pixel_threshold):
        """
        Computes the intersection between the semantic object mask (SAM) and the anomaly heatmap.
        
        Logic:
        1. Get the binary mask of the object using SAM (ensures we ignore background noise).
        2. Binarize the heatmap using the strict pixel_threshold.
        3. Return the intersection: pixels that are BOTH part of the object AND anomalous.
        
        Args:
            category (str): Object category (e.g., 'pcb1', 'candle').
            image_pil (PIL.Image): The input image.
            sam_engine (Sam3Engine): The SAM instance.
            heatmap (np.ndarray): The raw floating point heatmap.
            pixel_threshold (float): Threshold to decide if a pixel is anomalous.

        Returns:
            intersection_mask (np.ndarray): Binary mask (0 or 255) of valid anomalies.
        """
        # 1. Get the Semantic Object Mask (Foreground)
        obj_mask = self.get_object_mask(category, image_pil, sam_engine)
        
        # 2. Binarize the Heatmap (Anomaly Candidates)
        # Ensure heatmap matches object mask shape
        if heatmap.shape != obj_mask.shape:
            heatmap = cv2.resize(heatmap, (obj_mask.shape[1], obj_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        heatmap_binary = (heatmap > pixel_threshold).astype(np.uint8)

        # 3. Compute Intersection
        # We only care about anomalies that fall INSIDE the object structure.
        intersection_mask = cv2.bitwise_and(obj_mask, heatmap_binary)

        # Return as 0/255 uint8 mask
        return intersection_mask * 255
    def get_object_mask(self, category, image_pil, sam_engine):
        """Return a binary object mask (H, W) aligned to the input image."""
        img_np = np.array(image_pil)
        h, w = img_np.shape[:2]

        # Fallback: no SAM -> treat entire image as valid object area
        if sam_engine is None:
            return np.ones((h, w), dtype=np.uint8)

        cat_str = str(category).lower()
        prompt = get_object_prompt(cat_str)

        try:
            res = sam_engine.predict_mask(prompt)

            # Handle various return formats from sam_engine
            masks = res
            if isinstance(res, (tuple, list)) and len(res) > 0:
                masks = res[0]

            # If multiple masks, take the first (largest/most confident)
            if isinstance(masks, (tuple, list)):
                if len(masks) == 0:
                    return np.ones((h, w), dtype=np.uint8)
                masks = masks[0]

            if masks is None:
                return np.ones((h, w), dtype=np.uint8)

            mask = np.array(masks)
            if mask.ndim == 3:
                mask = mask.squeeze()
            mask = (mask > 0).astype(np.uint8)

            # Ensure same resolution as input image
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # If mask is degenerate (too small), fall back to full image
            if int(mask.sum()) < 10:
                return np.ones((h, w), dtype=np.uint8)

            return mask
        except Exception as e:
            print(f"⚠️ SAM object mask failed: {e}")
            return np.ones((h, w), dtype=np.uint8)
    def get_logic_analysis(self, category, image_pil, sam_engine):
        """
        统一的逻辑分析入口，支持 MVTec 和 VisA 数据集
        """
        cat_str = str(category).lower()
        
        # === MVTec Categories ===
        if cat_str == "cable":
            return self._analyze_cable_simple(image_pil, sam_engine)
        elif cat_str == "zipper":
            return self._analyze_zipper_simple(image_pil, sam_engine)
        elif cat_str == "pcb3":
            return self._analyze_pcb3(image_pil, sam_engine)
        # === VisA Categories ===
        elif cat_str == "candle":
            return self._analyze_candle_wick(image_pil, sam_engine)
            
        return None, None, None
    def _get_mask_safe(self, sam_engine, prompt, threshold=None):
        """Helper to safely get a binary mask from SAM."""
        try:
            if threshold:
                res = sam_engine.predict_mask(prompt, threshold=threshold)
            else:   
                res = sam_engine.predict_mask(prompt)
                
            if res is None: return None
            
            # Handle list/tuple returns
            m_raw = res[0] if isinstance(res, (tuple, list)) else res
            if m_raw is None: return None
            
            # Handle 3D arrays (C, H, W)
            if m_raw.ndim == 3: 
                m_final = np.max(m_raw, axis=0) 
            else: 
                m_final = m_raw
                
            return (m_final > 0).astype(np.uint8) * 255
        except Exception as e:
            print(f"Mask generation error for '{prompt}': {e}")
            return None
    # =========================================================
    # 4. VisA - PCB3 Logic (Body-Assisted Check) - CORRECTED
    # =========================================================
    def _analyze_pcb3(self, image_pil, sam_engine):
        """
        VisA PCB3 Logic (Location Aware):
        1. Detect Body, Pins, and Bulbs FIRST.
        2. Infer Global Orientations (Cross-reference sides: if Pins missing, use Bulb side to guess).
        3. Check Defects using these inferred locations to provide specific "Left/Right" reports.
        """
        img_np = np.array(image_pil)
        h_img, w_img = img_np.shape[:2]
        debug_vis = img_np.copy()
        report_lines = []
        
        # Defect accumulators
        defects = []

        # --- Helper: Get Location Label ---
        def get_pos_label(index, total_count, is_vertical=True):
            if total_count == 2:
                names = ["Top", "Bottom"] if is_vertical else ["Left", "Right"]
                return names[index] if index < 2 else f"#{index+1}"
            elif total_count == 3:
                names = ["Top", "Middle", "Bottom"] if is_vertical else ["Left", "Mid", "Right"]
                return names[index] if index < 3 else f"#{index+1}"
            return f"#{index+1}"

        # ---------------------------------------------------------
        # 1. Detection Phase (Gather all Data first)
        # ---------------------------------------------------------
        
        # A. Find Body
        mask_body = self._get_mask_safe(sam_engine, "the circuit board")
        if mask_body is None: mask_body = np.zeros((h_img, w_img), dtype=np.uint8)
        cnts_body, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Body Geometry Defaults
        body_center_x = w_img / 2
        body_left_x = 0; body_right_x = w_img
        
        if cnts_body:
            all_points = np.vstack(cnts_body)
            bx, by, bw, bh = cv2.boundingRect(all_points)
            body_center_x = bx + bw / 2
            body_left_x = bx; body_right_x = bx + bw
            cv2.drawContours(debug_vis, cnts_body, -1, (255, 0, 0), 1)
            cv2.rectangle(debug_vis, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)

        # B. Find Pins
        mask_pins = self._get_mask_safe(sam_engine, "the thin pin", threshold=0.75)
        if mask_pins is None: mask_pins = np.zeros((h_img, w_img), dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_pins = cv2.morphologyEx(mask_pins, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts_pins, _ = cv2.findContours(mask_pins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_pins = [c for c in cnts_pins if cv2.contourArea(c) > 30]
        valid_pins.sort(key=lambda c: cv2.boundingRect(c)[1]) 

        # C. Find Bulbs
        target_bulbs = { "Clear LED": "the clear bulb", "Black Sensor": "the black cap" }
        detected_bulbs = {} # Store contours: {'Clear LED': contour, ...}
        
        for name, prompt in target_bulbs.items():
            mask = self._get_mask_safe(sam_engine, prompt, threshold=0.35)
            cnts, _ = cv2.findContours(mask if mask is not None else np.zeros((h_img, w_img), dtype=np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c_max = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c_max) > 40:
                    detected_bulbs[name] = c_max

        # ---------------------------------------------------------
        # 2. Orientation Inference (The "Smart" Logic)
        # ---------------------------------------------------------
        
        # Determine Pin Side
        pin_side_str = "Unknown"
        if valid_pins:
            pin_x_avg = np.mean([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 for c in valid_pins])
            pin_side_str = "Right Edge" if pin_x_avg > body_center_x else "Left Edge"

        # Determine Bulb Side (Aggregate)
        bulb_side_str = "Unknown"
        if detected_bulbs:
            # Calculate average X of all found bulbs
            bulb_centers = []
            for c in detected_bulbs.values():
                bx, _, bw, _ = cv2.boundingRect(c)
                bulb_centers.append(bx + bw/2)
            bulb_x_avg = np.mean(bulb_centers)
            bulb_side_str = "Right Edge" if bulb_x_avg > body_center_x else "Left Edge"

        # *** Cross-Inference ***
        # If Pins are missing (Unknown), but Bulbs are found at Left, assume Pins are at Right (and vice versa)
        if pin_side_str == "Unknown" and bulb_side_str != "Unknown":
            pin_side_str = "Right Edge" if bulb_side_str == "Left Edge" else "Left Edge"
        
        if bulb_side_str == "Unknown" and pin_side_str != "Unknown":
            bulb_side_str = "Right Edge" if pin_side_str == "Left Edge" else "Left Edge"

        # ---------------------------------------------------------
        # 3. Defect Analysis (Using Inferred Locations)
        # ---------------------------------------------------------

        # --- A. Pins Analysis ---
        if len(valid_pins) < 3:
            # Now we can report WHERE they are missing from
            defects.append(f"Missing Pins ({3 - len(valid_pins)} missing) at {pin_side_str}")

        bent_pin_locs = []
        for i, c in enumerate(valid_pins):
            local_name = get_pos_label(i, 3, is_vertical=True)
            px, py, pw, ph = cv2.boundingRect(c)
            p_center_x = px + pw/2
            
            is_bent = False
            intrusion_buffer = 20 # Relaxed buffer
            
            # Check against inferred side
            if "Left" in pin_side_str:
                if p_center_x > (body_left_x + intrusion_buffer): is_bent = True
            elif "Right" in pin_side_str:
                if p_center_x < (body_right_x - intrusion_buffer): is_bent = True
            
            # Geometry check
            rect = cv2.minAreaRect(c); (w_r, h_r) = rect[1]
            ar = max(w_r,h_r)/(min(w_r,h_r)+0.01)
            hull = cv2.convexHull(c)
            sol = cv2.contourArea(c)/cv2.contourArea(hull) if cv2.contourArea(hull)>0 else 0
            if ar < 2.0 or sol < 0.75: is_bent = True

            color = (0, 0, 255) if is_bent else (0, 255, 0)
            if is_bent: bent_pin_locs.append(local_name)
            cv2.drawContours(debug_vis, [c], -1, color, 2)
            cv2.circle(debug_vis, (int(p_center_x), int(py+ph/2)), 3, (0, 255, 255), -1)

        if bent_pin_locs:
            defects.append(f"Misalignment or Deformation.Bent Pins at {pin_side_str} ({', '.join(bent_pin_locs)})")

        # --- B. Bulbs Analysis ---
        for name in target_bulbs.keys():
            if name not in detected_bulbs:
                # Report missing at the inferred Bulb Side
                defects.append(f"Missing {name} at {bulb_side_str}")
                continue
            
            # Check Geometry/Bent for found bulbs
            c = detected_bulbs[name]
            bx, by, bw, bh = cv2.boundingRect(c)
            b_center_x = bx + bw/2
            
            is_bulb_bent = False
            
            # 1. Aspect Ratio / Solidity
            rect = cv2.minAreaRect(c); (w_r, h_r) = rect[1]
            ar = max(w_r, h_r) / (min(w_r, h_r) + 0.01)
            hull = cv2.convexHull(c)
            sol = cv2.contourArea(c) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            if ar < 1.1 or sol < 0.82: is_bulb_bent = True
            
            # 2. Location Alignment (Dynamic check based on specific bulb center)
            # If the specific bulb is physically on the left, check left intrusion
            my_side = "Left" if b_center_x < body_center_x else "Right"
            bulb_buffer = 15
            
            if my_side == "Left":
                if b_center_x > (body_left_x + bulb_buffer): is_bulb_bent = True
            else:
                if b_center_x < (body_right_x - bulb_buffer): is_bulb_bent = True

            color = (0, 0, 255) if is_bulb_bent else (0, 255, 0)
            if is_bulb_bent:
                # Report specific bent location
                defects.append(f"Deformation.Bent/Misaligned {name} ({my_side} Edge)")
            
            cv2.drawContours(debug_vis, [c], -1, color, 2)
            cv2.putText(debug_vis, name, (bx, max(0, by-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ---------------------------------------------------------
        # 4. Final Verdict
        # ---------------------------------------------------------
        if not defects:
            final_verdict = "Global Status: Normal."
        else:
            final_verdict = f"Global Status: Defect detected. {'; '.join(defects)}."

        report_lines.append(f"Analysis Result:")
        report_lines.append(f"- Body Detected: {'Yes' if cnts_body else 'No'}")
        report_lines.append(f"- Pins: {len(valid_pins)}/3 found ({pin_side_str}).") # Added location to report
        report_lines.append(f"- Bulbs: {len(detected_bulbs)}/2 found.")
        if defects:
            report_lines.append(f"- Anomalies: {', '.join(defects)}")
        
        full_report = "[Logic Engine] Inspection Report:\n" + "\n".join(report_lines)
        del img_np, mask_body, mask_pins
        
        return full_report, Image.fromarray(debug_vis), final_verdict
    def _analyze_candle_wick(self, image_pil, sam_engine):
        """
        针对 VisA Candle 的细粒度逻辑分析：
        1. Short Wick (短芯)
        2. Missing Wick (缺芯)
        3. Irregular/Long Wick (异形/长芯) - 可选
        """
        img_np = np.array(image_pil)
        h_img, w_img = img_np.shape[:2]
        debug_vis = img_np.copy()
        report_lines = []
        defects_found = []

        # ---------------------------------------------------------
        # 1. SAM Segmentation
        # ---------------------------------------------------------
        prompts = {
            "whole_object": "all the round tea light candles",
            "wick": "the small white wick threads in the center of the tea light candles" 
        }

        # Step A: 获取所有蜡烛主体 (Reference Scale)
        res_candles = sam_engine.predict_mask(prompts["whole_object"])
        mask_candles = res_candles[0] if isinstance(res_candles, (tuple, list)) else res_candles
        if mask_candles is None:
            return "Error: No candles detected.", None, None
        if mask_candles.ndim == 3: mask_candles = mask_candles.squeeze()
        mask_candles = mask_candles.astype(np.uint8) * 255

        # Step B: 获取所有灯芯 (Targets)
        res_wicks = sam_engine.predict_mask(prompts["wick"])
        mask_wicks = res_wicks[0] if isinstance(res_wicks, (tuple, list)) else res_wicks
        if mask_wicks is None:
            mask_wicks = np.zeros_like(mask_candles) # 可能是全部缺失
        elif mask_wicks.ndim == 3:
            mask_wicks = mask_wicks.squeeze()
        mask_wicks = mask_wicks.astype(np.uint8) * 255

        # ---------------------------------------------------------
        # 2. Instance Analysis (Loop through each candle)
        # ---------------------------------------------------------
        # 找出独立的蜡烛个体
        contours_candles, _ = cv2.findContours(mask_candles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按位置排序：从上到下，从左到右
        candle_instances = []
        for cnt in contours_candles:
            if cv2.contourArea(cnt) < 1000: continue # 过滤噪点
            x, y, w, h = cv2.boundingRect(cnt)
            candle_instances.append({
                "cnt": cnt,
                "bbox": (x, y, w, h),
                "center": (x + w//2, y + h//2),
                "diameter": max(w, h) # 蜡烛直径估计
            })

        # 简单的排序逻辑：先按 Y 排序（分行），再按 X 排序
        candle_instances.sort(key=lambda k: (k['center'][1] // (h_img//2), k['center'][0]))

        report_lines.append(f"Detected {len(candle_instances)} candles.")

        for i, candle in enumerate(candle_instances):
            x, y, w, h = candle['bbox']
            diameter = candle['diameter']
            
            # 定义位置名称
            row_name = "Top" if candle['center'][1] < h_img / 2 else "Bottom"
            col_name = "Left" if candle['center'][0] < w_img / 2 else "Right"
            pos_name = f"{row_name}-{col_name} Candle"

            # 绘制蜡烛轮廓
            cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (200, 200, 200), 1)
            
            # --- ROI Check: 在当前蜡烛范围内找 Wick ---
            # 创建当前蜡烛的 Mask
            single_candle_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(single_candle_mask, [candle['cnt']], -1, 255, -1)
            
            # 取交集：Wicks AND Single_Candle
            current_wick_mask = cv2.bitwise_and(mask_wicks, mask_wicks, mask=single_candle_mask)
            
            # 分析该蜡烛内的灯芯
            cnts_wick, _ = cv2.findContours(current_wick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            status = "Normal"
            color = (0, 255, 0) # Green for Good
            metric_info = ""

            if not cnts_wick:
                # Case 1: 没找到灯芯 -> Missing Wick
                status = "MISSING_WICK"
                color = (0, 0, 255) # Red
                defects_found.append(f"{pos_name} (Missing Wick)")
            else:
                # 找到最大的一块作为主灯芯
                main_wick_cnt = max(cnts_wick, key=cv2.contourArea)
                
                # 计算灯芯的几何特征
                rect = cv2.minAreaRect(main_wick_cnt) # (center), (width, height), angle
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # 灯芯长度 (取矩形的长边)
                wick_len = max(rect[1])
                
                # 计算比率：灯芯长度 / 蜡烛直径
                # 正常灯芯通常占据直径的 20% - 30% 左右
                ratio = wick_len / diameter
                
                metric_info = f"Ratio: {ratio:.2f}"

                # === 核心判定逻辑 ===
                SHORT_WICK_THRESH = 0.22
                
                if ratio < SHORT_WICK_THRESH:
                    status = "SHORT_WICK"
                    color = (0, 0, 255) # Red
                    defects_found.append(f"{pos_name} (Short Wick or off center Wick detected, Ratio={ratio:.2f})")
                elif ratio >0.45:
                    status = "LONG_WICK"
                    color = (0, 0, 255) # Red
                    defects_found.append(f"{pos_name} (Long Wick or missing Wick detected, Ratio={ratio:.2f})")
                else:
                    # 如果不是短芯，检查是否是不规则
                    pass

                # 绘制灯芯
                cv2.drawContours(debug_vis, [box], 0, color, 2)

            # 在图上标注文字
            cv2.putText(debug_vis, f"{i+1}", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_vis, status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if metric_info:
                cv2.putText(debug_vis, metric_info, (x, y + h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            report_lines.append(f"- {pos_name}: {status} {metric_info}")

        # ---------------------------------------------------------
        # 3. Final Verdict
        # ---------------------------------------------------------
        final_verdict = None
        if len(defects_found) > 0:
            defect_desc = ", ".join(defects_found)
            final_verdict = f"Global Status: Defect detected. {defect_desc}."
            
            if "Short Wick" in defect_desc:
                report_lines.append("\n[Logic Engine Note]: The wick appears significantly shorter than the standard reference ratio.")
        else:
            final_verdict = "Global Status: Normal. All wicks are present and of standard length."

        # === [MEMORY FIX] 清理中间变量 ===
        del img_np, mask_candles, mask_wicks
        if 'res_candles' in locals():
            del res_candles
        if 'res_wicks' in locals():
            del res_wicks
        if 'contours_candles' in locals():
            del contours_candles

        return "\n".join(report_lines), Image.fromarray(debug_vis), final_verdict

    # =========================================================
    # 2. MVTec - Cable Logic
    # =========================================================
    def _analyze_cable_simple(self, image_pil, sam_engine):
        img_np = np.array(image_pil)
        h, w = img_np.shape[:2]
        
        # ---------------------------------------------------------
        # 1. SAM Segmentation & Bi-directional Safe Mask Construction
        # ---------------------------------------------------------
        prompts = {
            "outer_ring": "the large white circular object", 
            "copper_cores": "the inner copper strands inside the wire"
        }
        
        # --- A. Get Outer Sheath & Create Outer Exclusion Zone ---
        res_sheath = sam_engine.predict_mask(prompts["outer_ring"])
        mask_sheath = res_sheath[0] if isinstance(res_sheath, (tuple, list)) else res_sheath
        if mask_sheath is None: return "Error: Sheath not detected.", None, None
        if mask_sheath.ndim == 3: mask_sheath = mask_sheath.squeeze()
        mask_sheath = mask_sheath.astype(bool)

        # Strategy: Dilate sheath (push boundary inward) to prevent sampling near the edge
        kernel_sheath = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask_sheath_uint8 = mask_sheath.astype(np.uint8)
        mask_sheath_dilated = cv2.dilate(mask_sheath_uint8, kernel_sheath) > 0

        # --- B. Get Copper Cores & Create Inner Exclusion Zone ---
        res_copper = sam_engine.predict_mask(prompts["copper_cores"])
        mask_copper = res_copper[0] if isinstance(res_copper, (tuple, list)) else res_copper
        
        if mask_copper is not None:
            if mask_copper.ndim == 3: mask_copper = mask_copper.squeeze()
            # Strategy: Dilate copper (push boundary outward) to prevent sampling near the core
            kernel_copper = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) 
            mask_copper_uint8 = mask_copper.astype(np.uint8)
            mask_copper_dilated = cv2.dilate(mask_copper_uint8, kernel_copper) > 0
        else:
            mask_copper_dilated = np.zeros_like(mask_sheath, dtype=bool)

        # --- C. Construct Whole Solid Area ---
        sheath_uint8 = (mask_sheath.astype(np.uint8) * 255)
        mask_whole = np.zeros_like(mask_sheath, dtype=np.uint8)
        contours, _ = cv2.findContours(sheath_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours: 
            cv2.drawContours(mask_whole, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
        mask_whole = mask_whole.astype(bool)
        
        # --- D. Calculate "Absolutely Safe" Insulation Area ---
        # Logic: Whole Area - Dilated Sheath - Dilated Copper
        mask_safe_insulation = mask_whole & (~mask_sheath_dilated) & (~mask_copper_dilated)
        
        # Backup Mask: If erosion removes everything, fall back to standard mask
        mask_backup_insulation = mask_whole & (~mask_sheath) & (~mask_copper_dilated)

        # ---------------------------------------------------------
        # 2. Ray Intersection Midpoint
        # ---------------------------------------------------------
        M = cv2.moments(mask_whole.astype(np.uint8))
        if M["m00"] == 0: return "Error: Cable empty.", None, None
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        probes = {
            "Center Top": -90,    # 12 o'clock
            "Center Right": 30,   # 4 o'clock
            "Center Left": 150    # 8 o'clock
        }

        report_lines = [f"Geometry Center: ({cx}, {cy})"]
        debug_vis = img_np.copy()
        # Visualization: Darken unsafe areas for debugging
        debug_vis[~mask_safe_insulation] = debug_vis[~mask_safe_insulation] // 3
        cv2.circle(debug_vis, (cx, cy), 5, (255, 0, 255), -1)
        
        detected_results = {}

        for pos_name, angle_deg in probes.items():
            angle_rad = math.radians(angle_deg)
            dir_x = math.cos(angle_rad); dir_y = math.sin(angle_rad)
            
            # 1. Create Ray Mask
            ray_mask = np.zeros_like(mask_whole, dtype=np.uint8)
            end_x = int(cx + w * dir_x)
            end_y = int(cy + h * dir_y)
            cv2.line(ray_mask, (cx, cy), (end_x, end_y), 1, 2)
            
            # 2. Find Intersection
            intersect_mask = (ray_mask > 0) & mask_safe_insulation
            if not np.any(intersect_mask):
                intersect_mask = (ray_mask > 0) & mask_backup_insulation
            
            ys, xs = np.where(intersect_mask)
            
            if len(ys) > 0:
                mid_idx = len(ys) // 2
                sample_x = xs[mid_idx]
                sample_y = ys[mid_idx]
            else:
                # Fallback geometric estimation
                fallback_r = min(h, w) // 4
                sample_x = int(cx + fallback_r * dir_x)
                sample_y = int(cy + fallback_r * dir_y)

            # HSV Color Logic
            y1, y2 = max(0, sample_y - 5), min(h, sample_y + 5)
            x1, x2 = max(0, sample_x - 5), min(w, sample_x + 5)
            roi_img = img_np[y1:y2, x1:x2]
            roi_mask = mask_backup_insulation[y1:y2, x1:x2]
            
            if not np.any(roi_mask): valid_pixels = roi_img
            else: valid_pixels = roi_img[roi_mask]

            color_res = "VOID"
            draw_color = (50, 50, 50)

            if valid_pixels.size > 0:
                pixels_hsv = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
                h_val = np.mean(pixels_hsv[:, 0, 0])
                s_val = np.mean(pixels_hsv[:, 0, 1])
                v_val = np.mean(pixels_hsv[:, 0, 2])
                
                if v_val < 30:
                    color_res = "VOID"; draw_color = (50, 50, 50)
                elif s_val < 45: 
                    color_res = "BROWN"; draw_color = (139, 69, 19)
                else:
                    if 95 < h_val < 155:
                        color_res = "BLUE"; draw_color = (0, 0, 255)
                    elif 15 < h_val <= 95:
                        color_res = "YELLOW"; draw_color = (255, 255, 0)
                    else:
                        color_res = "BROWN"; draw_color = (139, 69, 19)
            
            detected_results[pos_name] = color_res
            report_lines.append(f"- {pos_name}: Found {color_res}")
            
            # Draw
            cv2.line(debug_vis, (cx, cy), (sample_x, sample_y), (150, 150, 150), 1)
            cv2.circle(debug_vis, (sample_x, sample_y), 6, draw_color, 2)
            cv2.putText(debug_vis, pos_name, (sample_x-30, sample_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(debug_vis, color_res, (sample_x-20, sample_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

        # ---------------------------------------------------------
        # 3. Final Verdict (UPDATED LOGIC)
        # ---------------------------------------------------------
        final_verdict = None
        missing_parts = [] 
        swap_details = []

        # Step A: 统计缺失的部分
        for pos, detected_color in detected_results.items():
            if detected_color == "VOID":
                missing_parts.append(pos)

        # Step B: 生成缺失描述
        if len(missing_parts) >= 2:
            missing_locs_str = ", ".join(missing_parts)
            final_verdict = f"Global Status: Missing cables or coppers detected at {missing_locs_str}."
        
        elif len(missing_parts) == 1:
            pos = missing_parts[0]
            expected_color = self.STANDARD_RULES.get(pos, "UNKNOWN")
            msg = (f"Missing cable or coppers detected in the expected {expected_color} insulation section at {pos} "
                   f"at the center of the cross-section, located at the approximate center of the image.")
            final_verdict = f"Global Status: {msg}"
            
        else:
            for pos, expected_color in self.STANDARD_RULES.items():
                detected = detected_results.get(pos)
                if detected != "VOID" and detected != expected_color:
                    swap_details.append(f"{pos} is {detected} (Expected {expected_color})")
            
            if swap_details:
                final_verdict = f"Global Status: Cable Swap detected. Details: {'; '.join(swap_details)}."
            else:
                final_verdict = "Global Status: No logical defects(no missing cable or cable swap)."

        # === [MEMORY FIX] 清理中间变量 ===
        del img_np, mask_sheath, mask_copper_dilated, mask_sheath_dilated
        del mask_whole, mask_safe_insulation, mask_backup_insulation
        if 'res_sheath' in locals():
            del res_sheath
        if 'res_copper' in locals():
            del res_copper
        if 'contours' in locals():
            del contours

        return "\n".join(report_lines), Image.fromarray(debug_vis), final_verdict
    
    # =========================================================
    # 3. MVTec - Zipper Logic
    # =========================================================
    def _analyze_zipper_simple(self, image_pil, sam_engine):
        """
        针对 Zipper 的几何逻辑检测
        1. Squeezed Teeth: 宽度小于平均值 (Narrowing/Constriction)
        2. Split Teeth: 宽度大于平均值 (Widening/Separation)
        """
        img_np = np.array(image_pil)
        h, w = img_np.shape[:2]
        
        # 1. SAM Segmentation: 获取拉链齿区域
        prompt = "the central interlocking zipper teeth"
        res_teeth = sam_engine.predict_mask(prompt)
        
        mask_teeth = res_teeth[0] if isinstance(res_teeth, (tuple, list)) else res_teeth
        if mask_teeth is None:
            return "Error: Zipper teeth not detected.", None, None
            
        if mask_teeth.ndim == 3: mask_teeth = mask_teeth.squeeze()
        mask_teeth = mask_teeth.astype(np.uint8) * 255

        # 2. 预处理：形态学闭运算
        # 使用垂直方向的长方形核，连接齿缝；对于 Split 情况，这会将分开的左右两边连成一个较宽的块
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        mask_solid = cv2.morphologyEx(mask_teeth, cv2.MORPH_CLOSE, kernel)
        
        # 3. 提取最大轮廓
        contours, _ = cv2.findContours(mask_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "Error: No zipper contour found.", None, None
        
        main_cnt = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(main_cnt)
        
        # 创建 Debug 视图
        debug_vis = img_np.copy()
        cv2.drawContours(debug_vis, [main_cnt], -1, (0, 255, 0), 1) # 绿色轮廓
        
        # 4. 几何分析：宽度轮廓
        roi_mask = mask_solid[y:y+h_box, x:x+w_box]
        row_widths = np.sum(roi_mask > 0, axis=1)
        
        valid_indices = np.where(row_widths > 0)[0]
        if len(valid_indices) < 50:
             return "Error: Zipper region too short.", None, None
             
        start_idx = int(len(valid_indices) * 0.1)
        end_idx = int(len(valid_indices) * 0.9)
        trimmed_widths = row_widths[valid_indices[start_idx:end_idx]]
        
        # 统计特征
        median_width = np.median(trimmed_widths)
        min_width = np.min(trimmed_widths)
        max_width = np.max(trimmed_widths)
        
        report_lines = [
            f"Median Width: {median_width:.1f} px",
            f"Range: [{min_width:.1f}, {max_width:.1f}] px",
        ]

        # 5. 双向判定逻辑
        # 阈值设定 (可微调)
        # Squeezed: 宽度 < 中位数 88%
        squeeze_threshold = median_width * 0.88
        # Split: 宽度 > 中位数 120% (开裂通常伴随显著变宽)
        split_threshold = median_width * 1.20
        
        is_squeezed = False
        is_split = False
        
        squeeze_y_coords = []
        split_y_coords = []

        for i in range(len(row_widths)):
            # 忽略顶底边缘
            if i < h_box * 0.05 or i > h_box * 0.95: continue
            
            width = row_widths[i]
            if width <= 0: continue
            
            # Check Squeezed (变窄)
            if width < squeeze_threshold:
                is_squeezed = True
                squeeze_y_coords.append(y + i)
            
            # Check Split (变宽)
            elif width > split_threshold:
                is_split = True
                split_y_coords.append(y + i)
        
        # 6. 绘制异常区域
        # Squeezed -> 红色 (Red)
        if squeeze_y_coords:
            min_y = min(squeeze_y_coords)
            max_y = max(squeeze_y_coords)
            cv2.rectangle(debug_vis, (x, min_y), (x+w_box, max_y), (255, 0, 0), 3)
            cv2.putText(debug_vis, "SQUEEZED", (x - 110, (min_y+max_y)//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Split -> 橙色 (Orange: BGR=0,165,255)
        if split_y_coords:
            min_y = min(split_y_coords)
            max_y = max(split_y_coords)
            cv2.rectangle(debug_vis, (x, min_y), (x+w_box, max_y), (0, 165, 255), 3)
            cv2.putText(debug_vis, "SPLIT/WIDE", (x + w_box + 10, (min_y+max_y)//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 7. 生成结论
        final_verdict = None
        if is_split:
            final_verdict = "Global Status: Split Teeth detected. Significant widening observed in zipper structure."
            report_lines.append("- Result: FAIL (Split/Widening)")
        elif is_squeezed:
            final_verdict = "Global Status: Squeezed Teeth detected. Significant narrowing observed in zipper structure."
            report_lines.append("- Result: FAIL (Squeezed/Narrowing)")
        else:
            final_verdict = "Global Status: Normal. Zipper teeth width is consistent."
            report_lines.append("- Result: PASS")

        # === [MEMORY FIX] 清理中间变量 ===
        del img_np, mask_teeth, mask_solid, roi_mask, row_widths
        if 'res_teeth' in locals():
            del res_teeth
        if 'contours' in locals():
            del contours
        if 'main_cnt' in locals():
            del main_cnt

        return "\n".join(report_lines), Image.fromarray(debug_vis), final_verdict
