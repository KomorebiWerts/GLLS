"""
SAM3 Engine - 优化版本 (框 + 点 + 负样本策略)

核心问题：
=========
对于弯曲的铜线等异常，矩形框会把其他正常部分也框进去。
SAM 基于图像边缘分割，可能会切出整个铜线而不是异常弯曲部分。

解决策略：
=========
1. 正样本框: 热力图区域的外接矩形（稍微扩大）
2. 正样本点: 热力图峰值位置（锚定"这才是我要的"）
3. 负样本点/框: 在矩形框内、但在热力图区域外的位置
   - 告诉 SAM "虽然在框内，但这些不是目标"

负样本采样策略：
==============
在 expanded_bbox 内部、但在 heatmap_contour 外部采样：
  ┌─────────────────────┐
  │  ●neg    ●neg       │  <- expanded_bbox
  │    ┌─────────┐      │
  │    │ ★pos   │      │  <- heatmap_contour
  │    │  异常   │      │
  │    └─────────┘      │
  │  ●neg        ●neg   │
  └─────────────────────┘

采样位置：
- 热力图轮廓外扩 10-15%，在这个环形区域采点
- 或者在 bbox 的四个角落附近采点
"""

import torch
import numpy as np
from PIL import Image
import os
import cv2

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class Sam3Engine:
    def __init__(self, checkpoint_path, device=None):
        """初始化 SAM3 模型"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SAM3 Engine] Loading model from: {checkpoint_path} to {self.device}")
        
        if "cuda" in str(self.device) and ":" in str(self.device):
            try:
                device_idx = int(str(self.device).split(":")[-1])
                torch.cuda.set_device(device_idx)
            except Exception:
                pass

        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=self.device
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = Sam3Processor(self.model)
        print("[SAM3 Engine] Model loaded successfully.")
        
        self.inference_state = None
        self.current_image_pil = None
        self._image_w = 0
        self._image_h = 0

    def set_image(self, image_path_or_pil):
        """设置当前处理的图片"""
        if isinstance(image_path_or_pil, str):
            self.current_image_pil = Image.open(image_path_or_pil).convert("RGB")
        else:
            self.current_image_pil = image_path_or_pil.convert("RGB") if hasattr(image_path_or_pil, 'convert') else image_path_or_pil
        
        self._image_w, self._image_h = self.current_image_pil.size
            
        if "cuda" in str(self.device) and ":" in str(self.device):
            try:
                device_idx = int(str(self.device).split(":")[-1])
                torch.cuda.set_device(device_idx)
            except Exception:
                pass

        with torch.inference_mode():
            self.inference_state = self.processor.set_image(self.current_image_pil)

    def _reset_prompts(self):
        """重置所有提示，准备新一轮预测"""
        if self.inference_state is not None:
            self.processor.reset_all_prompts(self.inference_state)

    def _pixel_to_normalized_box(self, x, y, w, h, padding_ratio=0.0):
        """将像素坐标转换为归一化的 [cx, cy, w, h] 格式"""
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(self._image_w - x, w + 2 * pad_w)
        h = min(self._image_h - y, h + 2 * pad_h)
        
        cx = (x + w / 2.0) / self._image_w
        cy = (y + h / 2.0) / self._image_h
        nw = w / self._image_w
        nh = h / self._image_h
        
        return [cx, cy, nw, nh]

    def _point_to_small_box(self, px, py, box_size_ratio=0.02):
        """将点坐标转换为小框"""
        cx = px / self._image_w
        cy = py / self._image_h
        nw = box_size_ratio
        nh = box_size_ratio
        return [cx, cy, nw, nh]

    def _sample_negative_points_in_margin(
        self, 
        contour, 
        bbox_xyxy, 
        heatmap=None,
        pixel_threshold=None,
        margin_ratio=0.15,
        num_points=4
    ):
        """
        在 bbox 内、但在 contour 外的区域采样负样本点
        
        策略：
        1. 创建 contour 的 mask
        2. 将 contour 向外扩展 margin_ratio
        3. 在 "扩展后的 contour" 和 "原始 contour" 之间的环形区域采样
        4. 如果环形区域太小，则在 bbox 四角采样
        
        Args:
            contour: 原始热力图轮廓
            bbox_xyxy: [x1, y1, x2, y2] 扩展后的 bbox
            heatmap: 热力图（可选，用于确保采样点热力值低）
            pixel_threshold: 像素阈值（可选）
            margin_ratio: 环形区域的宽度比例
            num_points: 采样点数量
        """
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        H, W = self._image_h, self._image_w
        
        # 创建 contour 的 mask
        contour_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # 膨胀 contour 创建外扩区域
        cx, cy, cw, ch = cv2.boundingRect(contour)
        dilate_size = max(3, int(max(cw, ch) * margin_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        dilated_mask = cv2.dilate(contour_mask, kernel)
        
        # 环形区域 = 膨胀后 - 原始
        margin_mask = dilated_mask.copy()
        margin_mask[contour_mask > 0] = 0
        
        # 限制在 bbox 内
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 255
        margin_mask = margin_mask & bbox_mask
        
        # 如果有热力图，进一步限制在低热力值区域
        if heatmap is not None and pixel_threshold is not None:
            low_heat_mask = (heatmap < pixel_threshold * 0.8).astype(np.uint8) * 255
            margin_mask = margin_mask & low_heat_mask
        
        # 在环形区域采样
        valid_coords = np.argwhere(margin_mask > 0)  # (y, x) format
        
        if len(valid_coords) >= num_points:
            # 均匀采样
            indices = np.linspace(0, len(valid_coords) - 1, num_points, dtype=int)
            sampled = valid_coords[indices]
            return [(int(x), int(y)) for y, x in sampled]
        
        # Fallback: 在 bbox 四角 + 边中点采样
        return self._sample_bbox_corners_and_edges(bbox_xyxy, contour_mask, num_points)

    def _sample_bbox_corners_and_edges(self, bbox_xyxy, contour_mask, num_points=4):
        """在 bbox 的角落和边缘采样（避开 contour 区域）"""
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        w, h = x2 - x1, y2 - y1
        
        # 候选点：四角 + 四边中点
        margin_x = max(3, int(w * 0.1))
        margin_y = max(3, int(h * 0.1))
        
        candidates = [
            # 四角
            (x1 + margin_x, y1 + margin_y),           # 左上
            (x2 - margin_x, y1 + margin_y),           # 右上
            (x1 + margin_x, y2 - margin_y),           # 左下
            (x2 - margin_x, y2 - margin_y),           # 右下
            # 四边中点
            ((x1 + x2) // 2, y1 + margin_y),          # 上中
            ((x1 + x2) // 2, y2 - margin_y),          # 下中
            (x1 + margin_x, (y1 + y2) // 2),          # 左中
            (x2 - margin_x, (y1 + y2) // 2),          # 右中
        ]
        
        # 过滤掉在 contour 内的点
        valid_points = []
        for px, py in candidates:
            px = max(0, min(self._image_w - 1, px))
            py = max(0, min(self._image_h - 1, py))
            if contour_mask[py, px] == 0:  # 不在 contour 内
                valid_points.append((px, py))
        
        # 如果有效点不够，就用所有候选点
        if len(valid_points) < num_points:
            valid_points = [(max(0, min(self._image_w-1, px)), 
                           max(0, min(self._image_h-1, py))) 
                          for px, py in candidates]
        
        return valid_points[:num_points]

    def _find_peak_points_in_contour(self, heatmap, contour, min_threshold, num_points=3):
        """在轮廓内部找到热力图的峰值点"""
        H, W = heatmap.shape
        
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        valid_region = (heatmap >= min_threshold) & (mask > 0)
        
        if not np.any(valid_region):
            # Fallback: 用轮廓中心
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return [(cx, cy)]
            x, y, w, h = cv2.boundingRect(contour)
            return [(x + w // 2, y + h // 2)]
        
        masked_heatmap = np.where(valid_region, heatmap, -np.inf)
        
        # 使用形态学找局部极大值
        kernel_size = max(5, min(H, W) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        from scipy import ndimage
        local_max = ndimage.maximum_filter(masked_heatmap, size=kernel_size)
        peaks = (masked_heatmap == local_max) & valid_region
        
        peak_coords = np.argwhere(peaks)
        if len(peak_coords) == 0:
            max_idx = np.unravel_index(np.argmax(masked_heatmap), masked_heatmap.shape)
            return [(int(max_idx[1]), int(max_idx[0]))]
        
        peak_values = [heatmap[py, px] for py, px in peak_coords]
        sorted_indices = np.argsort(peak_values)[::-1][:num_points]
        
        return [(int(peak_coords[idx][1]), int(peak_coords[idx][0])) for idx in sorted_indices]

    @torch.inference_mode()
    def predict_with_heatmap_guided(
        self,
        heatmap,                # 2D numpy array, 与图像同尺寸
        contour,                # cv2 轮廓 (热力图阈值化后的轮廓)
        pixel_threshold,        # 像素级阈值
        image_threshold=None,   # 图像级阈值（用于筛选峰值点）
        # 正样本配置
        num_positive_points=3,  # 正样本点数量
        box_padding=0.15,       # 框扩大比例
        positive_point_size=0.02,  # 正样本点框大小
        # 负样本配置
        use_negative_samples=True,  # 是否使用负样本
        num_negative_points=4,  # 负样本点数量
        negative_margin_ratio=0.12,  # 负样本采样的环形区域宽度
        negative_point_size=0.015,   # 负样本点框大小
        # SAM 配置
        threshold=0.4
    ):
        """
        [核心方法] 基于热力图智能引导 SAM3 分割
        
        工作流程：
        1. 从 contour 获取 bounding box，向外扩展 box_padding
        2. 在 contour 内找热力图峰值作为正样本点
        3. 在 bbox 内、contour 外采样负样本点
        4. 组合所有提示调用 SAM3
        
        Args:
            heatmap: 异常检测热力图
            contour: 热力图阈值化后的轮廓
            pixel_threshold: 像素级阈值
            image_threshold: 图像级阈值
            ... (其他参数见上方注释)
        """
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")
        
        if image_threshold is None:
            image_threshold = pixel_threshold
        
        # 重置之前的提示
        self._reset_prompts()
        
        # === Step 1: 获取 bounding box ===
        x, y, w, h = cv2.boundingRect(contour)
        
        # 扩展 bbox
        pad_w = int(w * box_padding)
        pad_h = int(h * box_padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(self._image_w, x + w + pad_w)
        y2 = min(self._image_h, y + h + pad_h)
        
        expanded_bbox = [x1, y1, x2, y2]
        
        # === Step 2: 找正样本点（热力图峰值）===
        positive_points = self._find_peak_points_in_contour(
            heatmap, contour,
            min_threshold=image_threshold,
            num_points=num_positive_points
        )
        
        # === Step 3: 采样负样本点 ===
        negative_points = []
        if use_negative_samples:
            negative_points = self._sample_negative_points_in_margin(
                contour=contour,
                bbox_xyxy=expanded_bbox,
                heatmap=heatmap,
                pixel_threshold=pixel_threshold,
                margin_ratio=negative_margin_ratio,
                num_points=num_negative_points
            )
        
        # === Step 4: 添加所有提示 ===
        # 4.1 主框（正样本）
        main_box = self._pixel_to_normalized_box(x1, y1, x2-x1, y2-y1, padding_ratio=0)
        output = self.processor.add_geometric_prompt(main_box, True, self.inference_state)
        
        # 4.2 正样本点
        for px, py in positive_points:
            point_box = self._point_to_small_box(px, py, box_size_ratio=positive_point_size)
            output = self.processor.add_geometric_prompt(point_box, True, self.inference_state)
        
        # 4.3 负样本点
        for px, py in negative_points:
            point_box = self._point_to_small_box(px, py, box_size_ratio=negative_point_size)
            output = self.processor.add_geometric_prompt(point_box, False, self.inference_state)
        
        return self._process_output(output, threshold)

    @torch.inference_mode()
    def predict_mask_with_boxes(self, boxes_norm, threshold=0.4):
        """[兼容方法] 使用归一化边界框列表进行预测"""
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")
        
        if not boxes_norm:
            return None, 0.0

        combined_mask_tensor = None
        max_score = 0.0

        for box in boxes_norm:
            try:
                self._reset_prompts()
                output = self.processor.add_geometric_prompt(box, True, self.inference_state)
                mask_np, score = self._process_output(output, threshold)
                
                if mask_np is not None:
                    mask_t = torch.from_numpy(mask_np).to(self.device)
                    if combined_mask_tensor is None:
                        combined_mask_tensor = mask_t
                    else:
                        combined_mask_tensor = torch.logical_or(combined_mask_tensor, mask_t)
                    max_score = max(max_score, score)
                        
            except Exception as e:
                print(f"  [SAM3 Engine] Box prompt failed: {e}")
                continue

        if combined_mask_tensor is None:
            return None, 0.0
            
        return combined_mask_tensor.cpu().numpy(), max_score

    def _process_output(self, output, threshold):
        """统一处理输出结果"""
        if isinstance(output, dict):
            if "masks" in output:
                masks = output["masks"]
                scores = output.get("scores", torch.ones(len(masks), device=self.device))
            elif "pred_masks" in output:
                masks = output["pred_masks"]
                scores = output.get("iou_scores", output.get("scores", torch.ones(len(masks), device=self.device)))
            else:
                return None, 0.0
        else:
            masks = output.pred_masks
            scores = getattr(output, 'iou_scores', getattr(output, 'scores', None))
            if scores is None:
                scores = torch.ones(len(masks), device=self.device)

        if torch.is_tensor(scores) and scores.ndim > 1:
            scores = scores.flatten()
            
        if len(masks) == 0:
            return None, 0.0
        
        valid_indices = torch.where(scores > threshold)[0]
        
        if len(valid_indices) == 0:
            best_idx = torch.argmax(scores)
            valid_indices = [best_idx]
        
        first_mask = masks[valid_indices[0]]
        if first_mask.ndim == 3:
            first_mask = first_mask.squeeze(0)
            
        combined_mask_tensor = torch.zeros_like(first_mask, dtype=torch.bool)
        max_score = 0.0
        
        for idx in valid_indices:
            current_score = scores[idx].item()
            if current_score > max_score:
                max_score = current_score
            
            current_mask = masks[idx]
            if current_mask.ndim == 3:
                current_mask = current_mask.squeeze(0)
            if current_mask.dtype != torch.bool:
                current_mask = current_mask > 0.0
            
            combined_mask_tensor = torch.logical_or(combined_mask_tensor, current_mask)

        mask_np = combined_mask_tensor.cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
            
        return mask_np, max_score

    def get_cutout_pil(self, mask_bool, padding=10):
        """获取裁剪后的图片（白色背景）"""
        if mask_bool is None or self.current_image_pil is None:
            return None

        image_np = np.array(self.current_image_pil)
        white_bg = np.full_like(image_np, 255)
        
        if mask_bool.ndim == 2:
            mask_3d = mask_bool[:, :, None]
        else:
            mask_3d = mask_bool
            
        result_np = np.where(mask_3d, image_np, white_bg)
        img_cutout = Image.fromarray(result_np.astype(np.uint8))
        
        coords = np.argwhere(mask_bool)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            w, h = img_cutout.size
            left = max(0, x_min - padding)
            top = max(0, y_min - padding)
            right = min(w, x_max + padding)
            bottom = min(h, y_max + padding)
            
            img_cutout = img_cutout.crop((left, top, right, bottom))
            
        return img_cutout

    def save_masked_cutout(self, mask_bool, save_path, padding=5):
        """保存裁剪后的图片"""
        img_cutout = self.get_cutout_pil(mask_bool, padding)
        if img_cutout:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img_cutout.save(save_path)
            print(f"    Saved: {os.path.basename(save_path)}")