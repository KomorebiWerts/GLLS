import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F

# 引入 SAM3 模块
# 保留原始导入
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class Sam3Engine:
    def __init__(self, checkpoint_path, device=None):
        """
        初始化 SAM3 模型
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Engine] Loading SAM3 model from: {checkpoint_path} to {self.device}")
        
        # [Fix] 尝试设置默认设备，防止部分内部操作走默认 cuda:0
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
        
        # 确保模型在正确的设备上并处于评估模式
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = Sam3Processor(self.model)
        print("[Engine] Model loaded successfully.")
        
        self.inference_state = None
        self.current_image_pil = None

    def set_image(self, image_path_or_pil):
        """
        设置当前处理的图片
        """
        if isinstance(image_path_or_pil, str):
            self.current_image_pil = Image.open(image_path_or_pil).convert("RGB")
        else:
            self.current_image_pil = image_path_or_pil
            
        # [Fix] 再次确保设备上下文
        if "cuda" in str(self.device) and ":" in str(self.device):
            try:
                device_idx = int(str(self.device).split(":")[-1])
                torch.cuda.set_device(device_idx)
            except Exception:
                pass

        with torch.inference_mode():
            self.inference_state = self.processor.set_image(self.current_image_pil)

        
    @torch.inference_mode()
    def predict_mask(self, prompt_text, threshold=0.4):
        """
        文本提示预测
        """
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")

        output = self.processor.set_text_prompt(state=self.inference_state, prompt=prompt_text)
        return self._process_output(output, threshold)
    
    @torch.inference_mode()
    def predict_mask_with_boxes(self, boxes_norm, threshold=0.4):
        """
        [New] 使用归一化边界框列表进行预测
        Args:
            boxes_norm: List of [cx, cy, w, h] (normalized 0-1)
        """
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")
        
        if not boxes_norm:
            return None, 0.0

        combined_mask_tensor = None
        max_score = 0.0

        # SAM3 支持一次性传入多个 Prompt，也可以循环调用
        # 为了稳健，我们这里对每个框分别调用并取并集
        for box in boxes_norm:
            try:
                output = self.processor.add_geometric_prompt(
                    box, 
                    True, # Label: Foreground 
                    self.inference_state
                )
                
                # 处理输出
                mask_np, score = self._process_output(output, threshold)
                
                if mask_np is not None:
                    # 转换为 Tensor 以便在 GPU 上做逻辑运算 (避免频繁 CPU/GPU 切换)
                    mask_t = torch.from_numpy(mask_np).to(self.device)
                    
                    if combined_mask_tensor is None:
                        combined_mask_tensor = mask_t
                    else:
                        combined_mask_tensor = torch.logical_or(combined_mask_tensor, mask_t)
                    
                    if score > max_score:
                        max_score = score
                        
            except Exception as e:
                print(f"  [SAM3 Engine] Box prompt failed: {e}")
                continue

        if combined_mask_tensor is None:
            return None, 0.0
            
        return combined_mask_tensor.cpu().numpy(), max_score
    @torch.inference_mode()
    def predict_mask_with_points(self, points, labels, threshold=0.4):
        """
        点提示预测
        """
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")

        # 尝试使用 add_geometric_prompt (处理为微小框)
        if hasattr(self.processor, "add_geometric_prompt"):
            W, H = self.current_image_pil.size
            pt = points[0] 
            box_size_w = W * 0.05
            box_size_h = H * 0.05
            box_prompt = [pt[0]/W, pt[1]/H, box_size_w/W, box_size_h/H]
            
            output = self.processor.add_geometric_prompt(
                box_prompt,
                True,
                self.inference_state
            )
            return self._process_output(output, threshold)

        # 兼容 set_point_prompt
        points_tensor = torch.tensor([points], dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor([labels], dtype=torch.int64, device=self.device)
        
        if hasattr(self.processor, "set_point_prompt"):
            output = self.processor.set_point_prompt(
                state=self.inference_state, 
                point_coords=points_tensor, 
                point_labels=labels_tensor
            )
        elif hasattr(self.processor, "set_click_prompt"):
             output = self.processor.set_click_prompt(
                state=self.inference_state, 
                point_coords=points_tensor, 
                point_labels=labels_tensor
            )
        elif hasattr(self.processor, "predict"):
            output = self.processor.predict(
                state=self.inference_state, 
                point_coords=points_tensor, 
                point_labels=labels_tensor
            )
        else:
            raise AttributeError("Sam3Processor missing point prompting method.")
            
        return self._process_output(output, threshold)

    def _process_output(self, output, threshold):
        """
        统一处理输出结果
        """
        # 解析不同API版本的返回格式
        if isinstance(output, dict):
            if "masks" in output:
                masks = output["masks"]
                scores = output["scores"] if "scores" in output else torch.ones(len(masks), device=self.device)
            elif "pred_masks" in output: 
                masks = output["pred_masks"]
                scores = output["iou_scores"]
            else:
                return None, 0.0
        else:
            masks = output.pred_masks
            scores = output.iou_scores

        if torch.is_tensor(scores) and scores.ndim > 1:
            scores = scores.flatten()
            
        if len(masks) == 0:
            return None, 0.0
        
        # 找到有效索引
        valid_indices = torch.where(scores > threshold)[0]
        
        if len(valid_indices) == 0:
            best_idx = torch.argmax(scores)
            valid_indices = [best_idx]
        
        combined_mask_tensor = torch.zeros_like(masks[0], dtype=torch.bool)
        max_score = 0.0
        
        for idx in valid_indices:
            current_score = scores[idx].item()
            if current_score > max_score:
                max_score = current_score
            
            current_mask = masks[idx]
            if current_mask.dtype != torch.bool:
                current_mask = current_mask > 0.0
            
            combined_mask_tensor = torch.logical_or(combined_mask_tensor, current_mask)

        mask_np = combined_mask_tensor.cpu().numpy()
        
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
            
        return mask_np, max_score

    def get_cutout_pil(self, mask_bool, padding=10):
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
        img_cutout = self.get_cutout_pil(mask_bool, padding)
        if img_cutout:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img_cutout.save(save_path)
            print(f"    Saved: {os.path.basename(save_path)}")