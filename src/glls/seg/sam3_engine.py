import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F

# Keep the upstream SAM3 imports explicit.
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class Sam3Engine:
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize the SAM3 model.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Engine] Loading SAM3 model from: {checkpoint_path} to {self.device}")
        
        # Keep internal CUDA operations on the requested device.
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
        
        # Ensure the model is on the selected device and in eval mode.
        self.model.to(self.device)
        self._align_fused_mlp_dtypes()
        self.model.eval()
        
        self.processor = Sam3Processor(self.model)
        print("[Engine] Model loaded successfully.")
        
        self.inference_state = None
        self.current_image_pil = None

    def _align_fused_mlp_dtypes(self):
        """
        SAM3's ViT MLP uses a fused addmm+activation kernel that emits bf16 on
        this environment. The following fc2 layers must match that activation
        dtype, otherwise first image forward fails with bf16/float32 mismatch.
        """
        if "cuda" not in str(self.device):
            return
        converted = 0
        for name, module in self.model.named_modules():
            if name.endswith("mlp.fc2"):
                module.to(dtype=torch.bfloat16)
                converted += 1
        if converted:
            print(f"[Engine] Aligned {converted} fused MLP projection layer(s) to bfloat16.")

    def set_image(self, image_path_or_pil):
        """
        Set the image used by subsequent prompt calls.
        """
        if isinstance(image_path_or_pil, str):
            self.current_image_pil = Image.open(image_path_or_pil).convert("RGB")
        else:
            self.current_image_pil = image_path_or_pil
            
        # Re-assert the CUDA context before SAM3 image preprocessing.
        if "cuda" in str(self.device) and ":" in str(self.device):
            try:
                device_idx = int(str(self.device).split(":")[-1])
                torch.cuda.set_device(device_idx)
            except Exception:
                pass

        with torch.inference_mode():
            self.inference_state = self.processor.set_image(self.current_image_pil)

    def _reset_prompts(self):
        if self.inference_state is None:
            return
        reset_fn = getattr(self.processor, "reset_all_prompts", None)
        if reset_fn is not None:
            reset_fn(self.inference_state)
        
    @torch.inference_mode()
    def predict_mask(self, prompt_text, threshold=0.4):
        """
        Predict a mask from a text prompt.
        """
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")

        self._reset_prompts()
        output = self.processor.set_text_prompt(state=self.inference_state, prompt=prompt_text)
        return self._process_output(output, threshold)
    
    @torch.inference_mode()
    def predict_mask_with_boxes(self, boxes_norm, threshold=0.4):
        """
        Predict masks from normalized bounding boxes.
        Args:
            boxes_norm: List of [cx, cy, w, h] (normalized 0-1)
        """
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")
        
        if not boxes_norm:
            return None, 0.0

        combined_mask_tensor = None
        max_score = 0.0

        # Call SAM3 once per box and merge masks for version-tolerant behavior.
        for box in boxes_norm:
            try:
                self._reset_prompts()
                output = self.processor.add_geometric_prompt(
                    box, 
                    True, # Label: Foreground 
                    self.inference_state
                )
                
                mask_np, score = self._process_output(output, threshold)
                
                if mask_np is not None:
                    # Keep mask union on-device to avoid repeated CPU/GPU transfers.
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
        Predict a mask from point prompts.
        """
        if self.inference_state is None:
            raise RuntimeError("Please call set_image() before predicting.")

        self._reset_prompts()

        # Prefer add_geometric_prompt by converting the point to a small box.
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

        # Fall back to point-prompt APIs used by older processor versions.
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
        Normalize SAM3 output variants into a single mask and score.
        """
        # Parse return formats across SAM3 API versions.
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
        
        # Keep all masks above threshold, or the best mask when none pass.
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
