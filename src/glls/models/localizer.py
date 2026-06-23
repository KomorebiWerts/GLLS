import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from .AdaptCLIP import adaptcliplib
from .AdaptCLIP import PQAdapter, TextualAdapter, VisualAdapter, fusion_fun
from .AdaptCLIP import get_transform as get_transform_adaptclip
from .heatmap_regions import HeatmapThresholds, adaptive_thresholds


def _load_abound_dependencies():
    try:
        from torch.distributions.multivariate_normal import MultivariateNormal
        from peft import LoraConfig, get_peft_model, TaskType
        import open_clip
        from .ABounD import VVCLIP_lib
        from .ABounD.prompt_generator import DynamicConceptFusion
        from .ABounD.utils import get_transform, generate_class_info
    except ImportError as exc:
        raise RuntimeError(
            "ABounD support is optional and is not included in the public "
            "AdaptCLIP release. Use --localizer adaptclip, or install the "
            "private ABounD package/checkpoints locally."
        ) from exc
    return {
        "MultivariateNormal": MultivariateNormal,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "TaskType": TaskType,
        "open_clip": open_clip,
        "VVCLIP_lib": VVCLIP_lib,
        "DynamicConceptFusion": DynamicConceptFusion,
        "get_transform": get_transform,
        "generate_class_info": generate_class_info,
    }


def _float_table(value):
    if not isinstance(value, dict):
        return {}
    table = {}
    for key, item in value.items():
        try:
            table[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return table


def _read_model_config(config_path):
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"⚠️ Warning: Failed to read model config at {path}: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _adaptclip_model_config(checkpoint_path):
    path = Path(str(checkpoint_path or ""))
    candidates = []
    if path.is_file():
        if path.parent.name == "checkpoints":
            candidates.append(path.parent.parent / "model_config.json")
        candidates.append(path.parent / "model_config.json")
    elif str(checkpoint_path or ""):
        candidates.append(path / "model_config.json")
        candidates.append(path.parent / "model_config.json")
    for candidate in candidates:
        config = _read_model_config(candidate)
        if config:
            return config
    return {}


class ABounD_Localizer():
    # === 1. 定义不同数据集的专属配置 ===
    _CONFIGS = {
        'mvtec': {
            'depth': 7,
            'n_ctx': 11,
            'spe': 4,
            'weights': [0.15, 0.35, 0.35, 0.15], # w0, w1, w2, w3
            'features_list': [6, 12, 18, 24],
            'num_visual_finetune_layers': 12
        },
        'visa': {
            'depth': 7,      # Visa 配置
            'n_ctx': 16,     # Visa 上下文通常更长
            'spe': 5,        # Visa spe 设置
            'weights': [0.15, 0.35, 0.35, 0.15], # 假设 Visa 权重
            'features_list': [6, 12, 18, 24],
            'num_visual_finetune_layers': 8  # Visa 可能 finetune 层数不同
        }
    }

    # === ABounD 专属阈值 (原 mcts_sam.py 中的数据，保持不变) ===
    # 图像级阈值 (Optimal Image Threshold)
    _IMAGE_THRESHOLDS = {
        # --- MVTec ---
        "bottle": 0.9793, "cable": 0.9399, "capsule": 0.8669, "carpet": 0.9738,
        "grid": 0.8922, "hazelnut": 0.8891, "leather": 1.1001, "metal_nut": 0.9746,
        "pill": 0.9254, "screw": 0.7795, "tile": 0.9947, "toothbrush": 0.9795,
        "transistor": 0.8117, "wood": 1.0526, "zipper": 0.7919,
        # --- VisA ---
        "candle": 0.9148, "cashew": 0.8979, "capsules": 0.90997, "chewinggum": 0.8573,
        "fryum": 0.82, "macaroni1": 0.8253, "macaroni2": 0.908, "pcb1": 0.9222,
        "pcb2": 0.9289, "pcb3": 0.9328, "pcb4": 0.8864, "pipe_fryum": 0.8505 
    }

    # 像素级阈值 (Best Pixel-F1 Threshold)
    _PIXEL_THRESHOLDS = {
        # --- MVTec ---
        "bottle": 0.8423, "cable": 0.8221, "capsule": 0.7557, "carpet": 1.0006,
        "grid": 0.9747, "hazelnut": 0.8070, "leather": 1.09, "metal_nut": 0.8439,
        "pill": 0.7117, "screw": 0.7, "tile": 0.8250, "toothbrush": 0.8754,
        "transistor": 0.6279, "wood": 0.92, "zipper": 0.6,
        # --- VisA ---
        "candle": 0.82, "cashew": 0.7635, "capsules": 0.8175, "chewinggum": 0.77,
        "fryum": 0.6274, "macaroni1": 0.74, "macaroni2": 0.9393, "pcb1": 0.8464,
        "pcb2": 0.8813, "pcb3": 0.78, "pcb4": 0.7571, "pipe_fryum": 0.7478
    }

    def __init__(self, args, device_id=0, device=None):
        self.args = args
        self._image_thresholds = dict(self._IMAGE_THRESHOLDS)
        self._pixel_thresholds = dict(self._PIXEL_THRESHOLDS)
        model_config = _read_model_config(Path(str(getattr(args, "save_path", ""))) / "model_config.json")
        thresholds = model_config.get("thresholds", {}) if isinstance(model_config, dict) else {}
        config_image_thresholds = _float_table(thresholds.get("image"))
        config_pixel_thresholds = _float_table(thresholds.get("pixel"))
        if config_image_thresholds:
            self._image_thresholds.update(config_image_thresholds)
        if config_pixel_thresholds:
            self._pixel_thresholds.update(config_pixel_thresholds)

        deps = _load_abound_dependencies()
        self._MultivariateNormal = deps["MultivariateNormal"]
        self._VVCLIP_lib = deps["VVCLIP_lib"]
        self._open_clip = deps["open_clip"]
        self._DynamicConceptFusion = deps["DynamicConceptFusion"]
        self._get_transform = deps["get_transform"]
        self._generate_class_info = deps["generate_class_info"]
        self._LoraConfig = deps["LoraConfig"]
        self._get_peft_model = deps["get_peft_model"]
        self._TaskType = deps["TaskType"]
        
        # [Device Setup]
        if device is not None:
            self.device = device
        else:
            self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        # === 2. 加载配置并覆盖 args 中的默认值 ===
        dataset_name = args.dataset.lower()
        if dataset_name not in self._CONFIGS:
            print(f"⚠️ Warning: Dataset '{dataset_name}' not in config. Defaulting to 'mvtec'.")
            self.cfg = self._CONFIGS['mvtec']
        else:
            self.cfg = self._CONFIGS[dataset_name]
        
        # 将配置参数绑定到 self
        self.depth = self.cfg['depth']
        self.n_ctx = self.cfg['n_ctx']
        self.spe = self.cfg['spe']
        self.w0, self.w1, self.w2, self.w3 = self.cfg['weights']
        self.features_list = self.cfg['features_list']
        self.num_visual_finetune_layers = self.cfg['num_visual_finetune_layers']
        self.image_size = args.image_size

        # === 3. Backbone checkpoint path ===
        self.checkpoint_full_path = os.path.join(args.checkpoint_path, "ViT-L-14-336px.pt")
        
        print(f"[{dataset_name.upper()}] Localizer Configured:")
        print(f"  -> Params: depth={self.depth}, ctx={self.n_ctx}, spe={self.spe}")
        print(f"  -> Loading Backbone: {self.checkpoint_full_path}")

        # === 4. 加载模型 ===
        VVCLIP_parameters = {
            "Prompt_length": self.n_ctx, 
            "learnabel_text_embedding_depth": self.depth,
            "learnabel_text_embedding_length": self.n_ctx
        }
        
        try:
            self.model, _ = self._VVCLIP_lib.load(
                "ViT-L/14@336px", 
                device=self.device, 
                design_details=VVCLIP_parameters,
                download_root=args.checkpoint_path 
            )
            self.tokenizer = self._open_clip.get_tokenizer("ViT-L-14")
            self.model.eval()
            self.model.to(self.device)

            if hasattr(self.model.visual, 'DAPM_replace'):
                self.model.visual.DAPM_replace(DPAM_layer=20)
        except Exception as e:
            print(f"❌ Error loading VVCLIP model: {e}")
            self.model = None

        if self.model and self.num_visual_finetune_layers > 0:
            target_modules_list = []
            total_blocks = len(self.model.visual.transformer.resblocks)
            for i in range(total_blocks - self.num_visual_finetune_layers, total_blocks):
                target_modules_list.extend([
                    f"visual.transformer.resblocks.{i}.attn.qkv",
                    f"visual.transformer.resblocks.{i}.attn.proj"
                ])
            
            lora_config = self._LoraConfig(
                r=4, lora_alpha=8, target_modules=target_modules_list,
                lora_dropout=0.25, bias="none", task_type=self._TaskType.FEATURE_EXTRACTION,
            )
            self.model = self._get_peft_model(self.model, lora_config)
            
            lora_path = os.path.join(args.save_path, args.dataset, f"final_vvclip_model_state_{args.dataset}.pth")
            if os.path.exists(lora_path):
                state_dict = torch.load(lora_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                print(f"⚠️ Warning: LoRA checkpoint not found at {lora_path}")

        embed_dim = 768
        self.dcf_module = self._DynamicConceptFusion(
            T_len_shared=self.n_ctx - self.spe, 
            T_len_specific=self.spe,
            embed_dim=embed_dim,
            depth=self.depth,
            text_embed_dim=embed_dim
        ).to(self.device)
        
        obj_list_full, _ = self._generate_class_info(args.dataset)
        if args.dataset in ['mvtec', 'visa']:
            obj_list_full.sort()

        for class_name in obj_list_full:
            self.dcf_module.add_class_prompts(class_name)

        dcf_path = os.path.join(args.save_path, args.dataset, f"final_soft_prompt_state_{args.dataset}.pth")
        if os.path.exists(dcf_path):
            self.dcf_module.load_state_dict(torch.load(dcf_path, map_location=self.device))
        else:
            print(f"⚠️ Warning: DCF module checkpoint not found at {dcf_path}")
        
        self.dcf_module.eval()

        self.obj_means_map_global = {}
        self.obj_covariances_map_global = {}
        self.visual_patch_bank_layer0_per_obj_global = {}
        self.visual_patch_bank_layer1_per_obj_global = {}
        self.visual_patch_bank_layer2_per_obj_global = {}
        self.visual_patch_bank_layer3_per_obj_global = {}
        self.text_prompts_pos_mem_bank_per_obj_global = {}
        self.text_prompts_neg_mem_bank_per_obj_global = {}

        mem_path = os.path.join(args.save_path, args.dataset, f"final_memory_bank_{args.dataset}.pt")
        if os.path.exists(mem_path):
            print(f"📥 Loading Memory Banks from: {mem_path}")
            try:
                mem_data = torch.load(mem_path, map_location=self.device)
                self.obj_means_map_global = mem_data.get('obj_means_map_global', {})
                self.obj_covariances_map_global = mem_data.get('obj_covariances_map_global', {})
                self.visual_patch_bank_layer0_per_obj_global = mem_data.get('visual_patch_bank_layer0_per_obj_global', {})
                self.visual_patch_bank_layer1_per_obj_global = mem_data.get('visual_patch_bank_layer1_per_obj_global', {})
                self.visual_patch_bank_layer2_per_obj_global = mem_data.get('visual_patch_bank_layer2_per_obj_global', {})
                self.visual_patch_bank_layer3_per_obj_global = mem_data.get('visual_patch_bank_layer3_per_obj_global', {})
                self.text_prompts_pos_mem_bank_per_obj_global = mem_data.get('text_prompts_pos_mem_bank_per_obj_global', {})
                self.text_prompts_neg_mem_bank_per_obj_global = mem_data.get('text_prompts_neg_mem_bank_per_obj_global', {})
            except Exception as e:
                print(f"❌ Error reading Memory Bank: {e}")
        else:
             print(f"⚠️ Warning: Memory bank not found at {mem_path}")

        prefix_texts = {
            "normal": ["normal ", "flawless ", "perfect "],
            "abnormal": ["defect ", "damaged ", "abnormal "]
        }
        self.prefix_embeddings = {"normal": [], "abnormal": []}
        if self.model:
            for state, texts_list in prefix_texts.items():
                for text in texts_list:
                    tokens = self.tokenizer([text]).to(self.device)
                    with torch.no_grad():
                        token_emb = self.model.token_embedding(tokens[0, 1:len(text.split()) + 1]).to(self.model.dtype)
                        self.prefix_embeddings[state].append(token_emb.unsqueeze(0))

            self.full_template = self.tokenizer(["normal " + "X " * (self.n_ctx + 2)]).to(self.device)
        self.preprocess, _ = self._get_transform(args)
    
    def get_threshold_config(self, category, heatmap=None):
        """
        Return traceable threshold configuration.

        Calibrated per-category thresholds are preserved exactly. The adaptive
        branch is only used for unknown categories, where the old behavior was a
        fixed 0.9/0.9 default.
        """
        image_thresholds = getattr(self, "_image_thresholds", self._IMAGE_THRESHOLDS)
        pixel_thresholds = getattr(self, "_pixel_thresholds", self._PIXEL_THRESHOLDS)
        if category in image_thresholds or category in pixel_thresholds:
            return HeatmapThresholds(
                image=float(image_thresholds.get(category, 0.9)),
                pixel=float(pixel_thresholds.get(category, 0.9)),
                source="abound:calibrated",
            )
        if heatmap is not None:
            return adaptive_thresholds(
                heatmap,
                default_image=0.9,
                default_pixel=0.9,
                source="abound:adaptive_unknown_category",
            )
        return HeatmapThresholds(image=0.9, pixel=0.9, source="abound:default_unknown_category")

    def get_thresholds(self, category):
        cfg = self.get_threshold_config(category)
        return cfg.image, cfg.pixel

    def predict_anomaly_map(self, image):
        if not self.model:
            print("⚠️ Localizer model not initialized. Returning empty map.")
            return np.zeros((self.image_size, self.image_size)), "unknown"
            
        if not self.obj_means_map_global:
            print("⚠️ Memory banks not loaded. Cannot perform localization. Returning empty map.")
            return np.zeros((self.image_size, self.image_size)), "unknown"

        with torch.no_grad():
            image_pil = image.convert("RGB")
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)

            global_feat, patch_features_list, patch_tokens_list = self.model.encode_image(
                image_tensor, self.features_list, DPAM_layer=20, ffn=False)
            global_feat = global_feat.mean(dim=1) if global_feat.ndim == 3 else global_feat
            global_feat_norm = F.normalize(global_feat, dim=-1)

            log_probs = []
            for obj_, mean in self.obj_means_map_global.items():
                cov = self.obj_covariances_map_global[obj_]
                lp = self._MultivariateNormal(mean.to(self.device), covariance_matrix=cov.to(self.device)).log_prob(global_feat_norm.squeeze())
                log_probs.append(lp)
            
            best_obj_name = list(self.obj_means_map_global.keys())[int(torch.stack(log_probs).argmax())]
            
            pos_p_inst, neg_p_inst, deep_txt_pos_test, deep_txt_neg_test = self.dcf_module(
                global_feat_norm, class_name=best_obj_name)

            text_pos_avg = self.text_prompts_pos_mem_bank_per_obj_global.get(best_obj_name)
            text_neg_avg = self.text_prompts_neg_mem_bank_per_obj_global.get(best_obj_name)
            vis0 = self.visual_patch_bank_layer0_per_obj_global.get(best_obj_name)
            vis1 = self.visual_patch_bank_layer1_per_obj_global.get(best_obj_name)
            vis2 = self.visual_patch_bank_layer2_per_obj_global.get(best_obj_name)
            vis3 = self.visual_patch_bank_layer3_per_obj_global.get(best_obj_name)
            
            retrieved_text_pos = text_pos_avg.mean(dim=0).to(self.device) if text_pos_avg is not None and text_pos_avg.numel() > 0 else None
            retrieved_text_neg = text_neg_avg.mean(dim=0).to(self.device) if text_neg_avg is not None and text_neg_avg.numel() > 0 else None

            pos_prefix_test = self.prefix_embeddings["normal"][0].expand(pos_p_inst.size(0), -1, -1).to(pos_p_inst.dtype)
            neg_prefix_test = self.prefix_embeddings["abnormal"][0].expand(neg_p_inst.size(0), -1, -1).to(neg_p_inst.dtype)
            
            pos_e_inst, _ = self.model.encode_text_with_prefix(
                pos_prefix_test, pos_p_inst, self.full_template, deep_compound_prompts_text=deep_txt_pos_test)
            neg_e_inst, _ = self.model.encode_text_with_prefix(
                neg_prefix_test, neg_p_inst, self.full_template, deep_compound_prompts_text=deep_txt_neg_test)
            
            pos_e_norm = F.normalize(pos_e_inst.squeeze(), dim=-1)
            neg_e_norm = F.normalize(neg_e_inst.squeeze(), dim=-1)

            final_pos = F.normalize((pos_e_norm + retrieved_text_pos) / 2 if retrieved_text_pos is not None else pos_e_norm, dim=-1)
            final_neg = F.normalize((neg_e_norm + retrieved_text_neg) / 2 if retrieved_text_neg is not None else neg_e_norm, dim=-1)
            text_feats = torch.stack([final_pos, final_neg], dim=1)

            num_patches_side = int((patch_features_list[0].size(1) - 1) ** 0.5)
            patch_norms = [F.normalize(p.squeeze(0)[1:], dim=-1) for p in patch_features_list]
            all_patches = torch.cat(patch_norms, dim=0)
            
            sim, _ = self._VVCLIP_lib.compute_similarity(all_patches.unsqueeze(0), text_feats.T)
            sim = sim.view(len(patch_norms), -1, 2)
            
            exp_n, exp_p = torch.exp(sim[..., 1]), torch.exp(sim[..., 0])
            text_map_comb = (exp_n / (exp_n + exp_p + 1e-8)).mean(0)
            text_map_comb = text_map_comb.view(1, 1, num_patches_side, num_patches_side)

            vis0_flat = vis0.reshape(-1, vis0.size(-1)).to(self.device)
            vis1_flat = vis1.reshape(-1, vis1.size(-1)).to(self.device)
            vis2_flat = vis2.reshape(-1, vis2.size(-1)).to(self.device)
            vis3_flat = vis3.reshape(-1, vis3.size(-1)).to(self.device)
            
            p0_norm = F.normalize(patch_tokens_list[0].squeeze(0)[1:], dim=-1)
            p1_norm = F.normalize(patch_tokens_list[1].squeeze(0)[1:], dim=-1)
            p2_norm = F.normalize(patch_tokens_list[2].squeeze(0)[1:], dim=-1)
            p3_norm = F.normalize(patch_tokens_list[3].squeeze(0)[1:], dim=-1)

            score0, _ = (1.0 - p0_norm @ vis0_flat.t()).min(dim=-1)
            score1, _ = (1.0 - p1_norm @ vis1_flat.t()).min(dim=-1)
            score2, _ = (1.0 - p2_norm @ vis2_flat.t()).min(dim=-1)
            score3, _ = (1.0 - p3_norm @ vis3_flat.t()).min(dim=-1)

            vis_map = ((self.w0 * score0 + self.w1 * score1 + self.w2 * score2 + self.w3 * score3) / 
                        (self.w0 + self.w1 + self.w2 + self.w3)).view(1, 1, num_patches_side, num_patches_side)

            final_map = text_map_comb + vis_map
            final_map_resized = F.interpolate(final_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            anomaly_map = final_map_resized.squeeze().cpu().numpy()
        
        return anomaly_map, best_obj_name

class AdaptCLIP_Localizer():
    """
    AdaptCLIP Zero-Shot Localizer (k_shots=0 mode).
    Strictly follows the logic in test.py for the no-memory branch.
    """
    _CONFIGS = {
        'default': {
            'image_size': 518,
            'features_list': [6, 12, 18, 24],
            'n_ctx': 12,
            'vl_reduction': 4,
            'sigma': 4,       
            'fusion_type': 'average_mean' 
        }
    }
    
    # AdaptCLIP thresholds calibrated on MMAD test data.
    # Image thresholds use image best-F1; pixel thresholds use anomaly-mask pixel best-F1.
    _IMAGE_THRESHOLDS_BY_SHOT = {
        0: {
            # --- VisA ---
            "candle": 0.492197,
            "capsules": 0.468662,
            "cashew": 0.462056,
            "chewinggum": 0.243132,
            "fryum": 0.294960,
            "macaroni1": 0.450678,
            "macaroni2": 0.432695,
            "pcb1": 0.418049,
            "pcb2": 0.324668,
            "pcb3": 0.399457,
            "pcb4": 0.401730,
            "pipe_fryum": 0.415181,
            # --- MVTec ---
            "bottle": 0.521602,
            "cable": 0.633561,
            "capsule": 0.662371,
            "carpet": 0.749844,
            "grid": 0.566117,
            "hazelnut": 0.643178,
            "leather": 0.793466,
            "metal_nut": 0.826857,
            "pill": 0.672302,
            "screw": 0.584296,
            "tile": 0.629624,
            "toothbrush": 0.642064,
            "transistor": 0.695740,
            "wood": 0.659023,
            "zipper": 0.601005,
        },
        1: {
            # --- VisA ---
            "candle": 0.218004,
            "capsules": 0.217441,
            "cashew": 0.187064,
            "chewinggum": 0.168488,
            "fryum": 0.176613,
            "macaroni1": 0.205907,
            "macaroni2": 0.203645,
            "pcb1": 0.238018,
            "pcb2": 0.193589,
            "pcb3": 0.220342,
            "pcb4": 0.218458,
            "pipe_fryum": 0.270058,
            # --- MVTec ---
            "bottle": 0.240687,
            "cable": 0.289088,
            "capsule": 0.193763,
            "carpet": 0.274225,
            "grid": 0.266388,
            "hazelnut": 0.357254,
            "leather": 0.356350,
            "metal_nut": 0.312918,
            "pill": 0.232602,
            "screw": 0.220036,
            "tile": 0.295386,
            "toothbrush": 0.287488,
            "transistor": 0.273907,
            "wood": 0.302756,
            "zipper": 0.177794,
        },
    }

    _PIXEL_THRESHOLDS_BY_SHOT = {
        0: {
            # --- VisA ---
            "candle": 0.438111,
            "capsules": 0.429234,
            "cashew": 0.333404,
            "chewinggum": 0.588078,
            "fryum": 0.283982,
            "macaroni1": 0.408517,
            "macaroni2": 0.476349,
            "pcb1": 0.272632,
            "pcb2": 0.347578,
            "pcb3": 0.260592,
            "pcb4": 0.337110,
            "pipe_fryum": 0.315041,
            # --- MVTec ---
            "bottle": 0.385266,
            "cable": 0.412920,
            "capsule": 0.422589,
            "carpet": 0.456064,
            "grid": 0.540987,
            "hazelnut": 0.470359,
            "leather": 0.634212,
            "metal_nut": 0.429602,
            "pill": 0.357995,
            "screw": 0.353140,
            "tile": 0.419157,
            "toothbrush": 0.369372,
            "transistor": 0.347801,
            "wood": 0.600865,
            "zipper": 0.223319,
        },
        1: {
            # --- VisA ---
            "candle": 0.147633,
            "capsules": 0.123421,
            "cashew": 0.117580,
            "chewinggum": 0.216259,
            "fryum": 0.101908,
            "macaroni1": 0.178247,
            "macaroni2": 0.166722,
            "pcb1": 0.163179,
            "pcb2": 0.144446,
            "pcb3": 0.133241,
            "pcb4": 0.131448,
            "pipe_fryum": 0.141771,
            # --- MVTec ---
            "bottle": 0.140620,
            "cable": 0.153849,
            "capsule": 0.110823,
            "carpet": 0.149830,
            "grid": 0.200830,
            "hazelnut": 0.198961,
            "leather": 0.227905,
            "metal_nut": 0.167134,
            "pill": 0.147910,
            "screw": 0.150780,
            "tile": 0.138218,
            "toothbrush": 0.180899,
            "transistor": 0.116187,
            "wood": 0.243519,
            "zipper": 0.091548,
        },
    }

    def __init__(self, args, device=None, pretrained_model='ViT-L/14@336px'):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = self._CONFIGS['default']
        self.k_shot = int(getattr(args, "k_shot", 0) or 0)
        self.active_category = None
        self.prompt_image_memory = {}
        self.prompt_patch_memory = {}
        self.last_image_score = None
        self._image_thresholds_by_shot = {
            int(shot): dict(table) for shot, table in self._IMAGE_THRESHOLDS_BY_SHOT.items()
        }
        self._pixel_thresholds_by_shot = {
            int(shot): dict(table) for shot, table in self._PIXEL_THRESHOLDS_BY_SHOT.items()
        }
        
        # Override default config with args if present
        if hasattr(args, 'image_size'):
            self.cfg['image_size'] = args.image_size
            
        checkpoint_path = getattr(args, 'checkpoint_path', None)
        model_config = _adaptclip_model_config(checkpoint_path)
        thresholds_by_shot = model_config.get("thresholds_by_shot", {}) if isinstance(model_config, dict) else {}
        if isinstance(thresholds_by_shot, dict):
            for shot_text, threshold_group in thresholds_by_shot.items():
                try:
                    shot = int(str(shot_text).replace("-shot", ""))
                except ValueError:
                    continue
                if not isinstance(threshold_group, dict):
                    continue
                image_table = _float_table(threshold_group.get("image"))
                pixel_table = _float_table(threshold_group.get("pixel"))
                if image_table:
                    self._image_thresholds_by_shot.setdefault(shot, {}).update(image_table)
                if pixel_table:
                    self._pixel_thresholds_by_shot.setdefault(shot, {}).update(pixel_table)
        
        print(f"[AdaptCLIP] Initializing Zero-Shot Localizer...")
        
        # 1. Load Backbone
        self.model, _ = adaptcliplib.load(pretrained_model, device=self.device)
        
        if '336px' in pretrained_model:
            self.DPAM_layer = 20
            self.patch_size = 14
            self.input_dim = 768
        else:
            self.DPAM_layer = 10
            self.patch_size = 16
            self.input_dim = 640 

        # 2. Apply DAPM (Core of AdaptCLIP)
        self.model.visual.DAPM_replace(DPAM_layer=self.DPAM_layer)

        # 3. Initialize Adapters
        self.textual_learner = TextualAdapter(self.model.to("cpu"), self.cfg['image_size'], self.cfg['n_ctx'])
        self.visual_learner = VisualAdapter(self.cfg['image_size'], self.patch_size, input_dim=self.input_dim, reduction=self.cfg['vl_reduction'])
        self.pq_learner = PQAdapter(
            self.cfg['image_size'],
            self.patch_size,
            context=True,
            input_dim=self.input_dim,
            mid_dim=128,
            layers_num=len(self.cfg['features_list']),
        )

        # 4. Load Trained Weights (Crucial)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[AdaptCLIP] Loading Adapter weights from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.textual_learner.load_state_dict(ckpt["textual_learner"], strict=False)
            self.visual_learner.load_state_dict(ckpt["visual_learner"], strict=False)
            if "pq_learner" in ckpt:
                self.pq_learner.load_state_dict(ckpt["pq_learner"], strict=False)
        else:
            print("⚠️ [AdaptCLIP] WARNING: No checkpoint found! Model will output random noise.")

        # 5. Model to Device & Eval
        self.model.to(self.device).eval()
        self.textual_learner.to(self.device).eval()
        self.visual_learner.to(self.device).eval()
        self.pq_learner.to(self.device).eval()

        # 6. Pre-compute Text Features (Optimization)
        print("[AdaptCLIP] Pre-computing static text embeddings...")
        self.textual_learner.prepare_static_text_feature(self.model)
        
        with torch.no_grad():
            learned_prompts, tokenized_prompts = self.textual_learner()
            self.learned_text_features = self.model.encode_text_learn(learned_prompts, tokenized_prompts).float()

        # 7. Setup Transform
        self.transform, _ = get_transform_adaptclip(image_size=self.cfg['image_size'])

    def configure_support(self, category, train_image_paths):
        self.active_category = category
        if self.k_shot <= 0:
            return
        selected = [str(path) for path in train_image_paths[:self.k_shot]]
        cache_key = (category, tuple(selected))
        if cache_key in self.prompt_image_memory:
            return
        image_feats = []
        patch_feats_by_layer = [[] for _ in self.cfg['features_list']]
        with torch.no_grad():
            for image_path in selected:
                with Image.open(image_path) as image:
                    img_tensor = self.preprocess(image.convert("RGB"))
                image_feat, patch_feats = self.model.encode_image(
                    img_tensor,
                    self.cfg['features_list'],
                    DPAM_layer=self.DPAM_layer,
                )
                image_feats.append(image_feat.squeeze(0).detach())
                for idx, patch_feat in enumerate(patch_feats):
                    patch_feats_by_layer[idx].append(patch_feat.squeeze(0).detach())
        if image_feats:
            self.prompt_image_memory[cache_key] = torch.stack(image_feats, dim=0).to(self.device)
            self.prompt_patch_memory[cache_key] = [
                torch.stack(layer_feats, dim=0).to(self.device)
                for layer_feats in patch_feats_by_layer
            ]

    def preprocess(self, image_pil):
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        return self.transform(image_pil).unsqueeze(0).to(self.device)

    def get_threshold_config(self, category, heatmap=None):
        """
        Retrieves traceable thresholds.

        AdaptCLIP uses shot-specific calibrated image and pixel thresholds.
        Unknown categories use an adaptive map-derived fallback.
        """
        shot_key = 1 if self.k_shot > 0 else 0
        image_by_shot = getattr(self, "_image_thresholds_by_shot", self._IMAGE_THRESHOLDS_BY_SHOT)
        pixel_by_shot = getattr(self, "_pixel_thresholds_by_shot", self._PIXEL_THRESHOLDS_BY_SHOT)
        image_thresholds = image_by_shot.get(shot_key, image_by_shot[0])
        pixel_thresholds = pixel_by_shot.get(shot_key, pixel_by_shot[0])
        if category in image_thresholds:
            img_thresh = float(image_thresholds[category])
            return HeatmapThresholds(
                image=img_thresh,
                pixel=float(pixel_thresholds[category]),
                source=f"adaptclip:calibrated:{shot_key}shot",
            )
        if heatmap is not None:
            return adaptive_thresholds(
                heatmap,
                default_image=0.5,
                default_pixel=0.45,
                source="adaptclip:adaptive_unknown_category",
            )
        return HeatmapThresholds(image=0.5, pixel=0.45, source="adaptclip:default_unknown_category")

    def get_thresholds(self, category):
        cfg = self.get_threshold_config(category)
        return cfg.image, cfg.pixel

    def predict_anomaly_map(self, image):
        self.last_image_score = None
        if isinstance(image, Image.Image):
            img_tensor = self.preprocess(image)
        elif isinstance(image, torch.Tensor):
            img_tensor = image.to(self.device)
            if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            query_feats, query_patch_feats = self.model.encode_image(
                img_tensor, self.cfg['features_list'], DPAM_layer=self.DPAM_layer
            )

            global_vl_logit, local_vl_map = self.visual_learner(
                query_feats, query_patch_feats, self.textual_learner.static_text_features
            )
            local_vl_map = local_vl_map[:, 1].detach()
            global_vl_score = global_vl_logit.softmax(-1)[:, 1].detach()

            global_tl_logit, local_tl_map = self.textual_learner.compute_global_local_score(
                query_feats, query_patch_feats, self.learned_text_features
            )
            local_tl_map = local_tl_map[:, 1].detach()
            global_tl_score = global_tl_logit.softmax(-1)[:, 1].detach()

            support_key = self._active_support_key()
            if self.k_shot > 0 and support_key is not None:
                prompt_feats = self.prompt_image_memory[support_key].unsqueeze(0)
                prompt_patch_feats = [
                    layer_memory.unsqueeze(0)
                    for layer_memory in self.prompt_patch_memory[support_key]
                ]
                global_pq_logit, local_pq_map_list, align_score_list = self.pq_learner(
                    query_feats,
                    query_patch_feats,
                    prompt_feats,
                    prompt_patch_feats,
                )
                local_pq_map_list = [item[:, 1].unsqueeze(1) for item in local_pq_map_list]
                local_pq_map = torch.concat(local_pq_map_list, dim=1).mean(dim=1).detach()
                align_score = fusion_fun(align_score_list, fusion_type="harmonic_mean")[:, 0]
                if isinstance(global_pq_logit, list):
                    global_pq_score = [item.softmax(-1).unsqueeze(-1) for item in global_pq_logit]
                    global_pq_score = torch.concat(global_pq_score, dim=-1).mean(dim=-1)[:, 1].detach()
                else:
                    global_pq_score = global_pq_logit.softmax(-1)[:, 1].detach()
                pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map, local_pq_map], fusion_type=self.cfg['fusion_type'])
                pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score], fusion_type="harmonic_mean")
                image_score_inputs = [global_vl_score, global_tl_score, global_pq_score]
                image_score_fusion = "few_shot"
            else:
                pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map], fusion_type=self.cfg['fusion_type'])
                image_score_inputs = [global_vl_score, global_tl_score]
                image_score_fusion = "zero_shot"

            pixel_map_np = pixel_anomaly_map.cpu().numpy()
            pixel_map_smooth = np.array([
                gaussian_filter(x, sigma=self.cfg['sigma']) for x in pixel_map_np
            ])
            pixel_anomaly_map = torch.from_numpy(pixel_map_smooth).to(self.device)

            pixel_anomaly_map = pixel_anomaly_map.unsqueeze(1) 
            pixel_anomaly_map_resized = F.interpolate(
                pixel_anomaly_map, size=(self.cfg['image_size'], self.cfg['image_size']), 
                mode='bilinear', align_corners=False
            )
            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(pixel_anomaly_map.shape[0], -1), dim=1)
            if image_score_fusion == "few_shot":
                image_anomaly_pred = fusion_fun(image_score_inputs, fusion_type=self.cfg['fusion_type'])
                image_anomaly_pred = fusion_fun([image_anomaly_pred, anomaly_map_max], fusion_type="harmonic_mean")
            else:
                image_anomaly_pred = fusion_fun([*image_score_inputs, anomaly_map_max], fusion_type=self.cfg['fusion_type'])
            self.last_image_score = float(image_anomaly_pred[0].detach().cpu())
            
            final_map = pixel_anomaly_map_resized[0, 0].cpu().numpy()
            best_obj_name = self.active_category

            return final_map, best_obj_name

    def _active_support_key(self):
        if not self.active_category:
            return None
        for key in self.prompt_image_memory:
            if key[0] == self.active_category:
                return key
        return None
