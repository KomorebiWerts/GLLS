import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from scipy.ndimage import gaussian_filter
import open_clip
from .ABounD import VVCLIP_lib
from .ABounD.prompt_generator import DynamicConceptFusion
from .ABounD.utils import get_transform, generate_class_info
from .AdaptCLIP import adaptcliplib
from .AdaptCLIP import TextualAdapter, VisualAdapter, fusion_fun
from .AdaptCLIP import get_transform as get_transform_adaptclip

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

        # === 3. 自动构建 Checkpoint 路径 ===
        ckpt_filename = f"{args.k_shot}shot.pt"
        self.checkpoint_full_path = os.path.join(args.checkpoint_path, ckpt_filename)
        
        print(f"[{dataset_name.upper()}] Localizer Configured:")
        print(f"  -> Params: depth={self.depth}, ctx={self.n_ctx}, spe={self.spe}")
        print(f"  -> Loading Checkpoint: {self.checkpoint_full_path}")

        # === 4. 加载模型 ===
        VVCLIP_parameters = {
            "Prompt_length": self.n_ctx, 
            "learnabel_text_embedding_depth": self.depth,
            "learnabel_text_embedding_length": self.n_ctx
        }
        
        try:
            self.model, _ = VVCLIP_lib.load(
                "ViT-L/14@336px", 
                device=self.device, 
                design_details=VVCLIP_parameters,
                download_root=args.checkpoint_path 
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
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
            
            lora_config = LoraConfig(
                r=4, lora_alpha=8, target_modules=target_modules_list,
                lora_dropout=0.25, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model = get_peft_model(self.model, lora_config)
            
            lora_path = os.path.join(args.save_path, args.dataset, f"final_vvclip_model_state_{args.dataset}.pth")
            if os.path.exists(lora_path):
                state_dict = torch.load(lora_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                print(f"⚠️ Warning: LoRA checkpoint not found at {lora_path}")

        embed_dim = 768
        self.dcf_module = DynamicConceptFusion(
            T_len_shared=self.n_ctx - self.spe, 
            T_len_specific=self.spe,
            embed_dim=embed_dim,
            depth=self.depth,
            text_embed_dim=embed_dim
        ).to(self.device)
        
        obj_list_full, _ = generate_class_info(args.dataset)
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
        self.preprocess, _ = get_transform(args)
    
    def get_thresholds(self, category):
        """
        Return tuple: (image_threshold, pixel_threshold)
        Uses the specific ABounD threshold tables.
        """
        img_thresh = self._IMAGE_THRESHOLDS.get(category, 0.9)
        pix_thresh = self._PIXEL_THRESHOLDS.get(category, 0.9)
        return img_thresh, pix_thresh

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
                lp = MultivariateNormal(mean.to(self.device), covariance_matrix=cov.to(self.device)).log_prob(global_feat_norm.squeeze())
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
            
            sim, _ = VVCLIP_lib.compute_similarity(all_patches.unsqueeze(0), text_feats.T)
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
    
    # === AdaptCLIP 专属 Image Thresholds (New Table) ===
    _IMAGE_THRESHOLDS = {
        # --- VisA ---
        "candle": 0.5107,
        "capsules": 0.5337,
        "cashew": 0.4507,
        "chewinggum": 0.381,
        "fryum": 0.5354,
        "macaroni1": 0.5302,
        "macaroni2": 0.9303,
        "pcb1": 0.6943,
        "pcb2": 0.7393,
        "pcb3": 0.7762,
        "pcb4": 0.8055,
        "pipe_fryum": 0.3994,
        # --- MVTec ---
        "bottle": 0.8507,
        "cable": 0.763,
        "capsule": 0.4503,
        "carpet": 0.2766,
        "grid": 0.3509,
        "hazelnut": 0.8747,
        "leather": 0.6126,
        "metal_nut": 0.7903,
        "pill": 0.5443,
        "screw": 0.7662,
        "tile": 0.6408,
        "toothbrush": 0.5659,
        "transistor": 0.7867,
        "wood": 0.784,
        "zipper": 0.2127
    }

    def __init__(self, args, device=None, pretrained_model='ViT-L/14@336px'):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = self._CONFIGS['default']
        
        # Override default config with args if present
        if hasattr(args, 'image_size'):
            self.cfg['image_size'] = args.image_size
            
        checkpoint_path = getattr(args, 'checkpoint_path', None)
        
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

        # 3. Initialize Adapters (Only Visual & Textual for 0-shot)
        self.textual_learner = TextualAdapter(self.model.to("cpu"), self.cfg['image_size'], self.cfg['n_ctx'])
        self.visual_learner = VisualAdapter(self.cfg['image_size'], self.patch_size, input_dim=self.input_dim, reduction=self.cfg['vl_reduction'])

        # 4. Load Trained Weights (Crucial)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[AdaptCLIP] Loading Adapter weights from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.textual_learner.load_state_dict(ckpt["textual_learner"], strict=False)
            self.visual_learner.load_state_dict(ckpt["visual_learner"], strict=False)
        else:
            print("⚠️ [AdaptCLIP] WARNING: No checkpoint found! Model will output random noise.")

        # 5. Model to Device & Eval
        self.model.to(self.device).eval()
        self.textual_learner.to(self.device).eval()
        self.visual_learner.to(self.device).eval()

        # 6. Pre-compute Text Features (Optimization)
        print("[AdaptCLIP] Pre-computing static text embeddings...")
        self.textual_learner.prepare_static_text_feature(self.model)
        
        with torch.no_grad():
            learned_prompts, tokenized_prompts = self.textual_learner()
            self.learned_text_features = self.model.encode_text_learn(learned_prompts, tokenized_prompts).float()

        # 7. Setup Transform
        self.transform, _ = get_transform_adaptclip(image_size=self.cfg['image_size'])

    def preprocess(self, image_pil):
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        return self.transform(image_pil).unsqueeze(0).to(self.device)

    def get_thresholds(self, category):
        """
        Retrieves the thresholds.
        Logic: Pixel Threshold = Image Threshold - 0.05
        """
        img_thresh = self._IMAGE_THRESHOLDS.get(category, 0.5) 
        pix_thresh = img_thresh - 0.05
        return img_thresh, pix_thresh

    def predict_anomaly_map(self, image):
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

            global_tl_logit, local_tl_map = self.textual_learner.compute_global_local_score(
                query_feats, query_patch_feats, self.learned_text_features
            )
            local_tl_map = local_tl_map[:, 1].detach()

            pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map], fusion_type=self.cfg['fusion_type'])

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
            
            final_map = pixel_anomaly_map_resized[0, 0].cpu().numpy()
            best_obj_name = None

            return final_map, best_obj_name