from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
import random


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


# implement attention module for v-v self-attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, 
                  layer_idx=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        # Visual prompt injection support
        self.layer_idx = layer_idx

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, whole=False, ffn=False, visual_prompts_info=None):
        # Handle visual prompt injection for BOTH paths in dual-path mode
        if visual_prompts_info is not None and isinstance(visual_prompts_info, tuple):
            deep_prompts, counter, num_prompts_injected, num_patches = visual_prompts_info
            
            # For dual path mode - handle both x and x_ori
            if isinstance(x, list):
                x, x_ori = x
                
                # Inject prompts at appropriate layers
                if counter < len(deep_prompts):
                    # Get current layer's visual prompt
                    visual_prompt = deep_prompts[counter]
                    L_prompt = visual_prompt.shape[0]
                    visual_prompt = visual_prompt.unsqueeze(1).expand(-1, x.shape[1], -1).to(x.dtype)
                    
                    if self.layer_idx == 0:
                        # First layer: concatenate at the END (following AdaCLIP pattern)
                        x = torch.cat([x, visual_prompt], dim=0)
                        x_ori = torch.cat([x_ori, visual_prompt], dim=0)
                    else:
                        # Subsequent layers: keep prompts after CLS + num_patches
                        # Token order should be: [CLS, patches[:num_patches], prompts, patches[num_patches:]]
                        cls_x = x[:1]  # CLS token
                        cls_x_ori = x_ori[:1]  # CLS token
                        
                        patches_x = x[1:1+num_patches]  # First num_patches
                        patches_x_ori = x_ori[1:1+num_patches]
                        
                        remaining_x = x[1+num_patches:]  # Remaining tokens (excluding previous prompts if any)
                        remaining_x_ori = x_ori[1+num_patches:]
                        
                        # If prompts were already injected, we need to skip them
                        if num_prompts_injected > 0:
                            remaining_x = remaining_x[num_prompts_injected:]
                            remaining_x_ori = remaining_x_ori[num_prompts_injected:]
                        
                        # Reconstruct with new order
                        x = torch.cat([cls_x, patches_x, visual_prompt, remaining_x], dim=0)
                        x_ori = torch.cat([cls_x_ori, patches_x_ori, visual_prompt, remaining_x_ori], dim=0)
                    
                    counter += 1
                    num_prompts_injected = L_prompt
                
                visual_prompts_info = (deep_prompts, counter, num_prompts_injected, num_patches)
                x = [x, x_ori]
            else:
                # Single path mode
                if counter < len(deep_prompts):
                    visual_prompt = deep_prompts[counter]
                    L_prompt = visual_prompt.shape[0]
                    visual_prompt = visual_prompt.unsqueeze(1).expand(-1, x.shape[1], -1).to(x.dtype)
                    
                    if self.layer_idx == 0:
                        # First layer: concatenate at the END
                        x = torch.cat([x, visual_prompt], dim=0)
                    else:
                        # Subsequent layers: insert after CLS + num_patches
                        cls_token = x[:1]
                        patches = x[1:1+num_patches]
                        remaining = x[1+num_patches:]
                        
                        if num_prompts_injected > 0:
                            remaining = remaining[num_prompts_injected:]
                        
                        x = torch.cat([cls_token, patches, visual_prompt, remaining], dim=0)
                    
                    counter += 1
                    num_prompts_injected = L_prompt
                
                visual_prompts_info = (deep_prompts, counter, num_prompts_injected, num_patches)

        # Process attention with dual paths
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                if not ffn:
                    x, x_ori = x
                    # Use x_ori for attention (VVCLIP design)
                    x_res = self.attention(self.ln_1(x_ori))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x = x + x_res
                    result = [x, x_ori]
                else:
                    x, x_ori_1 = x
                    x_res = self.attention(self.ln_1(x_ori_1))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori_1 + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x = x + x_res
                    x = x_res + x_ori_1
                    x = x + self.mlp(self.ln_2(x))
                    result = [x, x_ori]
            else:
                # Start of dual path
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x = x + x_res
                    result = [x, x_ori]
                else:
                    result = x
        else:
            # Single path (standard transformer block)
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            result = x
            
        # Return with updated visual prompts info if needed
        if visual_prompts_info is not None:
            return result, visual_prompts_info
        return result


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        pos_idx = torch.arange(1, self.positional_embedding.shape[0])
        random.shuffle(pos_idx)
        self.anomaly_pos = torch.cat(
            [self.positional_embedding[0].unsqueeze(dim=0), self.positional_embedding[pos_idx, :]], dim=0).cuda()

        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(width, layers, heads, need_weights=True)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.in_post = nn.InstanceNorm1d(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        

    @torch.no_grad()
    def DAPM_replace(self, DPAM_layer):
        if DPAM_layer is not None:
            for i in range(1, DPAM_layer):
                if i <= len(self.transformer.resblocks):
                    self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                    self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                    self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                    self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                    self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                    self.transformer.resblocks[-i].attn = self.attn

    def forward(self, x: torch.Tensor, features_list, ori_patch=False, proj_use=True, DPAM_layer=None, ffn=False,
                deep_compound_prompts_vision=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # Calculate num_patches
        num_patches = x.shape[1]  # grid ** 2
        
        # Add class token
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        # Handle position embedding resizing
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        pos = self.positional_embedding.to(x.dtype)
        x = x + pos
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Forward through transformer with deep prompts if provided
        if deep_compound_prompts_vision is not None:
            # Pass the tokens, deep_prompts, and initial counter to transformer
            [x, x_ori], patch_tokens = self.transformer([x, deep_compound_prompts_vision, 0], 
                                                       features_list, DPAM_layer=DPAM_layer, ffn=ffn)
        else:
            [x, x_ori], patch_tokens = self.transformer(x, features_list,
                                                       DPAM_layer=DPAM_layer, ffn=ffn)

        # Process outputs - extract tokens excluding prompts
        patch_token_list = []
        patch_token_memory = []
        
        for patch_token in patch_tokens:
            if isinstance(patch_token, list):
                # For dual path
                # Extract CLS + first num_patches (excluding any injected prompts)
                cls_token_0 = patch_token[0][:1]
                cls_token_1 = patch_token[1][:1]
                
                # Get the patches (they should be right after CLS)
                patches_0 = patch_token[0][1:num_patches+1]
                patches_1 = patch_token[1][1:num_patches+1]
                
                patch_only = torch.cat([cls_token_0, patches_0], dim=0)
                patch_only_memory = torch.cat([cls_token_1, patches_1], dim=0)
                
                normal_patch_token = self.ln_post(patch_only.permute(1, 0, 2)) @ self.proj
                patch_token_memory.append(patch_only_memory.permute(1, 0, 2))
            else:
                # Single path
                cls_token = patch_token[:1]
                patches = patch_token[1:num_patches+1]
                patch_only = torch.cat([cls_token, patches], dim=0)
                
                normal_patch_token = self.ln_post(patch_only.permute(1, 0, 2)) @ self.proj
                patch_token_memory.append(patch_only.permute(1, 0, 2))
            
            patch_token_list.append(normal_patch_token)
            
        patch_token_list   = torch.stack(patch_token_list,   dim=0)
        patch_token_memory = torch.stack(patch_token_memory, dim=0)
        
        return x_ori[0, :, :] @ self.proj, patch_token_list, patch_token_memory



class ResidualAttentionBlock_learnable_token(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.i = i
        self.compound_prompt_nctx = design_details['learnabel_text_embedding_length'] if design_details else 0
        self.text_layer = text_layer
        self.first_layer = (i == 0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        is_list_input = isinstance(inputs, list)

        if is_list_input:
            x = inputs[0]
            compound_prompts_deeper = inputs[1]
            counter = inputs[2]
            # Learnable token injection logic
            if not self.first_layer:
                if not (counter > len(compound_prompts_deeper) - 1):
                    prefix = x[:1, :, :]
                    suffix = x[1 + self.compound_prompt_nctx:, :, :]
                    textual_context = compound_prompts_deeper[counter]
                    textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).to(x.dtype)
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                    counter += 1
        else:
            x = inputs

        # Standard Attention + MLP
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        if is_list_input:
            return [x, compound_prompts_deeper, counter]
        else:
            return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, need_weights: bool = False,
                 design_details=None, text_layer=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.design_details = design_details
        
        print("text_layer", self.text_layer)
        if self.text_layer and (self.design_details is not None):
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlock_learnable_token(width, heads, attn_mask, self.design_details, text_layer, i=i)
                 for i in range(layers)])
        else:
            # For visual transformer with prompt support
            self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(width, heads, attn_mask, 
                                     layer_idx=i) 
                for i in range(layers)
            ])

    def forward(self, x: torch.Tensor, out_layers=[6, 12, 18, 24], DPAM_layer=None, ffn=False):
        visual_prompts_info = None
        
        # Handle visual prompts for vision transformer
        if not self.text_layer and isinstance(x, list) and len(x) == 3:
            # x = [tokens, deep_prompts, counter]
            tokens, deep_prompts, counter = x
            # Get num_patches from the token dimension (excluding CLS)
            num_patches = tokens.shape[0] - 1  # L - 1 (excluding CLS token)
            # Initialize with counter=0, num_prompts_injected=0, and num_patches
            visual_prompts_info = (deep_prompts, counter, 0, num_patches)
            x = tokens
        
        # Visual encoder forward
        if not self.text_layer:
            idx = 0
            out_tokens = []
            
            for r in self.resblocks:
                idx += 1
                
                if DPAM_layer is None:
                    # Original CLIP forward
                    if visual_prompts_info is not None:
                        x, visual_prompts_info = r(x, visual_prompts_info=visual_prompts_info)
                    else:
                        x = r(x)
                    if idx in out_layers:
                        out_tokens.append(x)
                else:
                    # VVCLIP forward with DPAM - properly handle visual prompts
                    if visual_prompts_info is not None:
                        result = r(x, ffn=ffn, visual_prompts_info=visual_prompts_info)
                        if isinstance(result, tuple) and len(result) == 2:
                            # Result contains updated visual_prompts_info
                            x, visual_prompts_info = result
                        else:
                            x = result
                    else:
                        x = r(x, ffn=ffn)
                    
                    if idx in out_layers:
                        if isinstance(x, list):
                            out_tokens.append([x[0].clone(), x[1].clone()])
                        else:
                            out_tokens.append(x)
            
            return x, out_tokens
            
        # Text encoder forward
        elif self.design_details is None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x
        # Insert learnable text embedding
        elif self.design_details is not None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x[0] if isinstance(x, list) else x

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype





from thop import profile


class VVCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details=None
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), text_layer=True, design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
    def encode_text_with_prompts(self, learnable_prompts, deep_compound_prompts_text=None):
        """
        Encode text using learnable prompts without needing a template. (Final Version)
        Args:
            learnable_prompts: Learnable prompt embeddings [N, L_prompt, E].
            deep_compound_prompts_text: Deep text prompts for subsequent layers (optional).
        """
        cast_dtype = self.transformer.get_cast_dtype()
        N = learnable_prompts.size(0)
        device = learnable_prompts.device

        # 1. Get SOT embedding from its fixed ID
        # For 'ViT-L-14' tokenizer, SOT is 49406.
        sot_id = 49406
        sot_token = torch.tensor([[sot_id]], device=device).expand(N, -1)
        sot_embedding = self.token_embedding(sot_token).to(cast_dtype)

        # 2. Get EOT embedding from its fixed ID
        # For 'ViT-L-14' tokenizer, EOT is 49407.
        eot_id = 49407
        eot_token = torch.tensor([[eot_id]], device=device).expand(N, -1)
        eot_embedding = self.token_embedding(eot_token).to(cast_dtype)

        # 3. Concatenate the meaningful parts: [SOT, prompts, EOT]
        x = torch.cat([sot_embedding, learnable_prompts, eot_embedding], dim=1)

        # 4. Handle padding and truncation to context_length
        if x.size(1) > self.context_length:
            x = x[:, :self.context_length, :]
        elif x.size(1) < self.context_length:
            padding_size = self.context_length - x.size(1)
            padding = torch.zeros(N, padding_size, x.size(2), device=device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

        # 5. Add positional encoding
        x = x + self.positional_embedding.to(cast_dtype)

        # 6. Pass through the transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 7. Final LayerNorm
        x = self.ln_final(x).type(self.dtype)

        # 8. Get the embedding at the EOT position
        eot_position = 1 + learnable_prompts.size(1)
        # Ensure the position doesn't exceed the context length
        if eot_position >= self.context_length:
            eot_position = self.context_length - 1

        final_eot_embedding = x[torch.arange(x.shape[0]), eot_position] @ self.text_projection

        return final_eot_embedding, x
    def encode_text_with_prefix(self, prefix_tokens, learnable_prompts, tokenized_full,
                                deep_compound_prompts_text=None):
        """
        Encode text with fixed prefix
        Args:
            prefix_tokens: "a photo of normal/abnormal" token embeddings [N, L_prefix, E]
            learnable_prompts: learnable prompts (includes visual tokens) [N, L_prompt, E]
            tokenized_full: full token sequence for EOT position [N, 77]
            deep_compound_prompts_text: deep text prompts (optional)
        """
        cast_dtype = self.transformer.get_cast_dtype()

        # Get SOT embedding
        sot_token = tokenized_full[0, 0]
        sot_embedding = self.token_embedding(sot_token.unsqueeze(0)).unsqueeze(0).to(cast_dtype)

        N = learnable_prompts.size(0)
        sot_embedding = sot_embedding.expand(N, -1, -1)

        # Concatenate: [SOT, prefix, learnable_prompts, EOT, padding]
        current_length = 1 + prefix_tokens.size(1) + learnable_prompts.size(1) + 1  # SOT + prefix + prompts + EOT
        padding_length = self.context_length - current_length

        # Get EOT and padding
        eot_token = tokenized_full[0, current_length - 1].unsqueeze(0)
        eot_embedding = self.token_embedding(eot_token).unsqueeze(0).to(cast_dtype).expand(N, -1, -1)

        if padding_length > 0:
            pad_tokens = tokenized_full[0, current_length:].unsqueeze(0).expand(N, -1)
            pad_embeddings = self.token_embedding(pad_tokens).to(cast_dtype)
            x = torch.cat([sot_embedding,learnable_prompts, prefix_tokens, eot_embedding, pad_embeddings], dim=1)
        else:
            x = torch.cat([sot_embedding, learnable_prompts, prefix_tokens, eot_embedding], dim=1)

        # Ensure length is 77
        if x.size(1) > self.context_length:
            x = x[:, :self.context_length, :]
        elif x.size(1) < self.context_length:
            padding_size = self.context_length - x.size(1)
            padding = torch.zeros(N, padding_size, x.size(2), device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

        # Add position encoding
        x = x + self.positional_embedding.to(cast_dtype)

        # Through Transformer
        x = x.permute(1, 0, 2)  # NLD -> LND

        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])

        x = x.permute(1, 0, 2)  # LND -> NLD

        # LayerNorm
        x = self.ln_final(x).type(self.dtype)

        # Get EOT position embedding
        eot_position = 1 + prefix_tokens.size(1) + learnable_prompts.size(1)
        eot_embedding = x[torch.arange(x.shape[0]), eot_position - 1] @ self.text_projection

        return eot_embedding, x

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_list=[], ori_patch=False, proj_use=True, DPAM_layer=None, ffn=False,
                     deep_compound_prompts_vision=None):
        return self.visual(image.type(self.dtype), feature_list, ori_patch=ori_patch, proj_use=proj_use,
                           DPAM_layer=DPAM_layer, ffn=ffn, deep_compound_prompts_vision=deep_compound_prompts_vision)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text