import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network block, used for instance-specific modulation.
    MLP(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float = 0.1, bias: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.w_down = nn.Linear(hidden_dim, out_dim, bias=bias)

        for m_linear in [self.w_gate, self.w_up, self.w_down]:
            init.kaiming_normal_(m_linear.weight.data)
            if m_linear.bias is not None:
                init.constant_(m_linear.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_val = self.act(self.w_gate(x))
        up_val = self.w_up(x)
        fused_val = gate_val * up_val
        fused_val = self.dropout(fused_val)
        output = self.w_down(fused_val)
        return output

class MixtureOfExpertsGating(nn.Module):
    """
    A module to generate shared prompts using a Mixture-of-Experts (MoE) approach.
    It uses visual tokens to generate gating weights for a set of learnable expert prompts.
    """
    def __init__(self, n_experts: int, prompt_len: int, embed_dim: int, vis_dim: int):
        super().__init__()
        self.n_experts = n_experts
        
        self.base_experts = nn.Parameter(torch.randn(n_experts, prompt_len, embed_dim))
        init.normal_(self.base_experts, std=0.02)

        self.gating_network = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 2),
            nn.ReLU(),
            nn.Linear(vis_dim // 2, n_experts)
        )

        for m in self.gating_network.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, vis_token: torch.Tensor) -> torch.Tensor:
        """
        Generates the final shared prompt based on visual features.
        Args:
            vis_token: Visual token features [B, D] or [D]
        Returns:
            Final prompt [B, L, D] or [L, D]
        """
        if vis_token.ndim == 1:
            vis_token = vis_token.unsqueeze(0)

        gating_logits = self.gating_network(vis_token)  # [B, n_experts]
        gating = F.softmax(gating_logits, dim=-1)
        
        final_prompt = torch.einsum('bk,kld->bld', gating, self.base_experts)
        
        return final_prompt.squeeze(0) if vis_token.shape[0] == 1 else final_prompt


class DynamicConceptFusion(nn.Module):
    """
    Implementation of the Dynamic Concept Fusion (DCF) module from the paper.
    - Uses a Mixture-of-Experts (MoE) for shared prompts.
    - Uses learnable vectors for class-specific prompts.
    - Generates deep prompts for hierarchical guidance.
    - Applies instance-specific modulation to shallow prompts via SwiGLU FFNs.
    """
    def __init__(self,
                 T_len_shared=18,
                 T_len_specific=8,
                 embed_dim=768,
                 text_embed_dim=768,
                 dropout_rate=0.1,
                 depth=9,
                 n_experts_shared=4,
                 **kwargs):
        super().__init__()

        # ===== Basic Parameters =====
        self.T_len_shared = T_len_shared
        self.T_len_specific = T_len_specific
        self.text_embed_dim = text_embed_dim
        self.depth = depth

        # ===== SHALLOW PROMPTS (Layer 0) =====
        # Shared Prompts via Mixture-of-Experts (MoE)
        self.shared_ctx_pos_generator = MixtureOfExpertsGating(
            n_experts_shared, T_len_shared, text_embed_dim, embed_dim
        )
        self.shared_ctx_neg_generator = MixtureOfExpertsGating(
            n_experts_shared, T_len_shared, text_embed_dim, embed_dim
        )
        
        # Class-Specific Prompts
        self.ctx_pos_specific = nn.ParameterDict()
        self.ctx_neg_specific = nn.ParameterDict()

        # ===== DEEP PROMPTS (Layers 1 to depth-1) =====
        if self.depth > 1:
            # Shared Deep Prompts via MoE
            self.deep_shared_pos_generators = nn.ModuleList([
                MixtureOfExpertsGating(n_experts_shared, T_len_shared, text_embed_dim, embed_dim)
                for _ in range(depth - 1)
            ])
            self.deep_shared_neg_generators = nn.ModuleList([
                MixtureOfExpertsGating(n_experts_shared, T_len_shared, text_embed_dim, embed_dim)
                for _ in range(depth - 1)
            ])

            # Class-Specific Deep Prompts
            self.deep_prompts_pos_specific = nn.ModuleDict()
            self.deep_prompts_neg_specific = nn.ModuleDict()

        # ===== Instance-Specific Modulation MLPs (for shallow prompts only) =====
        mlp_hidden = embed_dim * 2
        # Projections for the shared part of the shallow prompt
        self.fp_pos_mlp_shared = SwiGLUFFN(embed_dim, mlp_hidden, self.T_len_shared * text_embed_dim, dropout_rate)
        self.fp_neg_mlp_shared = SwiGLUFFN(embed_dim, mlp_hidden, self.T_len_shared * text_embed_dim, dropout_rate)
        # Projections for the specific part of the shallow prompt
        self.fp_pos_mlp_specific = SwiGLUFFN(embed_dim, mlp_hidden, self.T_len_specific * text_embed_dim, dropout_rate)
        self.fp_neg_mlp_specific = SwiGLUFFN(embed_dim, mlp_hidden, self.T_len_specific * text_embed_dim, dropout_rate)


    def add_class_prompts(self, class_name: str):
        """
        Adds new learnable prompts (both shallow and deep) for a given class name.
        """
        if class_name in self.ctx_pos_specific:
            return

        device = self.shared_ctx_pos_generator.base_experts.device

        # 1. Initialize Shallow Class-Specific Prompts
        new_pos_prompt = nn.Parameter(torch.randn(1, self.T_len_specific, self.text_embed_dim, device=device))
        new_neg_prompt = nn.Parameter(torch.randn(1, self.T_len_specific, self.text_embed_dim, device=device))
        init.normal_(new_pos_prompt, std=0.02)
        init.normal_(new_neg_prompt, std=0.02)
        self.ctx_pos_specific[class_name] = new_pos_prompt
        self.ctx_neg_specific[class_name] = new_neg_prompt

        # 2. Initialize Deep Class-Specific Prompts
        if self.depth > 1:
            pos_deep_list = nn.ParameterList()
            for _ in range(self.depth - 1):
                p = nn.Parameter(torch.empty(self.T_len_specific, self.text_embed_dim, device=device))
                nn.init.normal_(p, std=0.02)
                pos_deep_list.append(p)
            self.deep_prompts_pos_specific[class_name] = pos_deep_list

            neg_deep_list = nn.ParameterList()
            for _ in range(self.depth - 1):
                p = nn.Parameter(torch.empty(self.T_len_specific, self.text_embed_dim, device=device))
                nn.init.normal_(p, std=0.02)
                neg_deep_list.append(p)
            self.deep_prompts_neg_specific[class_name] = neg_deep_list


    def forward(self, vis_token: torch.Tensor, class_name: str):
        """
        Generates final shallow and deep prompts via hierarchical assembly.
        """
        assert class_name in self.ctx_pos_specific, f"Class '{class_name}' not initialized. Call add_class_prompts first."

        vis_token = vis_token.to(self.shared_ctx_pos_generator.base_experts.dtype)
        if vis_token.ndim == 1:
            vis_token = vis_token.unsqueeze(0)
        B = vis_token.size(0)

        # --- 1. ASSEMBLE SHALLOW PROMPTS (with instance modulation) ---
        # Generate shared prompts using MoE
        pos_shared_base = self.shared_ctx_pos_generator(vis_token)  # [B, T_len_shared, D]
        neg_shared_base = self.shared_ctx_neg_generator(vis_token)  # [B, T_len_shared, D]
        
        # Retrieve class-specific prompts
        pos_specific_base = self.ctx_pos_specific[class_name].expand(B, -1, -1)
        neg_specific_base = self.ctx_neg_specific[class_name].expand(B, -1, -1)

        # Generate instance-specific modulation offsets
        pos_offset_shared = self.fp_pos_mlp_shared(vis_token).view(B, self.T_len_shared, self.text_embed_dim)
        pos_offset_specific = self.fp_pos_mlp_specific(vis_token).view(B, self.T_len_specific, self.text_embed_dim)
        neg_offset_shared = self.fp_neg_mlp_shared(vis_token).view(B, self.T_len_shared, self.text_embed_dim)
        neg_offset_specific = self.fp_neg_mlp_specific(vis_token).view(B, self.T_len_specific, self.text_embed_dim)

        # Apply offsets and concatenate
        final_pos_shared = pos_shared_base + pos_offset_shared
        final_pos_specific = pos_specific_base + pos_offset_specific
        final_pos_prompt = torch.cat((final_pos_shared, final_pos_specific), dim=1)

        final_neg_shared = neg_shared_base + neg_offset_shared
        final_neg_specific = neg_specific_base + neg_offset_specific
        final_neg_prompt = torch.cat((final_neg_shared, final_neg_specific), dim=1)


        # --- 2. ASSEMBLE DEEP PROMPTS (without instance modulation) ---
        final_deep_pos_prompts = []
        final_deep_neg_prompts = []

        if self.depth > 1:
            for i in range(self.depth - 1):
                # Generate shared deep prompts using MoE
                deep_pos_shared = self.deep_shared_pos_generators[i](vis_token).squeeze(0)
                deep_neg_shared = self.deep_shared_neg_generators[i](vis_token).squeeze(0)
                
                # Retrieve class-specific deep prompts for the current layer
                deep_pos_specific = self.deep_prompts_pos_specific[class_name][i]
                deep_neg_specific = self.deep_prompts_neg_specific[class_name][i]
                
                # Combine shared and specific parts
                final_deep_pos_prompts.append(torch.cat((deep_pos_shared, deep_pos_specific), dim=0))
                final_deep_neg_prompts.append(torch.cat((deep_neg_shared, deep_neg_specific), dim=0))


        return final_pos_prompt, final_neg_prompt, final_deep_pos_prompts, final_deep_neg_prompts