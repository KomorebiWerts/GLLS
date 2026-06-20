import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Attack(object):
    def __init__(self, name, model):
        self.attack = name
        self.model = model
        
        # Attempt to set the device automatically
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = 'cpu'
            print("Warning: Model has no parameters. Setting device to 'cpu'.")
        except Exception as e:
            self.device = 'cpu'
            print(f"Failed to set device automatically due to {e}. Setting device to 'cpu'. Please use set_device() if needed.")

    def forward(self, *args, **kwargs):
        """Each subclass must override this method to define the attack's computation process."""
        raise NotImplementedError

    def set_device(self, device):
        """Manually set the device."""
        self.device = device
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# =================================================================================
#  Adversarial Boundary Forging (using fixed text features)
# =================================================================================
class AdversarialBoundaryForging(Attack):
    """
    Adversarial Boundary Forging (ABF) module based on PGD.
    This corresponds to the ABF module in the paper, which generates "fence features".
    """
    def __init__(self, model, eps=0.02, alpha=0.002, steps=20, beta=0.1, random_start=True):
        super().__init__("AdversarialBoundaryForging", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.beta = beta  # Corresponds to β in the paper's L_attack equation

    def forward(self, features_batch, pos_text_emb, neg_text_emb):
        """
        Executes the Adversarial Boundary Forging attack.
        
        Args:
            features_batch: Input visual features [B, D]
            pos_text_emb: Pre-calculated normal text embedding (normalized) [B, D] or [1, D]
            neg_text_emb: Pre-calculated abnormal text embedding (normalized) [B, D] or [1, D]
        
        Returns:
            adv_features_batch: Adversarially perturbed features (fence features) [B, D]
        """
        features_batch = features_batch.clone().detach().to(self.device)
        adv_features_batch = features_batch.clone().detach()
        
        # Ensure text embeddings are on the correct device
        pos_text_emb = pos_text_emb.to(self.device)
        neg_text_emb = neg_text_emb.to(self.device)
        
        # Expand text embeddings if they are [1, D] and batch size > 1
        if pos_text_emb.size(0) == 1 and features_batch.size(0) > 1:
            pos_text_emb = pos_text_emb.expand(features_batch.size(0), -1)
        if neg_text_emb.size(0) == 1 and features_batch.size(0) > 1:
            neg_text_emb = neg_text_emb.expand(features_batch.size(0), -1)

        if self.random_start:
            adv_features_batch += torch.empty_like(adv_features_batch).uniform_(-self.eps, self.eps)

        for _ in range(self.steps):
            adv_features_batch.requires_grad = True
            adv_features_norm = F.normalize(adv_features_batch, dim=-1)
            
            # L_balance: Encourages adversarial feature to be equidistant from normal and abnormal concepts
            sim_pos = F.cosine_similarity(adv_features_norm, pos_text_emb, dim=-1)
            sim_neg = F.cosine_similarity(adv_features_norm, neg_text_emb, dim=-1)
            balance_loss = torch.abs(sim_pos - sim_neg).mean()

            # L_dispersion: Encourages diversity among the generated boundary samples
            dispersion_loss = 0.0
            if adv_features_batch.size(0) > 1 and self.beta > 0:
                pairwise_distances = torch.pdist(adv_features_batch, p=2)
                dispersion_loss = -pairwise_distances.mean()
            
            # L_attack: The composite loss for the PGD attack
            total_loss = balance_loss  + self.beta * dispersion_loss
            
            grad = torch.autograd.grad(total_loss, adv_features_batch, retain_graph=False, create_graph=False)[0]
            
            adv_features_batch = adv_features_batch.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_features_batch - features_batch, min=-self.eps, max=self.eps)
            adv_features_batch = (features_batch + delta).detach()

        return adv_features_batch
