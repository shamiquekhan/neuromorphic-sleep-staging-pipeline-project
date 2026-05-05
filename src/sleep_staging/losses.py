"""Loss functions for sleep staging.

Provides a lightweight FocalLoss implementation compatible with CrossEntropyLoss API.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Args:
        gamma: focusing parameter (>0 reduces relative loss for well-classified examples)
        alpha: optional per-class weight tensor (shape [C]) or scalar
        reduction: 'mean'|'sum'|'none'
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), dtype=torch.float32)
        self.register_buffer("alpha", alpha) if isinstance(alpha, torch.Tensor) else setattr(self, "alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C)   targets: (N,)
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiLevelKDLoss(nn.Module):
    """Multi-level Knowledge Distillation with feature matching.
    
    Distills at multiple levels:
    1. Logits (soft targets with temperature)
    2. Feature maps (CNN representations)
    3. Relation knowledge (pairwise distances between samples)
    
    Based on: DistillSleepNet, attention transfer, RKD.
    Expected gain over single-level KD: +1-2% accuracy.
    
    Args:
        student_feat_dim: student CNN output dimension
        teacher_feat_dim: teacher CNN output dimension
        temperature: KD temperature (higher = softer)
        alpha: weight for hard CE loss
        beta: weight for KL divergence (soft targets)
        gamma: weight for feature matching loss
        delta: weight for relation knowledge distillation
        class_weights: per-class weights for CE
    """
    
    def __init__(
        self,
        student_feat_dim: int = 96,
        teacher_feat_dim: int = 256,
        temperature: float = 6.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.1,
        delta: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        if class_weights is None:
            cw = torch.ones(5, dtype=torch.float32, device=device)
        else:
            cw = class_weights.to(device) if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights, device=device)
        
        self.ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        
        # Feature projection: student features -> teacher dimension
        self.feature_projector = nn.Sequential(
            nn.Linear(student_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, teacher_feat_dim),
        )
    
    def forward(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        s_features: Optional[torch.Tensor] = None,
        t_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        focal_gamma: float = 1.5,
    ) -> dict:
        """Compute multi-level KD loss.
        
        Args:
            s_logits: student logits (B, T, C) or (B*T, C)
            t_logits: teacher logits (B, T, C) or (B*T, C)
            s_features: student features (B*T, feat_dim) before classification head
            t_features: teacher features (B*T, feat_dim) before classification head
            labels: ground truth labels (B*T,)
            focal_gamma: focal loss gamma parameter
            
        Returns:
            dict with loss components and total
        """
        
        # Ensure 2D shapes
        if s_logits.dim() == 3:
            B, T, C = s_logits.shape
            s_logits = s_logits.reshape(B * T, C)
            t_logits = t_logits.reshape(B * T, C)
            if labels is not None:
                labels = labels.reshape(B * T)
            if s_features is not None:
                s_features = s_features.reshape(B * T, -1)
            if t_features is not None:
                t_features = t_features.reshape(B * T, -1)
        
        losses = {}
        
        # 1. Hard CE (focal) on student predictions
        if labels is not None:
            ce_loss = self._focal_ce(s_logits, labels, focal_gamma)
            losses["ce"] = ce_loss
        else:
            ce_loss = torch.tensor(0.0, device=s_logits.device)
            losses["ce"] = ce_loss
        
        # 2. KL divergence on soft targets (logits)
        kl_loss = self._kl_distill(s_logits, t_logits)
        losses["kl"] = kl_loss
        
        # 3. Feature matching (L2 distance between projected features)
        feat_loss = torch.tensor(0.0, device=s_logits.device)
        if self.gamma > 0 and s_features is not None and t_features is not None:
            feat_loss = self._feature_matching(s_features, t_features)
        losses["feat"] = feat_loss
        
        # 4. Relation knowledge distillation (pairwise distance matching)
        rkd_loss = torch.tensor(0.0, device=s_logits.device)
        if self.delta > 0 and s_features is not None and t_features is not None:
            rkd_loss = self._relation_knowledge(s_features, t_features)
        losses["rkd"] = rkd_loss
        
        # Total weighted loss
        total_loss = (
            self.alpha * ce_loss +
            self.beta * kl_loss +
            self.gamma * feat_loss +
            self.delta * rkd_loss
        )
        
        return {
            "total": total_loss,
            "ce": float(ce_loss.detach().item()) if not torch.isnan(ce_loss) else 0.0,
            "kl": float(kl_loss.detach().item()) if not torch.isnan(kl_loss) else 0.0,
            "feat": float(feat_loss.detach().item()) if not torch.isnan(feat_loss) else 0.0,
            "rkd": float(rkd_loss.detach().item()) if not torch.isnan(rkd_loss) else 0.0,
        }
    
    def _focal_ce(self, logits: torch.Tensor, labels: torch.Tensor, gamma: float) -> torch.Tensor:
        """Focal cross-entropy loss."""
        ce = F.cross_entropy(logits, labels, weight=self.ce.weight, reduction="none")
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
            focal_w = (1.0 - pt).pow(gamma)
        loss = (focal_w * ce).mean()
        if torch.isnan(loss):
            loss = ce.mean()
        return loss
    
    def _kl_distill(self, s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
        """KL divergence for soft target matching."""
        s_logits = s_logits.clamp(min=-10, max=10) / self.T
        t_logits = t_logits.clamp(min=-10, max=10) / self.T
        
        s_soft = F.log_softmax(s_logits, dim=-1)
        t_soft = F.softmax(t_logits, dim=-1)
        
        loss = self.kl(s_soft, t_soft) * (self.T ** 2)
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=s_logits.device)
        return loss
    
    def _feature_matching(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Feature-level matching with projection and normalization."""
        # Project student features to teacher dimension
        s_proj = self.feature_projector(s_feat)
        
        # L2 normalize both
        s_norm = F.normalize(s_proj, p=2, dim=-1)
        t_norm = F.normalize(t_feat, p=2, dim=-1)
        
        loss = F.mse_loss(s_norm, t_norm)
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=s_feat.device)
        return loss
    
    def _relation_knowledge(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Relation knowledge distillation: match pairwise distances."""
        def _pairwise_dist(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """Compute normalized pairwise L2 distance matrix."""
            sq = feat.pow(2).sum(1, keepdim=True)
            dist = sq + sq.T - 2 * feat @ feat.T
            dist = dist.clamp(min=0).sqrt()
            mu = dist[dist > 0].mean() if (dist > 0).any() else 1.0
            dist = dist / (mu + eps)
            return dist
        
        try:
            s_dist = _pairwise_dist(s_feat)
            t_dist = _pairwise_dist(t_feat)
            loss = F.smooth_l1_loss(s_dist, t_dist)
            if torch.isnan(loss):
                loss = torch.tensor(0.0, device=s_feat.device)
            return loss
        except Exception:
            return torch.tensor(0.0, device=s_feat.device)
