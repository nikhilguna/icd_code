"""
Loss functions for multi-label ICD code prediction.

Includes:
- Weighted BCE loss for class imbalance
- Focal loss for hard examples
- Asymmetric loss for positive-negative imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiLabelBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss for multi-label classification.
    
    Supports:
    - Per-label positive weights for class imbalance
    - Label smoothing
    - Reduction options
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Args:
            pos_weight: Weight for positive samples per label (num_labels,)
            label_smoothing: Label smoothing factor (0 = no smoothing)
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss.
        
        Args:
            logits: Model output logits (batch, num_labels)
            targets: Binary targets (batch, num_labels)
            
        Returns:
            Loss value
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Focuses training on hard examples by down-weighting easy examples.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Based on: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            pos_weight: Additional per-label positive weights
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model output logits (batch, num_labels)
            targets: Binary targets (batch, num_labels)
            
        Returns:
            Loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Apply focal and alpha weights
        loss = alpha_weight * focal_weight * bce
        
        # Apply per-label weights if provided
        if self.pos_weight is not None:
            weight = self.pos_weight * targets + (1 - targets)
            loss = loss * weight
        
        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Uses different focusing parameters for positive and negative samples,
    which is beneficial when negatives vastly outnumber positives.
    
    Based on: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., 2021)
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping threshold for negatives
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Model output logits (batch, num_labels)
            targets: Binary targets (batch, num_labels)
            
        Returns:
            Loss value
        """
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping for negatives
        probs_neg = probs.clamp(min=self.clip)
        
        # Positive loss
        pos_loss = targets * torch.log(probs.clamp(min=1e-8))
        pos_loss = pos_loss * (1 - probs) ** self.gamma_pos
        
        # Negative loss
        neg_loss = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        neg_loss = neg_loss * probs_neg ** self.gamma_neg
        
        loss = -(pos_loss + neg_loss)
        
        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss that respects ICD code structure.
    
    ICD codes have a hierarchical structure (e.g., 410 -> 410.7 -> 410.71).
    This loss encourages consistent predictions across the hierarchy.
    
    Note: Requires a hierarchy mapping to be provided.
    """
    
    def __init__(
        self,
        hierarchy_matrix: torch.Tensor,
        alpha: float = 0.1,
        base_loss: str = "bce",
    ):
        """
        Args:
            hierarchy_matrix: Binary matrix (num_labels, num_labels) where
                             matrix[i,j] = 1 if label i is ancestor of label j
            alpha: Weight for hierarchy consistency loss
            base_loss: Base loss type ("bce" or "focal")
        """
        super().__init__()
        self.register_buffer("hierarchy_matrix", hierarchy_matrix)
        self.alpha = alpha
        
        if base_loss == "bce":
            self.base_loss = MultiLabelBCELoss()
        else:
            self.base_loss = FocalLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hierarchical loss.
        
        Args:
            logits: Model output logits (batch, num_labels)
            targets: Binary targets (batch, num_labels)
            
        Returns:
            Loss value
        """
        # Base classification loss
        cls_loss = self.base_loss(logits, targets)
        
        # Hierarchy consistency loss
        # If a child is predicted, ancestors should also be predicted
        probs = torch.sigmoid(logits)
        
        # For each label, compute max probability of its children
        # hierarchy_matrix[i,j] = 1 if i is ancestor of j
        child_probs = torch.matmul(self.hierarchy_matrix, probs.T).T  # (batch, num_labels)
        
        # Ancestor should have higher prob than children
        hierarchy_loss = F.relu(child_probs - probs).mean()
        
        return cls_loss + self.alpha * hierarchy_loss


def get_loss_function(
    loss_type: str = "bce",
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: One of "bce", "focal", "asymmetric"
        pos_weight: Optional positive class weights
        **kwargs: Additional arguments for specific losses
        
    Returns:
        Loss module
    """
    if loss_type == "bce":
        return MultiLabelBCELoss(pos_weight=pos_weight, **kwargs)
    elif loss_type == "focal":
        return FocalLoss(pos_weight=pos_weight, **kwargs)
    elif loss_type == "asymmetric":
        return AsymmetricLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

