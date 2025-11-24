"""
Segmentation losses combining cross-entropy and Dice components.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """
    Multiclass Dice loss operating on logits and integer targets.

    >>> _ = torch.manual_seed(0)
    >>> loss = DiceLoss(num_classes=2)
    >>> logits = torch.randn(1, 2, 4, 4)
    >>> targets = torch.zeros(1, 4, 4, dtype=torch.long)
    >>> round(loss(logits, targets).item(), 4)
    0.683
    """

    def __init__(self, num_classes: int, eps: float = 1e-6, ignore_index: Optional[int] = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets = targets.long()
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            if not mask.any():
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            probs = probs * mask.unsqueeze(1)
            targets = torch.where(mask, targets, torch.zeros_like(targets))
        one_hot = F.one_hot(targets.clamp(min=0, max=self.num_classes - 1), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    Combined cross-entropy and Dice loss with optional auxiliary output.

    >>> _ = torch.manual_seed(0)
    >>> loss_fn = SegmentationLoss(num_classes=2, ce_weight=1.0, dice_weight=1.0)
    >>> logits = torch.randn(1, 2, 4, 4)
    >>> targets = torch.zeros(1, 4, 4, dtype=torch.long)
    >>> round(loss_fn(logits, targets).item(), 4)
    1.6594
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        aux_weight: float = 0.4,
        class_weights: Optional[List[float]] = None,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor, persistent=False)
        else:
            self.class_weights = None
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.num_classes = num_classes

    def _ce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
        aux_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = 0.0
        if self.ce_weight:
            loss = loss + self.ce_weight * self._ce_loss(logits, targets)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if aux_logits is not None and aux_targets is not None and self.aux_weight > 0:
            if self.ce_weight:
                loss = loss + self.aux_weight * self.ce_weight * self._ce_loss(aux_logits, aux_targets)
            if self.dice_weight:
                loss = loss + self.aux_weight * self.dice_weight * self.dice(aux_logits, aux_targets)
        return loss
