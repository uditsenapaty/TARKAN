"""Unified BIO softmax head (paper Eq. 20); tagging loss is Eq. 21.

The separate linear ASC head in models.py is auxiliary training supervision only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import CONFIG, NUM_BIO_TAGS


class BIOTaggingHead(nn.Module):
    """Token-level unified BIO sentiment tagger (Eq. 20).

    Consumes the KAN-fused token representation h̃ of shape [..., d] and emits one
    distribution over the 7 unified BIO tags per token.
    """

    def __init__(self, d: int = None, num_tags: int = NUM_BIO_TAGS, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_tags)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.drop(h))  # [..., num_tags]

