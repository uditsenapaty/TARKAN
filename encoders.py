"""Frozen BERTweet and CLIP backbones; trainable projection layers."""
from __future__ import annotations
import torch
from torch import nn
from config import CONFIG

class TextEncoder(nn.Module):
    def __init__(self, model_id=None, hidden_dim=None, dropout=None, frozen=None):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_id or CONFIG.text_model_id,
                                             token=CONFIG.hf_token, torch_dtype=torch.float32)
        self.frozen = CONFIG.freeze_text if frozen is None else frozen
        self.bert.requires_grad_(not self.frozen)
        self.out_dim = self.bert.config.hidden_size
        d = hidden_dim or CONFIG.hidden_dim
        self.proj = nn.Identity() if self.out_dim == d else nn.Linear(self.out_dim, d)
        self.dropout = nn.Dropout(CONFIG.dropout if dropout is None else dropout)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.bert.eval()
        return self

    def forward(self, input_ids, attention_mask):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.dropout(self.proj(out))

class VisualEncoder(nn.Module):
    def __init__(self, model_id=None, hidden_dim=None, dropout=None, frozen=None):
        super().__init__()
        from transformers import CLIPVisionModel
        self.clip = CLIPVisionModel.from_pretrained(model_id or CONFIG.visual_model_id,
                                                   token=CONFIG.hf_token)
        self.frozen = CONFIG.freeze_visual if frozen is None else frozen
        self.clip.requires_grad_(not self.frozen)
        self.out_dim = self.clip.config.hidden_size
        self.proj = nn.Linear(self.out_dim, hidden_dim or CONFIG.hidden_dim)
        self.dropout = nn.Dropout(CONFIG.dropout if dropout is None else dropout)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.clip.eval()
        return self

    def forward(self, pixel_values):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            out = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        return self.dropout(self.proj(out))

