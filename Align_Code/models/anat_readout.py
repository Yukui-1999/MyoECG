from __future__ import annotations

import torch
import torch.nn as nn


class AnatomyQueryReadout(nn.Module):
    def __init__(self, embed_dim: int = 768, num_regions: int = 4, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_regions, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: torch.Tensor):
        batch_size = patch_tokens.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        anatomy_tokens, attn_weights = self.attn(queries, patch_tokens, patch_tokens, need_weights=True)
        return self.norm(anatomy_tokens), attn_weights
