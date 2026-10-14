from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


CONCEPT_GROUPS: Dict[str, List[int]] = {
    "lv_size": [0, 1, 2],
    "lv_function": [3, 4],
    "lv_mass": [5],
    "rv_size": [6, 7, 8],
    "rv_function": [9],
    "wt_regional": list(range(10, 26)),
    "wt_global": [26],
}


class ConceptQueryReadout(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.group_names = list(CONCEPT_GROUPS.keys())
        self.num_queries = len(self.group_names)
        self.queries = nn.Parameter(torch.randn(self.num_queries, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, len(indices)),
            )
            for name, indices in CONCEPT_GROUPS.items()
        })

    def forward(self, patch_tokens: torch.Tensor):
        batch_size = patch_tokens.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        concept_tokens, attn_weights = self.attn(queries, patch_tokens, patch_tokens, need_weights=True)
        concept_tokens = self.norm(concept_tokens)
        preds = []
        group_outputs = {}
        for query_idx, name in enumerate(self.group_names):
            out = self.heads[name](concept_tokens[:, query_idx])
            group_outputs[name] = out
            preds.append(out)
        pred_27 = torch.cat(preds, dim=-1)
        return concept_tokens, pred_27, group_outputs, attn_weights


def get_ecg_patch_tokens(ecg_model) -> torch.Tensor:
    if not hasattr(ecg_model, "latent"):
        raise RuntimeError("ECG model has no `latent`; call ECG_model(ecg) before concept readout.")
    latent = ecg_model.latent
    if latent.ndim == 4:
        return latent.reshape(latent.size(0), -1, latent.size(-1))
    if latent.ndim == 3:
        return latent
    raise RuntimeError(f"Unsupported ECG latent shape: {tuple(latent.shape)}")
