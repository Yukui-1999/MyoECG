from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, base_layer.out_features, bias=False)
        self.reset_lora_parameters()

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        lora_update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return self.base_layer(x) + lora_update


def _resolve_layers(layers, depth: int) -> List[int]:
    if layers is None:
        return list(range(depth))
    if isinstance(layers, str):
        text = layers.strip().lower()
        if text == "all":
            return list(range(depth))
        if text.startswith("last"):
            n = int(text.replace("last", ""))
            if n <= 0:
                raise ValueError(f"Invalid LoRA layer selector: {layers}")
            return list(range(max(0, depth - n), depth))
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    if isinstance(layers, Iterable):
        return [int(item) for item in layers]
    raise TypeError(f"Unsupported LoRA layers selector: {layers}")


def _as_list(value) -> List[str]:
    if value is None:
        return ["to_qkv"]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _replace_linear(parent: nn.Module, name: str, r: int, alpha: float, dropout: float):
    current = getattr(parent, name)
    if isinstance(current, LoRALinear):
        return False
    setattr(parent, name, LoRALinear(current, r=r, alpha=alpha, dropout=dropout))
    return True


def _replace_sequential_linear(parent: nn.Sequential, index: int, r: int, alpha: float, dropout: float):
    current = parent[index]
    if isinstance(current, LoRALinear):
        return False
    if not isinstance(current, nn.Linear):
        raise TypeError(f"Expected nn.Linear at sequential index {index}, got {type(current)}")
    parent[index] = LoRALinear(current, r=r, alpha=alpha, dropout=dropout)
    return True


def apply_lora_to_stmem(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: Sequence[str] | str = ("to_qkv",),
    layers: Sequence[int] | str = "last6",
) -> List[str]:
    depth = getattr(model, "depth", None)
    if depth is None:
        raise ValueError("LoRA STMEM wrapper expects a model with a `depth` attribute")
    selected_layers = _resolve_layers(layers, depth)
    targets = set(_as_list(target_modules))
    replaced = []

    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= depth:
            raise ValueError(f"LoRA layer index {layer_idx} is outside depth {depth}")
        block = getattr(model, f"block{layer_idx}")
        attn = block.attn.fn
        ff = block.ff.fn.net

        if "to_qkv" in targets and _replace_linear(attn, "to_qkv", r, alpha, dropout):
            replaced.append(f"block{layer_idx}.attn.fn.to_qkv")
        if "to_out" in targets and isinstance(attn.to_out, nn.Sequential):
            if _replace_sequential_linear(attn.to_out, 0, r, alpha, dropout):
                replaced.append(f"block{layer_idx}.attn.fn.to_out.0")
        if "mlp_fc1" in targets:
            if _replace_sequential_linear(ff, 0, r, alpha, dropout):
                replaced.append(f"block{layer_idx}.ff.fn.net.0")
        if "mlp_fc2" in targets:
            if _replace_sequential_linear(ff, 3, r, alpha, dropout):
                replaced.append(f"block{layer_idx}.ff.fn.net.3")

    print(f"Applied LoRA to {len(replaced)} modules: {replaced}")
    return replaced


def apply_lora_from_config(model: nn.Module, lora_config: dict | None):
    if not lora_config or not lora_config.get("enabled", False):
        return []
    return apply_lora_to_stmem(
        model,
        r=int(lora_config.get("r", lora_config.get("rank", 8))),
        alpha=float(lora_config.get("alpha", 16.0)),
        dropout=float(lora_config.get("dropout", 0.1)),
        target_modules=lora_config.get("target_modules", ["to_qkv"]),
        layers=lora_config.get("layers", "last6"),
    )


def is_lora_parameter_name(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def lora_parameter_summary(model: nn.Module) -> dict:
    lora_params = 0
    base_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        if is_lora_parameter_name(name):
            lora_params += n
        else:
            base_params += n
        if param.requires_grad:
            trainable_params += n
    return {
        "lora_params": lora_params,
        "base_params": base_params,
        "total_params": lora_params + base_params,
        "trainable_params": trainable_params,
        "lora_param_ratio": lora_params / max(lora_params + base_params, 1),
    }
