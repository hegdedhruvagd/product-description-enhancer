"""Small explicit LoRA implementation for PyTorch linear/Transformers Conv1D layers."""
from __future__ import annotations

import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if rank < 1 or alpha < 1: raise ValueError("LoRA rank and alpha must be positive.")
        weight = getattr(base_layer, "weight", None)
        if weight is None or weight.ndim != 2: raise TypeError("LoRA target must expose a 2D weight.")
        self.base_layer = base_layer
        # torch.nn.Linear stores [out,in]; Hugging Face GPT-2 Conv1D stores [in,out].
        self.fan_in_fan_out = not isinstance(base_layer, nn.Linear)
        shape = tuple(weight.shape)
        self.in_features, self.out_features = (shape if self.fan_in_fan_out else (shape[1], shape[0]))
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base_layer.parameters(): parameter.requires_grad_(False)

    def delta_weight(self):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return delta.T if self.fan_in_fan_out else delta

    def forward(self, x):
        return self.base_layer(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merged_layer(self):
        import copy
        merged = copy.deepcopy(self.base_layer)
        with torch.no_grad(): merged.weight.add_(self.delta_weight().to(merged.weight.dtype))
        return merged


def inject_lora(model: nn.Module, target_names=("c_attn",), rank=8, alpha=16, dropout=0.0):
    """Wrap matching projection modules in place; freeze all original model parameters."""
    for param in model.parameters(): param.requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules() if name.split(".")[-1] in target_names]
    if not targets: raise ValueError(f"No LoRA target modules found for {target_names}.")
    for name, module in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    return model


def parameter_report(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total, "percentage": 100 * trainable / max(total, 1)}


def adapter_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def merge_lora(model):
    """Replace each adapter wrapper with its merged base layer, in place."""
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name, wrapper in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, wrapper.merged_layer())
    return model
