"""
src/engines/quantized_linear.py
--------------------------------
Shared INT4 group-wise quantized Linear layer and replacement utilities.
Enables memory-efficient model loading on Kria KV260 ARM CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QuantizedLinear(nn.Module):
    """
    INT4 group-wise quantized Linear layer (inference).
    Uses pure PyTorch for compatibility with torch 2.0.1 and KV260 ARM CPU.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        w_bit: int = 4,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self.w_bit        = w_bit

        n_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("weight_int4", torch.zeros(out_features, n_groups * group_size, dtype=torch.int8))
        self.register_buffer("scale",       torch.ones(out_features, n_groups, dtype=torch.float32))
        self.register_buffer("act_scale",   torch.ones(in_features, dtype=torch.float32))
        if has_bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_features, dtype=torch.float32)))
        else:
            self.bias = None

    def dequantize_weight(self) -> torch.Tensor:
        out_f = self.out_features
        n_groups = (self.in_features + self.group_size - 1) // self.group_size

        # (out_f, n_groups, group_size)
        W = self.weight_int4.float().view(out_f, n_groups, self.group_size)
        # Dequantize: multiply by per-group scale
        W = W * self.scale.view(out_f, n_groups, 1)
        # Flatten and crop to in_features
        W = W.view(out_f, -1)[:, : self.in_features]
        # Undo activation-aware scaling
        W = W / self.act_scale.unsqueeze(0)
        return W

    @property
    def weight(self) -> torch.Tensor:
        """
        Expose a .weight attribute (dequantized FP32) for compatibility with
        framework code that accesses layer.weight directly (e.g. open_clip's
        get_cast_dtype() calls self.mlp.c_fc.weight.dtype).
        """
        return self.dequantize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.dequantize_weight().to(x.dtype)
        return F.linear(x, W, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        act_scales: torch.Tensor,
        group_size: int = 128,
        w_bit: int = 4,
    ) -> "QuantizedLinear":
        """Quantize a standard nn.Linear."""
        W = linear.weight.detach().float()
        out_f, in_f = W.shape

        a_scale = act_scales.float().clamp(min=1e-5)
        W_scaled = W * a_scale.unsqueeze(0)

        max_int = 2 ** (w_bit - 1) - 1
        n_groups = (in_f + group_size - 1) // group_size
        pad = n_groups * group_size - in_f
        if pad > 0:
            W_scaled = F.pad(W_scaled, (0, pad))

        W_g = W_scaled.view(out_f, n_groups, group_size)
        g_max = W_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = g_max / max_int
        W_q = (W_g / scale).round().clamp(-max_int, max_int).to(torch.int8)

        q_linear = cls(in_f, out_f, group_size, w_bit, has_bias=linear.bias is not None)
        q_linear.weight_int4.copy_(W_q.view(out_f, -1))
        q_linear.scale.copy_(scale.squeeze(-1))
        q_linear.act_scale.copy_(a_scale)
        if linear.bias is not None:
            q_linear.bias.data.copy_(linear.bias.data)
        return q_linear

def replace_with_quantized_linear(
    module: nn.Module,
    state_dict: dict,
    prefix: str = "",
    group_size: int = 128,
    w_bit: int = 4,
) -> None:
    """
    Recursively replaces nn.Linear with QuantizedLinear in module if matching
    keys exist in state_dict.
    """
    for name, child in list(module.named_children()):
        full_name = f"{prefix}{name}"
        weight_int4_key = f"{full_name}.weight_int4"
        if isinstance(child, nn.Linear) and weight_int4_key in state_dict:
            has_bias = f"{full_name}.bias" in state_dict
            q_linear = QuantizedLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                group_size=group_size,
                w_bit=w_bit,
                has_bias=has_bias,
            )
            setattr(module, name, q_linear)
        else:
            replace_with_quantized_linear(child, state_dict, f"{full_name}.", group_size, w_bit)
