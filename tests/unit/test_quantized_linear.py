"""
tests/unit/test_quantized_linear.py
-------------------------------------
Unit tests for the custom QuantizedLinear layer and replace_with_quantized_linear utility.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engines.quantized_linear import QuantizedLinear, replace_with_quantized_linear

pytestmark = pytest.mark.unit


def test_QL01_quantized_linear_from_linear():
    """Verify QuantizedLinear initialization from standard nn.Linear."""
    in_f = 256
    out_f = 128
    group_size = 64
    w_bit = 4

    linear = nn.Linear(in_f, out_f, bias=True)
    # Generate some mock activation scales
    act_scales = torch.ones(in_f) * 1.5

    q_linear = QuantizedLinear.from_linear(
        linear, act_scales, group_size=group_size, w_bit=w_bit
    )

    assert q_linear.in_features == in_f
    assert q_linear.out_features == out_f
    assert q_linear.group_size == group_size
    assert q_linear.w_bit == w_bit

    # Check buffers exist and have correct shapes
    # n_groups = (256 + 64 - 1) // 64 = 4
    n_groups = 4
    assert q_linear.weight_int4.shape == (out_f, n_groups * group_size)
    assert q_linear.scale.shape == (out_f, n_groups)
    assert q_linear.act_scale.shape == (in_f,)
    assert q_linear.bias is not None
    assert q_linear.bias.shape == (out_f,)

    # Verify weight values are within INT4 symmetric range [-7, 7]
    assert int(q_linear.weight_int4.min()) >= -7
    assert int(q_linear.weight_int4.max()) <= 7


def test_QL02_quantized_linear_forward():
    """Verify that QuantizedLinear forward pass produces expected output shape and values."""
    in_f = 64
    out_f = 32
    group_size = 32
    w_bit = 4

    linear = nn.Linear(in_f, out_f, bias=False)
    act_scales = torch.ones(in_f)

    q_linear = QuantizedLinear.from_linear(
        linear, act_scales, group_size=group_size, w_bit=w_bit
    )

    # Input tensor
    x = torch.randn(5, in_f)
    out = q_linear(x)

    assert out.shape == (5, out_f)

    # Manual dequantization for reference verification
    W_dequant = q_linear.dequantize_weight()
    expected_out = F.linear(x, W_dequant, None)

    assert torch.allclose(out, expected_out, atol=1e-5)


def test_QL03_replace_with_quantized_linear():
    """Verify replace_with_quantized_linear recursively swaps nn.Linear layers."""
    class MockSubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 30, bias=False)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = MockSubModel()
            self.fc3 = nn.Linear(30, 40)

    model = MockModel()

    # Create mock state dict containing quantized keys for fc1 and fc3 (but not fc2)
    state_dict = {
        "sub.fc1.weight_int4": torch.zeros(20, 128, dtype=torch.int8),
        "sub.fc1.scale": torch.ones(20, 1, dtype=torch.float32),
        "sub.fc1.act_scale": torch.ones(10, dtype=torch.float32),
        "sub.fc1.bias": torch.zeros(20, dtype=torch.float32),
        "fc3.weight_int4": torch.zeros(40, 128, dtype=torch.int8),
        "fc3.scale": torch.ones(40, 1, dtype=torch.float32),
        "fc3.act_scale": torch.ones(30, dtype=torch.float32),
        "fc3.bias": torch.zeros(40, dtype=torch.float32),
    }

    replace_with_quantized_linear(model, state_dict, group_size=128, w_bit=4)

    # Check fc1 and fc3 are replaced with QuantizedLinear
    assert isinstance(model.sub.fc1, QuantizedLinear)
    assert isinstance(model.fc3, QuantizedLinear)

    # Check fc2 remains nn.Linear because it's not in the state dict
    assert isinstance(model.sub.fc2, nn.Linear)

    # Test load_state_dict works cleanly on the modified model structure
    model.load_state_dict(state_dict, strict=False)
