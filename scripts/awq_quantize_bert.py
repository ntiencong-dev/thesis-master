"""
scripts/awq_quantize_bert.py
-----------------------------
AWQ-style INT4 group-wise quantization cho BLIP-1 BERT text encoder.

Phiên bản này KHÔNG phụ thuộc autoawq internals — dùng PyTorch thuần
để tương thích với torch 2.0.1 và bất kỳ phiên bản transformers nào.

Algorithm (AWQ-style weight-only quantization):
  1. Load BLIP-1 → extract text_encoder (BlipTextModel, BERT-base 12L×768d)
  2. Hook forward passes trên calibration texts → collect per-channel act scales
  3. Compute AWQ per-channel scale: s = mean(|act|)^α  (α=0.5, giống bài báo gốc)
  4. Group-wise INT4 symmetric quantization:
       - Chia weight rows thành groups kích thước group_size=128
       - w_q = clamp(round(w_scaled / scale_g), -7, 7)  (INT4 symmetric)
       - scale_g = max(|w_scaled|) / 7  (per group)
  5. Lưu (w_int4, scale_g, act_scale) vào blip1_text_awq.pt

Chạy trên PC (không cần Docker):
    source venv/bin/activate
    python scripts/awq_quantize_bert.py

Hoặc dùng master script:
    python scripts/quantize_all.py --step awq --option c

Output:
    exported_models/blip1_bert_awq_int4/blip1_text_awq.pt  (~110 MB)
    exported_models/blip1_bert_awq_int4/tokenizer_config.json

Load trong blip1_engine.py:
    engine_cfg["awq_bert_path"] = "exported_models/blip1_bert_awq_int4"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME   = "Salesforce/blip-itm-base-coco"
DEFAULT_OUT  = "exported_models/blip1_bert_awq_int4"
W_BIT        = 4
Q_GROUP_SIZE = 128
AWQ_ALPHA    = 0.5   # activation scale exponent (0.5 = geometric mean, per AWQ paper)

# Calibration: surveillance-domain texts matching deployment queries
_CALIB_TEXTS: List[str] = [
    "a person walking near the entrance",
    "security camera footage of an empty corridor",
    "a man carrying a bag",
    "a woman climbing a fence",
    "a crowd of people at the exit",
    "a delivery truck passing the gate",
    "someone running in the hallway",
    "a backpack left unattended",
    "a person looking at the camera",
    "a motorcycle near the curb",
    "an empty parking lot at night",
    "a person opening a door",
    "a red car parked near the fence",
    "two people talking near a door",
    "a person sitting on a bench",
    "a child playing in the yard",
    "surveillance video showing a theft",
    "a person in a yellow jacket",
    "a bicycle parked near the wall",
    "a guard standing at the gate",
] * 50   # 1000 samples — enough for stable activation stats


import sys
sys.path.append(".")
from src.engines.quantized_linear import QuantizedLinear


# ---------------------------------------------------------------------------
# Activation collection via hooks
# ---------------------------------------------------------------------------

def _collect_act_scales(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    target_module_type=nn.Linear,
    n_samples: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    Collect per-input-channel activation magnitudes for all target layers.

    Returns: { module_name → Tensor(in_features,) }
    """
    act_dict: Dict[str, List[torch.Tensor]] = {}
    handles  = []

    def _hook(name):
        def _fn(module, inp, out):
            # inp[0]: (B, seq, in_features) or (B, in_features)
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])
            # Running max of |activation|
            ch_max = x.abs().amax(dim=0)   # (in_features,)
            if name not in act_dict:
                act_dict[name] = [ch_max]
            else:
                act_dict[name].append(ch_max)
        return _fn

    for name, mod in model.named_modules():
        if isinstance(mod, target_module_type):
            handles.append(mod.register_forward_hook(_hook(name)))

    model.eval()
    with torch.no_grad():
        # Run in small batches
        bs = min(16, n_samples)
        input_ids      = inputs["input_ids"][:n_samples]
        attention_mask = inputs["attention_mask"][:n_samples]
        for i in range(0, len(input_ids), bs):
            batch_ids  = input_ids[i : i + bs]
            batch_mask = attention_mask[i : i + bs]
            try:
                model(input_ids=batch_ids, attention_mask=batch_mask)
            except Exception as exc:
                log.warning("Forward pass failed at batch %d: %s", i, exc)
                break

    for h in handles:
        h.remove()

    # Average max across batches → stable estimate
    return {
        name: torch.stack(tensors, dim=0).amax(dim=0)  # (in_features,)
        for name, tensors in act_dict.items()
        if tensors
    }


# ---------------------------------------------------------------------------
# Main quantization function
# ---------------------------------------------------------------------------

def quantize_blip1_bert(
    model_name:  str,
    output_dir:  str,
    w_bit:       int   = W_BIT,
    group_size:  int   = Q_GROUP_SIZE,
    awq_alpha:   float = AWQ_ALPHA,
    n_calib:     int   = 200,
) -> None:
    """
    Quantize BLIP-1 BERT text encoder with AWQ-style INT4 group quantization.
    """
    try:
        from transformers import BlipForImageTextRetrieval, BlipProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers not available. Install:\n"
            "  pip install 'transformers>=4.27,<=4.51.3'\n"
            f"Error: {exc}"
        ) from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading BLIP-1: %s", model_name)
    processor = BlipProcessor.from_pretrained(model_name)
    blip      = BlipForImageTextRetrieval.from_pretrained(model_name)
    blip.eval()

    text_encoder = blip.text_encoder
    text_proj    = blip.text_proj if hasattr(blip, "text_proj") else None
    log.info(
        "Text encoder: %d layers × %d-dim  (%d M params)",
        text_encoder.config.num_hidden_layers,
        text_encoder.config.hidden_size,
        sum(p.numel() for p in text_encoder.parameters()) // 1_000_000,
    )

    # ── 1. Tokenize calibration texts ────────────────────────────────────
    log.info("Tokenizing %d calibration texts ...", len(_CALIB_TEXTS[:n_calib]))
    inputs = processor(
        text=_CALIB_TEXTS[:n_calib],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    # ── 2. Collect activation scales ─────────────────────────────────────
    log.info("Collecting activation statistics (%d samples) ...", n_calib)
    act_scales = _collect_act_scales(
        text_encoder,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
        n_samples=min(n_calib, len(_CALIB_TEXTS)),
    )
    log.info("Collected stats for %d Linear layers.", len(act_scales))

    # Apply AWQ alpha: s = |act|^alpha (0.5 = geometric mean, per paper)
    act_scales_awq = {
        k: v.pow(awq_alpha).clamp(min=1e-5)
        for k, v in act_scales.items()
    }

    # ── 3. Quantize all Linear layers ────────────────────────────────────
    log.info("Quantizing %d Linear layers (INT%d, group=%d) ...",
             len(act_scales_awq), w_bit, group_size)

    n_quantized = 0
    total_linear = _count_linear(text_encoder)   # count BEFORE replacement
    for name, mod in list(text_encoder.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        a_scale = act_scales_awq.get(name)
        if a_scale is None:
            # No activation data → use uniform scale (RTN fallback)
            a_scale = torch.ones(mod.weight.shape[1])
        # Replace in-place: set attribute on parent module
        parent, attr = _get_parent_and_attr(text_encoder, name)
        if parent is None:
            continue
        q_linear = QuantizedLinear.from_linear(mod, a_scale, group_size, w_bit)
        setattr(parent, attr, q_linear)
        n_quantized += 1

    log.info("Quantized %d / %d linear layers.", n_quantized, total_linear)

    # ── 4. Validate quality ───────────────────────────────────────────────
    log.info("Validating quantized encoder quality ...")
    _validate_quality(
        blip, processor, text_encoder,
        test_texts=[
            "a person walking",
            "red car near the gate",
            "security camera footage",
            "person climbing fence",
            "delivery truck at gate",
        ],
    )

    # ── 5. Save ───────────────────────────────────────────────────────────
    save_path = out / "blip1_text_awq.pt"
    log.info("Saving quantized weights → %s", save_path)
    torch.save(
        {
            "text_encoder_state_dict": text_encoder.state_dict(),
            "text_proj_state_dict":    text_proj.state_dict() if text_proj else None,
            "quant_config": {
                "w_bit":      w_bit,
                "group_size": group_size,
                "awq_alpha":  awq_alpha,
                "model_name": model_name,
            },
            "act_scales":  {k: v.cpu() for k, v in act_scales.items()},
        },
        str(save_path),
    )
    processor.save_pretrained(str(out))

    _print_size_report(out, fp32_mb=440)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_parent_and_attr(root: nn.Module, full_name: str):
    """Return (parent_module, attr_name) for a dotted path."""
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        if hasattr(parent, p):
            parent = getattr(parent, p)
        else:
            return None, None
    return parent, parts[-1]


def _count_linear(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.Linear))


def _validate_quality(
    blip, processor, quantized_encoder, test_texts: List[str]
) -> None:
    """Compare ITC text embeddings: FP32 vs quantized."""
    # Reload FP32 text encoder for comparison
    from transformers import BlipForImageTextRetrieval
    fp32_model = BlipForImageTextRetrieval.from_pretrained(blip.config.name_or_path
                                                           if hasattr(blip.config, "name_or_path")
                                                           else MODEL_NAME)
    fp32_model.eval()

    inputs = processor(
        text=test_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=64,
    )

    with torch.no_grad():
        # FP32 reference
        fp32_out = fp32_model.text_encoder(**inputs).last_hidden_state[:, 0, :]
        fp32_proj = fp32_model.text_proj(fp32_out)
        fp32_emb  = F.normalize(fp32_proj.float(), dim=-1)

        # Quantized
        q_out  = quantized_encoder(**inputs).last_hidden_state[:, 0, :]
        if hasattr(blip, "text_proj"):
            q_proj = blip.text_proj(q_out)
        else:
            q_proj = q_out
        q_emb = F.normalize(q_proj.float(), dim=-1)

    cos = (fp32_emb * q_emb).sum(dim=-1)
    mean_cos = float(cos.mean())
    log.info("=" * 55)
    for txt, c in zip(test_texts, cos.tolist()):
        log.info("  cosine = %.4f  '%s'", c, txt)
    log.info("Mean cosine (FP32 vs INT4-AWQ): %.4f", mean_cos)
    if mean_cos > 0.97:
        log.info("✅ Quality PASS (> 0.97) — INT4 AWQ preserves embedding quality")
    elif mean_cos > 0.93:
        log.warning("⚠️  Quality MARGINAL (0.93–0.97) — acceptable for ARM deployment")
    else:
        log.error("❌ Quality FAIL (< 0.93) — try --group_size 256 or --w_bit 8")
    log.info("=" * 55)


def _print_size_report(out: Path, fp32_mb: float = 440) -> None:
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    size_mb = total / 1e6
    log.info("Output: %s", out)
    log.info("Total size: %.1f MB  (FP32: %.0f MB → %.1f%% reduction)",
             size_mb, fp32_mb, (1 - size_mb / fp32_mb) * 100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="AWQ-style INT4 quantization for BLIP-1 BERT text encoder"
    )
    p.add_argument("--model",       default=MODEL_NAME,   help="HuggingFace model id")
    p.add_argument("--output",      default=DEFAULT_OUT,  help="Output directory")
    p.add_argument("--w_bit",       type=int, default=W_BIT)
    p.add_argument("--group_size",  type=int, default=Q_GROUP_SIZE)
    p.add_argument("--awq_alpha",   type=float, default=AWQ_ALPHA,
                   help="Activation scale exponent α (0.5=AWQ paper default)")
    p.add_argument("--n_calib",     type=int, default=200,
                   help="Number of calibration texts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    quantize_blip1_bert(
        model_name  = args.model,
        output_dir  = args.output,
        w_bit       = args.w_bit,
        group_size  = args.group_size,
        awq_alpha   = args.awq_alpha,
        n_calib     = args.n_calib,
    )
