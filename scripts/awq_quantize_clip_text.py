"""
scripts/awq_quantize_clip_text.py
-----------------------------------
AWQ-style INT4 group-wise quantization cho CLIP text encoder (ViT-B/16, open_clip).

Phiên bản này KHÔNG phụ thuộc autoawq internals — dùng PyTorch thuần
để tương thích với torch 2.0.1 và bất kỳ phiên bản transformers nào.

Algorithm (AWQ-style weight-only quantization):
  1. Load CLIP model.transformer (CLIPTextTransformer, 12L×512d)
  2. Hook forward passes trên calibration texts → collect per-channel act scales
  3. Compute AWQ per-channel scale: s = mean(|act|)^α  (α=0.5, per AWQ paper)
  4. Group-wise INT4 symmetric quantization:
       - w_q = clamp(round(w_scaled / scale_g), -7, 7)  per group_size=128
       - scale_g = max(|w_scaled|) / 7
  5. Lưu (w_int4, scale_g, act_scale) vào clip_text_awq.pt

Chạy trên PC (không cần Docker):
    source venv/bin/activate
    python scripts/awq_quantize_clip_text.py

Output:
    exported_models/clip_text_awq_int4/clip_text_awq.pt  (~63 MB)

Load trong pc_engine.py / kria pipeline:
    engine_cfg["awq_text_path"] = "exported_models/clip_text_awq_int4"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME   = "ViT-B-16"
PRETRAINED   = "openai"
DEFAULT_OUT  = "exported_models/clip_text_awq_int4"
W_BIT        = 4
Q_GROUP_SIZE = 128
AWQ_ALPHA    = 0.5   # activation scale exponent

# Calibration texts — surveillance domain
_CALIB_TEXTS: List[str] = [
    "a person walking near the entrance",
    "a red car parked in the lot",
    "security footage of an empty corridor",
    "a man carrying a bag",
    "a woman climbing a fence",
    "a crowd of people at the exit",
    "a delivery truck passing the gate",
    "someone running in the hallway",
    "a dog on a leash",
    "bicycle parked near the wall",
    "a person in a yellow jacket",
    "two people talking near a door",
    "a fire hydrant on the sidewalk",
    "a child playing in the yard",
    "a car accident on the road",
    "a phone on the table",
    "a backpack left unattended",
    "a person looking at the camera",
    "a motorcycle near the curb",
    "an empty parking lot at night",
] * 50   # 1000 samples


import sys
sys.path.append(".")
from src.engines.quantized_linear import QuantizedLinear


# ---------------------------------------------------------------------------
# Activation collection via hooks
# ---------------------------------------------------------------------------

def _collect_act_scales(
    model: nn.Module,
    token_inputs: torch.Tensor,    # (N, context_length)
    encode_fn,                      # callable(tokens) → embeddings
    n_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    act_dict: Dict[str, List[torch.Tensor]] = {}
    handles  = []

    def _hook(name):
        def _fn(module, inp, out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])
            ch_max = x.abs().amax(dim=0)
            act_dict.setdefault(name, []).append(ch_max)
        return _fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(_hook(name)))

    model.eval()
    with torch.no_grad():
        bs = 32
        for i in range(0, min(n_samples, len(token_inputs)), bs):
            try:
                encode_fn(token_inputs[i : i + bs])
            except Exception as exc:
                log.warning("encode_fn failed at batch %d: %s", i, exc)
                break

    for h in handles:
        h.remove()

    return {
        k: torch.stack(v, dim=0).amax(dim=0)
        for k, v in act_dict.items()
        if v
    }


def _get_parent_and_attr(root: nn.Module, full_name: str):
    parts  = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None
        parent = getattr(parent, p)
    return parent, parts[-1]


# ---------------------------------------------------------------------------
# Main quantization function
# ---------------------------------------------------------------------------

def run_awq_clip_text(
    model_name:   str,
    pretrained:   str,
    output_dir:   str,
    w_bit:        int   = W_BIT,
    group_size:   int   = Q_GROUP_SIZE,
    awq_alpha:    float = AWQ_ALPHA,
    n_calib:      int   = 200,
) -> None:
    try:
        import open_clip
    except ImportError:
        raise ImportError("Install: pip install open-clip-torch")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading CLIP %s (%s) ...", model_name, pretrained)
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    text_transformer = model.transformer         # the actual transformer backbone
    text_projection  = model.text_projection     # (D → D) projection

    log.info(
        "Text transformer: %d M params",
        sum(p.numel() for p in text_transformer.parameters()) // 1_000_000,
    )

    # ── 1. Tokenize calibration texts ────────────────────────────────────
    log.info("Tokenizing %d calibration texts ...", min(n_calib, len(_CALIB_TEXTS)))
    tokens = tokenizer(_CALIB_TEXTS[:n_calib])   # (N, context_length)

    # ── 2. Collect activation scales (hook on text_transformer) ──────────
    log.info("Collecting activation statistics (%d samples) ...", n_calib)
    act_scales = _collect_act_scales(
        text_transformer,
        tokens,
        encode_fn=lambda t: model.encode_text(t),
        n_samples=n_calib,
    )
    log.info("Collected stats for %d Linear layers.", len(act_scales))

    # Apply AWQ alpha
    act_scales_awq = {
        k: v.pow(awq_alpha).clamp(min=1e-5)
        for k, v in act_scales.items()
    }

    # ── 3. Quantize all Linear layers in text_transformer ────────────────
    log.info("Quantizing %d Linear layers (INT%d, group=%d) ...",
             len(act_scales_awq), w_bit, group_size)
    n_quantized = 0
    for name, mod in list(text_transformer.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        a_scale = act_scales_awq.get(name, torch.ones(mod.weight.shape[1]))
        parent, attr = _get_parent_and_attr(text_transformer, name)
        if parent is None:
            continue
        setattr(parent, attr, QuantizedLinear.from_linear(mod, a_scale, group_size, w_bit))
        n_quantized += 1
    log.info("Quantized %d layers.", n_quantized)

    # ── 4. Validate ───────────────────────────────────────────────────────
    log.info("Validating embedding quality ...")
    _validate_quality(model, tokenizer, text_transformer, text_projection)

    # ── 5. Save ───────────────────────────────────────────────────────────
    save_path = out / "clip_text_awq.pt"
    torch.save(
        {
            "text_transformer_state_dict": text_transformer.state_dict(),
            "text_projection":             text_projection.detach(),
            "quant_config": {
                "w_bit":      w_bit,
                "group_size": group_size,
                "awq_alpha":  awq_alpha,
                "model_name": model_name,
                "pretrained": pretrained,
            },
            "vocab_size":     getattr(model, "vocab_size", 49408),
            "context_length": model.context_length,
        },
        str(save_path),
    )
    log.info("Saved: %s", save_path)
    _print_size_report(out, fp32_mb=250)


def _validate_quality(model, tokenizer, quantized_transformer, text_projection) -> None:
    """Compare FP32 vs quantized CLIP text embeddings."""
    test_texts = [
        "a person walking",
        "red car near the gate",
        "security camera footage",
        "person climbing fence",
        "delivery truck at gate",
    ]
    tokens = tokenizer(test_texts)

    # FP32 embeddings using quantized_transformer (now has QuantizedLinear layers)
    with torch.no_grad():
        q_emb = model.encode_text(tokens)
        q_emb = F.normalize(q_emb.float(), dim=-1)

    # Reload FP32 reference
    try:
        import open_clip
        fp32_model, _, _ = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        fp32_model.eval()
        with torch.no_grad():
            fp32_emb = fp32_model.encode_text(tokens)
            fp32_emb = F.normalize(fp32_emb.float(), dim=-1)

        cos = (fp32_emb * q_emb).sum(dim=-1)
        mean_cos = float(cos.mean())
        log.info("=" * 55)
        for txt, c in zip(test_texts, cos.tolist()):
            log.info("  cosine = %.4f  '%s'", c, txt)
        log.info("Mean cosine (FP32 vs INT4-AWQ): %.4f", mean_cos)
        if mean_cos > 0.97:
            log.info("✅ Quality PASS (> 0.97)")
        elif mean_cos > 0.93:
            log.warning("⚠️  Quality MARGINAL (0.93–0.97) — acceptable for Kria ARM")
        else:
            log.error("❌ Quality FAIL (< 0.93) — try --group_size 256 or --w_bit 8")
        log.info("=" * 55)
    except Exception as exc:
        log.warning("Could not load FP32 reference for comparison: %s", exc)


def _print_size_report(out: Path, fp32_mb: float) -> None:
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
        description="AWQ-style INT4 quantization for CLIP text encoder"
    )
    p.add_argument("--model_name",  default=MODEL_NAME,  help="open_clip model name")
    p.add_argument("--pretrained",  default=PRETRAINED,  help="open_clip pretrained tag")
    p.add_argument("--output",      default=DEFAULT_OUT, help="Output directory")
    p.add_argument("--w_bit",       type=int,   default=W_BIT)
    p.add_argument("--group_size",  type=int,   default=Q_GROUP_SIZE)
    p.add_argument("--awq_alpha",   type=float, default=AWQ_ALPHA)
    p.add_argument("--n_calib",     type=int,   default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_awq_clip_text(
        model_name  = args.model_name,
        pretrained  = args.pretrained,
        output_dir  = args.output,
        w_bit       = args.w_bit,
        group_size  = args.group_size,
        awq_alpha   = args.awq_alpha,
        n_calib     = args.n_calib,
    )
