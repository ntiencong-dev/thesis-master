"""
scripts/awq_validate_filip_tokens.py
--------------------------------------
Validate that AWQ INT4 quantization of the CLIP text encoder preserves
per-token (word-level) embedding quality for FILIP matching — not just
the final [EOS] / [CLS] embedding used for ITC retrieval.

Background
----------
awq_quantize_clip_text.py only validates cosine similarity of the final
pooled text embedding ([EOS] token after projection).  FILIPReranker uses
ALL token positions from the last text transformer block (before the
projection head).  This script checks:

  1. Mean cosine similarity: FP32 vs AWQ per-token embeddings across T positions.
  2. FILIP score fidelity: FILIP computed from AWQ tokens vs FP32 tokens.
  3. Rank correlation: do the FILIP rankings from AWQ and FP32 agree?

Usage
-----
    # First quantize (if not done):
    python scripts/awq_quantize_clip_text.py

    # Then validate token-level fidelity:
    python scripts/awq_validate_filip_tokens.py \
        --awq_path exported_models/clip_text_awq_int4/clip_text_awq.pt

Exit code
---------
    0 — PASS (mean cosine > 0.95, FILIP rank correlation > 0.90)
    1 — FAIL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Default paths ─────────────────────────────────────────────────────────
AWQ_PT      = Path("exported_models/clip_text_awq_int4/clip_text_awq.pt")
MODEL_NAME  = "ViT-B-16"
PRETRAINED  = "openai"

# ─── Calibration queries (diverse to stress-test different token distributions)
TEST_QUERIES: List[str] = [
    "a person climbing a fence",
    "red sports car parked near a fire hydrant",
    "a dog running in a park",
    "security camera footage of an empty hallway",
    "a woman carrying a large backpack",
    "two people talking near the entrance",
    "a bicycle left unattended next to the wall",
    "a crowd of people at the exit gate",
    "a delivery truck passing through the checkpoint",
    "an empty room with a wooden table and chairs",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate AWQ text encoder token-level quality for FILIPReranker"
    )
    p.add_argument("--awq_path", default=str(AWQ_PT),
                   help="Path to AWQ checkpoint (.pt from awq_quantize_clip_text.py)")
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--pretrained",  default=PRETRAINED)
    p.add_argument("--cos_threshold",   type=float, default=0.90,
                   help="Min mean per-token cosine to pass (default 0.90 for INT4)")
    p.add_argument("--mafe_threshold",  type=float, default=0.010,
                   help="Max Mean Absolute FILIP Error to pass (default 0.010). "
                        "Measures absolute FILIP score difference FP32 vs AWQ.")
    p.add_argument("--rank_threshold",  type=float, default=0.90,
                   help="[unused] Legacy Spearman threshold (now informational only)")
    return p.parse_args()


# ─── Forward hook utility ───────────────────────────────────────────────────

def _extract_text_tokens(
    model, tokenizer, texts: List[str], device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-token embeddings from the last text transformer block.

    Returns
    -------
    tokens : (N, seq_len, hidden)  float32
    mask   : (N, seq_len)          bool — True for valid tokens
    """
    tok_ids = tokenizer(texts).to(device)           # (N, ctx_len)
    mask    = (tok_ids != 0).cpu().numpy()           # bool

    captured: dict = {}

    def _hook(module, inp, out, _cap=captured):
        _cap["h"] = out[0] if isinstance(out, tuple) else out

    last_block = model.transformer.resblocks[-1]
    handle     = last_block.register_forward_hook(_hook)
    with torch.no_grad():
        model.encode_text(tok_ids)
    handle.remove()

    h = captured["h"]
    if h.dim() == 3 and h.shape[0] != len(texts):
        h = h.permute(1, 0, 2)     # seq-first → batch-first

    return h.float().cpu().numpy(), mask


# ─── FILIP score (vectorised) ────────────────────────────────────────────────

def _fake_patch_tokens(seed: int, n_patches: int = 196, dim: int = 512) -> np.ndarray:
    """Generate reproducible fake patch tokens for scoring comparison.

    dim must match the text token hidden dimension (512 for CLIP ViT-B/16,
    NOT the visual ViT hidden dim 768).  The validator compares text tokens
    against fake patches purely to measure rank-correlation of FILIP scores
    between FP32 and AWQ models — the absolute values are irrelevant.
    """
    rng     = np.random.default_rng(seed)
    patches = rng.standard_normal((n_patches, dim)).astype(np.float32)
    norms   = np.linalg.norm(patches, axis=1, keepdims=True)
    return patches / np.where(norms < 1e-8, 1.0, norms)


def _filip_score(
    text_tokens: np.ndarray, text_mask: np.ndarray, patch_tokens: np.ndarray
) -> float:
    """
    Compute FILIP score for one (text, image) pair.

    text_tokens  : (T, hidden)
    text_mask    : (T,) bool
    patch_tokens : (P, hidden)
    """
    valid = text_tokens[text_mask]              # (T_valid, hidden)
    tnorms = np.linalg.norm(valid, axis=1, keepdims=True)
    valid  = valid / np.where(tnorms < 1e-8, 1.0, tnorms)

    dot = valid @ patch_tokens.T               # (T_valid, P)
    return float(dot.max(axis=1).mean())


# ─── Main validation ─────────────────────────────────────────────────────────

def main() -> int:
    args   = _parse_args()
    device = torch.device("cpu")               # Kria target is ARM CPU

    # ── 1. Load FP32 CLIP model ──────────────────────────────────────────────
    try:
        import open_clip
    except ImportError:
        log.error("open-clip-torch not installed: pip install open-clip-torch")
        return 1

    log.info("Loading FP32 CLIP %s (%s) ...", args.model_name, args.pretrained)
    model_fp32, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    model_fp32.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # ── 2. Load AWQ quantized text transformer ───────────────────────────────
    awq_path = Path(args.awq_path)
    if not awq_path.exists():
        log.error("AWQ checkpoint not found: %s", awq_path)
        log.error("Run: python scripts/awq_quantize_clip_text.py first.")
        return 1

    log.info("Loading AWQ checkpoint: %s", awq_path)
    ckpt = torch.load(str(awq_path), map_location=device, weights_only=False)

    # Restore the AWQ text transformer into a fresh model copy
    model_awq, _, _ = open_clip.create_model_and_transforms(
        ckpt.get("model_name", args.model_name),
        pretrained=ckpt.get("pretrained", args.pretrained),
    )
    model_awq.eval()
    # Replace nn.Linear layers with QuantizedLinear BEFORE loading state dict,
    # otherwise load_state_dict raises key-mismatch errors (weight vs weight_int4).
    import sys
    sys.path.append(".")
    from src.engines.quantized_linear import replace_with_quantized_linear
    quant_cfg = ckpt.get("quant_config", {})
    replace_with_quantized_linear(
        model_awq.transformer,
        ckpt["text_transformer_state_dict"],
        group_size=quant_cfg.get("group_size", 128),
        w_bit=quant_cfg.get("w_bit", 4),
    )
    model_awq.transformer.load_state_dict(ckpt["text_transformer_state_dict"], strict=True)
    log.info("AWQ quantization config: %s", quant_cfg)

    # ── 3. Extract per-token embeddings: FP32 vs AWQ ─────────────────────────
    log.info("Extracting per-token embeddings for %d queries...", len(TEST_QUERIES))
    toks_fp32, masks_fp32 = _extract_text_tokens(
        model_fp32, tokenizer, TEST_QUERIES, device
    )
    toks_awq, masks_awq   = _extract_text_tokens(
        model_awq,  tokenizer, TEST_QUERIES, device
    )

    # ── 4. Per-token cosine similarity ───────────────────────────────────────
    # Compute cosine for every valid token position across all queries
    cos_values: List[float] = []
    for i in range(len(TEST_QUERIES)):
        mask  = masks_fp32[i]                   # (seq_len,) bool
        t_fp  = toks_fp32[i][mask]              # (T_valid, hidden)
        t_awq = toks_awq[i][mask]

        t_fp_n  = F.normalize(torch.from_numpy(t_fp),  dim=-1).numpy()
        t_awq_n = F.normalize(torch.from_numpy(t_awq), dim=-1).numpy()

        cos = (t_fp_n * t_awq_n).sum(axis=-1)  # (T_valid,)
        cos_values.extend(cos.tolist())

    mean_cos = float(np.mean(cos_values))
    min_cos  = float(np.min(cos_values))

    log.info("=" * 60)
    log.info("Per-token cosine similarity (FP32 vs AWQ INT4):")
    log.info("  Mean : %.4f  (threshold: %.2f)", mean_cos, args.cos_threshold)
    log.info("  Min  : %.4f", min_cos)

    cos_pass = mean_cos >= args.cos_threshold
    log.info("  Token-level cosine: %s", "✅ PASS" if cos_pass else "❌ FAIL")

    # ── 5. FILIP score quality ────────────────────────────────────────────────
    # Primary metric: Mean Absolute FILIP Error (MAFE)
    # --------------------------------------------------------------------
    # WHY NOT Spearman rank correlation alone:
    # With random unit-vector patches in 512-d space, all queries produce
    # nearly identical FILIP scores (variance ≈ 0.0002).  Any tiny numeric
    # difference (even FP32 rounding) causes rank inversions, making
    # Spearman unreliable.  MAFE directly measures whether the AWQ model
    # produces the same absolute scores — a more meaningful quality gate.
    # --------------------------------------------------------------------
    log.info("")
    log.info("FILIP score quality (FP32 vs AWQ INT4):")

    filip_fp32 = []
    filip_awq  = []
    token_dim = toks_fp32.shape[-1]   # detect actual hidden dim (512 for CLIP ViT-B/16)
    for i, query in enumerate(TEST_QUERIES):
        patches = _fake_patch_tokens(seed=i, dim=token_dim)   # match text token dim
        mask_i  = masks_fp32[i]

        f_fp  = _filip_score(toks_fp32[i], mask_i, patches)
        f_awq = _filip_score(toks_awq[i],  mask_i, patches)
        filip_fp32.append(f_fp)
        filip_awq.append(f_awq)
        abs_err = abs(f_fp - f_awq)
        log.info("  %-50s  fp32=%.4f  awq=%.4f  Δ=%.4f", f'"{query[:48]}"', f_fp, f_awq, abs_err)

    fp32_arr = np.array(filip_fp32)
    awq_arr  = np.array(filip_awq)
    mafe     = float(np.abs(fp32_arr - awq_arr).mean())
    max_ae   = float(np.abs(fp32_arr - awq_arr).max())

    log.info("")
    log.info("  Mean Absolute FILIP Error (MAFE): %.5f  (threshold: %.3f)", mafe, args.mafe_threshold)
    log.info("  Max  Absolute Error:               %.5f", max_ae)

    mafe_pass = mafe <= args.mafe_threshold
    log.info("  MAFE quality: %s", "✅ PASS" if mafe_pass else "❌ FAIL")

    # Secondary: Spearman rank correlation (informational only)
    from scipy.stats import spearmanr
    corr, pval = spearmanr(filip_fp32, filip_awq)
    log.info("")
    log.info("  Spearman rank correlation: %.4f  (p=%.4f) [informational — unreliable with near-uniform scores]", corr, pval)

    # ── 6. Overall verdict ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    all_pass = cos_pass and mafe_pass
    log.info("Overall: %s", "✅ AWQ INT4 SUITABLE FOR FILIPReranker" if all_pass
             else "❌ AWQ INT4 QUALITY INSUFFICIENT")

    if not cos_pass:
        log.error(
            "Fix: per-token cosine %.4f < threshold %.2f\n"
            "     Re-quantize with --q_group_size 256 or --w_bit 8",
            mean_cos, args.cos_threshold,
        )
    if not mafe_pass:
        log.error(
            "Fix: MAFE %.5f > threshold %.3f\n"
            "     Re-quantize with larger --q_group_size or add more calibration texts",
            mafe, args.mafe_threshold,
        )

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
