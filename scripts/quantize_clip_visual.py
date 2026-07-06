"""
scripts/quantize_clip_visual.py
--------------------------------
Post-Training Quantization (PTQ INT8) cho CLIP ViT-B/16 visual encoder
dùng vai_q_pytorch — cần thiết cho Option B & C Kria deployment.

Chạy TRONG Vitis AI Docker container:
  docker run \\
    -v $(pwd):/workspace \\
    -v /home/tienc/.cache/huggingface:/home/vitis-ai-user/.cache/huggingface \\
    -it xilinx/vitis-ai-pytorch-cpu:3.5.0 bash
  conda activate vitis-ai-pytorch
  pip install open-clip-torch -q
  cd /workspace
  python scripts/quantize_clip_visual.py

Tương tự quantize_blip1.py nhưng cho CLIP ViT-B/16 (open_clip).

Yêu cầu:
  exported_models/calibration_frames_clip.npy   (từ build_calibration_dataset.py --model clip)

Output:
  quantized/CLIPVisualWrapper_int.xmodel   — ready for vai_c_xir
  quantized/clip_quant_info.json
  
Sau đó compile:
  vai_c_xir -x quantized/CLIPVisualWrapper_int.xmodel \\
            -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \\
            -o compiled/ -n clip_vision
  # → compiled/clip_vision.xmodel
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME   = "ViT-B-16"
PRETRAINED   = "openai"
CALIB_NPY    = Path("exported_models/calibration_frames_clip.npy")
OUTPUT_DIR   = Path("quantized")
BATCH_SIZE   = 1
INPUT_SHAPE  = (1, 3, 224, 224)   # CLIP uses 224×224 (not 384 like BLIP-1)


# ---------------------------------------------------------------------------
# Wrapper: extracts [CLS] ITC embedding from CLIP ViT-B/16
# ---------------------------------------------------------------------------

class CLIPVisualWrapper(nn.Module):
    """
    CLIP visual encoder + ITC projection → (B, 512) L2-normalised.

    In open_clip's ViT-B-16:
      visual.trunk → ViT backbone (returns CLS token)
      visual.head  → projection head (Linear + LayerNorm, out=512)

    For models where visual.head doesn't exist, use visual.proj directly.
    """

    def __init__(self, clip_model) -> None:
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # open_clip visual forward returns (B, embed_dim) already normalized
        out  = self.visual(pixel_values)
        norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (out / norm).float()


class ManualMultiheadAttention(nn.Module):
    def __init__(self, mhsa: nn.MultiheadAttention) -> None:
        super().__init__()
        embed_dim      = mhsa.embed_dim
        self.num_heads = mhsa.num_heads
        self.head_dim  = embed_dim // self.num_heads
        self.scale     = float(self.head_dim) ** -0.5

        W        = mhsa.in_proj_weight.data.clone()
        has_bias = mhsa.in_proj_bias is not None
        b        = mhsa.in_proj_bias.data.clone() if has_bias else None

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)

        self.q_proj.weight = nn.Parameter(W[:embed_dim].clone())
        self.k_proj.weight = nn.Parameter(W[embed_dim : 2 * embed_dim].clone())
        self.v_proj.weight = nn.Parameter(W[2 * embed_dim :].clone())

        if has_bias:
            self.q_proj.bias = nn.Parameter(b[:embed_dim].clone())
            self.k_proj.bias = nn.Parameter(b[embed_dim : 2 * embed_dim].clone())
            self.v_proj.bias = nn.Parameter(b[2 * embed_dim :].clone())

        self.out_proj = copy.deepcopy(mhsa.out_proj)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask=None, need_weights=False, attn_mask=None) -> torch.Tensor:
        # Hardcode mọi thông số để XIR không thể hiểu nhầm
        # Use static shapes for batch size 1 to satisfy vai_c_xir
        B, T, C = 1, 197, 768
        H, Hd   = 12, 64
        k_len   = 197

        q = self.q_proj(query)  # (1, 197, 768)
        k = self.k_proj(key)    # (1, 197, 768)
        v = self.v_proj(value)  # (1, 197, 768)
        
        # Dùng .view() và .transpose() thay cho reshape/permute, kèm theo .contiguous()
        q = q.view(1, 197, 12, 64).transpose(1, 2)  # (1, 12, 197, 64)
        k = k.view(1, 197, 12, 64).transpose(1, 2)
        v = v.view(1, 197, 12, 64).transpose(1, 2)

        # Matmul với chiều tường minh
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale  
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (1, 12, 197, 64)

        # ÉP LIỀN MẠCH BỘ NHỚ bằng .contiguous() trước khi view lại
        out = out.transpose(1, 2).contiguous().view(1, 197, 768)
        
        return self.out_proj(out)


def _patch_visual_for_xir(wrapper: "CLIPVisualWrapper") -> None:
    """
    Make CLIP visual transformer XIR-compatible:

    Problem 1 — multi-output MHSA:
      Replace nn.MultiheadAttention with ManualMultiheadAttention.

    Problem 2 — seq-first permutations in VisionTransformer.forward():
      open_clip's VisionTransformer does x.permute(1,0,2) before its
      transformer (NLD→LND) and x.permute(1,0,2) after (LND→NLD).
      NNDCT v3.5 mis-represents this as a reshape in the xmodel, causing:
        'reshape-fix input {197,1,197,768} ≠ output {1,197,768}' in vai_c_xir
      Fix: patch VisionTransformer.forward() to stay batch-first throughout,
      and patch ResidualAttentionBlock.forward() to match (batch-first attn).
    """
    visual    = wrapper.visual
    resblocks = visual.transformer.resblocks

    # ── 1. Replace MHSA with batch-first ManualMultiheadAttention ────────────
    for block in resblocks:
        block.attn = ManualMultiheadAttention(block.attn)

        # Patch attention() to NOT do [0] and to stay batch-first
        def _bf_attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
            k_x = (
                self.ln_1_kv(k_x)
                if hasattr(self, "ln_1_kv") and k_x is not None
                else q_x
            )
            v_x = k_x if v_x is None else v_x
            attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
            return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)

        block.attention = types.MethodType(_bf_attention, block)

        # Patch ResidualAttentionBlock.forward() to stay batch-first (B, T, C)
        # Original open_clip forward assumes seq-first (T, B, C) input.
        def _bf_block_forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
            # q_x: (B, T, C)  — batch-first throughout
            x = self.ln_1(q_x)
            x = q_x + self.ls_1(self.attention(q_x=x, attn_mask=attn_mask))
            x = x + self.ls_2(self.mlp(self.ln_2(x)))
            return x

        block.forward = types.MethodType(_bf_block_forward, block)

    # ── 2. Patch VisionTransformer.forward() to remove LND permutations ──────
    #
    # Original (all open_clip versions):
    #   x = ...patch embed...
    #   x = x.permute(1, 0, 2)   # NLD → LND
    #   x = self.transformer(x)  # operates on LND
    #   x = x.permute(1, 0, 2)   # LND → NLD
    #
    # These permutes produce a reshape-fix node in NNDCT that vai_c_xir
    # can't compile ({197,1,197,768} → {1,197,768}).
    #
    # Fix: remove both permutes; resblocks are already patched for batch-first.
    #
    # API compatibility: avoid _expand_token (new) and _global_pool (new).
    # Use the original open_clip patterns that work on ALL versions.

    def _bf_vit_forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Patch embedding ──
        x = self.conv1(x)                                    # (1, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2).contiguous()        # (1, 196, 768)

        # ── CLS token ──
        cls_token = self.class_embedding.view(1, 1, 768).to(x.dtype)
        x = torch.cat([cls_token, x], dim=1)                 # (1, 197, 768)
        
        # ── Positional Embedding ──
        # Ép lên (1, 197, 768) để cấm PyTorch/XIR tự động đoán chiều
        pos_emb = self.positional_embedding.unsqueeze(0).to(x.dtype)
        x = x + pos_emb

        if hasattr(self, "patch_dropout"):
            x = self.patch_dropout(x)

        x = self.ln_pre(x)

        # ── Transformer ──
        x = self.transformer(x)                              

        # ── CLS pooling (Trích xuất token đầu tiên an toàn) ──
        x = x[:, 0, :] # (1, 768)
        x = x.view(1, 768) # Đảm bảo XIR hiểu đúng shape đầu ra

        x = self.ln_post(x)                         

        if self.proj is not None:
            x = x @ self.proj

        return x

    visual.forward = types.MethodType(_bf_vit_forward, visual)

    log.info(
        "[XIR patch] %d resblocks → batch-first ManualMHSA; VisionTransformer → no LND permute.",
        len(resblocks),
    )


def _load_wrapper() -> CLIPVisualWrapper:
    try:
        import open_clip
    except ImportError:
        raise ImportError("Run: pip install open-clip-torch -q")

    log.info("Loading CLIP %s (%s) ...", MODEL_NAME, PRETRAINED)
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, cache_dir="./model_cache"
    )
    model.eval().cpu()
    wrapper = CLIPVisualWrapper(model)
    wrapper.eval()

    # Patch MultiheadAttention for XIR single-output compatibility
    _patch_visual_for_xir(wrapper)

    log.info(
        "Wrapper loaded.  Visual params: %d M",
        sum(p.numel() for p in wrapper.parameters()) // 1_000_000,
    )
    return wrapper


# ---------------------------------------------------------------------------
# PTQ calibration
# ---------------------------------------------------------------------------

def run_calibration(output_dir: Path) -> None:
    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError:
        raise ImportError(
            "Must run inside Vitis AI Docker:\n"
            "  conda activate vitis-ai-pytorch"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = _load_wrapper()

    log.info("Loading calibration data: %s", CALIB_NPY)
    if not CALIB_NPY.exists():
        raise FileNotFoundError(
            f"{CALIB_NPY} not found.\n"
            "Run: python scripts/build_calibration_dataset.py --model clip"
        )
    calib_np = np.load(str(CALIB_NPY))
    n_calib  = int(os.environ.get("NLVS_CALIB_N", "200"))
    calib_tensor = torch.from_numpy(calib_np[:n_calib])
    log.info("Using %d calibration frames  shape=%s", n_calib, list(calib_tensor.shape))

    dummy_input = torch.randn(*INPUT_SHAPE)
    log.info("Setting up Vitis AI quantizer (mode=calib) ...")
    quantizer = torch_quantizer(
        quant_mode="calib", module=wrapper,
        input_args=(dummy_input,), output_dir=str(output_dir),
        quant_config_file=None,
    )
    quant_model = quantizer.quant_model

    log.info("Running calibration (%d frames, batch=%d) ...", n_calib, BATCH_SIZE)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(calib_tensor), BATCH_SIZE):
            batch = calib_tensor[i : i + BATCH_SIZE]
            quant_model(batch)
            if i % (BATCH_SIZE * 5) == 0:
                log.info("  %d / %d frames", i, n_calib)
    log.info("Calibration done in %.1f s", time.perf_counter() - t0)
    quantizer.export_quant_config()
    log.info("Quant config saved → %s/", output_dir)


# ---------------------------------------------------------------------------
# PTQ export → .xmodel
# ---------------------------------------------------------------------------

def run_export(output_dir: Path) -> None:
    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError:
        raise ImportError("Run inside Vitis AI Docker.")

    wrapper     = _load_wrapper()
    dummy_input = torch.randn(*INPUT_SHAPE)
    log.info("Setting up quantizer (mode=test/export) ...")
    quantizer = torch_quantizer(
        quant_mode="test", module=wrapper,
        input_args=(dummy_input,), output_dir=str(output_dir),
    )
    quant_model = quantizer.quant_model

    with torch.no_grad():
        quant_out = quant_model(dummy_input)
    log.info("Output shape: %s  (expect [1, 512])", list(quant_out.shape))

    log.info("Exporting xmodel ...")
    quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=True)

    xmodel_files = list(output_dir.glob("*.xmodel"))
    if not xmodel_files:
        raise RuntimeError("No .xmodel produced — check Vitis AI logs.")
    for f in xmodel_files:
        log.info("Produced: %s  (%.1f MB)", f, f.stat().st_size / 1e6)

    # Quality check vs FP32 reference
    ref_npy = Path("exported_models/clip_visual_fp32_ref.npy")
    ref_pv  = Path("exported_models/clip_ref_pixel_values.npy")
    if ref_npy.exists() and ref_pv.exists():
        log.info("Quality check vs FP32 reference ...")
        fp32_ref  = np.load(str(ref_npy))
        pv_batch  = torch.from_numpy(np.load(str(ref_pv)))
        with torch.no_grad():
            # [SỬA LỖI TẠI ĐÂY]: Chạy vòng lặp từng ảnh một (Batch Size = 1)
            int8_outs = []
            for i in range(pv_batch.shape[0]):
                single_input = pv_batch[i:i+1]  # Trích xuất 1 ảnh (1, 3, 224, 224)
                single_out = quant_model(single_input)
                int8_outs.append(single_out)
            
            # Ghép 16 kết quả lại thành 1 tensor lớn để tính toán so sánh
            int8_out = torch.cat(int8_outs, dim=0).numpy()
            
        cos_sims  = (fp32_ref * int8_out).sum(axis=1)
        mean_sim  = float(cos_sims.mean())
        min_sim   = float(cos_sims.min())
        log.info("  Mean cosine sim (INT8 vs FP32): %.4f", mean_sim)
        log.info("  Min  cosine sim:                %.4f", min_sim)
        log.info("  Quality: %s", "PASS ✅" if mean_sim > 0.95 else "WARN ⚠️ — recalibrate")
        stats = {
            "model": MODEL_NAME, "pretrained": PRETRAINED,
            "mean_cosine_sim": mean_sim, "min_cosine_sim": min_sim,
            "threshold": 0.95, "passed": mean_sim > 0.95,
        }
        (output_dir / "clip_quant_info.json").write_text(json.dumps(stats, indent=2))

    log.info("")
    log.info("=== Done. Next: compile xmodel ===")
    log.info("  vai_c_xir -x %s/*.xmodel \\", output_dir)
    log.info("            -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \\")
    log.info("            -o compiled/ -n clip_vision")


# ---------------------------------------------------------------------------
# Build FP32 reference embeddings (for quality check in run_export)
# ---------------------------------------------------------------------------

def build_fp32_reference(output_dir: Path, n_ref: int = 16) -> None:
    """Save FP32 CLIP visual embeddings as reference for INT8 quality check."""
    try:
        import open_clip
    except ImportError:
        log.warning("open_clip not available — skipping FP32 reference generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, cache_dir="./model_cache"
    )
    model.eval()

    if CALIB_NPY.exists():
        calib_np = np.load(str(CALIB_NPY))[:n_ref]
        pv = torch.from_numpy(calib_np)
    else:
        log.warning("Calibration .npy not found — using random tensor for reference")
        pv = torch.randn(n_ref, 3, 224, 224)

    with torch.no_grad():
        fp32_embs = model.visual(pv)
        fp32_embs = F.normalize(fp32_embs.float(), dim=-1).numpy()

    np.save(str(output_dir / "clip_visual_fp32_ref.npy"), fp32_embs)
    np.save(str(output_dir / "clip_ref_pixel_values.npy"), pv.numpy())
    log.info("FP32 reference saved → %s/clip_visual_fp32_ref.npy", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="PTQ INT8 quantization for CLIP ViT-B/16 visual encoder (Vitis AI)"
    )
    p.add_argument(
        "--step",
        choices=["calib", "export", "ref", "all"],
        default="all",
        help=(
            "calib: run PTQ calibration pass\n"
            "export: export .xmodel\n"
            "ref: build FP32 reference embeddings\n"
            "all: ref → calib → export (default)"
        ),
    )
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()
    out  = Path(args.output_dir)

    if args.step in ("ref", "all"):
        build_fp32_reference(Path("exported_models"))
    if args.step in ("calib", "all"):
        run_calibration(out)
    if args.step in ("export", "all"):
        run_export(out)
