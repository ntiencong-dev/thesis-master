"""
scripts/quantize_clip_rn50.py
--------------------------------
Post-Training Quantization (PTQ INT8) cho CLIP RN50 visual encoder
Tối ưu hóa kiến trúc CNN-based VLM cho DPU Xilinx Kria KV260.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Đổi sang mô hình RN50
MODEL_NAME   = "RN50"
PRETRAINED   = "openai"
CALIB_NPY    = Path("exported_models/calibration_frames_clip.npy")
OUTPUT_DIR   = Path("quantized")
BATCH_SIZE   = 1  # Ép cứng Batch Size = 1 cho Edge Deployment
INPUT_SHAPE  = (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Wrapper & Patch cho RN50
# ---------------------------------------------------------------------------

class CLIPVisualWrapper(nn.Module):
    """Bọc Visual Encoder và L2 Normalization."""
    def __init__(self, clip_model) -> None:
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.visual(pixel_values)
        norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (out / norm).float()


class ManualAttentionPool2d(nn.Module):
    """
    Patch riêng cho lớp AttentionPool2d ở cuối RN50 để tránh lỗi XIR.
    Xóa bỏ các hàm kích thước động và ép cấp phát bộ nhớ liền mạch (.contiguous).
    """
    def __init__(self, original_pool: nn.Module) -> None:
        super().__init__()
        self.embed_dim = original_pool.k_proj.in_features
        self.num_heads = original_pool.num_heads
        self.output_dim = original_pool.c_proj.out_features
        self.positional_embedding = copy.deepcopy(original_pool.positional_embedding)
        
        self.q_proj = copy.deepcopy(original_pool.q_proj)
        self.k_proj = copy.deepcopy(original_pool.k_proj)
        self.v_proj = copy.deepcopy(original_pool.v_proj)
        self.c_proj = copy.deepcopy(original_pool.c_proj)
        
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = float(self.head_dim) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input từ khối ResNet cuối cùng sẽ có shape tĩnh: (1, 2048, 7, 7)
        x = x.flatten(2).transpose(1, 2).contiguous() # (1, 49, 2048)
        
        # Mean pooling để tạo token tổng hợp
        cls_token = x.mean(dim=1, keepdim=True) # (1, 1, 2048)
        x = torch.cat([cls_token, x], dim=1) # (1, 50, 2048)
        
        # Cộng Positional Embedding với shape tĩnh
        pos_emb = self.positional_embedding.unsqueeze(0).to(x.dtype)
        x = x + pos_emb
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(1, 50, self.num_heads, self.head_dim).transpose(1, 2) # (1, 32, 50, 64)
        k = k.view(1, 50, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, 50, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(1, 50, self.embed_dim)
        out = self.c_proj(out)
        
        # Lấy token đầu ra để so sánh vector
        out = out[:, 0, :] # (1, 1024) - Lưu ý: RN50 của OpenAI trả ra vector 1024 chiều
        return out.view(1, self.output_dim)


def _load_wrapper() -> CLIPVisualWrapper:
    try:
        import open_clip
    except ImportError:
        raise ImportError("Run: pip install open-clip-torch -q")

    log.info("Loading CLIP %s (%s) ...", MODEL_NAME, PRETRAINED)
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval().cpu()
    
    # Can thiệp thay thế khối Attention cuối cùng của ResNet
    model.visual.attnpool = ManualAttentionPool2d(model.visual.attnpool)
    
    wrapper = CLIPVisualWrapper(model)
    wrapper.eval()
    return wrapper


# ---------------------------------------------------------------------------
# Quy trình PTQ (Calibration & Export)
# ---------------------------------------------------------------------------

def run_calibration(output_dir: Path) -> None:
    from pytorch_nndct.apis import torch_quantizer
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = _load_wrapper()

    calib_np = np.load(str(CALIB_NPY))
    n_calib  = int(os.environ.get("NLVS_CALIB_N", "200"))
    calib_tensor = torch.from_numpy(calib_np[:n_calib])

    dummy_input = torch.randn(*INPUT_SHAPE)
    quantizer = torch_quantizer(
        quant_mode="calib", module=wrapper,
        input_args=(dummy_input,), output_dir=str(output_dir),
        quant_config_file=None,
    )
    quant_model = quantizer.quant_model

    with torch.no_grad():
        for i in range(len(calib_tensor)):
            batch = calib_tensor[i : i + 1] # Loop batch = 1
            quant_model(batch)
    quantizer.export_quant_config()


def run_export(output_dir: Path) -> None:
    from pytorch_nndct.apis import torch_quantizer
    wrapper = _load_wrapper()
    dummy_input = torch.randn(*INPUT_SHAPE)
    
    quantizer = torch_quantizer(
        quant_mode="test", module=wrapper,
        input_args=(dummy_input,), output_dir=str(output_dir),
    )
    quant_model = quantizer.quant_model

    with torch.no_grad():
        quant_out = quant_model(dummy_input)

    quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=True)

    # Vòng lặp Quality check vs FP32 an toàn cho Batch Size = 1
    ref_npy = Path("exported_models/clip_visual_fp32_ref.npy")
    ref_pv  = Path("exported_models/clip_ref_pixel_values.npy")
    if ref_npy.exists() and ref_pv.exists():
        fp32_ref = np.load(str(ref_npy))
        pv_batch = torch.from_numpy(np.load(str(ref_pv)))
        with torch.no_grad():
            int8_outs = []
            for i in range(pv_batch.shape[0]):
                single_input = pv_batch[i:i+1]
                int8_outs.append(quant_model(single_input))
            int8_out = torch.cat(int8_outs, dim=0).numpy()
            
        cos_sims = (fp32_ref * int8_out).sum(axis=1)
        mean_sim = float(cos_sims.mean())
        log.info("  Mean cosine sim (INT8 vs FP32): %.4f", mean_sim)


def build_fp32_reference(output_dir: Path) -> None:
    import open_clip
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()

    if CALIB_NPY.exists():
        pv = torch.from_numpy(np.load(str(CALIB_NPY))[:16])
    else:
        pv = torch.randn(16, 3, 224, 224)

    with torch.no_grad():
        fp32_embs = model.visual(pv)
        fp32_embs = F.normalize(fp32_embs.float(), dim=-1).numpy()

    np.save(str(output_dir / "clip_visual_fp32_ref.npy"), fp32_embs)
    np.save(str(output_dir / "clip_ref_pixel_values.npy"), pv.numpy())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["calib", "export", "ref", "all"], default="all")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()
    out = Path(args.output_dir)

    if args.step in ("ref", "all"): build_fp32_reference(Path("exported_models"))
    if args.step in ("calib", "all"): run_calibration(out)
    if args.step in ("export", "all"): run_export(out)