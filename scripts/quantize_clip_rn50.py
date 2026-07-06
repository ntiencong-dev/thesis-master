"""
scripts/quantize_clip_rn50.py
--------------------------------
Post-Training Quantization (PTQ INT8) cho CLIP RN50 visual encoder.
Chiến lược tối ưu cho Kria KV260:
- CNN Backbone (ResNet stem + layers 1-4) -> Chạy trên DPU (INT8)
- AttentionPool2d + L2 Normalization -> Chạy trên CPU (FP32)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME  = "RN50"
PRETRAINED  = "openai"
CALIB_NPY   = Path("exported_models/calibration_frames_clip.npy")
OUTPUT_DIR  = Path("quantized/rn50")
BATCH_SIZE  = 1
INPUT_SHAPE = (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Wrapper: CLIPVisualDPUWrapper (Chỉ giữ lại CNN)
# ---------------------------------------------------------------------------

class CLIPVisualDPUWrapper(nn.Module):
    """
    Bọc CLIP RN50 visual encoder nhưng cắt bỏ khối AttentionPool2d.
    Đầu ra sẽ là feature map 3D cuối cùng của Layer 4 (B, 2048, 7, 7).
    """
    def __init__(self, visual_model) -> None:
        super().__init__()
        # Khởi tạo hàm ReLU cục bộ để tránh lỗi thiếu attribute của open_clip
        self.relu = nn.ReLU()
        
        # Gán trực tiếp các lớp có sẵn từ Stem
        self.conv1 = visual_model.conv1
        self.bn1 = visual_model.bn1
        self.conv2 = visual_model.conv2
        self.bn2 = visual_model.bn2
        self.conv3 = visual_model.conv3
        self.bn3 = visual_model.bn3
        self.avgpool = visual_model.avgpool
        
        # Các khối Convolution chính
        self.layer1 = visual_model.layer1
        self.layer2 = visual_model.layer2
        self.layer3 = visual_model.layer3
        self.layer4 = visual_model.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nối thủ công các lớp Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        
        # Nối các lớp Backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_components():
    try:
        import open_clip
    except ImportError:
        raise ImportError("Run: pip install open-clip-torch -q")

    log.info("Loading CLIP %s (%s) ...", MODEL_NAME, PRETRAINED)
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, cache_dir="./model_cache"
    )
    model.eval().cpu()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bóc tách và xuất khối AttentionPool2d để chạy riêng trên CPU
    cpu_pool_path = OUTPUT_DIR / "clip_rn50_attnpool.pt"
    torch.save(model.visual.attnpool.state_dict(), cpu_pool_path)
    log.info("Exported AttentionPool2d (CPU) to %s", cpu_pool_path)

    # Đóng gói phần CNN để lượng tử hóa
    wrapper = CLIPVisualDPUWrapper(model.visual)
    wrapper.eval()
    
    return wrapper, model.visual.attnpool


# ---------------------------------------------------------------------------
# PTQ — Calibration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PTQ — Calibration (Tích hợp Fast Finetuning / AdaQuant)
# ---------------------------------------------------------------------------

def run_calibration(output_dir: Path) -> None:
    from pytorch_nndct.apis import torch_quantizer

    wrapper, _ = _load_components()

    calib_np = np.load(str(CALIB_NPY))
    n_calib  = int(os.environ.get("NLVS_CALIB_N", "200"))
    calib_tensor = torch.from_numpy(calib_np[:n_calib])
    
    dummy_input = torch.randn(*INPUT_SHAPE)
    
    quantizer = torch_quantizer(
        quant_mode="calib", module=wrapper,
        input_args=(dummy_input,), output_dir=str(output_dir)
    )
    
    def forward_loop(model, data_tensor):
        model.eval()
        with torch.no_grad():
            for i in range(len(data_tensor)):
                model(data_tensor[i : i + 1])

    log.info("Bắt đầu khởi chạy Fast Finetuning (AdaQuant)...")
    
    # Đã bổ sung quantizer.quant_model vào tuple
    quantizer.fast_finetune(forward_loop, (quantizer.quant_model, calib_tensor))

    quantizer.export_quant_config()
    log.info("Fast Finetuning hoàn tất → %s", output_dir)


# ---------------------------------------------------------------------------
# PTQ — Export xmodel
# ---------------------------------------------------------------------------

def run_export(output_dir: Path) -> None:
    from pytorch_nndct.apis import torch_quantizer

    wrapper, cpu_pool = _load_components()
    dummy_input = torch.randn(*INPUT_SHAPE)

    quantizer = torch_quantizer(
        quant_mode="test", module=wrapper,
        input_args=(dummy_input,), output_dir=str(output_dir)
    )
    
    with torch.no_grad():
        quantizer.quant_model(dummy_input)

    quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=True)

    # Kiểm tra độ chính xác sau lượng hóa (Kép: DPU INT8 + CPU FP32)
    ref_npy = Path("exported_models/clip_visual_fp32_ref_rn50.npy")
    ref_pv  = Path("exported_models/clip_ref_pixel_values_rn50.npy")
    if ref_npy.exists() and ref_pv.exists():
        fp32_ref = np.load(str(ref_npy))
        pv_batch = torch.from_numpy(np.load(str(ref_pv)))
        
        int8_outs = []
        with torch.no_grad():
            # 1. Chạy phần CNN qua mô hình đã lượng tử hóa (Giả lập DPU)
            for i in range(pv_batch.shape[0]):
                int8_outs.append(quantizer.quant_model(pv_batch[i : i + 1]))
                
        dpu_features = torch.cat(int8_outs, dim=0).float()
        
        # 2. Đẩy qua khối AttentionPool2d trên CPU
        cpu_pool.eval()
        with torch.no_grad():
            final_out = cpu_pool(dpu_features)
            
        # 3. Chuẩn hóa L2 trên CPU
        final_out = F.normalize(final_out, dim=-1).numpy()
        
        cos_sims = (fp32_ref * final_out).sum(axis=1)
        mean_sim = float(cos_sims.mean())
        log.info("  Mean cosine sim (INT8 DPU + FP32 CPU vs Full FP32): %.4f", mean_sim)
        assert mean_sim > 0.90, "Quality check failed — cosine < 0.90"
        log.info("  ✅ Quality check passed.")
    else:
        log.warning("FP32 reference not found — skipping quality check.")

    log.info("Export done → %s", output_dir)


# ---------------------------------------------------------------------------
# Build FP32 reference embeddings
# ---------------------------------------------------------------------------

def build_fp32_reference(output_dir: Path) -> None:
    try:
        import open_clip
    except ImportError:
        raise ImportError("Run: pip install open-clip-torch -q")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, cache_dir="./model_cache"
    )
    model.eval()

    if CALIB_NPY.exists():
        pv = torch.from_numpy(np.load(str(CALIB_NPY))[:16])
    else:
        pv = torch.randn(16, 3, 224, 224)

    with torch.no_grad():
        fp32_embs = model.visual(pv)
        fp32_embs = F.normalize(fp32_embs.float(), dim=-1).numpy()

    np.save(str(output_dir / "clip_visual_fp32_ref_rn50.npy"), fp32_embs)
    np.save(str(output_dir / "clip_ref_pixel_values_rn50.npy"), pv.numpy())
    log.info("FP32 reference saved → %s", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PTQ INT8 — CLIP RN50 (CNN on DPU)")
    p.add_argument("--step",       choices=["calib", "export", "ref", "all"], default="all")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()
    out  = Path(args.output_dir)

    if args.step in ("ref",   "all"): build_fp32_reference(Path("exported_models"))
    if args.step in ("calib", "all"): run_calibration(out)
    if args.step in ("export","all"): run_export(out)