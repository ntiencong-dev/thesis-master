"""
scripts/export_blip1_for_vitis.py
-----------------------------------
Bước 1.1 — Export BLIP-1 ViT-B/16 visual encoder sang TorchScript FP32.
Chạy trên PC (không cần Vitis AI Docker).

Output:
  exported_models/blip1_visual_fp32.pt   — TorchScript traced model
  exported_models/blip1_visual_fp32_ref.npy  — 10 reference embeddings (PC FP32)
                                              dùng để validate sau PTQ

Usage:
  source venv/bin/activate
  python scripts/export_blip1_for_vitis.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME  = "Salesforce/blip-itm-base-coco"
OUTPUT_DIR  = Path("exported_models")
INPUT_SIZE  = 384          # BLIP-1 uses 384×384 (not 224×224 like CLIP)


# ---------------------------------------------------------------------------
# Wrapper: chỉ export visual branch (ViT-B/16 + ITC projection head)
# ---------------------------------------------------------------------------

class BlipVisualITCWrapper(torch.nn.Module):
    """
    Wrap BLIP-1 visual encoder + ITC projection head.

    Forward: (B, 3, 384, 384) float32 → (B, 256) float32 L2-normalised
    Only this part is acceleratable on DPU.
    BERT text encoder + ITM head stay on ARM CPU.
    """

    def __init__(self, blip_model) -> None:
        super().__init__()
        self.vision_model = blip_model.vision_model   # BlipVisionModel (ViT-B/16)
        self.vision_proj  = blip_model.vision_proj    # Linear(768 → 256)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, 3, 384, 384) — RGB, normalised by BlipProcessor
        vision_out = self.vision_model(pixel_values=pixel_values)
        cls_feat   = vision_out.last_hidden_state[:, 0, :]   # [CLS] → (B, 768)
        embed      = self.vision_proj(cls_feat)               # (B, 256)
        # L2 normalise (same as blip1_engine.py)
        norm  = embed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return embed / norm                                   # (B, 256) unit vector


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading BLIP-1 from %s …", MODEL_NAME)
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    blip  = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)
    blip.eval()
    blip.cpu()

    wrapper = BlipVisualITCWrapper(blip)
    wrapper.eval()

    # -----------------------------------------------------------------------
    # Verify wrapper output matches original model
    # -----------------------------------------------------------------------
    log.info("Verifying wrapper correctness …")
    proc = BlipProcessor.from_pretrained(MODEL_NAME)

    dummy_rgb = (np.random.rand(INPUT_SIZE, INPUT_SIZE, 3) * 255).astype(np.uint8)
    inputs    = proc(images=dummy_rgb, return_tensors="pt")
    pv        = inputs["pixel_values"]                # (1, 3, 384, 384)

    with torch.no_grad():
        ref_embed = wrapper(pv)

        # Original path for comparison
        vision_out_orig = blip.vision_model(pixel_values=pv)
        cls_orig        = vision_out_orig.last_hidden_state[:, 0, :]
        embed_orig      = blip.vision_proj(cls_orig)
        norm_orig       = embed_orig / embed_orig.norm(dim=-1, keepdim=True).clamp(1e-8)

    cos_sim = torch.nn.functional.cosine_similarity(ref_embed, norm_orig).item()
    log.info("Wrapper vs original cosine sim: %.6f (expect ≈ 1.0)", cos_sim)
    assert cos_sim > 0.9999, f"Wrapper mismatch! cos_sim={cos_sim}"

    # -----------------------------------------------------------------------
    # TorchScript trace
    # -----------------------------------------------------------------------
    log.info("Tracing TorchScript (input shape: 1×3×%d×%d) …", INPUT_SIZE, INPUT_SIZE)
    dummy_trace = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Disable MKLDNN before tracing — optimize_for_inference() converts Conv weights
    # to prim::ConstantMKLDNNTensor which is unreadable by Vitis AI Docker's PyTorch.
    _mkldnn_prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_trace)

    torch.backends.mkldnn.enabled = _mkldnn_prev
    # NOTE: do NOT call optimize_for_inference() here — it produces MKLDNN ops
    # that are incompatible with Vitis AI Docker (older PyTorch, no MKLDNN support).

    # Quick sanity check: traced output ≈ eager output
    with torch.no_grad():
        traced_out = traced(pv)
    cos_trace = torch.nn.functional.cosine_similarity(traced_out, ref_embed).item()
    log.info("Traced vs eager cosine sim: %.6f (expect ≈ 1.0)", cos_trace)
    assert cos_trace > 0.9999, f"Traced model mismatch! cos_sim={cos_trace}"

    # Save
    pt_path = output_dir / "blip1_visual_fp32.pt"
    traced.save(str(pt_path))
    log.info("Saved: %s", pt_path)

    # -----------------------------------------------------------------------
    # Generate reference embeddings for post-quantisation validation
    # Use real frames from calibration data (same distribution as PTQ calib).
    # -----------------------------------------------------------------------
    log.info("Generating 16 reference embeddings (PC FP32, real frames) …")
    calib_npy = output_dir.parent / "exported_models" / "calibration_frames_blip1.npy"
    if calib_npy.exists():
        log.info("  Using real calibration frames from %s", calib_npy)
        real_pv = np.load(str(calib_npy))[:16]          # (16, 3, 384, 384) already preprocessed
        ref_pv  = torch.from_numpy(real_pv)
    else:
        log.warning("  calibration_frames_blip1.npy not found — falling back to random frames")
        log.warning("  Run build_calibration_dataset.py first for a meaningful quality check.")
        ref_frames = [(np.random.rand(INPUT_SIZE, INPUT_SIZE, 3) * 255).astype(np.uint8)
                      for _ in range(16)]
        ref_inputs = proc(images=ref_frames, return_tensors="pt")
        ref_pv     = ref_inputs["pixel_values"]          # (16, 3, 384, 384)

    # Keep MKLDNN disabled for reference inference too
    torch.backends.mkldnn.enabled = False
    with torch.no_grad():
        ref_embeddings = traced(ref_pv).numpy()             # (16, 256)

    ref_path = output_dir / "blip1_visual_fp32_ref.npy"
    np.save(str(ref_path), ref_embeddings)
    log.info("Saved reference embeddings: %s  shape=%s", ref_path, ref_embeddings.shape)

    # Save the 16 pixel_values too (needed to reproduce on Kria for validation)
    pv_path = output_dir / "blip1_ref_pixel_values.npy"
    np.save(str(pv_path), ref_pv.numpy())
    log.info("Saved pixel_values for validation: %s", pv_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=== Export complete ===")
    log.info("  Model:         %s", MODEL_NAME)
    log.info("  Input shape:   (N, 3, %d, %d)  dtype=float32", INPUT_SIZE, INPUT_SIZE)
    log.info("  Output shape:  (N, 256)  L2-normalised float32")
    log.info("  Saved:         %s", pt_path)
    log.info("")
    log.info("Next step: python scripts/build_calibration_dataset.py")


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export BLIP-1 visual encoder to TorchScript")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help=f"Directory for output files (default: {OUTPUT_DIR})")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    export(Path(args.output_dir))
