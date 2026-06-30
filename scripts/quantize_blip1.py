"""
scripts/quantize_blip1.py
--------------------------
Bước 1.3 - Post-Training Quantization BLIP-1 visual encoder voi vai_q_pytorch.

Chay TRONG Vitis AI Docker container:
  docker run \
    -v $(pwd):/workspace \
    -v /home/tienc/.cache/huggingface:/home/vitis-ai-user/.cache/huggingface \
    -it xilinx/vitis-ai-pytorch-cpu:3.5.0 bash
  conda activate vitis-ai-pytorch
  pip install transformers -q
  cd /workspace
  python scripts/quantize_blip1.py

NOTE: vai_q_pytorch requires a live nn.Module - NOT a TorchScript .pt file.

Requires:
  exported_models/calibration_frames_blip1.npy
  HuggingFace cache: Salesforce/blip-itm-base-coco (mounted vao container)

Output:
  quantized/BlipVisualITCWrapper_int.xmodel  - ready for vai_c_xir
  quantized/quant_info.json
"""

from __future__ import annotations
import json, logging, os, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME  = "Salesforce/blip-itm-base-coco"
CALIB_NPY   = Path("exported_models/calibration_frames_blip1.npy")
OUTPUT_DIR  = Path("quantized")
BATCH_SIZE  = 1
INPUT_SHAPE = (1, 3, 384, 384)


class BlipVisualITCWrapper(nn.Module):
    """BLIP-1 visual encoder + ITC projection head -> (B, 256) L2-normalised."""
    def __init__(self, blip_model) -> None:
        super().__init__()
        self.vision_model = blip_model.vision_model
        self.vision_proj  = blip_model.vision_proj

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out      = self.vision_model(pixel_values=pixel_values)
        cls_feat = out.last_hidden_state[:, 0, :]
        embed    = self.vision_proj(cls_feat)
        norm     = embed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return embed / norm


def _load_wrapper() -> BlipVisualITCWrapper:
    try:
        from transformers import BlipForImageTextRetrieval
    except ImportError:
        raise ImportError("Run: pip install transformers -q")
    log.info("Loading BLIP-1 from HuggingFace cache ...")
    blip = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)
    blip.eval().cpu()

    # Apply strict static shapes for DPU compilation (BATCH_SIZE = 1)
    import types
    from typing import Optional, Tuple
    import torch.nn.functional as F

    def _bf_blip_embeddings_forward(self, pixel_values: torch.FloatTensor, **kwargs) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2).contiguous()

        class_embeds = self.class_embedding.view(1, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        
        pos_emb = self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        embeddings = embeddings + pos_emb
        return embeddings

    blip.vision_model.embeddings.forward = types.MethodType(_bf_blip_embeddings_forward, blip.vision_model.embeddings)

    def _bf_blip_attn_forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        tgt_len = hidden_states.size(1)
        embed_dim = self.embed_dim

        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.view(1, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3))
        attention_scores = attention_scores * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(1, tgt_len, embed_dim)

        output = self.projection(context_layer)
        return (output, attention_probs) if output_attentions else (output, None)

    for layer in blip.vision_model.encoder.layers:
        layer.self_attn.forward = types.MethodType(_bf_blip_attn_forward, layer.self_attn)
        
    log.info("[XIR patch] BLIP-1 embeddings and attention blocks patched for static shape (B=1, contiguous memory).")
    wrapper = BlipVisualITCWrapper(blip)
    wrapper.eval()
    log.info("Wrapper loaded. Params: %d M",
             sum(p.numel() for p in wrapper.parameters()) // 1_000_000)
    return wrapper


def run_calibration(output_dir: Path) -> None:
    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError:
        raise ImportError("Must run inside Vitis AI Docker with conda activate vitis-ai-pytorch")

    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = _load_wrapper()

    log.info("Loading calibration data: %s", CALIB_NPY)
    calib_np = np.load(str(CALIB_NPY))
    n_calib  = int(os.environ.get("NLVS_CALIB_N", "200"))
    calib_tensor = torch.from_numpy(calib_np[:n_calib])
    log.info("Using %d calibration frames", n_calib)

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
    log.info("Quant config saved -> %s/", output_dir)


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
    log.info("Output shape: %s  (expect [1, 256])", list(quant_out.shape))

    log.info("Exporting xmodel ...")
    quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=True)

    xmodel_files = list(output_dir.glob("*.xmodel"))
    if not xmodel_files:
        raise RuntimeError("No .xmodel produced - check Vitis AI logs above.")
    for f in xmodel_files:
        log.info("Produced: %s  (%.1f MB)", f, f.stat().st_size / 1e6)

    ref_npy = Path("exported_models/blip1_visual_fp32_ref.npy")
    ref_pv  = Path("exported_models/blip1_ref_pixel_values.npy")
    if ref_npy.exists() and ref_pv.exists():
        log.info("Quality check vs FP32 reference ...")
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
        min_sim  = float(cos_sims.min())
        log.info("  Mean cosine sim (INT8 vs FP32): %.4f", mean_sim)
        log.info("  Min  cosine sim:                %.4f", min_sim)
        log.info("  Quality: %s", "PASS" if mean_sim > 0.95 else "WARN - recalibrate")
        stats = {"mean_cosine_sim": mean_sim, "min_cosine_sim": min_sim,
                 "threshold": 0.95, "passed": mean_sim > 0.95}
        (output_dir / "quant_info.json").write_text(json.dumps(stats, indent=2))

    log.info("=== Done. Next: bash scripts/compile_blip1.sh ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["calib", "export", "all"], default="all")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()
    out  = Path(args.output_dir)
    if args.step in ("calib", "all"):
        run_calibration(out)
    if args.step in ("export", "all"):
        run_export(out)
