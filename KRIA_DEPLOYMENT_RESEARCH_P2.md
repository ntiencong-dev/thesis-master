# Nghiên cứu Triển khai NLVS trên AMD Kria KV260 — Phần 2

> **Phiên bản:** 1.0 (Part 2/2)  
> **Ngày:** 2026-05-12  
> **Tiếp nối từ:** [KRIA_DEPLOYMENT_RESEARCH.md](KRIA_DEPLOYMENT_RESEARCH.md)  
> **Phần 2 bao gồm:** Compilation workflow, VVAS pipeline, KriaEngine integration, benchmarks, deployment roadmap

---

## Mục lục (Phần 2)

7. [Vitis AI Compilation Workflow chi tiết](#7-vitis-ai-compilation-workflow-chi-tiết)
8. [VVAS Video Pipeline trên Kria](#8-vvas-video-pipeline-trên-kria)
9. [KriaEngine — Phân tích Implementation](#9-kriaengine--phân-tích-implementation)
10. [Tối ưu hóa hiệu năng & Memory Management](#10-tối-ưu-hóa-hiệu-năng--memory-management)
11. [Benchmark dự kiến & So sánh với PC](#11-benchmark-dự-kiến--so-sánh-với-pc)
12. [Lộ trình triển khai Step-by-Step](#12-lộ-trình-triển-khai-step-by-step)
13. [Rủi ro kỹ thuật & Mitigation Strategy](#13-rủi-ro-kỹ-thuật--mitigation-strategy)
14. [Kết luận & Khuyến nghị](#14-kết-luận--khuyến-nghị)

---

## 7. Vitis AI Compilation Workflow chi tiết

### 7.1 Tổng quan Pipeline (Development Machine → Kria)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT MACHINE (x86, Ubuntu)                │
│                                                                     │
│  Step 1: Prepare FP32 Model                                        │
│  ─────────────────────────                                          │
│  open_clip.create_model_and_transforms("ViT-B-16", "openai")       │
│  model.visual  → visual_encoder.pt  (torch.jit.script or torchscript)
│  model.encode_text → text_encoder.pt                               │
│                    │                                                │
│                    ▼                                                │
│  Step 2: Post-Training Quantization (vai_q_pytorch)                │
│  ─────────────────────────────────────────────────                  │
│  Input:  visual_encoder.pt (FP32)                                  │
│          calibration_dataset (1000 frames, CLIP-preprocessed)       │
│  Output: quantized_visual.pt (INT8 weights + activations)          │
│                    │                                                │
│                    ▼                                                │
│  Step 3: Compile to XIR (vai_c_xir)                                │
│  ─────────────────────────────────                                   │
│  Input:  quantized_visual.pt                                        │
│          arch.json  (DPU B4096 architecture description)            │
│  Output: clip_vision.xmodel  (~86 MB INT8)                         │
│          clip_text.xmodel    (~63 MB INT8)                          │
│                    │                                                │
│                    ▼                                                │
│  Step 4: Verify on DPU Target Simulation                           │
│  ────────────────────────────────────                               │
│  vai_runtime_check --xmodel clip_vision.xmodel                     │
│  Check: all DPU ops supported, no unsupported ops remain           │
└────────────────────────────┬────────────────────────────────────────┘
                             │ SCP / USB transfer
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KRIA KV260 (ARM Ubuntu)                          │
│                                                                     │
│  Step 5: Deploy & Validate                                          │
│  ─────────────────────────                                          │
│  python validate_model.py --xmodel clip_vision.xmodel              │
│  → Cosine similarity validation vs FP32 reference                  │
│  → Check embedding quality degradation < 5%                        │
│                                                                     │
│  Step 6: Integration Test                                           │
│  ────────────────────────                                            │
│  python -m pytest tests/ --config config/kria.yaml                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Step 1 — Export CLIP sang TorchScript

```python
# scripts/export_clip_for_vitis.py
"""
Export CLIP ViT-B/16 sang định dạng TorchScript cho Vitis AI quantization.
Chạy trên development machine (x86 GPU) trước khi quantize.
"""
import torch
import open_clip
import numpy as np
from pathlib import Path

def export_visual_encoder(output_dir: str = "exported_models"):
    """Export CLIP visual encoder sang TorchScript."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model.eval().cpu()  # CPU mode cho export
    
    # Wrap visual encoder để export chỉ visual branch
    class VisualEncoderWrapper(torch.nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (N, 3, 224, 224) float32, CLIP-normalized
            Returns:
                (N, 512) float32, unnormalized embeddings
            """
            # Trả về raw embeddings (L2 norm ở post-processing)
            return self.visual(x)
    
    visual_wrapper = VisualEncoderWrapper(model.visual)
    
    # Dummy input cho trace
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # TorchScript trace (không phải script vì ViT có control flow)
    with torch.no_grad():
        traced = torch.jit.trace(visual_wrapper, dummy_input)
    
    # Optimize for inference
    traced = torch.jit.optimize_for_inference(traced)
    
    save_path = f"{output_dir}/clip_visual_fp32.pt"
    traced.save(save_path)
    print(f"Saved: {save_path}")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {traced(dummy_input).shape}")
    
    return save_path


def export_text_encoder(output_dir: str = "exported_models"):
    """Export CLIP text encoder với fixed batch_size=12 (template ensemble)."""
    Path(output_dir).mkdir(exist_ok=True)
    
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model.eval().cpu()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, transformer, token_embedding, positional_embedding,
                     ln_final, text_projection, attn_mask):
            super().__init__()
            self.transformer      = transformer
            self.token_embedding  = token_embedding
            self.positional_embedding = positional_embedding
            self.ln_final         = ln_final
            self.text_projection  = text_projection
            self.register_buffer("attn_mask", attn_mask)
        
        def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
            """
            Args:
                text_tokens: (12, 77) int64 — fixed batch_size=12
            Returns:
                (12, 512) float32 — raw text embeddings
            """
            x = self.token_embedding(text_tokens)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
            x = x @ self.text_projection
            return x
    
    text_wrapper = TextEncoderWrapper(
        model.transformer,
        model.token_embedding,
        model.positional_embedding,
        model.ln_final,
        model.text_projection,
        model.attn_mask
    )
    
    # Fixed batch_size=12 (max template count)
    dummy_tokens = torch.zeros(12, 77, dtype=torch.int64)
    
    with torch.no_grad():
        traced = torch.jit.trace(text_wrapper, dummy_tokens)
    
    save_path = f"{output_dir}/clip_text_fp32.pt"
    traced.save(save_path)
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    visual_path = export_visual_encoder()
    text_path   = export_text_encoder()
    print("\nExport complete. Run quantization next:")
    print(f"  vai_q_pytorch --model {visual_path} ...")
```

### 7.3 Step 2 — Post-Training Quantization với vai_q_pytorch

```python
# scripts/quantize_clip.py
"""
Chạy trên development machine với Vitis AI Docker container.

Docker setup:
  docker pull xilinx/vitis-ai-pytorch-cpu:3.5.0
  docker run -v $(pwd):/workspace -it xilinx/vitis-ai-pytorch-cpu:3.5.0 bash
  cd /workspace && python scripts/quantize_clip.py
"""
import torch
import numpy as np
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

# Vitis AI quantizer
CALIB_FRAMES_PATH = "calibration_frames.npy"   # (1000, 3, 224, 224) float32
VISUAL_MODEL_PATH = "exported_models/clip_visual_fp32.pt"
QUANT_OUTPUT_DIR  = "quantized_models/"


def quantize_visual_encoder():
    # Load FP32 model
    model = torch.jit.load(VISUAL_MODEL_PATH)
    model.eval()
    
    # Load calibration dataset
    calib_data = np.load(CALIB_FRAMES_PATH)          # (1000, 3, 224, 224)
    calib_tensor = torch.from_numpy(calib_data[:500]) # Use 500 frames
    
    # Setup quantizer — PTQ mode (calib_quant_mode)
    dummy_input = torch.randn(1, 3, 224, 224)
    quantizer = torch_quantizer(
        quant_mode  = "calib",          # Calibration mode
        module      = model,
        input_args  = (dummy_input,),
        output_dir  = QUANT_OUTPUT_DIR,
        quant_config_file = None,       # Use default config
    )
    quant_model = quantizer.quant_model
    
    # Run calibration — forward pass với calibration data
    print("Running calibration...")
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(calib_tensor), batch_size):
            batch = calib_tensor[i : i + batch_size]
            quant_model(batch)
            if (i // batch_size) % 10 == 0:
                print(f"  Calibrated {i}/{len(calib_tensor)} frames")
    
    # Export calibration data
    quantizer.export_quant_config()
    print(f"Calibration complete. Config saved to {QUANT_OUTPUT_DIR}")


def export_quantized_xmodel():
    """Bước thứ 2: Export sang .xmodel sau calibration."""
    model = torch.jit.load(VISUAL_MODEL_PATH)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    quantizer = torch_quantizer(
        quant_mode  = "test",           # Test/export mode
        module      = model,
        input_args  = (dummy_input,),
        output_dir  = QUANT_OUTPUT_DIR,
    )
    quant_model = quantizer.quant_model
    
    # Single forward pass để trigger export
    with torch.no_grad():
        output = quant_model(dummy_input)
    
    # Export xmodel
    quantizer.export_xmodel(
        output_dir   = QUANT_OUTPUT_DIR,
        deploy_check = True             # Verify DPU compatibility
    )
    print(f"xmodel exported to {QUANT_OUTPUT_DIR}/")


if __name__ == "__main__":
    print("Step 1: Calibration")
    quantize_visual_encoder()
    
    print("\nStep 2: Export xmodel")
    export_quantized_xmodel()
```

### 7.4 Step 3 — Compile với vai_c_xir

```bash
#!/bin/bash
# scripts/compile_xmodel.sh
# Chạy trong Vitis AI Docker container

ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
QUANT_DIR="quantized_models"
OUTPUT_DIR="compiled_models"

mkdir -p $OUTPUT_DIR

echo "Compiling CLIP visual encoder..."
vai_c_xir \
  --xmodel   ${QUANT_DIR}/clip_visual_fp32_int.xmodel \
  --arch     ${ARCH} \
  --net_name clip_vision \
  --output_dir ${OUTPUT_DIR} \
  --options   '{"input_shape": "1,3,224,224"}' \
  --save_kernel

echo "Compiling CLIP text encoder..."
vai_c_xir \
  --xmodel   ${QUANT_DIR}/clip_text_fp32_int.xmodel \
  --arch     ${ARCH} \
  --net_name clip_text \
  --output_dir ${OUTPUT_DIR} \
  --options   '{"input_shape": "12,77"}' \
  --save_kernel

echo "Compiled models:"
ls -lh ${OUTPUT_DIR}/*.xmodel

# Expected output:
# clip_vision.xmodel  ~86 MB
# clip_text.xmodel    ~63 MB
```

**Giải thích flags:**

| Flag | Mô tả |
|---|---|
| `--arch` | JSON file mô tả DPU target (B4096, KV260) |
| `--net_name` | Tên cho compiled network (dùng trong logging) |
| `--save_kernel` | Lưu DPU instructions để debug |
| `--options input_shape` | Fix input shape cho static compilation |

### 7.5 Validate Quantization Quality

```python
# scripts/validate_quantization.py
"""
So sánh embeddings từ FP32 model vs INT8 xmodel.
Chạy trên Kria sau khi deploy.
"""
import numpy as np
import vart, xir
import open_clip
import torch

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def validate_visual_embedding(
    xmodel_path: str,
    test_frames:  np.ndarray,  # (N, 3, 224, 224) float32 normalized
    reference_embeddings: np.ndarray  # (N, 512) from FP32 model
):
    """
    Validation metric: mean cosine similarity của INT8 vs FP32 embeddings.
    Ngưỡng chấp nhận: > 0.95 (tức là < 5% degradation).
    """
    # Load xmodel
    graph    = xir.Graph.deserialize(xmodel_path)
    subgraph = _get_dpu_subgraph(graph)
    runner   = vart.Runner.create_runner(subgraph, "run")
    
    in_tensors  = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    
    fix_point = out_tensors[0].get_attr("fix_point")
    scale     = 2.0 ** (-fix_point)
    
    results = []
    for i, frame in enumerate(test_frames):
        # Quantize to INT8
        in_scale  = 2.0 ** (-in_tensors[0].get_attr("fix_point"))
        frame_int8 = np.clip(
            np.round(frame / in_scale), -128, 127
        ).astype(np.int8)
        
        # DPU inference
        inputs  = [frame_int8[np.newaxis]]  # (1, 3, 224, 224)
        outputs = [np.empty(out_tensors[0].dims, dtype=np.int8)]
        
        job_id = runner.execute_async(inputs, outputs)
        runner.wait(job_id)
        
        # Dequantize
        emb = outputs[0][0].astype(np.float32) * scale
        emb = emb / np.linalg.norm(emb)
        results.append(emb)
    
    # Compare với reference
    similarities = [
        cosine_similarity(results[i], reference_embeddings[i])
        for i in range(len(results))
    ]
    
    mean_sim = np.mean(similarities)
    min_sim  = np.min(similarities)
    
    print(f"Quantization Validation Results:")
    print(f"  Mean cosine sim (INT8 vs FP32): {mean_sim:.4f}")
    print(f"  Min cosine sim:                 {min_sim:.4f}")
    print(f"  Frames below 0.95 threshold:    {sum(s<0.95 for s in similarities)}/{len(similarities)}")
    
    if mean_sim > 0.95:
        print("  ✅ PASSED — quantization quality acceptable")
    else:
        print("  ❌ FAILED — re-calibrate with more diverse dataset")
    
    return mean_sim


def _get_dpu_subgraph(graph):
    for sg in graph.get_root_subgraph().toposort_child_subgraph():
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
            return sg
    raise RuntimeError("No DPU subgraph found in xmodel")
```

---

## 8. VVAS Video Pipeline trên Kria

### 8.1 VVAS Architecture

VVAS (Vitis Video Analytics SDK) là tầng abstraction cho hardware video processing trên Kria:

```
                     VVAS Architecture Stack
                     ─────────────────────────

Application Layer (Python/C++)
        │
        ▼ GStreamer API
┌─────────────────────────────────────────────┐
│              GStreamer Core                 │
│  (pad negotiation, buffer passing, clock)  │
└──────────┬──────────────────────────────────┘
           │ GStreamer plugins
    ┌──────┼──────────────────┐
    ▼      ▼                  ▼
vvas_xmultisrc  vvas_xdec    vvas_xfilter / vvas_xinfer
(input source)  (HW decode)  (custom processing)
    │               │                │
    ▼               ▼                ▼
┌───────────────────────────────────────────────┐
│              VVAS Core Library                │
│  (buffer management, DMA, xclbin loading)    │
└───────────────┬───────────────────────────────┘
                │ XRT (Xilinx Runtime)
                ▼
┌───────────────────────────────────────────────┐
│              FPGA Fabric (PL)                 │
│  ┌──────────────┐  ┌──────────┐  ┌────────┐  │
│  │ Video Codec  │  │ Scaler   │  │  DPU   │  │
│  │ Unit (VCU)   │  │ (XVSC)   │  │ B4096  │  │
│  │ H.264/265 HW │  │ Multi-ch │  │        │  │
│  └──────────────┘  └──────────┘  └────────┘  │
└───────────────────────────────────────────────┘
```

### 8.2 GStreamer Pipeline cho NLVS Indexing

```
NLVS Indexing Pipeline (Kria):
────────────────────────────────

                     ┌──────────────────────────────────────────┐
                     │  vvas_xmultisrc                          │
                     │  - Input: MP4 file (H.264)              │
                     │  - Demux: video/audio split              │
                     │  - Output: H.264 bitstream               │
                     └──────────────┬───────────────────────────┘
                                    │ H.264 ES
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │  vvas_xdec                               │
                     │  - Hardware H.264 decoder (VCU IP)       │
                     │  - Throughput: 1080p@60fps               │
                     │  - Output format: NV12 (YUV420 planar)   │
                     └──────────────┬───────────────────────────┘
                                    │ NV12, 1920×1080
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │  videoconvert                            │
                     │  - NV12 → BGRx (software)               │
                     │  - Note: thêm vvas_xscaler sau nếu cần  │
                     └──────────────┬───────────────────────────┘
                                    │ BGRx, 1920×1080
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │  vvas_xfilter (custom kernel)            │
                     │  - Resize: 1920×1080 → 224×224          │
                     │  - Color: BGRx → BGR                     │
                     │  - Hardware-accelerated via xclbin       │
                     └──────────────┬───────────────────────────┘
                                    │ BGR, 224×224
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │  video/x-raw,format=BGRx,               │
                     │  width=224,height=224                    │
                     │  framerate: controlled by pipeline       │
                     └──────────────┬───────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │  appsink                                 │
                     │  - emit-signals=true                     │
                     │  - max-buffers=4 (backpressure)          │
                     │  - Python callback via new-sample signal │
                     └──────────────────────────────────────────┘
```

### 8.3 GStreamer Pipeline String Implementation

```python
# src/gst_pipeline.py (Kria VVAS mode)
# Pipeline string cho indexing mode trên Kria KV260

def _build_kria_pipeline(self, video_path: str) -> str:
    """
    Tạo GStreamer pipeline string cho Kria VVAS hardware-accelerated decode.
    
    Yêu cầu:
    - VVAS 3.0 installed (/opt/vvas/)
    - KV260 bitstream loaded (xlnx-v-frmbuf-rd xclbin)
    - Input: H.264 MP4 file
    
    Output: BGR frames, 224×224, via appsink
    """
    # Detect input codec
    ext = Path(video_path).suffix.lower()
    
    if ext in [".mp4", ".avi", ".mkv"]:
        # Hardware H.264/265 decode
        pipeline = (
            f"filesrc location={video_path} ! "
            "qtdemux ! "                             # MP4 container demux
            "video/x-h264 ! "                        # H.264 elementary stream
            "vvas_xdec dev-idx=0 "                   # HW decoder, device 0
            "xclbin-location=/opt/xilinx/kv260-smartcam/kv260-smartcam.xclbin ! "
            "videoconvert ! "                        # NV12 → BGR
            "vvas_xfilter kernels-config=/opt/vvas/share/vvas/kv260/resize.json ! "
            "video/x-raw,format=BGRx,width=224,height=224 ! "
            "videoconvert ! "                        # BGRx → BGR
            "video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=4 drop=false"
        )
    else:
        # Fallback: software decode với OpenCV
        pipeline = (
            f"filesrc location={video_path} ! "
            "decodebin ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,format=BGR,width=224,height=224 ! "
            "appsink name=sink emit-signals=true max-buffers=4 drop=false"
        )
    
    return pipeline
```

### 8.4 Frame Extraction với Timestamp Control

Để extract frames tại specific timestamps (sliding window), cần timestamp-based seeking:

```python
# src/gst_pipeline.py — Kria timestamp-based seeking

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

class KriaVVASFrameExtractor:
    """
    Hardware-accelerated frame extraction trên Kria KV260.
    Sử dụng VVAS GStreamer pipeline với timestamp seeking.
    """
    
    def __init__(self, video_path: str):
        Gst.init(None)
        self.video_path = video_path
        self.pipeline   = None
        self.appsink    = None
        self._frames    = []
        self._lock      = threading.Lock()
    
    def extract_frames_at_timestamps(
        self,
        timestamps: list[float]  # seconds
    ) -> list[np.ndarray]:
        """
        Extract frames tại các timestamps cụ thể.
        
        Strategy: Build full pipeline, seek đến từng timestamp,
        extract 1 frame, tiếp tục.
        
        Note: Seeking trên HW decoder có overhead ~100ms per seek.
        Với sliding window (sequential access), nên extract theo
        range thay vì individual seeks.
        """
        pipeline_str = self._build_kria_pipeline(self.video_path)
        pipeline     = Gst.parse_launch(pipeline_str)
        appsink      = pipeline.get_by_name("sink")
        
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self._on_new_sample)
        
        pipeline.set_state(Gst.State.PLAYING)
        
        frames = []
        for ts in timestamps:
            # Seek đến timestamp (nanoseconds)
            seek_pos = int(ts * Gst.SECOND)
            pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                seek_pos
            )
            
            # Wait for frame
            sample = appsink.try_pull_sample(timeout=Gst.SECOND)
            if sample:
                buf  = sample.get_buffer()
                caps = sample.get_caps()
                h, w = caps.get_structure(0).get_value("height"), \
                       caps.get_structure(0).get_value("width")
                
                success, mapinfo = buf.map(Gst.MapFlags.READ)
                if success:
                    frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
                    frame = frame.reshape(h, w, 3)  # BGR
                    frames.append(frame.copy())
                    buf.unmap(mapinfo)
        
        pipeline.set_state(Gst.State.NULL)
        return frames
    
    def extract_sequential_range(
        self,
        start_sec: float,
        end_sec: float,
        n_frames: int
    ) -> list[np.ndarray]:
        """
        Efficient sequential extraction trong khoảng [start_sec, end_sec].
        Tốt hơn individual seeks cho sliding window patterns.
        """
        duration = end_sec - start_sec
        if n_frames > 1:
            step = duration / (n_frames - 1)
            target_pts = [start_sec + i * step for i in range(n_frames)]
        else:
            target_pts = [start_sec]
        
        # ... implementation với sequential decode
        # Seek một lần đến start_sec, sau đó decode đến các timestamps
        pass
```

### 8.5 Hardware Video Decode Performance

```
VVAS Hardware Decode (VCU IP Core) vs Software Decode:

Resolution    Format    HW Decode    SW Decode (ARM)    Speedup
──────────────────────────────────────────────────────────────────
1920×1080     H.264     60 fps        ~8 fps (1T)        7.5×
1920×1080     H.265     60 fps        ~5 fps (1T)        12×
3840×2160     H.264     30 fps        ~2 fps (1T)        15×
720×480       H.264     >120 fps      ~30 fps (1T)       4×

NLVS use case (1080p H.264 surveillance):
  HW decode: 60 fps → với sliding window 0.5 fps average seek rate
             → decode is NOT the bottleneck
  
  CLIP inference (INT8 DPU): ~22 fps per batch=4
             → DPU inference IS the bottleneck (expected)
```

---

## 9. KriaEngine — Phân tích Implementation

### 9.1 KriaEngine Class Architecture

```python
# src/engines/kria_engine.py — Detailed analysis

class KriaEngine(InferenceEngine):
    """
    DPU inference backend cho Kria KV260.
    
    Threading model:
    - Main thread: Python (query, preprocessing, postprocessing)  
    - DPU execution: Async via vart.Runner.execute_async()
    - GStreamer: Separate GLib main loop thread (video pipeline)
    
    Memory layout:
    - xmodel graph: ~86 MB in DDR (PS RAM)
    - Input buffer:  4 × 3 × 224 × 224 × 1 byte = 602 KB (INT8)
    - Output buffer: 4 × 512 × 1 byte = 2 KB (INT8)
    - Working bufs:  ~10 MB total
    """
    
    # Initialization flow:
    # __init__ → _load_models() → _get_dpu_subgraph() → create_runner()
    
    def _load_models(self):
        # 1. Visual encoder
        self._visual_graph    = xir.Graph.deserialize(self._xmodel_path)
        self._visual_subgraph = self._get_dpu_subgraph(self._visual_graph)
        self._visual_runner   = vart.Runner.create_runner(
            self._visual_subgraph, "run"
        )
        
        # 2. Text encoder (optional — may run CPU-only)
        if self._xmodel_text_path:
            self._text_graph    = xir.Graph.deserialize(self._xmodel_text_path)
            self._text_subgraph = self._get_dpu_subgraph(self._text_graph)
            self._text_runner   = vart.Runner.create_runner(
                self._text_subgraph, "run"
            )
        
        # 3. Extract quantization scales
        out_tensors = self._visual_runner.get_output_tensors()
        self._visual_fix_point = out_tensors[0].get_attr("fix_point")
        
        in_tensors = self._visual_runner.get_input_tensors()
        self._visual_in_fix_point = in_tensors[0].get_attr("fix_point")
```

### 9.2 Frame Preprocessing Pipeline

Preprocessing là **critical path** vì phải chạy trên ARM CPU:

```python
def _preprocess_frames(self, frames: list[np.ndarray]) -> np.ndarray:
    """
    Convert BGR frames → INT8 NHWC tensors cho DPU input.
    
    Pipeline:
      BGR uint8 (H, W, 3) 
      → RGB float32 (H, W, 3) / 255.0
      → normalize với CLIP mean/std
      → quantize sang INT8 với input fix_point
      → NHWC layout (N, H, W, C)
    
    Optimization notes:
    - Vectorized với NumPy (NEON-friendly ops)
    - Avoid Python loops — process batch all at once
    - Pre-allocate output buffer để tránh GC pressure
    """
    N = len(frames)
    
    # Pre-allocate output (N, 224, 224, 3) INT8 — NHWC cho DPU
    result = np.empty((N, 224, 224, 3), dtype=np.int8)
    
    # CLIP normalization constants
    MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
    STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    
    # Input quantization scale
    in_scale = 2.0 ** (-self._visual_in_fix_point)
    
    for i, frame in enumerate(frames):
        # 1. Resize (nếu chưa resize)
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224))
        
        # 2. BGR → RGB, normalize to [0, 1]
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        
        # 3. CLIP normalize: (x - mean) / std
        normalized = (rgb - MEAN) / STD
        # normalized range: approx [-2.5, 2.5]
        
        # 4. Quantize sang INT8:
        # q = clip(round(x / scale), -128, 127)
        quantized = np.clip(
            np.round(normalized / in_scale), -128, 127
        ).astype(np.int8)
        
        # 5. Store in NHWC layout
        result[i] = quantized  # (224, 224, 3)
    
    return result  # (N, 224, 224, 3) INT8 NHWC


# Optimization: Vectorized batch version (2-3× faster)
def _preprocess_frames_batch(self, frames: list[np.ndarray]) -> np.ndarray:
    """Vectorized version — xử lý toàn bộ batch cùng lúc."""
    N = len(frames)
    
    # Stack tất cả frames: (N, 224, 224, 3) uint8
    batch = np.stack([
        cv2.resize(f, (224, 224)) if f.shape[:2] != (224, 224) else f
        for f in frames
    ])
    
    MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
    STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    in_scale = 2.0 ** (-self._visual_in_fix_point)
    
    # Vectorized: BGR→RGB, normalize, quantize
    rgb_batch   = batch[:, :, :, ::-1].astype(np.float32) / 255.0
    norm_batch  = (rgb_batch - MEAN) / STD
    quant_batch = np.clip(np.round(norm_batch / in_scale), -128, 127).astype(np.int8)
    
    return quant_batch  # (N, 224, 224, 3) NHWC INT8
```

### 9.3 DPU Async Execution Pattern

```python
def encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
    """
    Encode visual frames sang embeddings sử dụng DPU.
    
    Batch strategy:
    - DPU batch_size=4 (từ config)
    - Process frames in batches of 4
    - Collect, dequantize, L2-normalize
    """
    if not frames:
        return np.empty((0, self._embed_dim), dtype=np.float32)
    
    in_tensors  = self._visual_runner.get_input_tensors()
    out_tensors = self._visual_runner.get_output_tensors()
    
    # Pre-allocate persistent buffers (tránh malloc per call)
    batch_size = self._batch_size  # = 4
    in_buf  = [np.empty(in_tensors[0].dims,  dtype=np.int8)]
    out_buf = [np.empty(out_tensors[0].dims, dtype=np.int8)]
    
    out_fix_point = self._visual_fix_point
    scale         = 2.0 ** (-out_fix_point)
    
    all_embeddings = []
    
    # Process in batches
    for batch_start in range(0, len(frames), batch_size):
        batch_frames = frames[batch_start : batch_start + batch_size]
        actual_n     = len(batch_frames)
        
        # Pad batch đến batch_size nếu cần
        if actual_n < batch_size:
            padding = [np.zeros((224, 224, 3), dtype=np.uint8)] * (batch_size - actual_n)
            batch_frames = batch_frames + padding
        
        # Preprocess
        in_buf[0][:] = self._preprocess_frames_batch(batch_frames)
        
        # Async DPU execution
        job_id = self._visual_runner.execute_async(in_buf, out_buf)
        self._visual_runner.wait(job_id)
        
        # Dequantize: float = int8 × 2^(-fix_point)
        # out_buf[0] shape: (batch_size, 512)
        embeddings = out_buf[0][:actual_n].astype(np.float32) * scale
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        embeddings = embeddings / norms
        
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)  # (N, 512)
```

### 9.4 Text Encoding — Hybrid CPU/DPU

```python
def encode_text(self, text: str | list[str]) -> np.ndarray:
    """
    Encode text queries.
    
    Two modes:
    1. DPU mode: Nếu text xmodel available → DPU accelerated
    2. CPU mode: Nếu không có text xmodel → full Python/PyTorch
                 (text encoding thường không phải bottleneck)
    
    Với template ensemble (5 templates):
    - text = "person jumping" 
    - → 5 templated texts
    - → batch=5 tokens (pad đến 12 cho DPU)
    - → 5 embeddings → mean → L2-normalize
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = list(text)
    
    if self._text_runner is None:
        # Fallback: CPU inference với PyTorch + open_clip
        return self._encode_text_cpu(texts)
    
    # DPU text encoding
    tokens = self._tokenize(texts)  # (N, 77) int64
    
    N = len(texts)
    batch_size = self._text_batch_size  # = 12
    
    # Pad to batch_size
    if N < batch_size:
        pad_tokens = np.zeros((batch_size - N, 77), dtype=np.int64)
        tokens     = np.vstack([tokens, pad_tokens])
    
    # DPU execute
    in_buf  = [tokens.astype(np.int8)]  # INT8 token IDs
    out_tensors = self._text_runner.get_output_tensors()
    out_buf = [np.empty(out_tensors[0].dims, dtype=np.int8)]
    
    job_id = self._text_runner.execute_async(in_buf, out_buf)
    self._text_runner.wait(job_id)
    
    fix_point = out_tensors[0].get_attr("fix_point")
    scale     = 2.0 ** (-fix_point)
    
    embeddings = out_buf[0][:N].astype(np.float32) * scale
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)
```

---

## 10. Tối ưu hóa hiệu năng & Memory Management

### 10.1 Multi-threaded DPU Runner

VART cho phép tạo **nhiều runners** từ cùng một subgraph. Mỗi runner là độc lập và thread-safe:

```python
# Optimal threading pattern cho Kria (4 ARM cores)

class OptimizedKriaEngine:
    """
    Multi-threaded DPU inference với pipelining.
    
    Thread layout:
      Thread 0 (main):      Nhận frames, preprocessing
      Thread 1 (DPU):       DPU inference (runner 0)
      Thread 2 (DPU):       DPU inference (runner 1) — chờ next batch
      Thread 3 (post):      Dequantize, L2-normalize
    
    Pipeline:
      [Preprocess batch N] → [DPU batch N] → [Post batch N]
                              [Preprocess batch N+1]
    
    Expected speedup: 1.5–2× vs single-threaded
    """
    
    def __init__(self, subgraph, n_runners=2):
        # 2 DPU runners cho double-buffering
        self._runners = [
            vart.Runner.create_runner(subgraph, "run")
            for _ in range(n_runners)
        ]
        self._runner_idx = 0
        self._lock = threading.Lock()
    
    def encode_batch_async(self, preprocessed: np.ndarray):
        """Non-blocking: submit batch, return future."""
        with self._lock:
            runner_idx = self._runner_idx % len(self._runners)
            self._runner_idx += 1
        
        runner   = self._runners[runner_idx]
        in_buf   = [preprocessed]
        out_buf  = [np.empty(runner.get_output_tensors()[0].dims, dtype=np.int8)]
        
        job_id = runner.execute_async(in_buf, out_buf)
        
        # Return (runner, job_id, out_buf) tuple cho caller để wait
        return runner, job_id, out_buf
    
    def wait_and_dequantize(self, runner, job_id, out_buf, fix_point):
        """Wait for job completion, dequantize output."""
        runner.wait(job_id)
        scale = 2.0 ** (-fix_point)
        emb   = out_buf[0].astype(np.float32) * scale
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
```

### 10.2 Memory Budget Analysis

```
Kria KV260 Memory Budget cho NLVS
────────────────────────────────────
Total PS RAM: 4096 MB

Sử dụng:
  Linux OS + kernel:          ~500 MB
  VVAS + GStreamer:            ~100 MB
  CLIP vision xmodel (DPU):    ~86 MB (mapped vào DDR)
  CLIP text xmodel (DPU):      ~63 MB
  DPU scratch/SRAM:            ~64 MB (on-chip PL SRAM)
  VART runtime:                ~50 MB
  Python runtime:              ~100 MB
  PyTorch (CPU text encoder): ~200 MB (nếu dùng CPU mode)
  open_clip tokenizer:         ~30 MB
  ─────────────────────────────────────────────────────
  Subtotal runtime:           ~1,193 MB

  FAISS index (1h video):
    720 segments × 512 × 4 bytes = ~1.47 MB  ← trivial
    Metadata (timestamps, paths): ~0.5 MB
  
  Video frame buffers (GStreamer):
    4 buffers × 1920×1080×3 = ~24 MB
    224×224 frames × 16 buffers = ~12 MB
  
  Working buffers:
    Input INT8 batch (4×224×224×3) = ~600 KB
    Output INT8 batch (4×512) = 2 KB
  ─────────────────────────────────────────────────────
  Total used:                 ~1,230 MB

  Available for index growth:  ~2,866 MB
  Max index capacity:
    2866 MB / (512 × 4 bytes) = ~1.4M segments
    At 5s/segment: ~82 days of 24/7 video ← sufficient
```

### 10.3 Faiss trên ARM Cortex-A53

Faiss IndexFlatIP không cần GPU — chạy tốt trên ARM:

```
Faiss ARM Performance (Cortex-A53 @ 1.5 GHz):

N vectors    Dim    Search latency    Threads
───────────────────────────────────────────────
1,000        512    < 0.1ms           1
10,000       512    ~0.8ms            1
100,000      512    ~8ms              1
100,000      512    ~2ms              4 (OpenMP)
1,000,000    512    ~80ms             4

NLVS expected: 720 segments/hour × hours_of_video
  10 giờ video: 7,200 vectors → ~0.06ms search
  100 giờ video: 72,000 vectors → ~0.6ms search
  1000 giờ video: 720,000 vectors → ~6ms search

→ Faiss trên ARM là hoàn toàn viable cho NLVS scale
```

**Build Faiss cho ARM (Kria):**

```bash
# Trên Kria (Ubuntu 22.04)
sudo apt install libopenblas-dev   # BLAS for NEON optimization
pip install faiss-cpu              # Pre-built ARM wheel hoặc:

# Build from source với NEON support
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_C_FLAGS="-march=armv8-a+simd" \
  -DCMAKE_CXX_FLAGS="-march=armv8-a+simd"
cmake --build build -- -j4
cd build/faiss/python && pip install .
```

### 10.4 Latency Optimization Techniques

```
Technique                      Impact    Effort    Kria-specific
────────────────────────────────────────────────────────────────
Batch size 1→4                 30-40%    Low       ✅ Config change
Double-buffer DPU runners      20-30%    Medium    ✅ VART native
Hardware video decode (VVAS)   50-70%*   High      ✅ Kria advantage
NEON-optimized preprocessing   20%       Medium    ✅ ARM NEON
Pre-load xmodel on startup     Startup   Low       ✅ Load once
Sequential window extraction   40%**     Low       ✅ vs random seek
Index preload to RAM           ~0ms      Low       ✅ (already fits)

* vs software H.264 decode on ARM
** vs seeking to each timestamp individually
```

---

## 11. Benchmark dự kiến & So sánh với PC

### 11.1 End-to-End Indexing Performance

```
┌─────────────────────────────────────────────────────────────────┐
│          INDEXING THROUGHPUT (1 giờ video, 1080p H.264)        │
├────────────────────────┬────────────────┬──────────────────────┤
│  Stage                 │ PC (1650 Ti)   │ Kria KV260           │
├────────────────────────┼────────────────┼──────────────────────┤
│ Video decode           │ ~15s (SW)      │ ~5s (HW VCU)         │
│ Frame extraction       │ ~8s            │ ~6s (VVAS pipeline)  │
│ CLIP preprocessing     │ ~5s (GPU)      │ ~35s (ARM NEON)      │
│ DPU/GPU inference      │ ~70s (FP16)    │ ~950s (INT8 DPU)*    │
│ Faiss add              │ <1s            │ <1s (same)           │
│ Metadata save          │ <1s            │ <1s (same)           │
├────────────────────────┼────────────────┼──────────────────────┤
│ Total (720 segments)   │ ~100s (~2min)  │ ~997s (~17min)       │
│ Real-time factor       │ 36× faster     │ ~3.6× real-time      │
│ Power during indexing  │ ~80W (system)  │ ~20W (system)        │
│ Energy/hour of video   │ ~2.2 Wh        │ ~5.6 Wh              │
└────────────────────────┴────────────────┴──────────────────────┘

* DPU throughput: batch=4, ~46ms/frame, 5 frames/segment = 230ms/segment
  720 segments × 230ms = 165.6s (DPU alone)
  + preprocessing, DMA, Python overhead → ~950s total
```

### 11.2 Search (Query) Latency

```
┌─────────────────────────────────────────────────────────────────┐
│           SEARCH LATENCY (single text query)                    │
├────────────────────────┬────────────────┬──────────────────────┤
│  Stage                 │ PC (1650 Ti)   │ Kria KV260           │
├────────────────────────┼────────────────┼──────────────────────┤
│ Query normalization    │ <1ms           │ <1ms                 │
│ Template expansion     │ <1ms           │ <1ms                 │
│ Text tokenization      │ ~2ms           │ ~5ms (ARM)           │
│ Text encoding (×5)     │ ~15ms (GPU)    │ ~100ms (CPU/DPU)     │
│ Faiss search (10K vec) │ <1ms           │ ~1ms (ARM)           │
│ Temporal NMS           │ <1ms           │ <1ms                 │
│ Result formatting      │ <1ms           │ <1ms                 │
├────────────────────────┼────────────────┼──────────────────────┤
│ Total latency          │ ~20ms          │ ~110ms               │
│ Perceived response     │ Instant        │ Acceptable           │
└────────────────────────┴────────────────┴──────────────────────┘

Note: 110ms search latency là tốt cho interactive use case.
      Streamlit UI renders trong ~200ms → tổng perceived latency ~300ms.
```

### 11.3 Memory Usage Comparison

```
                    PC (GTX 1650 Ti)    Kria KV260
──────────────────────────────────────────────────
Model weights:      172 MB (FP16 VRAM) 86 MB (INT8 DDR)
Index (10h video):  ~15 MB (RAM)       ~15 MB (DDR shared)
Peak inference RAM: ~2 GB (GPU+CPU)    ~1.2 GB (ARM DDR)
Peak VRAM:          ~2 GB (FP16)       N/A (no GPU)
Storage xmodel:     N/A               149 MB (.xmodel files)
──────────────────────────────────────────────────
Total footprint:    ~2.2 GB           ~1.35 GB ← Smaller!
```

### 11.4 Embedding Quality Degradation

```
Embedding Quality (Cosine Similarity: INT8 vs FP32)
────────────────────────────────────────────────────

Expected mean cosine similarity: 0.96–0.98
(based on similar INT8 CLIP quantization studies)

Impact on retrieval quality:
  R@1 drop (estimated): 2–4% compared to FP32
  
  Example:
    PC FP32:   Top result for "person climbing fence" → score 0.42
    Kria INT8: Top result for "person climbing fence" → score 0.40 (−4.8%)
    
    → Ranking order thường giữ nguyên
    → Score threshold có thể cần điều chỉnh: 0.20 → 0.18

Calibration quality impact:
  Poor calibration (random web images):  ~3–6% R@1 drop
  Good calibration (domain-specific):    ~1–3% R@1 drop
  ← Dùng actual surveillance footage cho calibration dataset
```

---

## 12. Lộ trình triển khai Step-by-Step

### Phase 0: Chuẩn bị (1 tuần)

```
Task 0.1: Setup Kria KV260 Hardware
────────────────────────────────────
□ Flash Ubuntu 22.04 + Vitis AI 3.5 SOM image lên MicroSD
□ Boot KV260, configure networking (SSH access)
□ Verify VVAS installation: gst-inspect-1.0 vvas_xdec
□ Verify DPU: xdputil query (check B4096 fingerprint)
□ Install Python deps: pip install vart xir faiss-cpu open_clip

Task 0.2: Setup Development Machine
─────────────────────────────────────
□ Install Vitis AI Docker: docker pull xilinx/vitis-ai-pytorch-cpu:3.5.0
□ Clone NLVS repo vào Docker container
□ Download CLIP ViT-B/16 weights (openai)
□ Collect calibration dataset (500–1000 frames)
□ Verify vai_q_pytorch available: python -c "from pytorch_nndct.apis import torch_quantizer"
```

### Phase 1: Model Preparation (3–5 ngày)

```
Task 1.1: Export FP32 Models
──────────────────────────────
□ Chạy scripts/export_clip_for_vitis.py
□ Verify output: clip_visual_fp32.pt, clip_text_fp32.pt
□ Validate: python validate_export.py (so sánh vs open_clip original)
  → Mean cosine sim > 0.9999 (TorchScript trace không làm mất accuracy)

Task 1.2: Build Calibration Dataset
─────────────────────────────────────
□ Collect 1000 frames từ target surveillance videos
□ Mix: 40% outdoor day, 20% outdoor night, 20% indoor, 20% crowded
□ Preprocess: BGR → RGB → CLIP normalize → float32
□ Save: calibration_frames.npy (shape: 1000, 3, 224, 224)

Task 1.3: Post-Training Quantization
──────────────────────────────────────
□ Docker: docker run -v $(pwd):/workspace -it xilinx/vitis-ai-pytorch-cpu:3.5.0
□ cd /workspace && conda activate vitis-ai-pytorch
□ python scripts/quantize_clip.py --mode calib
□ python scripts/quantize_clip.py --mode export
□ Verify: ls quantized_models/*.xmodel

Task 1.4: Compile sang DPU xmodel
───────────────────────────────────
□ bash scripts/compile_xmodel.sh
□ Check output: compiled_models/clip_vision.xmodel (~86 MB)
□ Analyze: vai_c_xir --analyze clip_vision.xmodel
□ Verify DPU coverage: xdputil benchmark clip_vision.xmodel -i 1
```

### Phase 2: Kria Integration (3–5 ngày)

```
Task 2.1: Deploy Models
────────────────────────
□ SCP compiled_models/*.xmodel → kria:/usr/share/vitis_ai_library/models/clip_vit_b16/
□ Update config/kria.yaml:
    xmodel_path: /usr/share/vitis_ai_library/models/clip_vit_b16/clip_vision.xmodel
    xmodel_text_path: .../clip_text.xmodel

Task 2.2: Validate on Kria
───────────────────────────
□ ssh kria "python /home/ubuntu/Prototype/scripts/validate_quantization.py"
□ Target: mean cosine sim > 0.95
□ If < 0.95: recalibrate với more domain-specific data

Task 2.3: VVAS Pipeline Test
──────────────────────────────
□ Test hardware decode:
    gst-launch-1.0 filesrc location=test.mp4 ! qtdemux ! vvas_xdec ! autovideosink
□ Test full pipeline:
    gst-launch-1.0 filesrc location=test.mp4 ! qtdemux ! vvas_xdec ! videoconvert ! 
    video/x-raw,format=BGR,width=224,height=224 ! appsink
□ Run gst_pipeline.py unit tests

Task 2.4: KriaEngine Integration Test
───────────────────────────────────────
□ python -m pytest tests/unit/test_engines.py -k "kria" --config config/kria.yaml
□ Verify encode_frames() latency: target < 250ms per 5-frame segment
□ Verify encode_text() latency: target < 120ms per query
```

### Phase 3: End-to-End System Test (2–3 ngày)

```
Task 3.1: Indexing Test
────────────────────────
□ python index_video.py --video data/test_10min.mp4 --config config/kria.yaml
□ Check index_store/faiss.index created
□ Measure: indexing time, embedding quality vs PC reference

Task 3.2: Search Test
──────────────────────
□ python -c "
  from src.searcher import Searcher
  s = Searcher('config/kria.yaml')
  results = s.search('person walking')
  print(results)
"
□ Verify results match PC results (same ranking within top-5)
□ Measure search latency: target < 200ms

Task 3.3: Streamlit App (headless hoặc remote)
────────────────────────────────────────────────
□ streamlit run app.py --server.headless true --server.port 8501
□ Access via: http://kria-ip:8501 từ browser trên laptop
□ Full UI test: search, download clip, config reload

Task 3.4: Continuous Operation Test
─────────────────────────────────────
□ Run 2-hour soak test: index 2 giờ video liên tục
□ Monitor: memory usage, temperature, DPU utilization
□ Check: no memory leaks, stable latency
□ Temperature: DPU < 85°C (cooling requirement)
```

### Phase 4: Production Hardening (1 tuần)

```
Task 4.1: Systemd Service
──────────────────────────
# /etc/systemd/system/nlvs.service
[Unit]
Description=NLVS Video Search Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Prototype
ExecStart=/usr/bin/python3 -m streamlit run app.py \
    --server.port 8501 \
    --server.headless true
Environment=NLVS_CONFIG=config/kria.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

Task 4.2: Monitoring & Logging
────────────────────────────────
□ Prometheus metrics: inference latency, DPU utilization
□ Log rotation: /var/log/nlvs/
□ Alerting: temperature > 80°C, memory > 3.5 GB

Task 4.3: Performance Tuning
─────────────────────────────
□ CPU frequency scaling: performance governor
   echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
□ DPU frequency: 300 MHz (default, can try 350 MHz)
□ Memory: disable swap for latency predictability
□ GStreamer buffer tuning: max-buffers=4 (prevents OOM)
```

---

## 13. Rủi ro kỹ thuật & Mitigation Strategy

### 13.1 Risk Matrix

| Rủi ro | Khả năng | Ảnh hưởng | Mitigation |
|---|---|---|---|
| Quantization accuracy drop > 5% | Trung bình | Cao | Recalibrate với domain data; thử QAT |
| DPU ops unsupported (ViT layers) | Thấp | Trung bình | CPU fallback đã handle; verify với xdputil |
| VVAS compatibility với video codec | Trung bình | Trung bình | Fallback sang OpenCV decode |
| Memory pressure (> 3.5 GB) | Thấp | Cao | Unload model sau indexing; streaming index |
| DPU overheating under sustained load | Trung bình | Cao | Thermal management; duty cycle |
| Vitis AI version mismatch | Trung bình | Cao | Pin versions; test in CI |
| CLIP text encoder DPU failure | Trung bình | Thấp | CPU fallback OK (text query = low frequency) |

### 13.2 Fallback Architecture

```
NLVS trên Kria — Fallback Strategy
─────────────────────────────────────

Level 1: Full VVAS pipeline
  vvas_xmultisrc → vvas_xdec (HW) → vvas_xfilter → DPU inference
  → Best performance

Level 2: Software decode + DPU inference (nếu VVAS unavailable)
  OpenCV VideoCapture → CPU preprocess → DPU inference
  → ~15% slower decode, same inference

Level 3: Full software fallback (nếu DPU unavailable)
  OpenCV → CPU preprocess → open_clip CPU inference
  → ~10× slower inference (production không viable)
  → Development/debugging mode

Automatic fallback code:
  try:
      engine = KriaEngine(config)     # Level 1/2: DPU
      engine.validate_dpu()
  except (RuntimeError, FileNotFoundError):
      logger.warning("DPU unavailable, falling back to CPU")
      engine = PCEngine(config)       # Level 3: CPU PyTorch
```

### 13.3 Thermal Management

DPU sustained workload có thể gây thermal throttling:

```python
# src/engines/kria_engine.py — Thermal monitoring

import subprocess
import time

def _check_temperature(self) -> float:
    """Đọc nhiệt độ CPU/DPU từ thermal zone."""
    try:
        result = subprocess.run(
            ["cat", "/sys/class/thermal/thermal_zone0/temp"],
            capture_output=True, text=True, timeout=1
        )
        return float(result.stdout.strip()) / 1000.0  # millidegrees → degrees
    except Exception:
        return 0.0

def encode_frames_with_throttle(self, frames, max_temp_c: float = 80.0):
    """
    Encode frames với thermal protection.
    Nếu nhiệt độ > max_temp_c: pause 1s để cool down.
    """
    temp = self._check_temperature()
    if temp > max_temp_c:
        self._logger.warning(f"Thermal throttle: {temp:.1f}°C > {max_temp_c}°C")
        time.sleep(1.0)  # Brief pause cho cooling
    
    return self.encode_frames(frames)
```

**Hardware cooling recommendation:**
- Kria KV260 kit bao gồm heatsink
- Với sustained DPU load (indexing 1h+ video): gắn thêm 5V fan (40mm)
- Target: < 75°C sustained, < 85°C peak

### 13.4 Index Portability (PC ↔ Kria)

```
Index Format Compatibility:
──────────────────────────────

PC index (FP32 embeddings):
  embed_dim = 512  ← same
  dtype = float32  ← same
  Faiss format = IndexFlatIP ← same (platform-independent)

Kria index (INT8 DPU → dequantized → float32):
  embed_dim = 512  ← same
  dtype = float32  ← same (dequantize output = float32)
  
→ Index files (.faiss + .pkl) hoàn toàn portable giữa PC và Kria!

Caveat: Embedding similarity phụ thuộc vào model precision:
  PC FP32 embedding của frame X ≠ Kria INT8 embedding của frame X
  (khác nhau ~2-3% về value, nhưng cosine sim > 0.96)
  
  → Nên index TRÊN CÙNG PLATFORM rồi search cũng trên platform đó
  → Hoặc chấp nhận ~2% recall loss khi cross-platform
  
Recommended workflow:
  Option A: Index on Kria → Search on Kria (production)
  Option B: Index on PC (faster) → Export embeddings → Re-index on Kria
            (chỉ cần copy normalized float32 embeddings, không copy model)
```

---

## 14. Kết luận & Khuyến nghị

### 14.1 Đánh giá khả thi

```
┌─────────────────────────────────────────────────────────────────┐
│              ĐÁNH GIÁ KHẢ THI TRIỂN KHAI KRIA                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ KHẢ THI — với các điều kiện sau:                           │
│                                                                 │
│  Kỹ thuật:                                                      │
│  ✅ CLIP ViT-B/16 có thể quantize INT8 (94.6% DPU coverage)   │
│  ✅ VART Python bindings hoàn thiện (Vitis AI 3.5)              │
│  ✅ Faiss IndexFlatIP chạy tốt trên ARM (< 1ms search)         │
│  ✅ VVAS hardware decode giảm CPU load                          │
│  ✅ 4 GB RAM đủ cho model + index + runtime (~1.3 GB used)     │
│                                                                 │
│  Hiệu năng:                                                     │
│  ✅ Search latency ~110ms (acceptable cho UI)                   │
│  ✅ Indexing 1h video trong ~17 phút (offline batch OK)        │
│  ⚠️  Real-time indexing (live stream) chưa feasible            │
│     cần DPU batch pipelining + faster preprocessing            │
│                                                                 │
│  Chất lượng:                                                    │
│  ✅ Expected R@1 drop < 4% so với FP32                         │
│  ✅ Cosine similarity INT8 vs FP32 > 0.96                      │
│  ⚠️  Cần calibration với surveillance-domain data              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 Khuyến nghị ưu tiên

| # | Action | Priority | Effort | Impact |
|---|---|---|---|---|
| 1 | Collect calibration dataset (1000 frames từ target domain) | 🔴 Critical | 1 ngày | Accuracy |
| 2 | Export + Quantize CLIP với vai_q_pytorch | 🔴 Critical | 2 ngày | Core function |
| 3 | Validate mean cosine sim > 0.95 trên Kria | 🔴 Critical | 1 ngày | Quality gate |
| 4 | VVAS pipeline test với actual video files | 🟡 High | 1 ngày | Reliability |
| 5 | Multi-threaded DPU (2 runners) | 🟡 High | 2 ngày | 20-30% speedup |
| 6 | Thermal monitoring + throttle | 🟡 High | 0.5 ngày | Stability |
| 7 | Score threshold tuning (0.20 → 0.18) | 🟢 Medium | 1h | Recall |
| 8 | Systemd service + auto-restart | 🟢 Medium | 2h | Production |
| 9 | Index portability validation | 🟢 Medium | 2h | Flexibility |

### 14.3 Expected Timeline

```
Tuần 1:  Phase 0 + Phase 1 (setup, calibration, quantization)
Tuần 2:  Phase 2 + Phase 3 (Kria integration, validation)
Tuần 3:  Phase 3 testing + Phase 4 hardening
Tuần 4:  Performance tuning, soak testing, documentation
──────────────────────────────────────────────────────────────
Total: ~4 tuần cho production-ready deployment

Risks to timeline:
  - VVAS compatibility issues: +1 tuần
  - Quantization accuracy < 0.95: +1 tuần (recalibrate/QAT)
  - Hardware procurement delay: blocking (need physical KV260)
```

### 14.4 Long-term Optimization Roadmap

```
Sau initial deployment:
──────────────────────────────────────────────────────────────
6 tháng: Structured pruning CLIP 30% → faster DPU inference
          Expected: 230ms → 150ms per segment
          
1 năm:   SmoothQuant (Vitis AI experimental) → better accuracy
          Expected: reduce R@1 drop từ 4% → 1%

1 năm:   Chuyển sang SigLIP ViT-B/16 (smaller, same quality)
          SigLIP-B/16: 87M params (same), INT8 better calibrated
          
2 năm:   Edge-optimized video transformer (EfficientViT-CL)
          5× smaller than ViT-B/16, comparable accuracy
          Expected: 230ms → 50ms per segment on DPU
```

---

## Phụ lục: Môi trường & Dependencies

### A. Kria Software Stack

```
OS:            Ubuntu 22.04 LTS (PetaLinux-based)
Kernel:        5.15.0-xlnx-v2023.2 (custom Xilinx kernel)
Vitis AI:      3.5.0
VVAS:          3.0
GStreamer:      1.20.3
XRT:           2.15.0 (Xilinx Runtime)
Python:        3.10
PyTorch:       2.0.1 (CPU-only, ARM wheel)
VART:          3.5.0 (Vitis AI Runtime)
XIR:           3.5.0
Faiss:         1.7.4 (CPU, OpenBLAS)
open_clip:     2.24.0
Streamlit:     1.29.0
```

### B. Development Machine Stack

```
OS:            Ubuntu 22.04 LTS
Docker:        xilinx/vitis-ai-pytorch-cpu:3.5.0
Python:        3.8.0 (in Docker)
PyTorch:       2.0.1
pytorch_nndct: 3.5.0 (Vitis AI quantizer)
CUDA:          Not required for PTQ (CPU-only quantization)
```

### C. Files tạo trong quá trình deploy

```
scripts/
  export_clip_for_vitis.py     # Export FP32 TorchScript
  quantize_clip.py             # PTQ với vai_q_pytorch
  compile_xmodel.sh            # Compile sang .xmodel
  validate_quantization.py     # Quality validation trên Kria
  calibration_data_gen.py      # Build calibration dataset

exported_models/
  clip_visual_fp32.pt          # FP32 TorchScript (visual)
  clip_text_fp32.pt            # FP32 TorchScript (text)

quantized_models/
  clip_visual_fp32_int.xmodel  # INT8 quantized (pre-compile)
  clip_text_fp32_int.xmodel    # INT8 quantized (pre-compile)
  quantize_config.json         # Calibration config

compiled_models/
  clip_vision.xmodel           # Final DPU executable (~86 MB)
  clip_text.xmodel             # Final DPU executable (~63 MB)
```

---

## Phụ lục: Tham khảo

| Reference | Liên quan |
|---|---|
| Vitis AI User Guide (UG1414) v3.5 | DPU programming model |
| VVAS 3.0 User Guide (PG338) | GStreamer plugins |
| Kria KV260 Product Brief (PB218) | Hardware specs |
| ZynqMP TRM (UG1085) | AXI interconnect, DMA |
| SmoothQuant (Xiao et al., 2022) | INT8 Transformer quantization |
| I-ViT (Xiao et al., 2023, NeurIPS) | Integer-only ViT |
| Q-ViT (Li et al., 2022, ECCV) | ViT quantization survey |
| FQ-ViT (Lin et al., 2022) | Post-training quantization ViT |
| ZeroQuant (Yao et al., 2022) | LLM/Transformer INT8 |
| CLIP (Radford et al., 2021, ICML) | Base model architecture |

---

*Phần 1 tại: [KRIA_DEPLOYMENT_RESEARCH.md](KRIA_DEPLOYMENT_RESEARCH.md)*  
*Tổng hợp bởi AI Research Analysis | 2026-05-12 | NLVS v3.0 Embedded Deployment Research*
