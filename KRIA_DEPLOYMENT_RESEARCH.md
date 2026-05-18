# Nghiên cứu Triển khai NLVS trên AMD Kria KV260

> **Phiên bản:** 1.0 (Part 1/2)  
> **Ngày:** 2026-05-12  
> **Hệ thống nguồn:** NLVS v3.0 (GTX 1650 Ti, Phase 3 hoàn thành)  
> **Mục tiêu:** Triển khai hoàn chỉnh pipeline tìm kiếm video bằng ngôn ngữ tự nhiên trên AMD Kria KV260 SOM  
> **Phương pháp:** INT8 Post-Training Quantization → DPU inference via Vitis AI 3.5 + VVAS video pipeline

---

## Mục lục

### Phần 1 (tài liệu này)
1. [Phân tích phần cứng AMD Kria KV260](#1-phân-tích-phần-cứng-amd-kria-kv260)
2. [Kiến trúc DPU B4096 — Deep Dive](#2-kiến-trúc-dpu-b4096--deep-dive)
3. [Vitis AI Ecosystem — Lý thuyết & Workflow](#3-vitis-ai-ecosystem--lý-thuyết--workflow)
4. [Lý thuyết Quantization cho Deep Neural Networks](#4-lý-thuyết-quantization-cho-deep-neural-networks)
5. [Quantization CLIP ViT-B/16 — Phân tích chuyên sâu](#5-quantization-clip-vit-b16--phân-tích-chuyên-sâu)
6. [Thách thức kỹ thuật đặc thù của Transformer trên DPU](#6-thách-thức-kỹ-thuật-đặc-thù-của-transformer-trên-dpu)

### Phần 2 (KRIA_DEPLOYMENT_RESEARCH_P2.md)
7. Vitis AI Quantization Workflow chi tiết
8. Compilation sang .xmodel với vai_c_xir
9. VVAS Video Pipeline trên Kria
10. KriaEngine Integration với NLVS
11. Tối ưu hóa hiệu năng & Memory Management
12. Benchmark dự kiến & So sánh với PC
13. Lộ trình triển khai step-by-step
14. Rủi ro kỹ thuật & Mitigation

---

## 1. Phân tích phần cứng AMD Kria KV260

### 1.1 Tổng quan SOM (System on Module)

AMD Kria KV260 là một **AI Starter Kit** xây dựng trên SOM K26, tích hợp Zynq UltraScale+ MPSoC — một kiến trúc heterogeneous kết hợp **Processing System (PS)** và **Programmable Logic (PL)** trên cùng một chip.

```
┌─────────────────────────────────────────────────────────────────┐
│                   AMD Kria KV260 SOM                           │
│                   Zynq UltraScale+ MPSoC                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────┐   │
│  │  Processing System  │    │    Programmable Logic (PL)   │   │
│  │       (PS)          │    │                              │   │
│  │                     │    │  ┌────────────────────────┐  │   │
│  │  ARM Cortex-A53 ×4  │    │  │   DPU B4096 (IP Core)  │  │   │
│  │  ARM Cortex-R5F ×2  │◄──►│  │   4096 DSP ops/cycle   │  │   │
│  │  Mali-400 MP2 GPU   │    │  │   Peak: ~1.4 TOPS INT8  │  │   │
│  │                     │    │  └────────────────────────┘  │   │
│  │  Memory:            │    │                              │   │
│  │  4 GB LPDDR4 (PS)   │    │  ┌────────────────────────┐  │   │
│  │  512 KB OCM         │    │  │  Video IP Cores        │  │   │
│  │                     │    │  │  H.264/265 Decoder     │  │   │
│  └─────────────────────┘    │  │  Multi-scaler (XVSC)   │  │   │
│                              │  │  Color Convert (VCSC)  │  │   │
│                              │  └────────────────────────┘  │   │
│                              │                              │   │
│                              │  ┌────────────────────────┐  │   │
│                              │  │  AXI Interconnect      │  │   │
│                              │  │  PS↔PL high-BW bridge  │  │   │
│                              │  └────────────────────────┘  │   │
│                              └──────────────────────────────┘   │
│                                                                 │
│  Storage: 8 GB eMMC + MicroSD + USB 3.0                        │
│  I/O: GbE, USB 3.0/2.0, MIPI CSI-2, DisplayPort, GPIO        │
│  Power: 15W typical, 25W max (vs GTX 1650 Ti: 55W)           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Thông số kỹ thuật chi tiết

| Thành phần | Specification | Ý nghĩa với NLVS |
|---|---|---|
| **CPU** | ARM Cortex-A53 × 4 @ 1.5 GHz | Chạy Python runtime, tokenizer, Faiss |
| **DPU** | B4096 @ 300 MHz | Inference CLIP visual encoder |
| **RAM (PS)** | 4 GB LPDDR4 @ 2133 MHz | Index storage, embeddings, model weights |
| **FPGA Fabric** | 256K LUT, 1.5K DSP48 | Video decode, pre/post-processing |
| **Bandwidth PS↔PL** | ~19.2 GB/s (AXI HP) | DMA transfer frames→DPU |
| **Storage** | 8 GB eMMC + MicroSD | Model files, video files, index |
| **Power** | 5W (idle) – 25W (full load) | ~4× tiết kiệm hơn GTX 1650 Ti setup |
| **OS** | Ubuntu 22.04 + PetaLinux | VVAS, Python 3.10, OpenCV |

### 1.3 So sánh tổng thể với môi trường PC

```
                   PC (GTX 1650 Ti)         Kria KV260
                   ─────────────────         ──────────
Peak INT8 TOPS:    ~16 TOPS (GPU)            ~1.4 TOPS (DPU)
FP16 TFLOPS:       2.9 TFLOPS               N/A (DPU INT8 only)
RAM:               8–16 GB system            4 GB LPDDR4
VRAM:              4 GB GDDR6               Shared PS RAM
Video Decode:      Software (OpenCV)        Hardware H.264/265 IP
Power:             ~100W (system)           ~15W (system)
Cost:              ~$300–400 (GPU)          ~$249 (KV260 kit)
Form factor:       Desktop / laptop         85×85mm SOM
──────────────────────────────────────────────────────────
Inference ratio:   1×  (baseline)           ~10–15× slower
Power ratio:       1×                       ~7× more efficient
TOPS/Watt:         0.16 TOPS/W              0.09 TOPS/W (DPU)
```

**Nhận xét:** Kria KV260 có throughput thấp hơn ~10–15× so với GTX 1650 Ti cho neural network inference, nhưng **tiêu thụ điện thấp hơn ~7×** và có **hardware video decode**, rất phù hợp cho ứng dụng giám sát edge (surveillance) chạy 24/7.

---

## 2. Kiến trúc DPU B4096 — Deep Dive

### 2.1 DPU là gì?

DPU (Deep learning Processing Unit) trong Vitis AI là một **IP Core** được viết bằng VHDL/Verilog và compile vào FPGA fabric của Zynq. Không giống GPU (fixed silicon), DPU là **programmable logic** — có thể customize số lượng cores, precision, và memory layout.

**B4096** là tên cấu hình DPU: **B** = base architecture, **4096** = số MAC operations mỗi clock cycle.

### 2.2 DPU B4096 — Kiến trúc nội bộ

```
                     DPU B4096 Internal Architecture
                     ──────────────────────────────

  AXI Memory Map Interface
  (từ PS hoặc DDR)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │                  Instruction Fetch Unit                 │
  │  Đọc DPU Instructions từ .xmodel compiled program      │
  └──────────────────────┬──────────────────────────────────┘
                         │ decoded ops
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │               Computation Engine                        │
  │                                                         │
  │   ┌──────────────────────────────────────────────┐     │
  │   │           CONV Engine                        │     │
  │   │  B4096 = 4096 MACs/cycle @ 300 MHz          │     │
  │   │  = 4096 × 300M × 2 = 2.46 TOPS INT8        │     │
  │   │                                              │     │
  │   │  Data format: INT8, NHWC layout              │     │
  │   │  Supports: Conv2D, DepthwiseConv, BatchNorm  │     │
  │   └──────────────────────────────────────────────┘     │
  │                                                         │
  │   ┌──────────────────────────────────────────────┐     │
  │   │         MISC Engine                          │     │
  │   │  Activation: ReLU, Sigmoid, Tanh (approx)  │     │
  │   │  Pooling: MaxPool, AvgPool                  │     │
  │   │  Elementwise: Add, Mul                       │     │
  │   └──────────────────────────────────────────────┘     │
  │                                                         │
  │   ┌──────────────────────────────────────────────┐     │
  │   │         LOAD / SAVE Engine                   │     │
  │   │  DMA: DDR ↔ Local Buffer (on-chip SRAM)     │     │
  │   │  Buffer size: ~9 MB on-chip                  │     │
  │   └──────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────┘
```

### 2.3 Supported và Unsupported Operations

Đây là điểm **cực kỳ quan trọng** vì không phải mọi layer trong CLIP đều có thể chạy trên DPU:

#### ✅ DPU Native Support (hardware-accelerated)

| Operation | Mô tả |
|---|---|
| `Conv2D` | Convolution 2D (standard, dilated, group) |
| `DepthwiseConv2D` | Depthwise separable convolution |
| `TransposedConv2D` | Deconvolution |
| `BatchNormalization` | Fused vào Conv (inference mode) |
| `ReLU`, `ReLU6` | Activation functions |
| `MaxPool2D`, `AvgPool2D` | Pooling |
| `Elementwise Add/Mul` | Skip connections (ResNet-style) |
| `Dense` / `Linear` (static) | Fully connected, nếu input shape cố định |
| `Concat` | Channel concatenation |

#### ❌ DPU Không hỗ trợ → Fallback CPU

| Operation | Lý do không hỗ trợ | Ảnh hưởng CLIP |
|---|---|---|
| `Softmax` | Non-linear, khó quantize với INT8 | Attention softmax |
| `LayerNormalization` | Cần FP32 precision | Mỗi Transformer block |
| `GELU` activation | Complex non-linear | MLP blocks |
| `MultiHeadAttention` (Q,K,V matmul) | Dynamic shapes | Toàn bộ attention |
| `Embedding lookup` | Index operation | Text tokenizer |
| `Dynamic shapes` | DPU cần static shapes | Batch dimension variable |

### 2.4 Hệ quả trực tiếp với CLIP ViT

ViT (Vision Transformer) chứa các thành phần:

```
ViT-B/16 Layer Structure:
  PatchEmbed (Conv2D 16×16)     → ✅ DPU
  CLS token + pos embedding     → ❌ CPU (embedding)
  ─── × 12 Transformer Blocks ───
    LayerNorm                   → ❌ CPU
    Q, K, V projections         → ✅ DPU (Linear, static)
    Scaled Dot-Product Attention→ ❌ CPU (Softmax, dynamic)
    Projection                  → ✅ DPU
    LayerNorm                   → ❌ CPU
    MLP fc1 (GELU)              → ⚠️ Partial (GELU → DPU approx)
    MLP fc2                     → ✅ DPU
  ─── end blocks ───
  LayerNorm + projection        → ❌ CPU
```

**Kết quả:** Vitis AI compiler sẽ tạo ra một **mixed-execution graph**:
- Các phần DPU-compatible được nhóm thành **DPU subgraphs** (chạy accelerated)
- Các phần không tương thích (LayerNorm, Softmax, GELU) chạy trên **CPU subgraphs**

```
                    xmodel Mixed Execution Graph
                    ─────────────────────────────
    Input (NHWC INT8)
         │
    ┌────▼────┐  PatchEmbed Conv (16×16)
    │  DPU 0  │  → output feature map (INT8)
    └────┬────┘
         │ dequantize → FP32
    ┌────▼────┐  LayerNorm + Q,K,V + Attention × 12
    │  CPU 0  │  (fallback, FP32)
    └────┬────┘
         │ requantize → INT8
    ┌────▼────┐  Linear projections in MLP blocks
    │  DPU 1  │  (chạy accelerated)
    └────┬────┘
         │  ... (interleaved) ...
    ┌────▼────┐  Final LayerNorm + proj head
    │  CPU N  │
    └────┬────┘
         │ FP32 output (512-d embedding)
```

**Quan trọng:** Càng nhiều CPU↔DPU transitions → càng tốn overhead (DMA transfer, dequantize/requantize). Đây là lý do latency của ViT trên DPU kém hơn CNN đáng kể.

---

## 3. Vitis AI Ecosystem — Lý thuyết & Workflow

### 3.1 Tổng quan Vitis AI 3.5

Vitis AI là framework của AMD (trước đây là Xilinx) để deploy AI models lên FPGA/SoC. Gồm các thành phần:

```
Vitis AI 3.5 Ecosystem
──────────────────────

[Development Machine (x86)]          [Target Device (Kria)]
┌─────────────────────────┐          ┌──────────────────────┐
│   Vitis AI Toolchain    │          │   Vitis AI Runtime   │
│                         │          │                      │
│  ┌───────────────────┐  │          │  ┌────────────────┐  │
│  │  vai_q_pytorch    │  │          │  │  VART (C++)    │  │
│  │  (quantizer)      │  │   .xmodel│  │  vart::Runner  │  │
│  └────────┬──────────┘  │ ────────►│  └────────────────┘  │
│           │ int8 model  │          │                      │
│  ┌────────▼──────────┐  │          │  ┌────────────────┐  │
│  │  vai_c_xir        │  │          │  │  VART Python   │  │
│  │  (compiler/xir)   │  │          │  │  bindings      │  │
│  └────────┬──────────┘  │          │  └────────────────┘  │
│           │ .xmodel     │          │                      │
│  ┌────────▼──────────┐  │          │  ┌────────────────┐  │
│  │  Vitis AI Model   │  │          │  │  VVAS 3.0      │  │
│  │  Zoo (pretrained) │  │          │  │  GStreamer      │  │
│  └───────────────────┘  │          │  └────────────────┘  │
└─────────────────────────┘          └──────────────────────┘

Tools used:
  vai_q_pytorch  : PyTorch → INT8 quantized model (QAT/PTQ)
  vai_c_xir      : Quantized model → .xmodel (DPU executable)
  VART           : Runtime để execute .xmodel trên DPU
  XIR            : XModel Intermediate Representation (graph format)
  VVAS           : Vitis Video Analytics SDK (GStreamer plugins cho Kria)
```

### 3.2 XIR — XModel Intermediate Representation

XIR là định dạng graph trung gian của Vitis AI, tương tự ONNX nhưng dành riêng cho AMD DPU:

```
XIR Graph Structure:
─────────────────────
Graph
  └── Root Subgraph
        ├── Subgraph_0  (device="CPU")    ← preprocessing ops
        ├── Subgraph_1  (device="DPU")    ← DPU-compatible ops
        ├── Subgraph_2  (device="CPU")    ← LayerNorm, Attention
        ├── Subgraph_3  (device="DPU")    ← more DPU ops
        └── ...

DPU Subgraph attributes:
  - "device": "DPU"
  - "dpu_fingerprint": unique identifier cho bitstream match
  - Input tensors: shape, dtype (INT8), fix_point (quantization scale)
  - Output tensors: shape, dtype (INT8), fix_point

fix_point là số bits sau decimal point trong fixed-point representation:
  value_float = value_int8 × 2^(-fix_point)
  Ví dụ: fix_point=4 → scale=2^(-4)=0.0625
          INT8 value 64 → float 64 × 0.0625 = 4.0
```

### 3.3 VART — Vitis AI Runtime

VART (Vitis AI Runtime) là C++ library để execute .xmodel trên DPU. Python bindings được expose qua `vart` module:

```python
import vart
import xir

# Load compiled xmodel
graph    = xir.Graph.deserialize("clip_vision.xmodel")

# Lấy DPU subgraph đầu tiên
def get_dpu_subgraph(graph):
    for sg in graph.get_root_subgraph().toposort_child_subgraph():
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
            return sg
    raise RuntimeError("No DPU subgraph found")

subgraph = get_dpu_subgraph(graph)

# Tạo DPU runner (thread-safe, có thể tạo nhiều runners)
runner = vart.Runner.create_runner(subgraph, "run")

# Lấy tensor specs
input_tensors  = runner.get_input_tensors()   # List[xir.Tensor]
output_tensors = runner.get_output_tensors()  # List[xir.Tensor]

# Allocate buffers
inputs  = [np.empty(t.dims, dtype=np.int8) for t in input_tensors]
outputs = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]

# Fill input (đã quantize sang INT8)
inputs[0][:] = preprocessed_frames_int8

# Async execution (non-blocking)
job_id = runner.execute_async(inputs, outputs)

# Wait for completion
runner.wait(job_id)

# Dequantize output
fix_point = output_tensors[0].get_attr("fix_point")
scale     = 2.0 ** (-fix_point)
features  = outputs[0].astype(np.float32) * scale
```

### 3.4 VVAS — Vitis Video Analytics SDK

VVAS (Vitis Video Analytics SDK) là tập hợp **GStreamer plugins** được AMD cung cấp cho Kria, sử dụng hardware video IP cores:

```
VVAS GStreamer Plugin Set:
──────────────────────────

vvas_xmultisrc     : Multi-source input (file, RTSP, USB camera)
vvas_xdec          : Hardware H.264/H.265 decoder (VCU IP core)
                     Throughput: 4K@60fps H.264 hoặc 1080p@120fps
vvas_xfilter       : Chạy custom VVAS kernels (xclbin-based)
vvas_xscaler       : Hardware multi-channel image scaler (XVSC IP)
                     Resize nhiều luồng đồng thời trên PL
vvas_xoverlay      : Overlay bounding boxes / text
vvas_xinfer        : Inference plugin (gọi VART runner từ GStreamer)
vvas_xmetaaffixer  : Attach metadata (bounding boxes, timestamps)

Pipeline ví dụ cho NLVS indexing:
  vvas_xmultisrc → vvas_xdec → vvas_xscaler → appsink
  │                │             │               │
  Input video      H.265 decode  Resize to       Python receives
  (H.264 file)     HW accel.     224×224 NV12    BGR frames
```

---

## 4. Lý thuyết Quantization cho Deep Neural Networks

### 4.1 Tại sao cần Quantization?

Neural networks được train với **FP32** (32-bit floating point). DPU chỉ hỗ trợ **INT8** (8-bit integer). Quantization là quá trình chuyển đổi có kiểm soát:

$$\text{FP32} \xrightarrow{\text{quantize}} \text{INT8} \xrightarrow{\text{inference}} \text{INT8} \xrightarrow{\text{dequantize}} \text{FP32}$$

**Lợi ích:**
- **Tốc độ:** INT8 MACs nhanh hơn FP32 MACs 2–4× trên phần cứng tương thích
- **Memory:** INT8 weights = ¼ dung lượng FP32 weights
- **Power:** INT8 operations tiêu thụ điện thấp hơn ~3× so với FP32
- **Bandwidth:** ¼ bandwidth cho weight loading từ DDR

**Đánh đổi:**
- **Accuracy drop:** Thường 0.5–2% trên classification tasks; có thể lớn hơn với models phức tạp như Transformer
- **Calibration cost:** Cần calibration dataset để determine quantization ranges

### 4.2 Uniform Affine Quantization (Vitis AI Default)

Vitis AI sử dụng **symmetric uniform quantization** với **power-of-2 scaling** (gọi là "fixed-point quantization"):

**Forward pass (inference):**

$$x_q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right), -128, 127\right)$$

$$x_{\text{dequant}} = x_q \cdot S$$

Trong đó $S = 2^{-\text{fix\_point}}$ là **quantization scale** (scale factor).

**Quantization error:**

$$\epsilon = x - x_{\text{dequant}} = x - \text{round}(x / S) \cdot S$$

Với uniform quantization, $|\epsilon| \leq S/2 = 2^{-\text{fix\_point}-1}$.

**Fixed-point representation trong Vitis AI:**

Vitis AI biểu diễn mỗi tensor bằng một giá trị `fix_point` nguyên:

```
fix_point = 7  →  S = 2^(-7) = 0.0078125  →  range [-1.0, 0.992]
fix_point = 4  →  S = 2^(-4) = 0.0625     →  range [-8.0, 7.9375]
fix_point = 1  →  S = 2^(-1) = 0.5        →  range [-64, 63.5]
fix_point = -2 →  S = 2^2   = 4.0         →  range [-512, 508]

Công thức: range = [-128 × S, 127 × S]
```

**Tại sao power-of-2?** Multiplication/division bởi $2^k$ có thể thực hiện bằng **bit shift** — rất nhanh trên FPGA fabric.

### 4.3 Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)

Vitis AI hỗ trợ cả hai phương pháp:

#### Post-Training Quantization (PTQ) — `vai_q_pytorch` default

```
Workflow:
1. Load pre-trained FP32 model (weights đã train)
2. Pass calibration dataset (100–1000 samples) qua model
3. Thu thập activation statistics: min, max, histogram
4. Tính optimal fix_point cho mỗi tensor:
   fix_point = floor(log2(128 / max(|activations|)))
5. Export quantized model

Ưu điểm:
  - Không cần retrain (chỉ vài phút)
  - Đơn giản, không cần training infrastructure

Nhược điểm:
  - Accuracy drop lớn hơn QAT (1–3%)
  - Kém hơn với non-uniform distributions (CLIP attention scores)
```

#### Quantization-Aware Training (QAT) — Fine-tuning với fake quantization

```
Workflow:
1. Load pre-trained FP32 model
2. Insert "fake quantization" nodes vào computation graph:
   x_fake_quant = dequantize(quantize(x))  ← simulate INT8 during training
3. Fine-tune trên task-specific dataset (thường 10% original LR, 10 epochs)
4. Export quantized model

Ưu điểm:
  - Accuracy drop nhỏ hơn PTQ (0.1–0.5%)
  - Model học cách compensate quantization errors

Nhược điểm:
  - Cần dataset + training time
  - Phức tạp hơn
  - Với frozen CLIP: fine-tune làm mất zero-shot properties
```

**Khuyến nghị cho NLVS:** Dùng **PTQ** với calibration dataset gồm 500–1000 frames từ target video domain (surveillance footage). Không fine-tune CLIP để giữ zero-shot generalization.

### 4.4 Calibration Dataset cho NLVS

Calibration dataset phải representative cho **distribution của activations** trong inference thực tế:

```python
# calibration_dataset.py
import cv2
import numpy as np
from pathlib import Path

def build_calibration_dataset(
    video_dir: str,
    n_frames: int = 1000,
    target_size: tuple = (224, 224)
) -> list:
    """
    Lấy mẫu frames ngẫu nhiên từ các video surveillance.
    
    Yêu cầu:
    - Đa dạng về lighting (day/night, indoor/outdoor)
    - Đa dạng về nội dung (người đi bộ, xe cộ, cảnh tĩnh)
    - Đại diện cho domain target của NLVS
    """
    frames = []
    video_files = list(Path(video_dir).glob("**/*.mp4"))
    
    frames_per_video = n_frames // len(video_files)
    
    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        indices = np.linspace(0, total-1, frames_per_video, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                # BGR → RGB → normalize (CLIP preprocessing)
                rgb   = frame[:, :, ::-1].astype(np.float32) / 255.0
                mean  = np.array([0.48145466, 0.4578275, 0.40821073])
                std   = np.array([0.26862954, 0.26130258, 0.27577711])
                norm  = (rgb - mean) / std
                frames.append(norm)
        cap.release()
    
    return frames[:n_frames]
```

### 4.5 Quantization Sensitivity Analysis

Không phải mọi layer đều nhạy cảm như nhau với quantization. Phân tích sensitivity giúp quyết định layer nào giữ FP32:

**Mixed-precision strategy cho Transformer:**

```
Layer Type         Sensitivity   Strategy
─────────────────────────────────────────
PatchEmbed Conv    Low           INT8 ✅
MLP fc1, fc2      Low-Medium    INT8 ✅
Q, K, V proj      Medium        INT8 ✅ (với careful calibration)
Attention Softmax  High          FP32 (CPU fallback anyway)
LayerNorm         High          FP32 (CPU fallback anyway)
Output projection  Low           INT8 ✅
L2 normalization   N/A           CPU (post-processing)

→ Accuracy impact chủ yếu từ MLP và projection layers
→ Softmax/LayerNorm đã trên CPU → không ảnh hưởng
```

### 4.6 Quantization-Induced Accuracy Drop — Phân tích lý thuyết

Với CLIP ViT-B/16 INT8 trên surveillance video (không phải ImageNet):

**Nguồn error chính:**

1. **Rounding error trong MLP weights:**

$$\Delta W = W_{fp32} - \text{round}(W_{fp32} / S_W) \cdot S_W$$

Expected $|\Delta W|_{mean} \approx S_W / 4$ (uniform distribution assumption)

2. **Accumulated error qua 12 blocks:**

$$\epsilon_{\text{total}} \approx \sqrt{12} \cdot \epsilon_{\text{single block}}$$

(các errors partially cancel do random nature)

3. **Clipping error** nếu fix_point không tối ưu:

Vitus AI chooses: $\text{fix\_point} = \lfloor \log_2(128 / \max(|x|)) \rfloor$

Nếu distribution có outliers (rare large activations) → fix_point nhỏ hơn → coarser quantization cho phần lớn activations.

**Kinh nghiệm thực tế từ literature:**
- ViT-B/16 INT8 PTQ trên ImageNet: ~0.5–1.5% accuracy drop
- CLIP ViT-B/16 INT8 zero-shot retrieval: ~2–4% R@1 drop (ước tính từ các nghiên cứu tương tự)
- Với surveillance domain (distribution khác ImageNet): có thể 3–5% drop

---

## 5. Quantization CLIP ViT-B/16 — Phân tích chuyên sâu

### 5.1 CLIP Visual Encoder — Layer-by-layer Analysis

```
CLIP ViT-B/16 Visual Encoder
────────────────────────────
Total params: ~86M
INT8 size: ~86 MB (vs 172 MB FP16, 344 MB FP32)

Input: (N, 3, 224, 224) FP32
  │
  ▼ PatchEmbed
  Conv2D(3, 768, kernel=16, stride=16) → (N, 196, 768)
    ─ Quantizable? ✅ YES (standard conv)
    ─ DPU support? ✅ YES
  │
  ▼ CLS token prepend + PositionEmbedding
    ─ Quantizable? INT8 embedding ✅
    ─ DPU support? ❌ NO (embedding table lookup)
    ─ → Chạy CPU, cast về FP32
  │
  ▼ TransformerBlock × 12
  │   ├── LayerNorm(768)
  │   │     ─ DPU support? ❌ NO
  │   │     ─ → CPU, FP32
  │   │
  │   ├── Self-Attention
  │   │     Q_proj: Linear(768, 768) ─ DPU ✅ (if static shape)
  │   │     K_proj: Linear(768, 768) ─ DPU ✅
  │   │     V_proj: Linear(768, 768) ─ DPU ✅
  │   │     Scaled Dot-Product:
  │   │       QK^T / sqrt(d) ─ DPU ✅ (matmul)
  │   │       Softmax         ─ CPU ❌
  │   │       AV              ─ DPU ✅ (matmul)
  │   │     Out_proj: Linear(768, 768) ─ DPU ✅
  │   │
  │   ├── LayerNorm(768)     ─ CPU ❌
  │   │
  │   └── MLP
  │         fc1: Linear(768, 3072) ─ DPU ✅
  │         GELU                   ─ DPU ⚠️ (approximated)
  │         fc2: Linear(3072, 768) ─ DPU ✅
  │
  ▼ LayerNorm(768) ─ CPU ❌
  │
  ▼ Projection: Linear(768, 512) ─ DPU ✅
  │
  ▼ L2 Normalize ─ CPU (Python, post-processing)

Output: (N, 512) FP32, L2-normalised
```

### 5.2 DPU Utilization Estimate

Ước tính % FLOPs chạy trên DPU vs CPU:

```
Operation            FLOPs/layer    Count    DPU?    DPU FLOPs
────────────────────────────────────────────────────────────────
PatchEmbed Conv      2×196×768×3    1        ✅      1.18B
Q,K,V proj          2×196×768×768  3×12     ✅      1.74B
Matmul QK^T         2×196²×64      12       ✅      0.06B
Matmul AV           2×196×196×64   12       ✅      0.06B
Out proj            2×196×768×768  12       ✅      0.58B
MLP fc1             2×196×768×3072 12       ✅      2.33B
MLP fc2             2×196×3072×768 12       ✅      2.33B
────────────────────────────────────────────────────────────────
Total DPU FLOPs:                                   ~8.28B

LayerNorm (×25)      196×768×~10   25        ❌     ~0.38B (CPU)
Softmax (×12)        196×196×~5    12        ❌     ~0.02B (CPU)
GELU (×12)           196×3072      12        ❌     ~0.07B (CPU)
────────────────────────────────────────────────────────────────
Total CPU FLOPs:                                   ~0.47B

DPU utilization: 8.28 / (8.28 + 0.47) ≈ 94.6%
```

**Kết luận:** ~94.6% FLOPs chạy trên DPU — đây là con số rất tốt. Tuy nhiên, **latency thực tế** bị ảnh hưởng bởi **số lần CPU↔DPU handoff** (25+ lần LayerNorm) chứ không chỉ FLOPs ratio.

### 5.3 Latency Model cho DPU Inference

```
Total latency = DPU compute + CPU fallback + DMA overhead

DPU compute:
  FLOPs = 8.28B INT8 ops
  DPU throughput = 4096 MAC/cycle × 300 MHz = 1.23 TOPS
  Time = 8.28B / (1.23 × 10^12) = 6.7ms/frame (theoretical)
  With memory bandwidth bound: × 2–3 → 13–20ms/frame (practical)

CPU fallback (ARM A53 @ 1.5 GHz, single-threaded NEON):
  FLOPs = 0.47B FP32 ops
  Throughput ≈ 6 GFLOPS (optimistic, NEON FP32)
  Time = 0.47B / 6G = 78ms/frame (pure CPU)
  
  Mitigation: NEON SIMD + multi-threading → ~20-30ms

DMA overhead (25 CPU↔DPU transitions):
  Each DMA: transfer ~196×768×1 byte = ~150 KB
  AXI bandwidth: ~4 GB/s → 150 KB / 4 GB/s = 37 μs
  25 transitions × 37 μs = ~0.9ms total DMA

Estimated total: 20ms (DPU) + 25ms (CPU) + 1ms (DMA)
= ~46ms per frame (ViT-B/16)
= ~230ms per segment (5 frames, sequential)
```

**So sánh:**
```
                GTX 1650 Ti (FP16)    Kria DPU B4096 (INT8)
────────────────────────────────────────────────────────────
Per frame:      2–5ms                  ~46ms
Per segment:    10–25ms                ~230ms
Batch throughput: 32 frames/batch     4 frames/batch
Indexing 1h video: 1–2 min            ~15–20 min
Search latency: 15–30ms               ~100–200ms
```

---

## 6. Thách thức kỹ thuật đặc thù của Transformer trên DPU

### 6.1 Vấn đề LayerNorm

LayerNorm là thách thức lớn nhất khi deploy Transformer lên DPU:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

**Tại sao không thể INT8?**
- $\mu$ và $\sigma^2$ là **instance-level statistics** — thay đổi theo từng input, không thể precompute
- Phép chia với $\sqrt{\sigma^2 + \epsilon}$ cần FP32 precision
- Nếu quantize sang INT8: round($\sigma^2$) mất độ chính xác nghiêm trọng vì $\sigma^2$ thường rất nhỏ

**Giải pháp nghiên cứu:**

1. **I-ViT (Xiao et al., 2023) — Integer-only Quantization:**
   - Xấp xỉ sqrt bằng integer iterative method
   - LayerNorm toàn bộ bằng integer arithmetic
   - Accuracy drop < 0.5% trên ImageNet
   - Chưa available trong Vitis AI mainstream

2. **SmoothQuant (Xiao et al., 2022):**
   - Chuyển quantization difficulty từ activations sang weights
   - $Y = X W = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$
   - Trong đó $s$ là per-channel scaling factor
   - Supported trong Vitis AI 3.5 (experimental)

3. **Hiện tại (Vitis AI default):**
   - LayerNorm → CPU subgraph (FP32) → xacceptable quality, higher latency

**Thực tiễn:** Với DPU B4096 trên Kria, LayerNorm CPU fallback là **chấp nhận được** vì 24 LayerNorm operations chiếm ~0.38B FLOPs so với tổng 8.75B FLOPs. Tuy nhiên, **context switch overhead** (24 lần) là vấn đề latency thực sự.

### 6.2 Vấn đề Attention Softmax với Long Sequences

Softmax trong attention:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Tại sao khó quantize?**
- $e^{x_i}$ exponential function có range rất rộng ($e^{-127}$ đến $e^{127}$)
- Tổng $\sum_j e^{x_j}$ cần FP32 accumulation
- INT8 overflow ngay với chỉ một vài large inputs

**Giải pháp:**

1. **FlashAttention-style chunking:** Tính softmax theo chunks để tránh overflow — nhưng yêu cầu programming model phức tạp hơn

2. **INT16 intermediate:** Dùng INT16 cho exp và sum — không supported trên DPU B4096

3. **CPU fallback (current approach):** Softmax trên ARM NEON, ~2ms per block × 12 blocks = 24ms overhead

**Ảnh hưởng:** Với ViT-B/16 (196 patches), softmax matrix là 196×196 — không quá lớn, overhead manageable. Với ViT-L/14 (256 patches), overhead lớn hơn nhưng vẫn feasible.

### 6.3 Dynamic Shape Problem

DPU yêu cầu **tất cả tensor shapes phải static** tại compile time. CLIP text encoder có vấn đề này:

```
Text Encoder input:
  tokens: (batch_size, sequence_length) 
  Sequence length = 77 (CLIP max)
  
  Vấn đề: batch_size là dynamic (1 query → batch=1, 
                                  12 templates → batch=12)

Giải pháp:
  Option A: Compile với batch_size=1, loop 12 times
    → 12 × DPU inference calls = overhead
    
  Option B: Compile với batch_size=12 (max templates)
    → Pad batch khi cần < 12 queries
    → Single DPU call
    → Khuyến nghị ✅

  Option C: Compile multiple xmodels (batch=1,4,8,12)
    → Select at runtime based on actual batch
    → Flexibility cao nhưng tốn storage
```

**Implementation trong KriaEngine:**

```python
# Compile với batch=12 để match template ensemble
# kria.yaml:
#   engine:
#     batch_size: 12   # = len(_CLIP_TEMPLATES)
#     xmodel_text_path: clip_text_b12.xmodel

# Trong encode_text():
def encode_text(self, texts):
    if isinstance(texts, str):
        texts = [texts]
    
    # Pad to batch_size=12 if needed
    original_n = len(texts)
    if len(texts) < self._batch_size:
        texts = texts + [""] * (self._batch_size - len(texts))
    
    # Single DPU call với batch=12
    ...
    
    # Return chỉ original_n results
    return results[:original_n]
```

### 6.4 Memory Bandwidth Bottleneck

DPU B4096 trên Kria KV260 bị **memory bandwidth bound** chứ không phải compute bound:

```
Compute capacity: 4096 MACs/cycle × 300 MHz = 1.23 TOPS
Memory bandwidth: AXI HP × 2 = ~8 GB/s (PS DDR4 to PL)

ViT-B/16 weight transfer per inference:
  Weights: ~86 MB (INT8)
  If weights cached on-chip (BRAM): ~4.5 MB on-chip
  Remaining 81.5 MB from DDR: 81.5 MB / 8 GB/s = ~10ms
  
Arithmetic Intensity:
  FLOPs / bytes = 8.28B / 86M = 96 ops/byte
  DPU roof: 1.23T / 8G = 154 ops/byte
  
  → 96 < 154: MEMORY BOUND ← bottleneck là memory bandwidth
  
Tối ưu hóa:
  1. Tăng batch_size: 1→4 amortizes weight loading
     4 frames × 86MB ÷ 4 = 21.5 MB effective/frame
     4 frames cùng computation: 4 × 6.7ms / 4 = 6.7ms (không đổi)
     Nhưng thực tế: memory access amortized → ~30% faster
     
  2. Weight compression: không available trên DPU (cần INT8 exact)
  
  3. Model pruning: Reduce weight size → reduce memory bandwidth
     Structured pruning 30%: 86MB → 60MB → 30% bandwidth reduction
```

### 6.5 GELU Approximation trên DPU

GELU (Gaussian Error Linear Unit) được dùng trong CLIP MLP:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Vấn đề:** `erf()` không implement trực tiếp trong DPU hardware.

**Giải pháp Vitis AI — Piecewise Linear Approximation:**

Vitis AI compiler tự động xấp xỉ GELU bằng piecewise linear function:

$$\text{GELU\_approx}(x) \approx \begin{cases} 0 & x < -3 \\ 0.5 \cdot x & -3 \leq x \leq 0 \quad \text{(rough)} \\ x & x > 3 \end{cases}$$

Approximation chính xác hơn dùng hardswish: $\text{GELU}(x) \approx x \cdot \sigma(1.702x)$ với sigmoid approximated bằng lookup table.

**Accuracy impact:** ~0.2–0.5% drop so với exact GELU — acceptable.

---

*Tài liệu tiếp tục tại: [KRIA_DEPLOYMENT_RESEARCH_P2.md](KRIA_DEPLOYMENT_RESEARCH_P2.md)*  
*Part 2 bao gồm: Compilation workflow, VVAS integration, KriaEngine code, benchmarks, deployment roadmap*
