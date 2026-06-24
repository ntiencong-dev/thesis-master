# KV260 Deployment — Kế hoạch triển khai NLVS

> **Ngày:** 2026-06-20  
> **Board:** AMD Kria KV260 (Zynq UltraScale+ MPSoC, DPU B4096)  
> **Model:** BLIP-1 ViT-B/16 (`Salesforce/blip-itm-base-coco`)  
> **Config hiện tại:** `config/kria_blip1.yaml`

---

## 1. Kiến trúc hệ thống trên KV260

```
┌─────────────────────────────────────────────────────────────┐
│                    KV260 Runtime                            │
│                                                             │
│   Camera / File                                             │
│        │                                                    │
│        ▼ OpenCV (video_backend: opencv)                     │
│   OverlapCaptureDaemon   (capture_daemon.py)                │
│        │  segment .mp4 files → ./segments/                  │
│        ▼                                                     │
│   PersistentJobQueue     (job_queue.py / SQLite WAL)        │
│        │                                                     │
│        ▼                                                     │
│   ContinuousIndexer      (continuous_indexer.py)            │
│        │                                                     │
│        ├─ KriaEngine.encode_frames()  ←── DPU B4096 (INT8) │
│        │    BLIP-1 ViT-B/16 visual encoder (.xmodel)        │
│        │    Input : BGR frames, batch=4, NHWC INT8          │
│        │    Output: 256-dim L2-norm vectors (float32)       │
│        │                                                     │
│        └─ BERT text encoder + ITM head  ←── ARM CPU        │
│             (BlipProcessor, HuggingFace transformers)       │
│                                                             │
│   Qdrant (nlvs_segments_kria_blip1, dim=256, HNSW cosine)  │
│        │                                                     │
│   Searcher + BLIP1ITMReranker  (searcher.py / reranker.py) │
│        │  12 prompt templates → encode_text → query_points  │
│        │  stage-2 ITM reranking (alpha=0.6, top_k=50)       │
│        ▼                                                     │
│   FastAPI / Streamlit UI  (api/main.py, app.py)             │
└─────────────────────────────────────────────────────────────┘
```

### Phân chia công việc CPU vs DPU

| Thành phần | Nơi chạy | Ghi chú |
|---|---|---|
| Visual encoder ViT-B/16 | **DPU B4096 (INT8)** | `.xmodel` cần compile trước |
| BERT text encoder (ITC) | **ARM Cortex-A53 (CPU)** | HuggingFace transformers |
| ITM cross-attention | **ARM Cortex-A53 (CPU)** | ~46 ms/candidate |
| Soft-max pooling | **ARM CPU** | `base_engine.py` (hardware-agnostic) |
| Qdrant vector search | **ARM CPU** | Docker container |
| Temporal NMS | **ARM CPU** | `searcher.py` |
| Video decode | **OpenCV (ARM CPU)** | VVAS upgrade sau |

**Blocker hiện tại:** File `.xmodel` cho BLIP-1 ViT-B/16 chưa được compile.  
→ Trước mắt dùng `engine.type: pc` + `device: cpu` trong `kria_blip1.yaml` để validate pipeline.

---

## 2. Các phase triển khai

---

### Phase 0 — Kiểm tra board (ngay bây giờ, ~1 ngày)

**Mục tiêu:** Board KV260 hoạt động, Qdrant chạy được, test pipeline với CPU engine.

```bash
# ---- Trên KV260 (SSH vào board) ----

# 1. Kiểm tra OS và Python
uname -a                    # phải là ARM64 Ubuntu 22.04
python3 --version           # >= 3.10

# 2. Clone repo
git clone <repo_url> ~/Prototype
cd ~/Prototype
python3 -m venv venv && source venv/bin/activate

# 3. Cài dependencies (CPU-only, không cần GPU)
pip install -r requirements.txt
# Nếu transformers chưa có:
pip install transformers>=4.27 torch --index-url https://download.pytorch.org/whl/cpu

# 4. Chạy Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:v1.9.0

# 5. Tạm thời dùng PC engine (CPU) để kiểm tra pipeline
# Sửa kria_blip1.yaml:
#   engine:
#     type: pc          ← thay kria bằng pc
#     device: cpu

# 6. Test server
CONFIG=config/kria_blip1.yaml uvicorn api.main:app --host 0.0.0.0 --port 8000

# 7. Chạy unit tests
pytest tests/unit/ -m unit -q
```

**Tiêu chí pass:** Server khởi động, unit tests xanh, Qdrant accessible.

---

### Phase 1 — Compile BLIP-1 ViT-B/16 sang .xmodel (~3–5 ngày)

> Thực hiện trên **máy PC x86** với Vitis AI Docker.

#### 1.1 Export BLIP-1 visual encoder sang TorchScript

```python
# Chạy trên PC (không cần trong Docker — chỉ cần PyTorch FP32)
import torch
from transformers import BlipForImageTextRetrieval, BlipProcessor

model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
model.eval()

class BlipVisualWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.vision_model = m.vision_model
        self.vision_proj  = m.vision_proj        # Linear(768 → 256)

    def forward(self, pixel_values):
        out    = self.vision_model(pixel_values=pixel_values)
        cls_f  = out.last_hidden_state[:, 0, :]  # [CLS] token
        embed  = self.vision_proj(cls_f)
        norm   = embed / embed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return norm  # (B, 256) float32

wrapper = BlipVisualWrapper(model)
dummy   = torch.randn(1, 3, 384, 384)           # BLIP-1 input size = 384×384

with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy)
    traced.save("blip1_visual_fp32.pt")
```

> **Lưu ý:** BLIP-1 dùng input size **384×384** (khác CLIP 224×224). Cập nhật `_preprocess_frames` trong `kria_engine.py` nếu cần.

#### 1.2 Thu thập calibration dataset

```bash
# Lấy 1000 frames từ video surveillance thực tế
# Format: (1000, 3, 384, 384) float32, RGB, BLIP-normalized
# Lưu: calibration_frames_blip1.npy
```

#### 1.3 Quantize với vai_q_pytorch

```bash
# Trong Vitis AI Docker
docker pull xilinx/vitis-ai-pytorch-cpu:3.5.0
docker run -v $(pwd):/workspace -it xilinx/vitis-ai-pytorch-cpu:3.5.0 bash
conda activate vitis-ai-pytorch

python - <<'EOF'
from pytorch_nndct.apis import torch_quantizer
import torch, numpy as np

model  = torch.jit.load("/workspace/blip1_visual_fp32.pt")
calib  = torch.from_numpy(np.load("/workspace/calibration_frames_blip1.npy")[:500])
dummy  = torch.randn(1, 3, 384, 384)

# Bước 1: Calibration
q = torch_quantizer("calib", model, (dummy,), output_dir="quantized/")
qm = q.quant_model
with torch.no_grad():
    for i in range(0, 500, 16):
        qm(calib[i:i+16])
q.export_quant_config()

# Bước 2: Export xmodel
q2 = torch_quantizer("test", model, (dummy,), output_dir="quantized/")
with torch.no_grad():
    q2.quant_model(dummy)
q2.export_xmodel(output_dir="quantized/", deploy_check=True)
EOF
```

#### 1.4 Compile sang DPU xmodel

```bash
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"

vai_c_xir \
  --xmodel   quantized/blip1_visual_fp32_int.xmodel \
  --arch     $ARCH \
  --net_name blip1_vision \
  --output_dir compiled/ \
  --options  '{"input_shape": "4,3,384,384"}'

# Output: compiled/blip1_vision.xmodel (~86 MB INT8)
```

---

### Phase 2 — Deploy xmodel lên KV260 (~1–2 ngày)

```bash
# ---- Trên PC ----
# Copy xmodel lên board
scp compiled/blip1_vision.xmodel ubuntu@kria-ip:/usr/share/vitis_ai_library/models/blip1_vit_b16/

# ---- Trên KV260 ----
# Kiểm tra DPU fingerprint khớp với xmodel
xdputil query
xdputil benchmark /usr/share/vitis_ai_library/models/blip1_vit_b16/blip1_vision.xmodel -i 1

# Validate chất lượng embedding
python3 - <<'EOF'
import numpy as np, vart, xir

def load_runner(path):
    g = xir.Graph.deserialize(path)
    for sg in g.get_root_subgraph().toposort_child_subgraph():
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
            return vart.Runner.create_runner(sg, "run")

runner = load_runner("/usr/share/vitis_ai_library/models/blip1_vit_b16/blip1_vision.xmodel")
print("DPU runner loaded OK")
it = runner.get_input_tensors()
ot = runner.get_output_tensors()
print(f"Input:  {it[0].dims}  dtype={it[0].dtype}")
print(f"Output: {ot[0].dims}  dtype={ot[0].dtype}")
EOF

# Khi xmodel đã validate: Bật lại KriaEngine trong config
# kria_blip1.yaml:
#   engine:
#     type: kria          ← đổi lại từ pc
```

---

### Phase 3 — Tích hợp đầy đủ và chạy end-to-end (~2–3 ngày)

```bash
# ---- Trên KV260 ----
source ~/Prototype/venv/bin/activate
cd ~/Prototype

# 1. Khởi động Qdrant
docker start <qdrant-container>   # hoặc docker run lại nếu chưa có

# 2. Index một đoạn video test
CONFIG=config/kria_blip1.yaml python index_video.py --dir ./segments

# 3. Kiểm tra số vector đã index
python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient('http://localhost:6333')
print(c.get_collection('nlvs_segments_kria_blip1').points_count)
"

# 4. Chạy API server
CONFIG=config/kria_blip1.yaml uvicorn api.main:app \
    --host 0.0.0.0 --port 8000

# 5. Test search từ laptop (qua mạng LAN)
curl -X POST http://kria-ip:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "person walking", "top_k": 5}'

# 6. Streamlit UI (truy cập từ laptop qua LAN)
CONFIG=config/kria_blip1.yaml streamlit run app.py \
    --server.headless true --server.port 8501
# Mở browser: http://kria-ip:8501
```

---

### Phase 4 — Đo benchmark và so sánh (~1 ngày)

Chạy các test sau để có số liệu cho thesis:

```bash
# Đo latency indexing (1 segment = 10s video, 5 frames)
time python3 -c "
import numpy as np, yaml, cv2
from src.engines.factory import create_engine

cfg = yaml.safe_load(open('config/kria_blip1.yaml'))
engine = create_engine(cfg['engine'])

frames = [np.random.randint(0,255,(384,384,3),dtype=np.uint8) for _ in range(5)]
import time; t0=time.perf_counter()
for _ in range(20):
    engine.encode_frames(frames)
print(f'encode_frames 5 frames: {(time.perf_counter()-t0)/20*1000:.1f} ms avg')
"

# Đo latency search (text query)
time python3 -c "
from src.searcher import Searcher
s = Searcher('config/kria_blip1.yaml')
import time; t0=time.perf_counter()
for _ in range(10):
    s.search('person walking near fence')
print(f'search latency: {(time.perf_counter()-t0)/10*1000:.1f} ms avg')
"
```

**Chỉ tiêu mục tiêu (thesis):**

| Metric | Mục tiêu |
|---|---|
| encode_frames (5 frames) | < 250 ms |
| encode_text (1 query) | < 150 ms (ARM CPU) |
| Search latency (end-to-end) | < 500 ms |
| Mean cosine sim INT8 vs FP32 | > 0.95 |
| R@1 drop so với PC FP32 | < 4% |
| Power consumption | < 20W |

---

## 3. Sơ đồ luồng dữ liệu đầy đủ

```
[Camera / MP4 file]
      │
      ▼ OpenCV capture (capture_daemon.py)
[Segment .mp4  ─── 60s, stride 50s ───]
      │
      ▼ SQLite WAL job queue (job_queue.py)
[ContinuousIndexer thread]
      │
      ├─ extract 5 frames / segment (OpenCV)
      │
      ├─ KriaEngine.encode_segment_frames(frames)
      │     → soft-max pooling T=0.5   (base_engine.py)
      │     → DPU encode_frames()      (kria_engine.py)
      │     → INT8 NHWC → DPU runner → INT8 out → dequant → float32
      │     → L2-norm → 256-dim vector
      │
      ▼ Qdrant upsert (collection: nlvs_segments_kria_blip1, dim=256)
[Vector stored]

[User query: "người mặc áo đỏ đi xe máy"]
      │
      ▼ _normalize_query() (searcher.py)
      ├─ 12 prompt templates
      ├─ BERT encode_text() ← ARM CPU (transformers)
      ├─ mean of 12 vectors → 256-dim query vector
      │
      ▼ Qdrant query_points() (HNSW cosine, top 50)
      │
      ▼ temporal_nms() → top-k
      │
      ▼ BLIP1ITMReranker.rerank() (ARM CPU)
      │   _extract_best_frame() → ITM score/candidate
      │   combined = 0.6×cosine + 0.4×itm_prob
      │
      ▼ RerankResult list → API response / Streamlit UI
```

---

## 4. Files quan trọng cần theo dõi

| File | Vai trò |
|---|---|
| `config/kria_blip1.yaml` | Config chính cho KV260 |
| `src/engines/kria_engine.py` | DPU runner, preprocess, dequant |
| `src/engines/blip1_engine.py` | BLIP-1 model (PC reference) |
| `src/engines/base_engine.py` | Soft-max pooling (dùng chung) |
| `src/engines/factory.py` | `create_engine(config)` |
| `src/searcher.py` | Query pipeline, Qdrant search |
| `src/reranker.py` | BLIP1ITMReranker stage-2 |
| `api/main.py` | FastAPI endpoints |

---

## 5. Checklist tổng hợp

```
Phase 0 — Board setup
  [ ] KV260 SSH được, Ubuntu 22.04 xác nhận
  [ ] Python venv + requirements.txt cài xong
  [ ] Qdrant Docker chạy được trên board
  [ ] Server khởi động với engine.type: pc + device: cpu
  [ ] Unit tests pass trên ARM

Phase 1 — Compile xmodel (trên PC)
  [ ] export blip1_visual_fp32.pt (TorchScript)
  [ ] calibration_frames_blip1.npy thu thập (≥500 frames, surveillance domain)
  [ ] PTQ calibration hoàn thành (vai_q_pytorch)
  [ ] xmodel exported (quantized/)
  [ ] Compile thành công (compiled/blip1_vision.xmodel)

Phase 2 — Deploy xmodel
  [ ] SCP xmodel lên KV260 /usr/share/vitis_ai_library/...
  [ ] xdputil benchmark pass
  [ ] Mean cosine sim INT8 vs FP32 > 0.95
  [ ] engine.type đổi lại thành kria trong kria_blip1.yaml

Phase 3 — End-to-end
  [ ] index_video.py chạy được với KriaEngine
  [ ] Search API trả kết quả đúng
  [ ] Streamlit UI accessible từ mạng LAN

Phase 4 — Benchmark
  [ ] encode_frames latency đo xong
  [ ] Search latency đo xong
  [ ] So sánh kết quả PC vs KV260 (R@1, latency, power)
  [ ] Số liệu ghi vào thesis
```

---

*Tham khảo chi tiết: [KRIA_DEPLOYMENT_RESEARCH.md](KRIA_DEPLOYMENT_RESEARCH.md) | [KRIA_DEPLOYMENT_RESEARCH_P2.md](KRIA_DEPLOYMENT_RESEARCH_P2.md) | [SYSTEM_KNOWLEDGE_V4.md](SYSTEM_KNOWLEDGE_V4.md)*
