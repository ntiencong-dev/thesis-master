# NLVS — System Context & Architecture Document

> **Phiên bản:** v4.0  
> **Ngày cập nhật:** 2026-05-21  
> **Mục đích:** Context input cho AI Agent trong các session làm việc tiếp theo.  
> Đọc file này là đủ để hiểu thiết kế, tiếp tục code hoặc debug mà không cần đọc lại toàn bộ codebase.

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Bài toán giải quyết

**NLVS (Natural Language Video Search)** — hệ thống tìm kiếm video bằng ngôn ngữ tự nhiên trên luồng camera giám sát liên tục. Người dùng gõ query như *"người mặc áo đỏ"* hoặc *"someone waving hand"* và nhận về các đoạn video khớp nhất kèm timestamp để seek.

Hai chế độ hoạt động:
- **Pre-recorded**: upload file `.mp4` → index → search.
- **Continuous stream**: camera ghi liên tục → segment files → auto-index background → real-time searchable.

### 1.2 Tech Stack

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10 |
| Visual-Language Model | **EVA-CLIP ViT-L/14** (`EVA02-L-14/merged2b_s4b_b131k`) via `open_clip` |
| Vector DB | **Qdrant** `http://localhost:6333`, collection `nlvs_segments` (sole backend since v3.0) |
| Video decode | OpenCV (`cv2.VideoCapture`) hoặc GStreamer |
| Job queue | **SQLite WAL** (`segments/job_queue.db`) |
| Web UI | **Streamlit** v1.32.0 (`localhost:8501`) |
| REST API | **FastAPI** + Uvicorn (`localhost:8000`) |
| Scene detection | PySceneDetect `ContentDetector` (optional) |
| Query translation | `deep_translator` (Vietnamese → English) |
| 2-stage reranker | **BLIP-2** `blip2-opt-2.7b` (optional, lazy-load) |
| Caption augment | **BLIP-1** `blip-image-captioning-base` (optional) |
| Edge deployment | AMD Kria KV260 / VART + XIR (`.xmodel`) |
| Container | Docker (multi-stage: `base-pc` / `base-kria`) |
| Hardware (dev) | WSL2 Ubuntu 24.04 · GTX 1650 Ti 4 GB VRAM · CUDA |

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1 Mô hình kiến trúc tổng thể

**Event-driven Async Pipeline** với 3 layer độc lập:

```
┌────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Capture                                             │
│  OverlapCaptureDaemon (ffmpeg subprocess, RTSP/V4L2)           │
│  FolderWatcherDaemon  (poll shared folder từ Windows bridge)   │
│                    ↓ enqueue(segment_path)                     │
├────────────────────────────────────────────────────────────────┤
│  LAYER 2 — Async Indexing (SQLite job queue + thread pool)     │
│  PersistentJobQueue → ContinuousIndexer (indexer thread)       │
│  CircuitBreaker (depth 20/5 open/close thresholds)             │
│                    ↓ upsert vector                             │
├────────────────────────────────────────────────────────────────┤
│  LAYER 3 — Search & UI                                         │
│  NLVideoSearcher ← Qdrant HNSW                                │
│  Streamlit UI  /  FastAPI REST                                 │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Luồng dữ liệu (Data Flow)

#### Indexing path (continuous stream)
```
Camera (USB/RTSP)
  └→ [Windows host] windows_capture_server.py
        └→ cv2.VideoCapture (DirectShow)
              └→ writes cam01_<ts>_<seq>.mp4 to shared folder
                    └→ [WSL2] FolderWatcherDaemon (polls every 2s, stable-size 2s)
                          └→ PersistentJobQueue.enqueue(job)
                                └→ ContinuousIndexer._indexer_thread()
                                      └→ VideoProcessor → sliding-window frames (5 frames/segment)
                                            └→ PCEngine.encode_frames() → EVA-CLIP vision encoder
                                                  └→ mean-pool → L2-norm → vector ∈ ℝ^768
                                                        └→ Qdrant.upsert()
```

#### Search path
```
User query (text)
  └→ _normalize_query() — strip imperative prefixes ("find me...")
        └→ deep_translator (VI → EN, optional)
              └→ EVA-CLIP text encoder (12 prompt templates → mean-pool → L2-norm)
                    └→ Qdrant.query_points(vector, limit=top_k) — HNSW cosine search
                          └→ temporal_nms (IoU threshold 0.30) — merge overlapping windows
                                └→ Optional: BLIP-2 reranker (stage-2 VQA)
                                      └→ SearchResult list (score, cam_id, video_path, start_time)
                                            └→ Streamlit UI / FastAPI JSON response
```

### 2.3 Concurrency model

```
Main process (Streamlit/FastAPI)
├─ _load_searcher() [@st.cache_resource — singleton, survives Streamlit reruns]
│   └─ NLVideoSearcher (model loaded once in ~25s, stays in VRAM)
│
└─ ContinuousIndexer (launched khi user nhấn "Start Pipeline")
    ├─ Thread-1: watchdog / FolderWatcherDaemon (polling loop)
    └─ Thread-2: indexer_thread (dequeue → encode → upsert)
```

**Thread-safety**: `PersistentJobQueue` dùng SQLite với WAL mode + `threading.Lock`. Qdrant upsert là thread-safe qua HTTP.

---

## 3. CHI TIẾT CÁC THÀNH PHẦN

### 3.1 Module map

| File | Class/Function | Chức năng |
|---|---|---|
| `src/searcher.py` | `NLVideoSearcher` | Orchestrator chính — index + search, kết nối Qdrant |
| `src/searcher.py` | `_normalize_query()` | Xóa prefix imperative ("find me", "show me", ...) |
| `src/searcher.py` | `SearchResult` | Dataclass kết quả tìm kiếm |
| `src/feature_extractor.py` | `CLIPFeatureExtractor` | Encode frames/text bằng open_clip, batch inference, FP16 |
| `src/indexer.py` | `SegmentMeta` | Metadata mỗi segment (cam_id, path, timestamps) — VideoIndex removed v3.0 |
| `src/indexer.py` | `temporal_nms()` | Temporal Non-Maximum Suppression theo IoU |
| `src/indexer.py` | `calculate_iou()` | Tính IoU của 2 segments theo thời gian |
| `src/job_queue.py` | `PersistentJobQueue` | SQLite WAL queue, idempotent enqueue (UNIQUE), retry/backoff |
| `src/job_queue.py` | `CircuitBreaker` | Tạm dừng enqueue khi depth ≥ 20, mở lại khi ≤ 5 |
| `src/continuous_indexer.py` | `ContinuousIndexer` | 2-thread daemon: watchdog + indexer (persist thread removed v3.0) |
| `src/capture_daemon.py` | `OverlapCaptureDaemon` | ffmpeg subprocess, tail-only 10s overlap segments |
| `src/capture_daemon.py` | `FolderWatcherDaemon` | Poll shared folder, stable-size check, enqueue |
| `src/video_processor.py` | `VideoProcessor` | OpenCV frame extraction, sliding-window, resize 224×224 |
| `src/scene_segmenter.py` | `SceneSegmenter` | PySceneDetect ContentDetector (HSV diff), fallback uniform |
| `src/gst_pipeline.py` | `VideoPipeline` | Factory: OpenCV / GStreamer / VVAS backend |
| `src/reranker.py` | `BLIP2Reranker` | 2-stage VQA rerank, lazy VRAM load/unload |
| `src/caption_augmenter.py` | `CaptionAugmenter` | Hybrid embedding: α·visual + (1-α)·caption |
| `src/engines/factory.py` | `create_engine()` | Factory pattern cho 6 engine types |
| `src/engines/pc_engine.py` | `PCEngine` | Adapter: `CLIPFeatureExtractor` → `InferenceEngine` |
| `src/engines/kria_engine.py` | `KriaEngine` | DPU B4096 via VART/XIR, `.xmodel` inference |
| `src/engines/xclip_engine.py` | `XCLIPEngine` | Microsoft X-CLIP (temporal video understanding) |
| `src/engines/siglip_engine.py` | `SigLIPEngine` | Google SigLIP (sigmoid loss, không dùng softmax) |
| `src/engines/languagebind_engine.py` | `LanguageBindEngine` | LanguageBind multi-modal |
| `src/engines/intern_video2_engine.py` | `InternVideo2Engine` | InternVideo2 (video-specific pretraining) |
| `api/main.py` | FastAPI app | REST: `/health`, `/index`, `/search`, `/thumbnail`, `/debug/query` |
| `app.py` | Streamlit app | Web UI: upload, pipeline control, search, result display |
| `windows_capture_server.py` | CLI script | Windows-side bridge: DirectShow → shared folder MP4 |
| `index_video.py` | CLI script | Batch index video files từ command line |

### 3.2 Cổng giao tiếp

| Protocol | Port | Mục đích |
|---|---|---|
| HTTP/REST | 8000 | FastAPI (`/search`, `/index`, `/health`, `/thumbnail`) |
| HTTP | 8501 | Streamlit web UI |
| HTTP | 6333 | Qdrant REST API (internal) |
| File system | — | Job queue SQLite, capture segments, Qdrant storage volume |

### 3.3 Segment capture strategy (Tail-Only Overlap)

```
SEGMENT_SEC = 60s, STRIDE_SEC = 50s, OVERLAP = 10s

t=0  ──[cam01_0000000000_00000.mp4 : 0s→60s]
t=50 ──────────────[cam01_0000000050_00001.mp4 : 50s→110s]
t=100──────────────────────────[cam01_0000000100_00002.mp4 : 100s→160s]

Mục đích: đảm bảo event xảy ra ở biên segment không bị mất.
```

### 3.4 Job Queue state machine

```
pending → processing → done
                    ↘ failed (retry < MAX_RETRIES=3, backoff=30×2^(n-1) s)
                            ↘ dead (retry ≥ 3, hoặc FileNotFoundError)
```

---

## 4. THUẬT TOÁN AI & XỬ LÝ DỮ LIỆU

### 4.1 Mô hình đang dùng (production)

**EVA-CLIP ViT-L/14** — `open_clip` model `EVA02-L-14`, weights `merged2b_s4b_b131k`  
- Embedding dim: **768**  
- ImageNet zero-shot: 79.8% (vs 68.3% của ViT-B/16)  
- VRAM FP16: ~1.4 GB trên GTX 1650 Ti  
- Load time: ~25s (cached bởi `@st.cache_resource`)

### 4.2 Video embedding pipeline

```python
# Mỗi segment (window_sec=10s):
frames = [5 frames BGR 224×224]  # frames_per_window=5

# Visual encoding
frame_embeds = model.encode_image(frames)  # shape (5, 768), FP16
segment_embed = mean_pool(frame_embeds)    # shape (768,)
segment_embed = L2_normalize(segment_embed) # cosine-ready

# Stored to Qdrant with payload:
# {cam_id, video_path, relative_start, relative_end, 
#  segment_wall_start, absolute_start, absolute_end}
```

### 4.3 Text query embedding (12-template ensemble)

```python
templates = [
    "{}",                               # bare query
    "a photo of {}",
    "a video frame of {}",
    "security camera footage of {}",
    "surveillance video showing {}",
    "a person {}",
    "someone is {}",
    # ... 12 total
]
text_embeds = [model.encode_text(t.format(query)) for t in templates]
query_embed = L2_normalize(mean_pool(text_embeds))  # shape (768,)
```

Kỹ thuật này (CLIP paper §3.2) cải thiện recall ~3–5% trên action/event queries.

### 4.4 Temporal NMS (Non-Maximum Suppression)

Sau Qdrant search trả về top-K candidates, áp dụng temporal NMS để dedup:
```
IoU(A, B) = overlap_duration / union_duration
Nếu IoU > 0.30 → giữ segment có score cao hơn, drop cái còn lại
```

### 4.5 Adaptive threshold

Khi `adaptive_threshold=True`: nếu top-1 score < `score_threshold` (0.10), tự động hạ threshold xuống `top1_score × 0.8` để vẫn trả về kết quả thay vì empty.

### 4.6 2-stage reranker (Phase 3, optional)

```
Stage 1: EVA-CLIP cosine → Top-50 candidates  (fast, ms)
Stage 2: BLIP-2 VQA → "Does scene contain <query>?" → yes/no log-prob
Score = 0.6 × cosine + 0.4 × blip_score
```
**VRAM constraint**: BLIP-2 OPT-2.7B (8-bit) = 2.8 GB + EVA-CLIP 1.4 GB > 4 GB → dùng `device="cpu"` cho BLIP-2 hoặc unload CLIP trước.

### 4.7 Scene-aware segmentation (Phase 2, optional)

PySceneDetect `ContentDetector` (HSV histogram diff, threshold=27.0):
- Scene < `max_scene_sec`: 1 segment per scene
- Scene > `max_scene_sec`: subdivide với sliding-window
- Fallback: uniform sliding-window khi PySceneDetect không có

### 4.8 Tối ưu hóa GPU

- Toàn bộ inference chạy `torch.no_grad()` + `float16` (FP16)
- Batch size 16 frames/call để amortize CUDA kernel launch
- `@st.cache_resource` giữ model singleton trong process memory — không reload giữa các Streamlit reruns
- Qdrant HNSW index trên disk → search O(log N), không cần load toàn bộ vector vào RAM

---

## 5. KIẾN TRÚC TRIỂN KHAI & HẠ TẦNG

### 5.1 Cấu hình chính xác (production hiện tại — pc.yaml)

```yaml
engine:
  type: pc
  model_name: EVA02-L-14
  pretrained: merged2b_s4b_b131k
  device: cuda
  batch_size: 16
  frames_per_window: 5

pipeline:
  video_backend: opencv
  window_sec: 10.0
  overlap_ratio: 0.30        # stride = 7.0s

index:
  embed_dim: 768
  backend: qdrant
  qdrant_url: http://localhost:6333
  qdrant_collection: nlvs_segments

search:
  top_k: 5
  score_threshold: 0.10      # EVA-CLIP screen-capture scores ~0.15-0.18
  nms_iou_threshold: 0.30
```

### 5.2 Docker

```
Dockerfile (multi-stage, --build-arg TARGET=pc|kria)

base-pc  : nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
           + PyTorch cu118 + Python 3.10 + GStreamer

base-kria: xilinx/vitis-ai-cpu:3.5.0
           + open_clip + transformers

final    : WORKDIR /app, COPY src/ api/ config/ app.py
           CMD uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Qdrant chạy **riêng biệt** như Docker container:
```bash
docker run -d --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v /home/tienc/Prototype/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```
`qdrant_storage/` mount vào `/qdrant/storage` bên trong container → data persist qua restart.

### 5.3 WSL2 USB bridge

**Vấn đề**: WSL2 `vhci_hcd` không hỗ trợ isochronous USB → `ECONNRESET (-104)` khi dùng V4L2 với webcam UVC.

**Giải pháp**:
```
Windows host:
  windows_capture_server.py --output \\wsl$\Ubuntu\home\tienc\Prototype\segments
    → cv2.VideoCapture(0) DirectShow
    → ghi cam01_*.mp4 vào shared folder

WSL2:
  FolderWatcherDaemon(watch_dir="./segments")
    → poll 2s, stable-size check 2s
    → PersistentJobQueue.enqueue()
```

### 5.4 Kria KV260 edge deployment — Đánh giá trạng thái sẵn sàng

#### Những gì đã hoàn chỉnh (code-ready)

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| `KriaEngine` (`src/engines/kria_engine.py`) | ✅ Code hoàn chỉnh | VART/XIR runner, `encode_frames()`, `encode_text()`, INT8 dequant + L2-norm |
| `kria.yaml` config | ✅ Hoàn chỉnh | xmodel paths, DPU batch=4, VVAS pipeline elements |
| `VideoPipeline` VVAS backend | ✅ Code có | `gstreamer_vvas` backend, fallback về OpenCV nếu GStreamer không có |
| `ContinuousIndexer`, `PersistentJobQueue`, `NLVideoSearcher` | ✅ Hardware-agnostic | Hoạt động giống hệt PC, chỉ thay engine |
| `Dockerfile` (base-kria) | ✅ Có | `xilinx/vitis-ai-cpu:3.5.0` base |

#### Blockers cần giải quyết trước khi deploy lên Kria

**Blocker 1 — File `.xmodel` chưa được tạo (Critical)**

Đây là blocker lớn nhất. CLIP ViT-B/16 cần trải qua pipeline:
```
1. Quantization:
   vai_q_pytorch --model clip_vit_b16 --calib-dataset <imagenet_subset>
   → INT8 quantized model (accuracy drop ~2-3%)

2. Compilation:
   vai_c_xir -m clip_quantized.pth -a DPU_B4096 -o ./compiled/
   → clip_vision.xmodel  (vision encoder)
   → clip_text.xmodel    (text encoder)

3. Deploy:
   cp clip_*.xmodel /usr/share/vitis_ai_library/models/clip_vit_b16/
```
Cần: máy có Vitis AI 3.5 toolchain + calibration dataset (1000 ảnh ImageNet).

**Blocker 2 — `vart` + `xir` Python packages chỉ có trên Kria hardware**

`KriaEngine.__init__()` sẽ raise `ImportError` trên mọi máy không phải Kria — đây là intentional (factory tự fallback sang `PCEngine`). Tuy nhiên không thể test KriaEngine trên PC.

**Blocker 3 — VVAS GStreamer plugins**

`vvas_xdec` (hardware H.264 decoder), `vvas_xfilter` (FPGA scaler NV12→BGR) chỉ có trên Kria với VVAS 2.0+ installed. Không có → fallback về OpenCV (software decode, chậm hơn ~3×).

**Blocker 4 — Embedding dimension mismatch**

| Platform | Model | dim | Index |
|---|---|---|---|
| PC (dev) | EVA-CLIP ViT-L/14 | 768 | Qdrant collection `nlvs_segments` |
| Kria | CLIP ViT-B/16 INT8 | **512** | Phải tạo collection riêng, không tương thích |

→ Khi deploy Kria: đổi `qdrant_collection: nlvs_segments_kria` trong `kria.yaml` và re-index toàn bộ.

**Blocker 5 — Text tokenizer dependency**

`encode_text()` trong `KriaEngine` vẫn gọi `open_clip.get_tokenizer("ViT-B-16")` để tokenize trên CPU trước khi gửi vào DPU. Cần `open_clip` installed trên Kria (đã có trong Dockerfile).

#### Kết luận: Chưa thể chạy ngay

**Luồng dữ liệu tổng thể** (capture → queue → index → search) **đã sẵn sàng 100%** — chỉ cần thay engine. Nhưng để KriaEngine hoạt động cần: Kria hardware + xmodel files + VVAS plugins. Ước tính công sức: 2–3 ngày nếu có Kria board sẵn.

**Workaround để test ngay**: chạy `pc.yaml` trên Kria với `engine.type: pc` — chậm hơn ~10× nhưng hoạt động được ngay để validate pipeline end-to-end.

---

## 6. TRẠNG THÁI HIỆN TẠI & CÁC ĐẶC ĐIỂM QUAN TRỌNG

### 6.1 Trạng thái hệ thống (tính đến v2.9)

| Thành phần | Trạng thái |
|---|---|
| Qdrant | ✅ running, `restart=unless-stopped`, **90 points**, collection `nlvs_segments` |
| EVA-CLIP | ✅ loaded via `@st.cache_resource`, CUDA, embed_dim=768 |
| Streamlit | ✅ `localhost:8501` |
| FastAPI | ✅ `localhost:8000` |
| Job queue DB | `./segments/job_queue.db` |
| Test suite | 220 passed, 14 skipped |

### 6.2 API breaks đã fix (quan trọng để không bị fix lại)

- `qdrant-client >= 1.18.0`: `client.search()` đã bị **xóa** → dùng `client.query_points()`. Xem `src/searcher.py:_search_qdrant()`.
- `st.cache_resource.clear()` trong Streamlit 1.32 **không phải coroutine** — warning `expire_cache was never awaited` là internal Streamlit bug.
- `_reset_searcher()` trong `app.py` **không gọi** `_load_searcher.clear()` nữa (gây reload model 25s + warning). Thay vào đó: reset `_index`, set `_qdrant_client=None`, gọi `_init_qdrant()`.

### 6.3 Score threshold

EVA-CLIP scores trên screen-capture video = **~0.15–0.18** (thấp hơn real camera ~0.25–0.35). Threshold = **0.10** (không phải 0.20 như `kria.yaml` default).

### 6.4 Files không commit vào git

- `qdrant_storage/` — vector data, không cần commit (regenerate được từ segments)
- `segments/*.mp4` — raw capture files
- `venv/` — Python virtualenv

---

## 7. PATTERNS & CONVENTIONS QUAN TRỌNG

### 7.1 Engine interface contract

Mọi engine phải implement `InferenceEngine` (abstract base):
```python
class InferenceEngine:
    @property
    def embed_dim(self) -> int: ...
    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray: ...
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray: ...
```
Output luôn là `np.ndarray float32 L2-normalized`.

### 7.2 Config-driven design

Toàn bộ behavior quyết định bởi YAML config:
```python
NLVideoSearcher.from_config("config/pc.yaml")  # production
NLVideoSearcher.from_params(...)               # programmatic (tests)
```

### 7.3 Qdrant-only architecture (v3.0)

Từ v3.0, Qdrant là **sole vector store** (Faiss/VideoIndex đã bị xóa):
```python
_init_qdrant()   # kết nối, tạo collection nếu chưa có; raises RuntimeError nếu không kết nối được
# store: _qdrant_client.upsert()
# search: _qdrant_client.query_points()
```

### 7.4 SegmentMeta schema (payload trong Qdrant)

```json
{
  "cam_id": "cam01_1716000000_00001",
  "video_path": "/home/tienc/Prototype/segments/cam01_1716000000_00001.mp4",
  "relative_start": 0.0,
  "relative_end": 10.0,
  "segment_wall_start": 1716000000.0,
  "absolute_start": 1716000000.0,
  "absolute_end": 1716000010.0
}
```

`relative_start/end`: dùng để seek trong file. `absolute_start/end`: unix timestamp cho wall-clock timeline.

### 7.5 Test structure

```
tests/
├── unit/          # mock-based, không cần hardware
│   ├── test_capture_daemon.py     (12 tests, CD-01..12)
│   ├── test_searcher_unit.py      (QN-21, QN-22: query_points API)
│   ├── test_continuous_indexer.py (CI-07, CI-08: mark_dead)
│   ├── test_engines.py
│   ├── test_feature_extractor.py
│   ├── test_indexer.py
│   └── ...
└── integration/   # cần Qdrant running
    ├── test_api.py
    └── test_searcher.py
```

Chạy: `venv/bin/pytest -x -q` (root `pytest.ini` cấu hình path).

---

## 9. LÝ THUYẾT VLM & CƠ CHẾ TÌM KIẾM VIDEO

### 9.1 Vision-Language Model là gì?

**VLM (Vision-Language Model)** là mô hình học cách ánh xạ ảnh và văn bản vào cùng một không gian vector (embedding space) sao cho nội dung ngữ nghĩa tương đồng → vector gần nhau; nội dung khác nhau → vector xa nhau.

**Intuition**: sau khi train, vector của từ `"dog"` sẽ gần với vector của ảnh con chó hơn là vector của ảnh con mèo — dù hai thứ đó hoàn toàn khác kiểu dữ liệu (text vs pixels).

### 9.2 CLIP — Contrastive Language-Image Pretraining

CLIP (OpenAI, 2021) là nền tảng của toàn bộ hệ thống NLVS.

#### Kiến trúc dual-encoder

```
                    ┌─────────────────────┐
Image Input ───────▶│  Vision Encoder     │──▶ v ∈ ℝ^D
(224×224×3 pixels)  │  ViT / ResNet       │
                    └─────────────────────┘
                             ↕ Contrastive Loss
                    ┌─────────────────────┐
Text Input  ───────▶│  Text Encoder       │──▶ t ∈ ℝ^D
("a red car")       │  Transformer (12L)  │
                    └─────────────────────┘
```

Hai encoder **độc lập hoàn toàn** — không có cross-attention giữa vision và text trong quá trình encode. Chúng chỉ được "kéo" về gần nhau thông qua loss function.

#### Contrastive Loss (InfoNCE)

Với một batch N cặp (image_i, text_i):
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j) / \tau)}$$

Trong đó:
- $\text{sim}(v, t) = \frac{v \cdot t}{|v||t|}$ — cosine similarity
- $\tau$ — nhiệt độ (temperature), learnable parameter
- Phân tử: score của cặp đúng (positive pair)
- Phân mẫu: tổng score trên tất cả cặp sai trong batch (negatives)

**Hiệu ứng**: mô hình học cách tối đa hóa similarity của cặp đúng và tối thiểu hóa với tất cả cặp sai trong cùng batch. Scale batch lên đến 32,768 cặp → mô hình thấy rất nhiều negatives → embedding space cực kỳ discriminative.

#### Training data

CLIP gốc train trên **400 triệu** cặp (image, text) thu thập từ internet. Không cần human annotation — text là alt-text hoặc caption gốc của ảnh. Đây là **weakly supervised** learning ở quy mô lớn.

EVA-CLIP (hệ thống này đang dùng) train trên **2 tỷ** cặp với kiến trúc EVA-02 (modified ViT với thêm relative position embedding, SwiGLU activation).

### 9.3 Vision Encoder — ViT (Vision Transformer)

#### Patch embedding

```
Input image: 224×224×3
       ↓ chia thành 16×16 patches → 196 patches
       ↓ mỗi patch 16×16×3 = 768 pixels → flatten → Linear projection → d=768
       ↓ thêm [CLS] token ở đầu → sequence length = 197
       ↓ thêm position embedding (learned)
       ↓ vào Transformer stack
```

Đây là lý do tại sao input phải là đúng 224×224 — để tạo ra đúng 196 patches.

#### Transformer layers (ViT-L/14 — model hệ thống đang dùng)

| Config | ViT-B/16 (Kria) | ViT-L/14 (PC, EVA-CLIP) |
|---|---|---|
| Layers | 12 | 24 |
| Hidden dim | 768 | 1024 |
| Attention heads | 12 | 16 |
| Patch size | 16×16 | 14×14 |
| Patches per image | 196 | 256 |
| Output dim | 512 | **768** |
| Params | ~86M | ~307M |

**ViT-L/14 tốt hơn ViT-B/16** vì: nhiều layers hơn → học được feature trừu tượng hơn; patch nhỏ hơn (14 vs 16) → resolution cao hơn → bắt được chi tiết nhỏ.

#### Self-attention — cơ chế "nhìn toàn bộ ảnh"

Mỗi patch "nhìn" tất cả patches còn lại qua self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Patch ở vị trí "mắt người" sẽ attend mạnh tới patch "khuôn mặt" và "tóc" → model tự học được spatial relationship mà không cần convolution.

#### [CLS] token là embedding đầu ra

Token đặc biệt `[CLS]` không tương ứng với patch nào. Sau 24 layers của Transformer, giá trị của `[CLS]` đã aggregate thông tin từ toàn bộ ảnh qua attention → đây chính là **visual embedding** $v \in \mathbb{R}^{768}$ được đưa vào Qdrant.

### 9.4 Text Encoder — Masked Transformer

Text encoder của CLIP là Transformer 12 layers, hidden dim 512 → projected ra 768d (để match visual encoder).

```
"person running in park"
       ↓ BPE tokenization (Byte-Pair Encoding)
[49406, 2533, 2761, 530, 2447, 49407, 0, 0, ..., 0]  (length=77)
  ^                                          ^
[SOS]                                      [EOS]
       ↓ token embedding + position embedding
       ↓ 12-layer Transformer
       ↓ lấy giá trị tại vị trí [EOS] → text embedding t ∈ ℝ^768
```

**Tại sao lấy [EOS] thay vì [CLS]?** CLIP text encoder dùng causal masking (GPT-style), không có [CLS]. Token [EOS] attend được toàn bộ sequence → aggregate thông tin tốt nhất.

### 9.5 Tại sao có thể tìm kiếm video bằng text?

Sau khi train contrastive trên 2B cặp, embedding space đạt được tính chất quan trọng:

**1. Cross-modal alignment**: $\text{sim}(v_\text{"red car"}, t_\text{"red car"}) \approx 0.85$ trong khi $\text{sim}(v_\text{"red car"}, t_\text{"blue bus"}) \approx 0.20$

**2. Zero-shot transfer**: mô hình chưa bao giờ thấy "surveillance camera footage" trong training nhưng vẫn match được vì nó học được concept-level understanding, không phải pixel-level pattern matching.

**3. Compositional understanding**: "người đội mũ đỏ ngồi trên ghế" = composition của nhiều concepts mà model đã học. EVA-CLIP xử lý tốt hơn baseline CLIP nhờ train trên dataset đa dạng hơn.

### 9.6 Từ Image Search → Video Search: vấn đề temporal

CLIP là **image model** — nó không hiểu thời gian. Video là chuỗi frames theo thời gian. Hệ thống NLVS giải quyết gap này bằng:

#### Sliding-window + mean pooling (hiện tại)

```
Video file (60s)
  ↓ sliding window: window=10s, stride=7s
  ↓ 8 windows: [0-10s], [7-17s], [14-24s], ..., [50-60s]
  ↓ mỗi window: extract 5 frames (tại t=0,2.5,5,7.5,10s)
  ↓ encode 5 frames → 5 vectors ∈ ℝ^768
  ↓ mean pooling → 1 vector ∈ ℝ^768
  ↓ L2 normalize → segment embedding
```

**Ưu điểm**: đơn giản, hiệu quả, không cần model đặc biệt.  
**Nhược điểm**: mean pooling "làm mờ" temporal order — không phân biệt được "người A đứng dậy rồi chạy" vs "người A chạy rồi đứng". Cũng không xử lý tốt action kéo dài rất ngắn (< 1s) hoặc rất dài (> 10s).

#### Tại sao 5 frames thay vì 1?

Lấy 1 frame duy nhất có thể trúng frame mờ/transitional. 5 frames → mean pooling → embedding ổn định hơn, đại diện tốt hơn cho nội dung của cả window.

#### Các model tiên tiến xử lý temporal tốt hơn (engine alternatives)

| Model | Cách xử lý temporal | Trong codebase |
|---|---|---|
| CLIP (baseline) | Không có | `PCEngine` với ViT-B/16 |
| EVA-CLIP | Không có, nhưng visual feature tốt hơn | `PCEngine` với EVA02-L-14 (**đang dùng**) |
| X-CLIP | Thêm cross-frame attention module, train trên video | `XCLIPEngine` |
| InternVideo2 | Pre-train trên 12M video clips, temporal contrastive | `InternVideo2Engine` |
| LanguageBind | Unify video/audio/depth/thermal + language | `LanguageBindEngine` |

### 9.7 Tại sao L2 normalize?

Sau khi encode, áp dụng L2 normalization:
$$\hat{v} = \frac{v}{\|v\|_2}$$

Kết quả: $\|\hat{v}\|_2 = 1$ — tất cả vectors nằm trên unit hypersphere $S^{D-1}$.

**Lợi ích**:
- Cosine similarity = inner product: $\text{cos}(\hat{v}, \hat{t}) = \hat{v} \cdot \hat{t}$
- Qdrant HNSW với `Distance.COSINE` cho kết quả cosine similarity đúng trên unit vectors
- Loại bỏ ảnh hưởng của "độ lớn" vector — chỉ hướng mới quan trọng
- Tất cả scores nằm trong $[-1, 1]$, dễ set threshold

### 9.8 Tại sao dùng 12 prompt templates thay vì query gốc?

Đây là kỹ thuật **prompt ensembling** từ CLIP paper §3.2.

**Vấn đề**: CLIP train trên text dạng "a photo of a {class}" nhiều hơn là bare word "dog". Nếu chỉ encode "dog" → text embedding lệch khỏi vùng mà visual encoder expect.

**Giải pháp**: tạo 12 biến thể → mean pool text embeddings:
$$t_\text{final} = \text{L2\_norm}\left(\frac{1}{12}\sum_{k=1}^{12} \text{TextEnc}(\text{template}_k.\text{format}(\text{query}))\right)$$

**Kết quả thực nghiệm**: +3-5% recall trên action queries, đặc biệt với queries như "someone climbing" hay "security camera footage of running person".

Thêm surveillance-specific templates ("security camera footage of {}", "surveillance video showing {}") giúp vì trong training data của EVA-CLIP có rất nhiều surveillance/dashcam footage với captions kiểu này.

### 9.9 Adaptive threshold & vấn đề distribution shift

**Vấn đề**: scores từ screen-capture video (~0.15–0.18) thấp hơn đáng kể so với natural video (~0.25–0.35). Nguyên nhân:
- Screen-capture: thêm compression artifact, UI elements, text overlays
- Lighting không tự nhiên (monitor glow)
- No depth cues → visual features ít discriminative

**Adaptive threshold** trong `NLVideoSearcher.search()`:
```python
if results and results[0].score < self._score_threshold:
    # Hạ threshold xuống 80% của top-1 score
    effective_threshold = results[0].score * 0.80
```
Kỹ thuật này giữ ít nhất 1 kết quả ngay cả khi toàn bộ scores thấp, thay vì trả về empty list.

### 9.10 Temporal NMS — giải quyết vấn đề overlapping windows

Do sliding-window có overlap 30%, nhiều windows liên tiếp có thể match cùng một event:

```
Query: "người ngã"

Window [14-24s]: score=0.42  ← best match
Window [21-31s]: score=0.39  ← overlap 3s với window trên
Window [28-38s]: score=0.21  ← khác event

IoU([14-24], [21-31]) = overlap/union = 3/(14+3) = 0.176 < 0.30 → KEEP
```

Nếu IoU > 0.30: chỉ giữ window có score cao hơn. Đây là temporal analog của NMS trong object detection (chỉ khác: 1D thay vì 2D bounding boxes).

### 9.11 BLIP-2 reranker — tại sao cần stage 2?

CLIP là **bi-encoder**: text và image encode riêng lẻ → không có cross-modal interaction trong forward pass → bỏ mất fine-grained alignment.

BLIP-2 là **cross-encoder**: frame và query được xử lý **jointly** qua Q-Former (Querying Transformer):
```
Frame + Query → Q-Former (cross-attention) → "Yes, this scene contains {query}" / "No"
```

Cross-attention captures chi tiết như: "người mặc áo **đỏ**" vs "người mặc áo **xanh**" — điều mà bi-encoder dễ bỏ qua vì màu sắc là fine-grained feature.

**Trade-off**: BLIP-2 forward pass ~500ms/frame (CPU) vs CLIP search ~2ms. Dùng như stage-2 trên top-50 candidates thay vì toàn bộ index.

### 9.12 Tóm tắt: tại sao hệ thống hoạt động được

```
1. EVA-CLIP học semantic alignment từ 2B (image, text) pairs
   → "áo đỏ" → vector gần với ảnh chứa áo đỏ

2. Text query → encode → query vector q ∈ ℝ^768
   Video segment → encode → segment vector s ∈ ℝ^768

3. cosine_sim(q, s) = q · s (sau L2-norm)
   Nếu > 0.10 → segment này chứa nội dung khớp query

4. HNSW trong Qdrant tìm top-K nearest neighbors trong O(log N)
   → instant search dù có 90+ segments

5. Temporal NMS loại bỏ duplicate → clean result list
```

Toàn bộ không cần training trên domain cụ thể — đây là **zero-shot retrieval** nhờ vào general visual-semantic understanding của EVA-CLIP.

---

## 8. HƯỚNG PHÁT TRIỂN TIẾP THEO (Phase roadmap)

| Phase | Tính năng | Ước tính gain |
|---|---|---|
| Phase 1 ✅ | EVA-CLIP upgrade, 12-template ensemble, VI→EN translation | Recall +12%, Precision +8% |
| Phase 2 ✅ | Scene-aware segmentation (PySceneDetect) | Index size -30%, Precision +5% |
| Phase 3 ✅ | BLIP-2 2-stage reranker, caption augmentation (arch ready, opt-in) | Precision@5 +15-20% |
| Phase 4 🔲 | RTSP multi-camera, timeline visualization, export clips | UX |
| Edge ✅ | Kria KV260 KriaEngine skeleton, VVAS pipeline config | — |

---

## 10. CHANGELOG

### v3.0 (Qdrant-only refactor)
- **Removed**: Faiss/VideoIndex/ScalableVideoIndex từ toàn bộ codebase
- **Changed**: `_init_qdrant()` — raises `RuntimeError` nếu Qdrant không kết nối được (không còn fallback)
- **Changed**: `from_params()` — bỏ `index_dir`, thêm `qdrant_url`, `qdrant_collection` params
- **Changed**: `ContinuousIndexer` — 2 threads (watchdog + indexer), bỏ persist thread
- **Changed**: `save_index()` / `load_index()` — REMOVED từ `NLVideoSearcher`
- **Tests updated**: `test_indexer.py`, `test_continuous_indexer.py`, `test_searcher_unit.py`, `test_searcher.py`, `conftest.py`
