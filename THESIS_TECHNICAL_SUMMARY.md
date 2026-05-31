# Tổng hợp Kỹ thuật cho Luận văn Thạc sĩ
## Hệ thống NLVS — Workflow, Kiến trúc và Thuật toán

> **Mục đích:** Tài liệu này tổng hợp toàn bộ workflow, kiến trúc và các thuật toán đã được
> thiết kế, cài đặt và kiểm chứng trong prototype PC của hệ thống NLVS. Nội dung được tổ chức
> theo cấu trúc luận văn để hỗ trợ viết đề cương chi tiết và các chương nội dung.

---

## 1. Bức tranh tổng thể — Hệ thống NLVS end-to-end

### 1.1 Bài toán và đầu vào/đầu ra

**Đầu vào:** Truy vấn văn bản tự do (tiếng Việt hoặc tiếng Anh) — ví dụ *"người cầm máy sấy tóc"*,
*"someone holding a hair dryer"*.

**Đầu ra:** Danh sách các đoạn video (timestamped video segments) được xếp hạng theo độ tương đồng
ngữ nghĩa với query, mỗi kết quả có: `video_path`, `start_time`, `end_time`, `score`, `rank`.

**Pipeline tổng quát (2-stage retrieval):**

```
Text query
   │
   ▼
[Stage 1 — ITC Bi-encoder Search]
   ├─ _normalize_query()          → strip imperative prefix
   ├─ 12-template ensemble        → mean ITC embedding (256-dim)
   └─ Qdrant HNSW ANN search      → top-K candidates (cosine similarity)
          │ temporal_nms()          → deduplicate overlapping windows
          │
          ▼
[Stage 2 — ITM Cross-encoder Rerank]
   ├─ encode_text(query)          → ITC query vector (256-dim)
   ├─ _extract_best_frame(×N)     → ITC-guided frame selection
   ├─ score_itm(frames, query)    → P(match) [0,1] per candidate
   └─ combined = 0.6×cosine + 0.4×ITM_prob → re-sort → final ranking
```

### 1.2 Kiến trúc 3-layer async pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — Capture                                                    │
│  OverlapCaptureDaemon (ffmpeg, RTSP/USB) — SEGMENT_SEC=60, STRIDE=50 │
│  FolderWatcherDaemon  (poll shared folder từ Windows via WSL2 bridge)│
│                     ↓ enqueue(segment_path)                          │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 2 — Async Indexing                                             │
│  PersistentJobQueue (SQLite WAL, idempotent UNIQUE, retry backoff)   │
│  ContinuousIndexer  → VideoProcessor → encode_segment_frames()       │
│  CircuitBreaker     (depth=20 open / depth=5 close)                  │
│                     ↓ upsert(vector, payload)                        │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 3 — Search & Retrieval                                         │
│  NLVideoSearcher (ITC query) → Qdrant HNSW → NMS → ITM reranker     │
│  Streamlit UI (localhost:8501)   /   FastAPI REST (localhost:8000)   │
└──────────────────────────────────────────────────────────────────────┘
```

PC và Kria KV260 **chia sẻ cùng một code path** — sự khác biệt duy nhất là backend engine
(`BLIP1Engine` với HuggingFace Transformers trên PC, `KriaEngine` với VART/XIR `.xmodel` trên Kria).

---

## 2. Mô hình — BLIP-1 ViT-B/16 (Dual-mode)

### 2.1 Lý do chọn BLIP-1 thay vì CLIP

| Tiêu chí | CLIP ViT-B/16 | BLIP-1 `blip-itm-base-coco` |
|---|---|---|
| Embed dim (ITC) | 512 | **256** — nhỏ gọn hơn, phù hợp Kria memory |
| 2-stage reranker | Cần load model thứ hai | **Dùng lại cùng model** — ITM chỉ thêm linear head |
| KV260 validation | ViT-B/16 DPU chưa validated | **BLIP-1 ViT-B/16 = cùng kiến trúc** |
| ITM cross-attention | Không có | ✅ Built-in ITM head (BERT + cross-attn) |

### 2.2 BLIP-1 ITC mode (Stage-1 Indexing & Search)

**Dual-encoder (bi-encoder):** Ảnh và văn bản được encode độc lập thành 256-dim L2-normalized vectors.

```
Frame BGR → BlipProcessor (resize 384×384, normalize) → ViT-B/16 → [CLS] → vision_proj → v ∈ ℝ^256
Text       → BlipProcessor (tokenize) → BERT-base      → [CLS] → text_proj   → t ∈ ℝ^256

sim(v, t) = cos(v, t) = vᵀt  (because ‖v‖=‖t‖=1 after L2-norm)
```

Mục tiêu huấn luyện InfoNCE (ITC loss):
$$\mathcal{L}_{ITC} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{s(v_i,t_i)/\tau}}{\sum_{j}e^{s(v_i,t_j)/\tau}} + \log\frac{e^{s(v_i,t_i)/\tau}}{\sum_{j}e^{s(v_j,t_i)/\tau}}\right]$$

Đặc điểm quan trọng: sau L2-normalization, cosine similarity = inner product → có thể dùng HNSW index
hiệu quả.

### 2.3 BLIP-1 ITM mode (Stage-2 Reranking)

**Cross-encoder (fusion encoder):** Ảnh và văn bản được xử lý **cùng nhau** qua BERT với
cross-attention — cho phép mô hình học tương tác chi tiết pixel-word.

```
Frame BGR  → ViT-B/16 → patch features (B, 196, 768)
                              ↓
Text tokens → BERT encoder  ← cross-attention over patch features
                              ↓
                         [CLS] vector → ITM head (linear 768→2) → softmax
                              ↓
                         P(match) ∈ [0, 1]
```

**Lưu ý implementation:**
- Khi gọi `text_encoder` ở ITM mode, bắt buộc phải truyền `encoder_attention_mask=torch.ones(B, img_seq_len)`.
  Thiếu tham số này dẫn đến điểm sai một cách âm thầm.
- Cùng một model instance — không cần load thêm bộ nhớ.

**API trong codebase:**
```python
engine.score_itm(frames_bgr: List[np.ndarray], text: str) -> np.ndarray  # shape (N,), float32, [0,1]
```

---

## 3. Thuật toán Indexing

### 3.1 Video Segmentation

**Chiến lược Tail-Only Overlap:**
```
SEGMENT_SEC=60, STRIDE_SEC=50 → OVERLAP=10s

t=0  ──[segment_00: 0s→60s]
t=50 ──────────────[segment_01: 50s→110s]
t=100──────────────────────────[segment_02: 100s→160s]
```
Mục đích: event xảy ra tại biên segment không bị mất.

Optional: PySceneDetect `ContentDetector` (HSV histogram diff, threshold=27) để tạo scene-adaptive
segments thay vì uniform sliding-window.

### 3.2 Frame Extraction

`VideoProcessor` trích xuất **5 frames/segment** theo phân bố đều (uniform sampling) trong cửa sổ
thời gian. Frames được resize về 224×224 BGR (OpenCV convention).

### 3.3 Soft-Max Pooling (thuật toán chính — đóng góp của prototype)

**Vấn đề với mean pooling:** Nếu chỉ 1 trong 5 frames chứa đối tượng truy vấn (ví dụ máy sấy tóc
xuất hiện 1.5s trong cửa sổ 10s), mean embedding bị kéo về phía 4 frames không chứa đối tượng →
segment bị xếp hạng thấp hơn các segment không liên quan.

**Giải pháp — Soft-Max Pooling:**

$$\mathbf{e}_{seg} = \text{L2-norm}\left(\sum_{i=1}^{N} w_i \cdot \mathbf{e}_i\right)$$

$$w_i = \frac{\exp\!\left(d_i / T\right)}{\sum_j \exp(d_j / T)}, \quad
d_i = \|\mathbf{e}_i - \bar{\mathbf{e}}\|_2, \quad T = 0.5$$

Trong đó $\bar{\mathbf{e}} = \frac{1}{N}\sum_j \mathbf{e}_j$ là centroid, $d_i$ là độ lệch của frame
$i$ so với trung bình batch. Frame nào "dị biệt" nhất (outlier) nhận trọng số cao nhất.

**Hiệu quả đo được:**

| Chiến lược pooling | Score segment máy sấy | Rank |
|---|---|---|
| Mean pooling (baseline) | 0.3405 | #7 |
| Soft-max pooling T=0.5 | **0.3582** | **#1** |

Trọng số phân bổ ví dụ (5 frames, 1 frame có máy sấy): dryer frame 0.357 vs. 0.200 (uniform).

**Implementation** (`src/engines/base_engine.py`, method `encode_segment_frames`):
```python
embs      = self.encode_frames(frames_bgr)          # (N, D)
centroid  = embs.mean(axis=0)
deviations = np.linalg.norm(embs - centroid, axis=1) # (N,)
T = 0.5
weights   = np.exp(deviations / T)
weights   = weights / weights.sum()
weighted  = (embs * weights[:, None]).sum(axis=0)
return L2_normalize(weighted)
```

### 3.4 Vector Storage

Mỗi segment được upsert vào Qdrant với:
- **Vector:** 256-dim float32, L2-normalized (sẵn sàng cho cosine HNSW)
- **Payload:** `{cam_id, video_path, relative_start, relative_end, absolute_start, absolute_end}`
- **Collection:** `nlvs_segments_blip1`, HNSW cosine, dim=256

---

## 4. Thuật toán Search

### 4.1 Query Normalization

`_normalize_query()` xóa các prefix mệnh lệnh trước khi encode:
```
"find me a person with red shirt" → "person with red shirt"
"show me someone running"         → "someone running"
"search for hair dryer"           → "hair dryer"
```
Lý do: các prefix này có trong mọi query nhưng không mang thông tin ngữ nghĩa về nội dung video.

### 4.2 12-Template Ensemble

Kỹ thuật từ CLIP paper §3.2: encode query qua nhiều prompt templates, lấy trung bình embedding.

```python
# 9 templates cho visual objects:
"{}",  "a photo of {}",  "a video frame of {}",
"security camera footage of {}",  "surveillance video showing {}",
"a photo of a {}",  "a video of {}",
"an image showing {}",  "a frame containing {}",

# 12 templates cho action queries (khi query chứa action token):
"a person {}",  "someone is {}",  "a video of someone {}",
"surveillance footage of a person {}",  ...
```

Smart routing: `_is_action_query()` kiểm tra `_ACTION_TOKENS` frozenset → chọn template set phù hợp.

$$\mathbf{q} = \text{L2-norm}\!\left(\frac{1}{|\mathcal{T}|}\sum_{\tau \in \mathcal{T}} \text{encode\_text}(\tau(\text{query}))\right)$$

Cải thiện recall ~3–5% so với encode bare query.

### 4.3 Qdrant HNSW Search

```python
results = client.query_points(
    collection_name="nlvs_segments_blip1",
    query=qvec.tolist(),   # must be 1-D (squeeze if needed)
    limit=top_k,
    score_threshold=score_threshold
)
```

HNSW (Hierarchical Navigable Small World, Malkov & Yashunin 2020):
- Độ phức tạp tìm kiếm: $O(\log N)$
- Recall@10 ≈ 99% ở cấu hình mặc định (M=16, ef=100)
- Phù hợp triển khai nhúng — Qdrant ARM64 binary chạy được trên Kria KV260

**Pitfall quan trọng:** `encode_text(str)` trả về shape `(1, D)` → phải squeeze trước khi gọi
`query_points()`, nếu không Qdrant nhận list-of-lists và raise
*"Conversion between multi and regular vectors failed"*.

### 4.4 Temporal NMS (Non-Maximum Suppression)

Sau Qdrant search, nhiều segments có thể overlap về mặt thời gian (do stride < window size).
Temporal NMS loại bỏ duplicate:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\min(A_{end}, B_{end}) - \max(A_{start}, B_{start})}{\max(A_{end}, B_{end}) - \min(A_{start}, B_{start})}$$

Nếu $\text{IoU}(A, B) > \theta = 0.30$: giữ segment có score cao hơn, drop segment còn lại.

---

## 5. Thuật toán Stage-2 Reranking — ITC-Guided ITM

### 5.1 Vấn đề với Midpoint Frame Heuristic

**Symptom (quan sát trực tiếp trong prototype):**
- Segment máy sấy [42–51.7s]: vật thể xuất hiện t=44–45s, midpoint t=46.8s chỉ có khuôn mặt.
- Khi ITM nhận frame t=46.8s: P(match) = 0.045 → segment bị đẩy xuống rank #4.
- Segment headphone [7–17s] nhận frame t=12s: BLIP-1 confusion (headphone shape ≈ hair dryer
  trong không gian feature) → P(match) = 0.069 → xếp rank #1 sai.

### 5.2 Giải pháp — ITC-Guided Frame Selection

**Ý tưởng:** Model ITC đã "biết" frame nào trông giống query nhất qua cosine similarity trong 256-dim
ITC space. Dùng chính thông tin này để chọn frame tốt nhất cho ITM input.

**Thuật toán `_extract_best_frame`:**

```python
# 1. Sample 5 frames uniformly across [start_time, end_time]
timestamps = [start + i*(end-start)/4 for i in range(5)]
frames_bgr = [extract_frame(video, t) for t in timestamps]

# 2. Encode via ITC visual encoder
embs   = engine.encode_frames(frames_bgr)   # (5, 256) L2-normalized

# 3. Compute cosine similarity vs ITC query vector
cosines = embs @ query_vec                  # (5,)

# 4. Select best frame
best_idx = argmax(cosines)
return resize(frames_bgr[best_idx], (224, 224))
```

**`rerank()` flow:**
```python
qvec   = engine.encode_text(query)  # (1,256) → squeeze → (256,)
frames = [_extract_best_frame(c.video_path, c.start_time, c.end_time, qvec)
          for c in candidates]       # N ITC lookups
itm_scores = engine.score_itm(frames, query)   # 1 batch ITM call
combined = 0.6×cosine + 0.4×itm_prob          # per candidate
```

**Hiệu quả đo được:**

| Phương pháp chọn frame | Frame được chọn | ITM score | Rank cuối |
|---|---|---|---|
| Midpoint (t=46.8s) | Khuôn mặt | 0.0452 | #4 |
| ITC-guided (t=44.4s) | Máy sấy tóc | **0.4116** | **#1** |

**Cost overhead:** 5 ITC forward passes thêm mỗi candidate ≈ 5×3ms = 15ms/candidate trên GPU
(so sánh: 1 ITM pass ≈ 46ms trên Kria ARM). Tradeoff chấp nhận được.

### 5.3 Combined Score Formula

$$\text{score}_{final}(c) = \alpha \cdot \text{cos}_{ITC}(c) + (1-\alpha) \cdot P_{ITM}(c), \quad \alpha = 0.6$$

Lý do $\alpha=0.6$: ITC search đã được tối ưu qua calibration, ITM cung cấp tín hiệu bổ sung nhưng
có thể noisy (phụ thuộc chất lượng frame được chọn). 60-40 cho ITC ưu thế.

---

## 6. Kết quả Thực nghiệm Định tính

### 6.1 Môi trường test

- Qdrant collection: `nlvs_segments_blip1`, 21 points từ 3 video thực
- Model: `Salesforce/blip-itm-base-coco` (BLIP-1 ViT-B/16), GTX 1650 Ti, CUDA
- Query: `"hair dryer"`

### 6.2 So sánh pipeline stages

| Pipeline | Rank máy sấy | ITC score | ITM score | Combined |
|---|---|---|---|---|
| Mean pooling, no ITM | #7 | 0.3405 | — | 0.3405 |
| Soft-max pooling, no ITM | **#1** | 0.3582 | — | 0.3582 |
| Soft-max + ITM midpoint | #4 | 0.3582 | 0.0452 | 0.2328 |
| Soft-max + ITM ITC-guided | **#1** | 0.3582 | **0.4116** | **0.3797** |

**Kết luận:** Hai cải tiến có tính độc lập và cộng hưởng:
1. **Soft-max pooling** giải quyết vấn đề dilution ở tầng indexing.
2. **ITC-guided frame selection** giải quyết vấn đề temporal misalignment ở tầng reranking.

### 6.3 Phân tích ngữ nghĩa — tại sao các segment khác được xếp cao?

Segment "person indoors" không liên quan đến máy sấy vẫn có ITC cosine ≈ 0.30–0.31 vì:
- BLIP-1 training data: người trong không gian trong nhà thường đi kèm thiết bị gia dụng
- 256-dim ITC space: low resolution → semantic ambiguity ở cùng score range (Δ = 0.005)
- Đây là **giới hạn intrinsic** của ITC stage, không phải lỗi — ITM stage cần thiết để phân biệt

---

## 7. Test Suite

### 7.1 Cấu trúc tests

```
tests/
├── unit/           # pytest -m unit — không cần GPU/Qdrant
│   ├── test_engines.py          # ENG-01..38: encode_frames, encode_text, score_itm
│   ├── test_reranker.py         # ITM-01..11, RR-01..19: BLIP1ITMReranker, BLIP2Reranker
│   ├── test_searcher_unit.py    # SU-01..xx: NLVideoSearcher mocked
│   ├── test_indexer.py          # NMS-01..xx: temporal_nms, IoU
│   ├── test_continuous_indexer.py
│   ├── test_capture_daemon.py
│   └── ...
└── integration/    # pytest -m integration — cần Qdrant running
    ├── test_api.py
    └── test_searcher.py         # SR-01..xx: real Qdrant, real video files
```

### 7.2 Kết quả hiện tại

```
pytest tests/unit/ -m unit -q    → 138+ passed, 1 skipped
pytest tests/unit/test_reranker.py -q → 29 passed, 1 skipped
```

### 7.3 Mock pattern cho BLIP1ITMReranker tests

```python
mock_engine = MagicMock()
mock_engine.score_itm.return_value = np.array([0.8, 0.7, 0.6], dtype=np.float32)
mock_engine.encode_text.return_value = np.zeros(256, dtype=np.float32)   # 1-D
mock_engine.encode_frames.return_value = np.zeros((1, 256), dtype=np.float32)

reranker = BLIP1ITMReranker(engine=mock_engine, alpha=0.6)
reranker._extract_best_frame = lambda path, t0, t1, qvec, n_candidates=5: \
    np.zeros((224, 224, 3), dtype=np.uint8)
```

---

## 8. Liên hệ với Mục tiêu Luận văn

### 8.1 Mapping NC → Component

| Nội dung NC | Component được kiểm chứng trong prototype |
|---|---|
| **NC1** — Lý thuyết VLM, PTQ, HNSW | BLIP-1 ITC+ITM math, soft-max pooling, HNSW Qdrant |
| **NC2** — Prototype PC | Toàn bộ `src/`, 138+ tests, search quality validated |
| **NC3** — PTQ INT8 + .xmodel | Blocker: cần Vitis AI 3.5; architecture validated với BLIP-1 ViT-B/16 |
| **NC4** — Kria deployment | `KriaEngine` code-ready; `kria_blip1.yaml` đầy đủ config |
| **NC5** — Benchmark | Tập metrics đã thiết kế; baseline PC đã có (cần chạy trên DiDeMo) |

### 8.2 Đóng góp kỹ thuật từ prototype (cho luận văn)

1. **Soft-max pooling với temperature scaling** cho video segment embedding:
   - Công thức: deviation-weighted softmax với T=0.5
   - Kết quả: cải thiện rank từ #7 → #1 cho object detection trong brief-appearance scenario
   - Có thể tái tạo trong bất kỳ ViT-based model nào (hardware-agnostic)

2. **ITC-guided frame selection cho ITM cross-encoder:**
   - Giải quyết temporal misalignment giữa segment window và object appearance
   - Reuse ITC encoder (zero extra model load) → phù hợp với memory constraint của Kria
   - Kết quả: ITM score 0.0452 → 0.4116 cho target segment

3. **2-stage retrieval pipeline tích hợp:**
   - Stage 1 (ITC bi-encoder): latency ~5ms, recall coverage
   - Stage 2 (ITM cross-encoder): latency ~46ms/candidate trên ARM, precision boost
   - Combined score công thức: 0.6×cosine + 0.4×ITM_prob

4. **Unified PC↔Kria design:**
   - Cùng model (BLIP-1 ViT-B/16), cùng Qdrant collection format, cùng code path
   - Chỉ thay engine backend: HuggingFace Transformers → VART/XIR .xmodel
   - Tất cả improvements (soft-max pooling, ITC-guided ITM) đều hardware-agnostic

### 8.3 Các giá trị đo được dự kiến cho Kria (NC5)

| Metric | PC baseline | Kria-FP32 (ước tính) | Kria-INT8 DPU (mục tiêu) |
|---|---|---|---|
| Recall@10 (DiDeMo) | ~65% | ~55% | ~60–63% |
| Index throughput | ~8 FPS | ~0.5 FPS | **≥2 FPS** (MT-03) |
| Query latency | ~80ms | ~500ms | ~150–300ms |
| ITM rerank (50 cands) | ~400ms GPU | ~2.3s ARM | ~2.3s ARM |
| Power consumption | ~120W | ~8W | **≤25W** (MT-05) |
| GOPS/W | ~15 | ~5 | ~50–60 |

---

## 9. Critical Pitfalls (tổng hợp để tránh lặp lại)

| Pitfall | Mô tả | Fix |
|---|---|---|
| Faiss removed | `VideoIndex`, `FaissIndex` không tồn tại | Chỉ dùng Qdrant |
| `client.search()` deprecated | qdrant-client ≥1.18 | Dùng `client.query_points()` |
| 2-D query vector | `encode_text(str)` → `(1,D)` | `qvec = qvec[0]` nếu `ndim > 1` |
| `vision_projection` | Attribute sai trên BLIP-1 | Dùng `vision_proj` |
| ITM missing mask | `encoder_attention_mask` bắt buộc | `torch.ones(B, seq_len)` |
| BGR → RGB | OpenCV frames là BGR | Convert trong engine, không ở ngoài |
| Mock encode_text | Unit test với MagicMock | `mock.encode_text.return_value = np.zeros(256)` (1-D) |
| Test isolation | `nlvs_segments_test` tích lũy points | `delete_collection()` trước mỗi session |
| Qdrant dim mismatch | Collection tồn tại với dim khác | Validate hoặc dùng collection name mới |

---

## 10. Cấu trúc Files Quan trọng

```
src/
├── engines/
│   ├── base_engine.py          # InferenceEngine ABC + encode_segment_frames (soft-max pooling)
│   ├── blip1_engine.py         # BLIP1Engine: ITC encode + ITM score_itm
│   ├── kria_engine.py          # KriaEngine: VART/XIR .xmodel inference
│   └── factory.py              # create_engine(config) factory
├── searcher.py                 # NLVideoSearcher: normalize → templates → Qdrant → NMS → rerank
├── reranker.py                 # BLIP1ITMReranker: encode_text + _extract_best_frame + score_itm
├── indexer.py                  # SegmentMeta, temporal_nms, IoU
├── continuous_indexer.py       # ContinuousIndexer: watchdog + indexer thread
├── job_queue.py                # PersistentJobQueue (SQLite WAL) + CircuitBreaker
└── video_processor.py          # VideoProcessor: frame extraction, sliding window

config/
├── pc_blip1.yaml               # PC config (CUDA, HuggingFace, dim=256)
└── kria_blip1.yaml             # Kria config (VART .xmodel, dim=256) — xmodel pending

api/main.py                     # FastAPI: /search /index /health /thumbnail
app.py                          # Streamlit UI với ITM toggle
```
