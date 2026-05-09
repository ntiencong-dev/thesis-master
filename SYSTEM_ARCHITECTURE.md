# Kiến trúc & Kỹ thuật AI — Hệ thống Tìm kiếm Video bằng Ngôn ngữ Tự nhiên (NLVS)

> **Mục đích tài liệu:** Mô tả chi tiết mọi thuật toán, kỹ thuật AI đang được sử dụng trong từng thành phần, đồng thời phân tích điểm yếu và đề xuất cải tiến cụ thể để nâng cao độ chính xác tìm kiếm.

---

## 1. Sơ đồ Kiến trúc Tổng thể

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         NLVS — PIPELINE TỔNG THỂ                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐               ║
║  │  VIDEO FILE │────▶│  VIDEO PROCESSOR│────▶│ FEATURE EXTRACTOR│               ║
║  │  (.mp4 ...) │     │  (src/video_    │     │ (src/feature_    │               ║
║  └─────────────┘     │  processor.py)  │     │  extractor.py)   │               ║
║                      └────────┬────────┘     └────────┬─────────┘               ║
║                               │                        │                         ║
║                    [VideoSegment list]          [float32 embeddings              ║
║                    video_id, start, end,         shape: (N, 512)                 ║
║                    frame (224×224 BGR)]          L2-normalised]                  ║
║                                                          │                       ║
║  ┌─────────────┐                               ┌────────▼─────────┐             ║
║  │  TEXT QUERY │────▶ [CLIP Text Encoder] ────▶│   FAISS INDEX    │             ║
║  │  (free text)│      shape: (1, 512)           │  (src/indexer.py)│             ║
║  └─────────────┘                               │  IndexFlatIP     │             ║
║                                                └────────┬─────────┘             ║
║                                                          │                       ║
║                                                 [Top-K (score, meta)]            ║
║                                                          │                       ║
║                                                ┌─────────▼────────┐             ║
║                                                │    SEARCHER      │             ║
║                                                │ (src/searcher.py)│             ║
║                                                └─────────┬────────┘             ║
║                                                          │                       ║
║                                          ┌───────────────▼────────────────┐     ║
║                                          │    OUTPUT: SearchResult list   │     ║
║                                          │  [rank, score, video_id,       │     ║
║                                          │   start_time, end_time]        │     ║
║                                          └────────────────────────────────┘     ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  GIAO DIỆN:  Streamlit Web UI (app.py)   │   CLI Tool (index_video.py)          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Chi tiết Từng Component và Kỹ thuật AI

---

### 2.1 Video Processor — `src/video_processor.py`

#### Vai trò
Phân tách video thô thành các đơn vị nhỏ (segment) có thể biểu diễn bằng vector. Đây là **bước tiền xử lý** quyết định độ phủ (recall) của toàn bộ hệ thống.

#### Kỹ thuật đang dùng

| Kỹ thuật | Mô tả | Tham số hiện tại |
|---|---|---|
| **Keyframe Sampling (1-FPS)** | Lấy 1 frame đại diện mỗi giây. Nhanh, ít tốn bộ nhớ. | `fps_mode=1.0` |
| **Sliding Window** | Chia video thành các cửa sổ chồng lấp nhau. Bắt được các hành động nằm trên ranh giới 2 window. | `window_sec=5.0`, `stride_sec=2.5` (50% overlap) |
| **OpenCV Seek** | Dùng `CAP_PROP_POS_FRAMES` để seek chính xác đến frame tương ứng với timestamp. | — |
| **Bilinear Resize** | Resize về 224×224 bằng `cv2.INTER_LINEAR` — chuẩn đầu vào của CLIP. | target=(224,224) |

#### Sơ đồ Sliding Window

```
Video timeline (30 giây, window=5s, stride=2.5s, overlap=50%):

t=0    2.5   5    7.5   10   12.5   ...
│─────────│         Window 1 [0→5s]
      │─────────│   Window 2 [2.5→7.5s]
            │─────────│ Window 3 [5→10s]
                  │─────────│ Window 4 [7.5→12.5s]

Frame đại diện = middle frame của mỗi window
```

#### Vấn đề & Điểm yếu hiện tại
- **Single-frame representation:** Mỗi window chỉ được biểu diễn bởi 1 frame (frame giữa). Nếu hành động xảy ra ở đầu hoặc cuối window, frame giữa có thể không chứa hành động đó.
- **No scene change detection:** Không phát hiện cảnh thay đổi đột ngột — một window có thể chứa 2 cảnh hoàn toàn khác nhau.
- **Fixed window size:** Không thích ứng với tốc độ diễn ra của sự kiện. Hành động nhanh (0.5s) bị hòa loãng trong window 5s.

---

### 2.2 Feature Extractor — `src/feature_extractor.py`

#### Vai trò
**Trung tâm của toàn hệ thống.** Ánh xạ cả ảnh lẫn văn bản vào cùng một không gian vector 512 chiều, cho phép tính toán độ tương đồng giữa query text và frame video.

#### Mô hình: CLIP ViT-B/16

**CLIP (Contrastive Language-Image Pretraining)** — OpenAI, 2021.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLIP JOINT EMBEDDING SPACE                       │
│                                                                     │
│  Image Input (224×224)          Text Input ("người leo rào")        │
│        │                                   │                        │
│  ┌─────▼──────────────────┐   ┌────────────▼──────────────┐        │
│  │   VISION ENCODER       │   │    TEXT ENCODER           │        │
│  │   ViT-B/16             │   │    Transformer 12L        │        │
│  │                        │   │                           │        │
│  │ Patch size: 16×16      │   │ Vocab: 49,408 tokens      │        │
│  │ Patches: 14×14 = 196   │   │ Context len: 77 tokens    │        │
│  │ Hidden dim: 768        │   │ Hidden dim: 512            │        │
│  │ Layers: 12             │   │ Heads: 8                  │        │
│  │ Heads: 12              │   │                           │        │
│  │ Output: 512-d          │   │ Output: 512-d             │        │
│  └──────────┬─────────────┘   └─────────────┬─────────────┘        │
│             │                               │                       │
│             ▼                               ▼                       │
│       Image Embedding (512-d)    Text Embedding (512-d)            │
│             │                               │                       │
│             └─────────── Cosine Similarity ──────────────┘          │
│                           score ∈ [−1, 1]                           │
└─────────────────────────────────────────────────────────────────────┘
```

#### Kiến trúc Vision Encoder: Vision Transformer (ViT-B/16)

```
Input image (224×224×3)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Patch Embedding                                  │
│  Split → 196 patches (16×16)                      │
│  Linear projection → 768-d tokens                │
│  + [CLS] token + Positional Encoding              │
└───────────────────────────────────────────────────┘
        │
        ▼  × 12 Transformer Blocks
┌───────────────────────────────────────────────────┐
│  Multi-Head Self-Attention (12 heads)             │
│  LayerNorm → MLP (FFN) → LayerNorm                │
│  Residual connections                             │
└───────────────────────────────────────────────────┘
        │
        ▼
  [CLS] token output → Linear projection → 512-d
        │
        ▼
  L2 Normalisation → Image Embedding
```

#### Kiến trúc Text Encoder: Transformer

```
Text prompt → BPE Tokenization (max 77 tokens)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Token Embedding + Positional Encoding            │
└───────────────────────────────────────────────────┘
        │
        ▼  × 12 Transformer Blocks
┌───────────────────────────────────────────────────┐
│  Causal Self-Attention (masked)                   │
│  LayerNorm → MLP → LayerNorm                      │
└───────────────────────────────────────────────────┘
        │
        ▼
  [EOS] token output → Linear projection → 512-d
        │
        ▼
  L2 Normalisation → Text Embedding
```

#### Training Objective của CLIP: Contrastive Loss

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(v_i, t_j)/\tau)}$$

Trong đó:
- $v_i$ = image embedding thứ $i$
- $t_i$ = text embedding khớp với ảnh $i$
- $\tau$ = temperature parameter (học được)
- $N$ = batch size (lên đến 32,768 trong paper gốc)

#### Kỹ thuật tối ưu hiệu suất đang dùng

| Kỹ thuật | Mục đích | Vị trí code |
|---|---|---|
| `torch.no_grad()` | Không tính gradient → tiết kiệm ~50% VRAM | `encode_frames()`, `encode_text()` |
| `float16` (half precision) | Giảm VRAM từ ~900MB → ~450MB cho ViT-B/16 | `self._model.half()` |
| Batch processing | Xử lý 32 frames/lần → amortise CUDA overhead | `batch_size=32` |
| L2 Normalisation | Chuẩn hoá về unit sphere → cosine sim = dot product | `F.normalize(..., dim=-1)` |

#### Vấn đề & Điểm yếu hiện tại
- **Single-frame per segment:** Embed 1 frame thay vì embed trung bình nhiều frame — mất thông tin temporal.
- **ViT-B/16 là mô hình nhỏ:** Chỉ 151M params, độ chính xác thấp hơn ViT-L/14 (427M) hoặc ViT-bigG.
- **No video-specific pretraining:** CLIP được train trên ảnh tĩnh, không có thông tin temporal. X-CLIP hoặc Video-CLIP sẽ tốt hơn cho video.
- **Truncation 77 tokens:** Query dài sẽ bị cắt bớt.

---

### 2.3 Indexer — `src/indexer.py`

#### Vai trò
Lưu trữ và tìm kiếm hiệu quả trên tập hợp lớn các vector embedding.

#### Kỹ thuật: Faiss `IndexFlatIP`

**Faiss (Facebook AI Similarity Search)** — tìm kiếm nearest neighbor chính xác (exact search) trên không gian vector.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FAISS INDEXFLATIP                                 │
│                                                                      │
│  Stored vectors: N × 512  (float32)                                  │
│                                                                      │
│  Query: q ∈ ℝ^512                                                    │
│                                                                      │
│  Score(q, vᵢ) = q · vᵢ  =  Σ qₖ × vᵢₖ   (inner product)            │
│                k=1..512                                              │
│                                                                      │
│  Vì q và vᵢ đều L2-normalised:                                      │
│  q · vᵢ = ||q|| · ||vᵢ|| · cos(θ) = cos(θ)  ∈ [−1, 1]              │
│                                                                      │
│  → IndexFlatIP trên unit vectors = Cosine Similarity Search          │
│                                                                      │
│  Algorithm: Brute-force (O(N·D)) — chính xác 100%                   │
│  N = số segments, D = 512                                            │
└──────────────────────────────────────────────────────────────────────┘
```

#### So sánh các loại Faiss Index

| Index | Tốc độ | Độ chính xác | RAM | Phù hợp khi |
|---|---|---|---|---|
| **IndexFlatIP** *(đang dùng)* | O(N) chậm | 100% (exact) | O(N·D) | N < 100K vectors |
| IndexIVFFlat | O(N/n_list) nhanh hơn | ~95-99% | O(N·D) | 100K < N < 10M |
| IndexHNSWFlat | O(log N) rất nhanh | ~98% | O(N·D·M) | N > 1M, latency nhạy |
| IndexIVFPQ | O(N/n_list) nhanh | ~90-95% | O(N·M/8) rất thấp | Edge deployment |

**Kết luận:** Với video ngắn (< vài giờ), `IndexFlatIP` cho kết quả chính xác nhất. Cần nâng cấp lên `IndexIVFFlat` hoặc `IndexHNSWFlat` khi indexing > 50 video dài.

#### Vấn đề & Điểm yếu hiện tại
- **No post-processing/reranking:** Kết quả raw từ Faiss chưa được lọc hay sắp xếp lại.
- **No duplicate segment merging:** Nhiều windows chồng lấp có thể trả về cùng một thời điểm với score khác nhau.
- **Flat index không scale:** N > 500K vectors sẽ chậm rõ rệt.

---

### 2.4 Searcher — `src/searcher.py`

#### Vai trò
Orchestrator nối toàn bộ pipeline. Không chứa thuật toán AI riêng, nhưng quyết định **chiến lược phân tách video** và **post-processing kết quả**.

#### Flow tìm kiếm

```
query = "người leo rào"
        │
        ▼
CLIPFeatureExtractor.encode_text(query)
→ query_vec: shape (1, 512), L2-normalised
        │
        ▼
VideoIndex.search(query_vec, top_k=5)
→ [(score₁, meta₁), (score₂, meta₂), ...]
  sorted by cosine similarity (descending)
        │
        ▼
SearchResult list
[rank=1, score=0.312, video_id="cam01",
 start_time=42.5s, end_time=47.5s]
```

#### Vấn đề & Điểm yếu hiện tại
- **No temporal NMS (Non-Maximum Suppression):** Nếu "người leo rào" xảy ra ở giây 44-46, có thể có 3-4 windows chồng lấp cùng xuất hiện trong top-5, lãng phí kết quả.
- **No score threshold:** Không lọc kết quả có score quá thấp (noise).
- **No query expansion:** Không mở rộng query với các từ đồng nghĩa.

---

## 3. Luồng Dữ liệu Chi tiết (Data Flow)

```
VIDEO FILE
    │
    │  OpenCV VideoCapture
    ▼
FRAME EXTRACTION (video_processor.py)
    │  Sliding Window / 1-FPS
    │  BGR uint8 (224×224)
    ▼
CLIP VISION ENCODER (feature_extractor.py)
    │  PIL RGB → ViT-B/16 preprocess
    │  [norm, resize, centercrop, to_tensor]
    │  ViT forward pass (float16, CUDA)
    │  [CLS] projection → 512-d
    │  F.normalize → unit vector
    ▼
EMBEDDINGS numpy float32 (N×512)
    │
    │  faiss.IndexFlatIP.add()
    ▼
FAISS INDEX (indexer.py)
    │  stored: float32 matrix (N×512)
    │  pickled metadata: List[SegmentMeta]
    │

─── QUERY TIME ────────────────────────────────────

TEXT QUERY "người leo rào"
    │
    │  CLIP BPE Tokenizer → token ids (77)
    ▼
CLIP TEXT ENCODER (feature_extractor.py)
    │  Transformer forward pass (float16, CUDA)
    │  [EOS] projection → 512-d
    │  F.normalize → unit vector
    ▼
QUERY VECTOR (1×512)
    │
    │  faiss IndexFlatIP.search(q, k)
    │  inner product = cosine sim (O(N·D))
    ▼
TOP-K RESULTS [(score, SegmentMeta), ...]
    │
    ▼
SearchResult(rank, score, video_id, start_time, end_time)
    │
    ▼
UI / CLI OUTPUT
```

---

## 4. Phân tích Độ chính xác Hiện tại

### 4.1 Thế mạnh
| Điểm mạnh | Lý do |
|---|---|
| **Open-vocabulary** | CLIP train trên 400M cặp ảnh-text → hiểu ngôn ngữ tự do, không cần danh sách nhãn cố định |
| **Zero-shot** | Không cần fine-tune cho domain cụ thể |
| **Multilingual (hạn chế)** | Tiếng Việt có thể hoạt động do CLIP train trên dữ liệu đa ngôn ngữ từ web, nhưng kém hơn tiếng Anh |
| **Exact search** | IndexFlatIP đảm bảo 100% recall về mặt vector similarity |

### 4.2 Điểm yếu & Impact

| Vấn đề | Impact | Mức độ |
|---|---|---|
| Single frame / window | Miss hành động nhanh | 🔴 Cao |
| No temporal NMS | Top-5 bị trùng lặp | 🔴 Cao |
| ViT-B/16 nhỏ | Độ chính xác embedding thấp | 🟡 Trung bình |
| No score threshold | Kết quả noise lẫn vào | 🟡 Trung bình |
| Tiếng Việt yếu | Query tiếng Việt kém hơn tiếng Anh | 🟡 Trung bình |
| Fixed window | Hành động < 1s bị hòa loãng | 🟡 Trung bình |
| No reranking | Thứ tự kết quả chưa tối ưu | 🟢 Thấp |

---

## 5. Đề xuất Cải tiến (Roadmap)

### 5.1 Cải tiến Nhanh (1-2 ngày, impact cao)

#### A. Temporal Non-Maximum Suppression (NMS)
Gộp các kết quả chồng lấp nhau về thời gian để không lãng phí top-K:

```python
# Sau khi search(), gộp các segments chồng lấp > 50%
def temporal_nms(results: List[SearchResult], iou_thresh=0.5) -> List[SearchResult]:
    kept = []
    for r in sorted(results, key=lambda x: -x.score):
        overlap = False
        for k in kept:
            if k.video_id == r.video_id:
                inter = max(0, min(k.end_time, r.end_time) - max(k.start_time, r.start_time))
                union = max(k.end_time, r.end_time) - min(k.start_time, r.start_time)
                if inter / union > iou_thresh:
                    overlap = True
                    break
        if not overlap:
            kept.append(r)
    return kept
```

#### B. Multi-frame Averaging per Segment
Thay vì 1 frame, lấy 3 frames (đầu/giữa/cuối window) và trung bình embedding:

```python
# Lấy 3 frames và average embedding
frames = [frame_at(t_start + window*0.1),
          frame_at(t_start + window*0.5),
          frame_at(t_start + window*0.9)]
embs = extractor.encode_frames(frames)   # (3, 512)
segment_emb = embs.mean(axis=0)          # (512,)
segment_emb /= np.linalg.norm(segment_emb)  # re-normalise
```

#### C. Score Threshold
Lọc kết quả có cosine similarity < 0.2 (CLIP score thường < 0.3 cho unrelated content):

```python
results = [r for r in raw_results if r.score >= 0.20]
```

#### D. Query Translation (Tiếng Việt → Tiếng Anh)
Dùng `deep_translator` để dịch query sang tiếng Anh trước khi encode:

```python
from deep_translator import GoogleTranslator
en_query = GoogleTranslator(source='vi', target='en').translate(query)
vec = extractor.encode_text(en_query)
```

---

### 5.2 Cải tiến Trung hạn (1-2 tuần, impact rất cao)

#### E. Nâng cấp Model lên ViT-L/14

| Model | EMBED_DIM | Params | Zero-shot ImageNet | VRAM (float16) |
|---|---|---|---|---|
| ViT-B/16 *(hiện tại)* | 512 | 151M | 68.3% | ~450 MB |
| **ViT-L/14** | 768 | 427M | 75.3% | ~1.4 GB |
| ViT-L/14@336px | 768 | 427M | 76.6% | ~1.4 GB |
| ViT-bigG-14 | 1280 | 1.8B | 80.1% | ~5 GB (quá VRAM) |

**Khuyến nghị:** ViT-L/14 với float16 chỉ tốn ~1.4 GB VRAM → vẫn nằm trong giới hạn 4 GB của GTX 1650 Ti.

```python
# Thay đổi trong feature_extractor.py
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
EMBED_DIM  = 768
```

#### F. Video-aware Model: X-CLIP hoặc InternVideo

**X-CLIP** (Microsoft, 2022) được fine-tune trên video datasets (Kinetics-400/600), hiểu temporal context tốt hơn CLIP thuần:

```
CLIP Image Encoder  ──▶  Cross-frame Attention ──▶ Video Embedding
                          (temporal reasoning)
CLIP Text Encoder   ──▶  Prompt Engineering   ──▶ Text Embedding
```

```python
# Cài đặt
pip install git+https://github.com/microsoft/VideoX.git

# Thay CLIPFeatureExtractor bằng XCLIPFeatureExtractor
from transformers import XCLIPModel, XCLIPProcessor
model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch16")
```

#### G. Adaptive Window Sizing (Scene-based Segmentation)

Dùng PySceneDetect để phát hiện cảnh, sau đó tạo windows theo ranh giới cảnh tự nhiên:

```python
from scenedetect import detect, ContentDetector
scenes = detect(video_path, ContentDetector(threshold=30.0))
# → [(FrameTimecode(0), FrameTimecode(150)), (FrameTimecode(150), ...)]
```

---

### 5.3 Cải tiến Dài hạn (1-2 tháng)

#### H. Two-stage Retrieval: Recall + Reranking

```
Stage 1: CLIP (fast, recall)
  → Top-50 candidates from Faiss
        │
        ▼
Stage 2: Cross-encoder (slow, precision)
  → Rerank Top-50 với mô hình mạnh hơn
  → Output Top-5 chính xác
```

**Cross-encoder candidates:**
- `BLIP-2` (Salesforce): hỏi trực tiếp "Does this image contain [query]?" → Yes/No score
- `LLaVA` (Large Language Vision Assistant): mô tả frame rồi so sánh với query

#### I. Dense Retrieval với Video-specific Training

Fine-tune CLIP trên dữ liệu surveillance/action video với nhãn:
- Dataset: Kinetics-700, UCF-101, ActivityNet Captions
- Loss: InfoNCE contrastive loss trên video-text pairs

#### J. Nâng cấp Faiss Index cho Dataset Lớn

```python
# Khi N > 100K segments
import faiss
quantizer = faiss.IndexFlatIP(768)
index = faiss.IndexIVFFlat(quantizer, 768, n_list=1024)
index.train(all_embeddings)  # cần train trước
index.nprobe = 64            # trade-off: speed vs accuracy
```

---

## 6. Ma trận Đánh giá Cải tiến

| Cải tiến | Effort | Impact Recall | Impact Precision | Ưu tiên |
|---|---|---|---|---|
| A. Temporal NMS | Thấp | — | 🔴 +Cao | ⭐⭐⭐⭐⭐ |
| B. Multi-frame avg | Thấp | 🔴 +Cao | 🔴 +Cao | ⭐⭐⭐⭐⭐ |
| C. Score threshold | Rất thấp | — | 🟡 +Trung bình | ⭐⭐⭐⭐ |
| D. Query translation (VI→EN) | Thấp | 🟡 +Trung bình | 🟡 +Trung bình | ⭐⭐⭐⭐ |
| E. ViT-L/14 | Thấp | 🔴 +Cao | 🔴 +Cao | ⭐⭐⭐⭐ |
| F. X-CLIP | Trung bình | 🔴 +Cao | 🔴 +Cao | ⭐⭐⭐ |
| G. Scene detection | Trung bình | 🟡 +Trung bình | 🟡 +Trung bình | ⭐⭐⭐ |
| H. Two-stage reranking | Cao | — | 🔴 +Rất cao | ⭐⭐ |
| I. Fine-tuning | Rất cao | 🔴 +Cao | 🔴 +Cao | ⭐ |

---

## 7. Tóm tắt Stack Kỹ thuật

```
┌────────────────────────────────────────────────────────┐
│               TECHNOLOGY STACK SUMMARY                 │
├────────────────────────────┬───────────────────────────┤
│ Component                  │ Algorithm / Library       │
├────────────────────────────┼───────────────────────────┤
│ Video decoding             │ OpenCV VideoCapture       │
│ Frame segmentation         │ Sliding Window / 1-FPS   │
│ Image preprocessing        │ CLIP standard transforms  │
│ Vision backbone            │ ViT-B/16 (Transformer)   │
│ Text backbone              │ CLIP Transformer 12L      │
│ Embedding space            │ Joint 512-d (cosine sim)  │
│ Training paradigm          │ Contrastive (InfoNCE)     │
│ Similarity metric          │ Cosine Similarity         │
│ Vector database            │ Faiss IndexFlatIP         │
│ Search algorithm           │ Brute-force dot product   │
│ Compute precision          │ FP16 (inference), FP32    │
│ Acceleration               │ CUDA (GTX 1650 Ti)        │
│ Web UI                     │ Streamlit                 │
└────────────────────────────┴───────────────────────────┘
```

---

*Tài liệu tạo ngày: 2026-05-03 | Prototype v1.0 | GTX 1650 Ti (4GB VRAM)*
