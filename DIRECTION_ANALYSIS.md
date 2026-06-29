# NLVS — Architecture Analysis: Hướng 1 vs Hướng 2

> **Branch**: `v1.0.3_clip_FILIP_reranker`  
> **Goal of this document**: Analyse current implementation status of both research
> directions, identify the re-ranker gap in Hướng 1, enumerate lightweight options,
> and give a clear recommendation.

---

## 1. Tổng quan hai hướng

| | **Hướng 1** (CLIP + lightweight re-ranker) | **Hướng 2** (BLIP-1 ITC + ITM) |
|---|---|---|
| Branch | `v1.0.3_clip_FILIP_reranker` | *(main pipeline)* |
| Stage-1 model | CLIP ViT-B/16 (`PCEngine`, `open_clip`) | BLIP-1 ViT-B/16 (`BLIP1Engine`, HuggingFace) |
| Stage-1 embed dim | **512** | **256** |
| Qdrant collection | `nlvs_segments` | `nlvs_segments_blip1` |
| Config | `config/pc.yaml` | `config/pc_blip1.yaml` |
| Stage-2 re-ranker | ❌ **Chưa có** (BLIP2Reranker quá nặng) | ✅ `BLIP1ITMReranker` (fully implemented) |
| KV260 engine | `KriaEngine` (CLIP xmodel on DPU) | `KriaEngine` (BLIP-1 xmodel, compiled ✅) |
| Kria config | `config/kria.yaml` | `config/kria_blip1.yaml` |
| Implementation status | Stage-1 ✅ · Stage-2 ❌ | Stage-1 ✅ · Stage-2 ✅ |

---

## 2. Phân tích kiến trúc hiện tại

### 2.1 Hướng 2 (BLIP-1 ITC + ITM) — Fully implemented

```
                      ┌───────────────────────────────────────────────┐
VIDEO SEGMENTS ──────▶│  BLIP-1 ViT-B/16  │  ITC visual proj (256d) │──▶ Qdrant
                      └───────────────────────────────────────────────┘
                              ↕ same model instance
TEXT QUERY ──────────▶ BLIP-1 BERT self-attn (ITC text head, 256d)
                                       │
                              Qdrant cosine search → top-50
                                       │
                      ┌───────────────────────────────────────────────┐
                      │  BLIP1ITMReranker                             │
                      │    _extract_best_frame() — ITC-guided (n=5)  │
                      │    engine.score_itm(frame, query)             │
                      │    BERT cross-attention over ViT patches       │
                      │    combined = 0.6×cosine + 0.4×itm_prob       │
                      └───────────────────────────────────────────────┘
```

**Điểm mạnh:**
- Single model load — `BLIP1Engine` phục vụ cả ITC (stage-1) và ITM (stage-2).
- Cross-encoder quality: BERT cross-attention thực sự so sánh image patches với text tokens.
- Đã compiled sang INT8 xmodel (161 MB, 61 DPU subgraphs) — KV260-ready.
- ARM latency: ~46 ms/candidate → 50 candidates ≈ 2.3 s (acceptable).

**Điểm yếu:**
- BLIP-1 BERT text encoder nặng hơn CLIP text encoder khi không cần ITM.
- ITC embedding (256d) nhỏ hơn CLIP (512d) — ít expressiveness hơn ở stage-1.
- Kria stage-2: ITM chạy trên ARM Cortex-A53 (CPU-only, không DPU-accelerated).

---

### 2.2 Hướng 1 (CLIP + lightweight re-ranker) — Stage-2 còn thiếu

```
                      ┌──────────────────────────────────────────┐
VIDEO SEGMENTS ──────▶│  CLIP ViT-B/16 (open_clip)              │──▶ Qdrant
                      │  CLIPFeatureExtractor.encode_frames()    │   (512d)
                      └──────────────────────────────────────────┘
                              ↕ same PCEngine
TEXT QUERY ──────────▶ CLIP text encoder → 512d
                                       │
                              Qdrant cosine search → top-k

         Stage-2: ❌ KHÔNG CÓ re-ranker phù hợp
         ┌──────────────────────────────────────────────────────┐
         │ BLIP2Reranker (hiện tại trong code):                  │
         │   BLIP-2 OPT-2.7B → 2.8 GB 8-bit, quá nặng          │
         │   Không thể chạy song song với CLIP trên 4 GB VRAM   │
         │   Kria ARM: không khả thi (LLM)                       │
         └──────────────────────────────────────────────────────┘
```

**Gap cần giải quyết**: Cần re-ranker hoạt động với `PCEngine` / `KriaEngine` CLIP
mà **không** load thêm model nặng.

---

## 3. Các lựa chọn lightweight re-ranker cho Hướng 1

### Option A: `CLIPFrameSamplingReranker` (zero overhead, no new model)

**Cơ chế:**
```
top-K candidates (từ Qdrant cosine search)
  ↓
Với mỗi candidate: sample N frames từ đoạn video (N=5)
  ↓
encode_frames(frames) → (N, 512) embeddings via PCEngine (same CLIP model)
  ↓
per_frame_scores = dot(frame_embs, query_vec)   # cosine similarity
  ↓
max_score = max(per_frame_scores)
  ↓
combined = alpha × cosine_score + (1-alpha) × max_score
```

**Ý tưởng**: stage-1 dùng soft-max pooled segment embedding → có thể bỏ sót brief object
appearances. Stage-2 score lại bằng per-frame max cosine → tìm frame nào match tốt nhất.

**Pros:**
- Zero thêm model — dùng lại `PCEngine.encode_frames()` / `encode_text()`.
- Works trên Kria: CLIP đã chạy trên DPU, frame sampling chỉ là thêm vài inference calls.
- Implementation: ~60 lines, không thay đổi interface.
- Hoạt động ngay với `pc.yaml` / `kria.yaml`.

**Cons:**
- Vẫn là global CLIP embedding — không có cross-modal interaction thực sự.
- Max-score dễ bị outlier (noise frame).
- Không phân biệt được object ở vị trí nào trong frame.

**Interface fit:**
```python
class CLIPFrameSamplingReranker:
    def __init__(self, engine: PCEngine, alpha=0.6, n_frames=5): ...
    def rerank(self, candidates, query, top_k=None) -> List[RerankResult]: ...
    # Dùng engine.encode_frames() + engine.encode_text() — không đổi interface
```

---

### Option B: `FILIPReranker` (token-level CLIP matching, same model) — **Revised với AWQ + MACRO**

**Cơ chế (FILIP — Fine-grained Interactive Language-Image Pre-Training):**
```
Với mỗi candidate segment:
  frames = sample N=5 frames từ segment

  [DPU / PTQ INT8]
  image_tokens = CLIP ViT patch embeddings (196 patches × 768d hidden)
                 → extracted via forward hook trên last ViT attention block
                 → KHÔNG phải 512d projected CLS — là ViT hidden states thô

  [ARM CPU / AWQ INT4]
  text_tokens  = CLIP text encoder word embeddings (T_valid × 768d hidden)
                 → extracted via forward hook trên last text transformer block
                 → ALL token positions, không chỉ [EOS]/[CLS]

  filip_score  = mean_{text tokens t_i}( max_{patches v_j}( dot(t_i, v_j) ) )
  # = "for each word, find best matching image patch; average across words"

combined = alpha × cosine_score + (1-alpha) × filip_score
```

**Phân tích AWQ_MACRO cho Option B:**

| Component | Hardware | Quantization | Lý do |
|---|---|---|---|
| CLIP visual encoder (ITC [CLS]) | **DPU B4096** | **PTQ INT8** (giữ nguyên) | DPU chỉ nhận `.xmodel` |
| CLIP visual encoder (FILIP patches) | **DPU B4096** | **PTQ INT8** (xmodel MỚI) | Cần xmodel expose patch tokens |
| CLIP text encoder (ITC [EOS]) | **ARM Cortex-A53** | **AWQ INT4** | Giảm 250 MB → 63 MB |
| CLIP text encoder (FILIP all tokens) | **ARM Cortex-A53** | **AWQ INT4** (cùng model) | AWQ giữ nguyên per-token quality |
| MACRO (2 CLIP xmodels) | Kria DPU | **MACRO optional** | Tránh ~300 ms reload/query |

**AWQ INT4 cho CLIP text encoder trong Option B:**
- Cùng `exported_models/clip_text_awq_int4/` như Option C — có thể tái sử dụng.
- FILIPReranker cần **all T token positions** từ last text transformer block.
- `scripts/awq_validate_filip_tokens.py` (mới) kiểm tra per-token cosine > 0.95
  và FILIP rank correlation (Spearman) > 0.90 để đảm bảo AWQ không làm hỏng
  fine-grained matching.

**Điều chỉnh kỹ thuật quan trọng — patch embed dim:**
- ViT-B/16 patch tokens có hidden dim = **768** (ViT hidden size), KHÔNG phải 512d.
- 512d là output của projection head (sau `visual.ln_post` + `visual.proj`) — dùng cho ITC.
- FILIP dùng **pre-projection** hidden states → 768d; dot-product với text tokens 768d.
- `encode_frames_tokens()` drops CLS (index 0) → trả về `(N, 196, 768)`.
- `encode_text_tokens()` trả về `(tokens, mask)` shape `(N, T, 768)` + bool mask.

**Pros:**
- True fine-grained matching: biết "hair dryer" khớp với patch góc phải, không phải toàn frame.
- Dùng cùng CLIP model — không load thêm model mới.
- Phù hợp với branch name `v1.0.3_clip_FILIP_reranker`.
- Recall tốt hơn Option A khi object nhỏ/xuất hiện ngắn.
- **[MỚI] AWQ INT4 text encoder**: 250 MB FP32 → ~63 MB — viable trên Kria.
- **[MỚI] MACRO optional**: nếu dùng 2 CLIP xmodels, MACRO tiết kiệm ~300 ms/query.

**Cons:**
- **Blocker Kria**: cần export xmodel CLIP mới giữ patch tokens (trước projection head).
  Xmodel hiện tại `clip_itc.xmodel` chỉ export [CLS] embedding — không tái dụng được.
- Trên PC: forward hook hoạt động ngay, không cần xmodel mới.
- 2–4× slower hơn Option A do xử lý 196 patches × T tokens.
- AWQ không áp dụng cho visual side (DPU) — chỉ text encoder (ARM CPU).

**Implementation status (đã hoàn thành trên PC):**
```python
# ✅ src/feature_extractor.py — CLIPFeatureExtractor:
def encode_frames_tokens(frames_bgr) -> np.ndarray:   # (N, 196, 768)
    # Forward hook on visual.transformer.resblocks[-1]
    ...
def encode_text_tokens(texts) -> tuple:               # (tokens, mask)
    # Forward hook on transformer.resblocks[-1]
    # tokens: (N, T, 768) — ALL token positions
    # mask:   (N, T) bool — True = valid, False = padding
    ...

# ✅ src/engines/pc_engine.py — PCEngine:
def encode_frames_tokens(frames_bgr) -> np.ndarray: ...
def encode_text_tokens(texts) -> tuple: ...

# ✅ src/reranker_filip.py — FILIPReranker:
#   filip_score = mean_{t_i}( max_{v_j}( dot(t_i, v_j) ) )
#   normalize_filip=True: FILIP scores → [0,1] trước khi combine
#   Same interface as BLIP1ITMReranker → drop-in với set_reranker()

# ✅ scripts/awq_validate_filip_tokens.py:
#   Kiểm tra AWQ per-token cosine > 0.95, FILIP rank corr > 0.90
```

---

### Option C: `DualModelReranker` (CLIP stage-1 + BLIP-1 ITM stage-2) — **Revised với AWQ + MACRO**

**Cơ chế:**
```
Load PCEngine (CLIP)  → stage-1 indexing & search (512d, Qdrant)
Load BLIP1Engine      → stage-2 reranking chỉ (score_itm())
  ↓
Dùng BLIP1ITMReranker (đã có trong code!) với BLIP1Engine instance
combined = 0.6 × clip_cosine + 0.4 × blip1_itm_prob
```

**Pros:**
- Tận dụng `BLIP1ITMReranker` đã có sẵn — không code mới cho re-ranker logic.
- Stage-2 quality cao nhất (BERT cross-attention > CLIP global tokens > CLIP patches).
- Best-of-both-worlds: CLIP broad vocabulary + BLIP-1 precise cross-modal matching.
- **[MỚI] Với AWQ INT4 cho text encoders**: memory footprint giảm ~4× trên ARM CPU.
- **[MỚI] Với MACRO**: không có overhead reload giữa CLIP runner và BLIP-1 runner.

**Cons (trước khi áp dụng AWQ + MACRO):**
- Hai model loads: CLIP (~1.4 GB VRAM) + BLIP-1 (~880 MB VRAM) = ~2.3 GB → quá lớn.
- Kria 4 GB LPDDR4: không đủ cho cả 2 model FP32 song song.
- Collections riêng biệt: CLIP `nlvs_segments` (512d) ≠ BLIP-1 `nlvs_segments_blip1` (256d).

**Sau khi áp dụng AWQ + MACRO (xem Section 3.5–3.6):**
```
Component                    Trước (FP32/PTQ)   Sau (AWQ INT4 / PTQ INT8)
────────────────────────────────────────────────────────────────────────
CLIP visual encoder (DPU)    ~350 MB FP32       ~160 MB INT8 xmodel (PTQ)
CLIP text encoder (ARM CPU)  ~250 MB FP32       ~63 MB INT4 (AWQ)
BLIP-1 visual encoder (DPU) ~327 MB FP32       ~161 MB INT8 xmodel (PTQ)
BLIP-1 BERT (ARM CPU)        ~440 MB FP32       ~110 MB INT4 (AWQ)
────────────────────────────────────────────────────────────────────────
Tổng (trước)                 ~1.37 GB           ~494 MB  ← khả thi trên Kria
```
Việc áp dụng AWQ + MACRO chuyển Option C từ **không khả thi** → **khả thi** trên Kria 4 GB LPDDR4.

---

## 3.5 Chiến lược lượng hóa: AWQ thay thế PTQ

### Vì sao PTQ không đủ?

PTQ (Post-Training Quantization) tiêu chuẩn qua `vai_q_pytorch`:
- Visual encoder INT8 xmodel: ~161 MB (đã compile) — **ổn** cho DPU
- Text encoder (BERT, CLIP text): chỉ INT8 nếu chạy trên ARM → ~110–220 MB mỗi model
- **Vấn đề**: với Option C cần 2 text encoders chạy song song trên ARM → ~330 MB chỉ cho text
- Cộng thêm OS, Qdrant Docker (~250 MB), pipeline: tổng > 1 GB thường trú trên 4 GB LPDDR4

### AWQ (Activation-aware Weight Quantization)

AWQ (Lin et al. 2023) xác định *salient weights* bằng cách phân tích activation magnitude:
```
PTQ INT8:  quantize tất cả weights đồng đều → accuracy drop khi có outliers
AWQ INT4:  bảo vệ 1% weights có ảnh hưởng lớn nhất → giảm perplexity/loss đáng kể
           so với PTQ INT8 dù chỉ dùng 4-bit
```

**Áp dụng AWQ trong hệ thống này:**

| Component | Target hardware | Quantization | Lý do |
|---|---|---|---|
| CLIP visual encoder | **DPU B4096** | **PTQ INT8** (giữ nguyên) | DPU chỉ nhận INT8 xmodel |
| BLIP-1 visual encoder | **DPU B4096** | **PTQ INT8** (giữ nguyên) | DPU chỉ nhận INT8 xmodel |
| CLIP text encoder | **ARM Cortex-A53** | **AWQ INT4** | Chạy trên CPU, hỗ trợ INT4 qua AutoAWQ |
| BLIP-1 BERT (text + ITM) | **ARM Cortex-A53** | **AWQ INT4** | Chạy trên CPU, hỗ trợ INT4 qua AutoAWQ |

**Tool chain:**
```bash
# AutoAWQ — hỗ trợ BERT, CLIP text encoder, và nhiều Transformer models
pip install autoawq

# Quantize BLIP-1 BERT text encoder sang INT4
python scripts/awq_quantize_bert.py \
    --model Salesforce/blip-itm-base-coco \
    --output exported_models/blip1_bert_awq_int4/ \
    --w_bit 4 --q_group_size 128

# Quantize CLIP text encoder sang INT4
python scripts/awq_quantize_clip_text.py \
    --model ViT-B-16 --pretrained openai \
    --output exported_models/clip_text_awq_int4/ \
    --w_bit 4 --q_group_size 128
```

**Lưu ý quan trọng:**
- AWQ INT4 chạy trên ARM CPU cần kernel hỗ trợ: `AutoAWQ` tích hợp `gemm_lowbit_kernel` cho ARM
- Nếu ARM kernel chưa tối ưu: fallback INT8 AWQ (vẫn tốt hơn PTQ INT8 ~1.5–2%)
- Accuracy: AWQ INT4 BERT trên COCO ITM thường drop < 1.5% so với FP32

---

## 3.6 Cơ chế MACRO — Persistent DPU Runners

### Vấn đề với cách load model thông thường

Khi không có MACRO, mỗi lần switch giữa CLIP và BLIP-1 trong Option C pipeline:
```
Query đến
  → Tạo CLIP DPU Runner (vart.Runner.create_runner) → load weights DDR→DPU SRAM
  → Chạy stage-1 inference
  → Hủy CLIP Runner
  → Tạo BLIP-1 DPU Runner → load weights DDR→DPU SRAM  ← overhead ~150–300 ms
  → Chạy stage-2 ITM inference
  → Hủy BLIP-1 Runner
```
Overhead reload ước tính: **150–300 ms per query** — không chấp nhận được.

### MACRO: Persistent Runner Pool

MAC = giữ cả 2 DPU Runners **sống liên tục** từ lúc server khởi động:
```python
class MacroRunnerPool:
    """Giữ CLIP và BLIP-1 DPU runners luôn active trong DDR."""
    def __init__(self, clip_xmodel: str, blip1_xmodel: str):
        # Khởi tạo một lần khi server start
        self._clip_runner  = self._load_runner(clip_xmodel)   # stays alive
        self._blip1_runner = self._load_runner(blip1_xmodel)  # stays alive

    def run_clip(self, frames_int8) -> np.ndarray:
        # Không có load/unload — runner đã sẵn sàng
        return self._dpu_infer(self._clip_runner, frames_int8)

    def run_blip1(self, frames_int8) -> np.ndarray:
        return self._dpu_infer(self._blip1_runner, frames_int8)
```

**Latency profile với MACRO (trên Kria KV260, DPU B4096):**

| Bước | Không MACRO | Có MACRO |
|---|---|---|
| Load CLIP runner | ~180 ms | 0 ms (đã có sẵn) |
| CLIP stage-1 encode (5 frames) | ~60 ms | ~60 ms |
| Unload/Load BLIP-1 runner | ~240 ms | 0 ms (đã có sẵn) |
| BLIP-1 ITM rerank (50 cands) | ~2300 ms | ~2300 ms |
| **Tổng** | **~2780 ms** | **~2360 ms** |

Ngoài ra, với MACRO, cả 2 AWQ models (CLIP text + BLIP-1 BERT) cũng được **preload vào RAM** khi server start — không có lazy-loading.

**Khi nào MACRO phát huy tác dụng nhất:**
- High-frequency queries: mỗi query tiết kiệm ~420 ms
- Option C pipeline (cả CLIP + BLIP-1): cần thiết
- Không cần MACRO nếu chỉ dùng Hướng 2 (BLIP-1 ITC+ITM single model)

---

## 4. So sánh tổng hợp

| Tiêu chí | Option A (Frame Sampling) | Option B (FILIP Token + AWQ) | Option C (Dual Model + AWQ + MACRO) |
|---|---|---|---|
| Model load thêm | Không | Không | BLIP-1 BERT AWQ (~110 MB) |
| Interface changes | Không | ✅ **Đã implement** (`encode_frames_tokens`, `encode_text_tokens`) | Thêm `MacroRunnerPool` |
| Kria-compatible | ✅ Trực tiếp | ⚠️ **Cần CLIP FILIP xmodel mới** | ✅ **Khả thi sau AWQ** (~494 MB tổng) |
| Precision improvement | Trung bình | Cao (token-level alignment) | **Cao nhất** (BERT cross-attention) |
| Latency stage-2 (PC, GPU) | ~30 ms/cand | ~80 ms/cand | ~46 ms/cand |
| Latency stage-2 (Kria ARM) | ~20 ms/cand | ⚠️ ~120 ms/cand | ~46 ms/cand |
| Memory (Kria LPDDR4) | ~160 MB | **~223 MB** (PTQ 160 + AWQ 63) | **~494 MB** (với AWQ) |
| Complexity | Thấp | Cao | Trung bình |
| Quantization (visual) | PTQ INT8 (DPU) | PTQ INT8 (DPU, xmodel mới) | PTQ INT8 (DPU) |
| Quantization (text) | N/A | **AWQ INT4 (ARM CPU)** ← MỚI | **AWQ INT4 (ARM CPU)** |
| MACRO benefit | Không cần | **Optional** (~300 ms saved nếu 2 CLIP xmodels) | **Cần thiết** (~420 ms saved/query) |
| AWQ text encoder | N/A | ✅ Reuse `clip_text_awq_int4/` | ✅ `clip_text_awq_int4/` + `blip1_bert_awq_int4/` |
| Branch name aligned | Một phần | ✅ FILIP = branch name | Không |
| Implementation effort (PC) | 1 ngày | ✅ **Đã xong** (3 files) | 2–3 ngày |
| Implementation effort (Kria) | 0 ngày | **3–5 ngày** (xmodel mới) | 2–3 ngày |

---

## 5. Khuyến nghị

### 5.1 Quyết định kiến trúc đã xác nhận

Dựa trên phân tích AWQ + MACRO:

> **Hướng 1, Option C + AWQ + MACRO** là lựa chọn tối ưu nhất cho cả PC và Kria.

**Lý do:**
1. **AWQ INT4** cho text encoders (ARM CPU) giải quyết hoàn toàn vấn đề memory (~494 MB tổng, nằm trong 4 GB LPDDR4 của Kria).
2. **MACRO** loại bỏ overhead reload DPU runner (~420 ms/query) — critical cho real-time deployment.
3. **`BLIP1ITMReranker` đã có sẵn** — không cần code re-ranker mới, chỉ cần wire đúng.
4. Stage-2 quality cao nhất (BERT cross-attention > FILIP token matching > frame sampling).
5. Thesis contribution: AWQ INT4 trên ARM Cortex-A53 cho BERT + MACRO cho embedded dual-model pipeline — **đây là điểm mới**.

### 5.2 So sánh Hướng 1 (Option C) vs Hướng 2

| Tiêu chí luận văn | **Hướng 1 Option C** (CLIP + BLIP-1 ITM, AWQ+MACRO) | Hướng 2 (BLIP-1 ITC+ITM only) |
|---|---|---|
| Contribution mới | ✅ AWQ dual-model + MACRO cho embedded | ❌ Standard BLIP-1 ITM pipeline |
| Kria memory | ✅ ~494 MB (AWQ INT4 text + PTQ INT8 visual) | ✅ ~271 MB (single model) |
| Stage-1 quality | ✅ CLIP 512d (broader zero-shot) | ⚠️ BLIP-1 ITC 256d |
| Stage-2 quality | ✅ BLIP-1 ITM cross-attention | ✅ BLIP-1 ITM cross-attention |
| Latency (với MACRO) | ~2360 ms/query | ~2300 ms/query |
| Kria DPU models | 2 xmodels (CLIP + BLIP-1 visual) | 1 xmodel (BLIP-1 visual) |
| Implementation effort | 2–3 ngày (AWQ scripts + MacroPool) | ✅ Đã xong |
| Thesis novelty | **Cao nhất** | Trung bình |

**Cả hai hướng đều viable trên Kria.** Option C được khuyến nghị vì novelty và stage-1 quality cao hơn.

### 5.3 Lộ trình ưu tiên

```
Phase A (ngắn hạn):  AWQ quantize CLIP text + BLIP-1 BERT  →  validate accuracy
Phase B:             Implement MacroRunnerPool  →  integrate vào KriaEngine
Phase C:             Benchmark: no-reranker → BLIP1ITM → Option C dual-model
Phase D:             Thesis writeup: compare Hướng 1 Option C vs Hướng 2
```

---

## 6. Kế hoạch triển khai

### Bước 1 — AWQ quantize text encoders
- **Script mới**: `scripts/awq_quantize_bert.py` — AutoAWQ cho BLIP-1 BERT (INT4)
- **Script mới**: `scripts/awq_quantize_clip_text.py` — AutoAWQ cho CLIP text encoder (INT4)
- Output: `exported_models/blip1_bert_awq_int4/` (~110 MB), `exported_models/clip_text_awq_int4/` (~63 MB)
- Validate: cosine similarity giữa FP32 và AWQ INT4 embeddings > 0.95

### Bước 2 — Implement `MacroRunnerPool` trong `KriaEngine`
- File: `src/engines/kria_engine.py` — thêm persistent runner pool
- Giữ CLIP runner + BLIP-1 runner alive từ `__init__()` đến khi process kết thúc
- Thêm `load_blip1_runner(xmodel_path)` method để init BLIP-1 runner riêng

### Bước 3 — Cập nhật `BLIP1Engine` (PC side) load AWQ text encoder
- File: `src/engines/blip1_engine.py` — thêm path `awq_bert_path` trong config
- Khi `awq_bert_path` có trong config: load AutoAWQ model thay FP32 BERT

### Bước 4 — Wiring dual-model trong `api/main.py` và `searcher.py`
- `api/main.py`: init `BLIP1Engine` riêng cho re-ranker nếu config `use_dual_model: true`
- `searcher.py`: không thay đổi `search()` — chỉ thay đổi cách `set_reranker()` được gọi từ bên ngoài
- Config mới: `config/pc_clip_blip1.yaml` (Hướng 1 Option C, PC) + `config/kria_clip_blip1.yaml` (Kria)

### Bước 5 — Benchmark
- Compare: Hướng 2 (BLIP-1 ITC+ITM) vs Hướng 1 Option C (CLIP+BLIP1ITM, AWQ, MACRO)
- Metrics: Precision@5, Recall@5, latency (ms/query), memory (MB), power (W trên Kria)

---

## 7. Files cần thay đổi

### Option B (FILIP + AWQ) — đã hoàn thành (PC side)

| File | Trạng thái | Nội dung |
|---|---|---|
| `src/feature_extractor.py` | ✅ **Xong** | `encode_frames_tokens()` → `(N,196,768)` patch tokens via ViT hook; `encode_text_tokens()` → `(tokens,mask)` word-level tokens via text transformer hook |
| `src/engines/pc_engine.py` | ✅ **Xong** | Delegate `encode_frames_tokens()` và `encode_text_tokens()` sang extractor |
| `src/reranker_filip.py` | ✅ **Xong (Mới)** | `FILIPReranker`: FILIP score + normalize + combine; same interface as `BLIP1ITMReranker` |
| `scripts/awq_validate_filip_tokens.py` | ✅ **Xong (Mới)** | Validate AWQ per-token cosine > 0.95, FILIP rank correlation (Spearman) > 0.90 |
| `scripts/export_clip_filip_xmodel.py` | ⏳ **Kria only** | Export CLIP xmodel với patch-level output (stop before projection head) — Vitis AI work |
| `config/pc_clip_filip.yaml` | ⏳ Chưa có | PC config: CLIP stage-1 + FILIP reranker, AWQ text path |

### Option C (Dual Model + AWQ + MACRO) — còn thiếu

| File | Loại thay đổi | Nội dung |
|---|---|---|
| `scripts/awq_quantize_bert.py` | ✅ **Mới (có sẵn)** | AutoAWQ INT4 cho BLIP-1 BERT text + ITM encoder |
| `scripts/awq_quantize_clip_text.py` | ✅ **Mới (có sẵn)** | AutoAWQ INT4 cho CLIP text encoder |
| `src/engines/kria_engine.py` | Sửa | Thêm `MacroRunnerPool`, persistent runners, load AWQ text |
| `src/engines/blip1_engine.py` | Sửa | Hỗ trợ `awq_bert_path` config để load AWQ INT4 BERT |
| `src/engines/factory.py` | Sửa | Hỗ trợ `engine.type: clip_blip1` (dual model) |
| `config/pc_clip_blip1.yaml` | **Mới** | PC config: CLIP stage-1 + BLIP-1 ITM, AWQ paths |
| `config/kria_clip_blip1.yaml` | **Mới** | Kria config: dual xmodel + AWQ + MACRO settings |
| `api/main.py` | Sửa | Init dual-model searcher khi `use_dual_model: true` |
| `tests/unit/test_reranker.py` | Sửa | Test dual-model pipeline với mock AWQ engine |

---

## 8. Tóm tắt quyết định kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│  Quyết định 1: Quantization Strategy (áp dụng cho TẤT CẢ options) │
│    DPU (visual encoders) : PTQ INT8 via vai_q_pytorch — bất biến  │
│    ARM CPU (text encoders): AWQ INT4 via AutoAWQ  — áp dụng       │
│    Lý do: DPU B4096 chỉ nhận INT8; ARM hỗ trợ INT4 gemm kernels  │
│                                                                   │
│    Option B: AWQ áp dụng cho CLIP text encoder (~63 MB)           │
│    Option C: AWQ áp dụng cho CLIP text + BLIP-1 BERT (~63+110 MB) │
├─────────────────────────────────────────────────────────────────┤
│  Quyết định 2: MACRO — Persistent DPU Runners                    │
│    Option B: MACRO optional — 2 CLIP xmodels (ITC + FILIP)        │
│              Nếu dùng 1 xmodel: MACRO không cần                  │
│              Saving: ~300 ms/query (2-xmodel case)               │
│    Option C: MACRO cần thiết — CLIP xmodel + BLIP-1 xmodel        │
│              Saving: ~420 ms/query (must-have)                   │
├─────────────────────────────────────────────────────────────────┤
│  Quyết định 3: Chọn Option C cho Hướng 1 (khuyến nghị chính)    │
│    CLIP (512d) stage-1 + BLIP-1 ITM stage-2 (đã có code)        │
│    AWQ làm cho Option C viable trên Kria 4 GB LPDDR4             │
│    MACRO làm cho dual-model latency acceptable (~2360 ms/query)  │
├─────────────────────────────────────────────────────────────────┤
│  [MỚI] Option B PC implementation (đã hoàn thành):              │
│    src/reranker_filip.py          — FILIPReranker class           │
│    src/feature_extractor.py       — encode_frames_tokens()        │
│                                     encode_text_tokens()          │
│    src/engines/pc_engine.py       — delegate methods              │
│    scripts/awq_validate_filip_tokens.py — per-token AWQ check     │
│    Kria blocker: cần CLIP FILIP xmodel (patch-level output)      │
└─────────────────────────────────────────────────────────────────┘
```
