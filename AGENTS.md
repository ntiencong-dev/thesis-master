# NLVS — Agent Instructions

Natural Language Video Search (NLVS) system: text queries retrieve timestamped video segments via BLIP-1 ITC embeddings + Qdrant vector search. PC prototype validates the pipeline; the same model (BLIP-1 ViT-B/16) is compiled to INT8 xmodel for AMD Kria KV260 deployment. See [SYSTEM_KNOWLEDGE_V4.md](SYSTEM_KNOWLEDGE_V4.md) for full architecture.

## Commands

```bash
# Tests (run from project root with venv active)
pytest tests/unit/ -m unit          # fast, no GPU/disk needed
pytest tests/integration/ -m integration
pytest tests/ -q                    # full suite (138+ passed, 1 skipped expected for reranker)

# API server — PC prototype
CONFIG=config/pc_blip1.yaml uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API server — Kria KV260 (after xmodel is compiled)
CONFIG=config/kria_blip1.yaml uvicorn api.main:app --host 0.0.0.0 --port 8000

# Streamlit demo
streamlit run app.py

# Index a video
CONFIG=config/pc_blip1.yaml python index_video.py --dir ./segments

# Qdrant (must be running before server/tests that hit it)
docker run -p 6333:6333 qdrant/qdrant:v1.9.0
```

## Architecture

3-layer async pipeline (PC and Kria share the same code path — only the engine backend differs):

```
Camera/File → capture_daemon.py (OverlapCaptureDaemon)
           → job_queue.py (SQLite WAL, PersistentJobQueue)
           → continuous_indexer.py (background thread)
           → BLIP1Engine.encode_frames()  ←── PC: HuggingFace transformers (CUDA/CPU)
             KriaEngine.encode_frames()   ←── Kria: DPU B4096 via VART/XIR (.xmodel)
           → 256-dim L2-norm ITC vector
           → Qdrant (nlvs_segments_blip1, HNSW cosine, dim=256)

Query text → searcher.py (_normalize_query → 12 prompt templates → encode_text → 256-dim)
          → Qdrant.query_points() → temporal_nms() → SearchResult list
          → BLIP1ITMReranker.rerank()  ← stage-2 ITM cross-encoder (alpha=0.6)
          → RerankResult list (combined score = 0.6×cosine + 0.4×ITM_prob)
```

Key source files: [src/searcher.py](src/searcher.py), [src/engines/blip1_engine.py](src/engines/blip1_engine.py), [src/engines/kria_engine.py](src/engines/kria_engine.py), [src/reranker.py](src/reranker.py), [src/continuous_indexer.py](src/continuous_indexer.py), [api/main.py](api/main.py).

## Configuration

Set via `CONFIG=config/<file>.yaml` env var. Every PC config has a corresponding Kria config with the same model architecture and embed_dim — Qdrant collections are directly interchangeable between them.

| File | Target | Engine backend | embed_dim | Qdrant collection |
|---|---|---|---|---|
| `config/pc_blip1.yaml` | PC / laptop (CUDA or CPU) | `blip1` — HuggingFace transformers | **256** | `nlvs_segments_blip1` |
| `config/kria_blip1.yaml` | AMD Kria KV260 (DPU B4096) | `kria` — VART/XIR .xmodel | **256** | `nlvs_segments_blip1` |

Both configs use `Salesforce/blip-itm-base-coco` (BLIP-1 ViT-B/16). The PC config runs the full PyTorch model; the Kria config runs the visual encoder on the DPU and the BERT text encoder + ITM head on ARM Cortex-A53 CPU.

**Kria xmodel blocker:** `kria_blip1.yaml` is fully configured but the `.xmodel` for BLIP-1 ViT-B/16 is not yet compiled with Vitis AI 3.5. Use `engine.type: pc` with `device: cpu` in `kria_blip1.yaml` for pipeline validation on Kria until the xmodel is ready.

## Engine System

All engines implement `InferenceEngine` ABC ([src/engines/base_engine.py](src/engines/base_engine.py)):
- `encode_frames(frames_bgr: List[np.ndarray]) → np.ndarray` — shape `(N, embed_dim)`, float32, L2-normalized, BGR input
- `encode_text(texts: str | List[str]) → np.ndarray` — shape `(N, embed_dim)`, always returns 2-D even for a single string
- `encode_segment_frames(frames_bgr)` — **soft-max pooling** (shared default impl, defined in `base_engine.py`): encode all N frames → compute deviation from centroid → temperature-scaled softmax (T=0.5) → weighted sum → L2-normalize. Biases toward the most "extreme" frame (e.g. a brief object appearance) instead of plain mean pooling which dilutes rare events.

`BLIP1Engine` additionally exposes:
- `score_itm(frames_bgr, text) → np.ndarray` — shape `(N,)` float32 [0,1], ITM cross-encoder scores used by `BLIP1ITMReranker`

Create via factory: `create_engine(config)` in [src/engines/factory.py](src/engines/factory.py). Registered types: `blip1`, `kria`, `pc` (legacy EVA-CLIP), `siglip`, `xclip`, `languagebind`.

## Reranker System

`src/reranker.py` — stage-2 reranker for the BLIP-1 pipeline:

`BLIP1ITMReranker(engine, alpha=0.6)` — takes the **same** `BLIP1Engine` instance used for indexing (no second model load).

**Frame selection:** `_extract_best_frame(path, t0, t1, query_vec, n_candidates=5)` — samples 5 frames uniformly across the segment, encodes them via ITC (`encode_frames`), computes cosine vs. `query_vec`, and passes the highest-cosine frame to ITM. This outperforms a fixed midpoint heuristic when the query object appears only briefly (e.g. dryer visible 1-2 s inside a 10 s window).

**ITC vector:** `rerank()` calls `encode_text(query)` once to obtain the ITC query vector used in frame selection. `encode_text` returns shape `(1, D)`; the reranker squeezes it to `(D,)` before computing cosines.

Combined score: `alpha × cosine_score + (1 − alpha) × itm_prob`

Enable in config: `search.use_itm_reranker: true`, `search.itm_top_k: 50`, `search.itm_alpha: 0.6`. Attach in code: `searcher.set_reranker(BLIP1ITMReranker(engine))`.

## Critical Pitfalls

**Faiss is removed.** `VideoIndex`/`ScalableVideoIndex`/`FaissIndex` do NOT exist. Qdrant is the sole vector backend. Any Faiss import will break tests.

**`api/main.py` uses Qdrant, not Faiss.** `_index.total_vectors()` was removed in v3.0. Use `_qdrant_total(searcher)` helper (defined in `api/main.py`) which calls `client.get_collection(...).points_count`.

**`_search_qdrant` needs a 1-D vector.** `encode_text(str)` returns shape `(1, D)`. Passing it directly to `query_points(query=vec.tolist())` sends a list-of-lists, which Qdrant treats as a multi-vector query and raises *"Conversion between multi and regular vectors failed"*. `_search_qdrant` squeezes `qvec = qvec[0]` when `qvec.ndim > 1`.

**Qdrant dimension mismatch.** `_init_qdrant()` validates the existing collection's `embed_dim` on startup and raises `RuntimeError("...dim=<existing>...")` on mismatch. Delete the collection or use a different `qdrant_collection` name to resolve.

**`from_params()` has no `index_dir`.** Removed in v3.0. Use `qdrant_collection="<name>"` to select a collection (important for test isolation).

**Query normalization.** `_normalize_query()` in [src/searcher.py](src/searcher.py) strips imperative prefixes ("find", "show me", "search for") before encoding. Never encode raw user input directly.

**BLIP-1 uses `vision_proj`, not `image_projection` or `vision_projection`.** In `blip1_engine.py`, the ITC visual projection head is accessed as `self._model.vision_proj`. Any other attribute name causes `AttributeError`.

**BLIP-1 ITM cross-attention requires `encoder_attention_mask`.** Pass `torch.ones(B, img_seq_len)` as `encoder_attention_mask` when calling `text_encoder` in ITM mode. Omitting it produces silently wrong scores.

**`_extract_best_frame` mock must be patched in tests.** When unit-testing `BLIP1ITMReranker`, mock `engine.encode_text.return_value = np.zeros(256, dtype=np.float32)` (1-D, not 2-D) and patch `reranker._extract_best_frame = lambda path, t0, t1, qvec, n_candidates=5: np.zeros((224,224,3), dtype=np.uint8)`. Patching only `_extract_frame` is insufficient since `rerank()` now calls `_extract_best_frame`.

**`searcher_with_real_video` fixture clears `nlvs_segments_test` first.** The fixture is `scope="session"` and calls `QdrantClient.delete_collection("nlvs_segments_test")` before indexing. Without this, each test run appends vectors and point counts drift across runs (e.g. SR04 expects ≤45 but sees 120 on the 4th run).

**watchdog is optional.** If not installed, `ContinuousIndexer` falls back to 5 s polling — expected, not a bug.

**Qdrant API.** Use `client.query_points()` not `client.search()` (deprecated in qdrant-client ≥1.18).

## Conventions

- Embeddings: always `np.ndarray` float32, L2-normalized, never stored as raw logits
- Frame color space: BGR (OpenCV) throughout pipeline — convert to RGB only inside engine (`f[:, :, ::-1]` in BLIP-1 engine)
- Logging: `logger = logging.getLogger(__name__)` in every module; prefix with `[ClassName]`
- Test IDs: `test_<CATEGORY><NN>_<description>` (e.g., `test_NMS02_identical_pair`)
- `SegmentMeta` (dataclass in [src/indexer.py](src/indexer.py)) — canonical metadata type during indexing
- `SearchResult` / `RerankResult` (dataclasses in [src/searcher.py](src/searcher.py) / [src/reranker.py](src/reranker.py)) — canonical output types

## Thesis Context

This codebase is the **PC prototype** for a master's thesis on deploying a BLIP-1 ViT-B/16 NLVS system on AMD Kria KV260. The unified design principle: **every PC config has a direct Kria deployment equivalent** — same model architecture, same embed_dim, same Qdrant collection format.

| Stage | PC (prototype) | Kria KV260 (target) |
|---|---|---|
| Visual encoder | BLIP-1 ViT-B/16 (HuggingFace, CUDA) | BLIP-1 ViT-B/16 INT8 (DPU B4096, .xmodel) |
| Text encoder + ITM | BLIP-1 BERT (HuggingFace, CUDA) | BLIP-1 BERT (ARM Cortex-A53 CPU) |
| Segment pooling | Soft-max pooling T=0.5 (base_engine.py) | Same (hardware-agnostic) |
| ITM frame selection | ITC-guided best-frame (n=5) | Same — ITC on DPU, cosine on CPU |
| Index | Qdrant `nlvs_segments_blip1` 256-dim | Same collection (vectors portable) |
| Config | `config/pc_blip1.yaml` | `config/kria_blip1.yaml` |

The thesis research contribution is the INT8 quantization + Vitis AI DPU compilation of BLIP-1 ViT-B/16, and the latency/accuracy tradeoff analysis on Kria. See [DE_CUONG_LVTHS.md](DE_CUONG_LVTHS.md) for the thesis outline.


python windows_capture_server.py --output \\wsl$\Ubuntu\home\tienc\Prototype\segments --cam 0 --seg-sec 60