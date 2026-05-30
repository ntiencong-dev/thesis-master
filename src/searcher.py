"""
searcher.py — config-driven, engine-agnostic orchestrator.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import yaml
from tqdm import tqdm
from .engines.base_engine import InferenceEngine
from .engines.factory import create_engine
from .gst_pipeline import VideoPipeline
from .indexer import SegmentMeta, temporal_nms, calculate_iou
from .video_processor import VideoSegment

# ---------------------------------------------------------------------------
# Query normalisation helpers
# ---------------------------------------------------------------------------

# Imperative prefixes that add no semantic meaning for CLIP's text encoder
_QUERY_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"find(?:\s+(?:the|me|a|an|all|all\s+the|me\s+the))?"
    r"|show(?:\s+(?:me|the|a|an|me\s+the))?"
    r"|search(?:\s+for)?"
    r"|get(?:\s+me(?:\s+the)?)?"
    r"|look(?:\s+for)?"
    r"|locate|detect|identify"
    r"|can\s+you\s+(?:find|show|search\s+for)?"
    r"|please\s+(?:find|show)?"
    r"|i\s+(?:want|need)\s+to\s+see"
    r")\s+",
    re.IGNORECASE,
)

# CLIP prompt templates — averaging their embeddings improves recall on
# action and event queries (mirrors the ensemble strategy from the CLIP paper).
# Phase 1: expanded from 5 → 12 templates, adding action/event and
# surveillance-specific phrasings (RESEARCH_SOTA_NLVS.md §I3).
_CLIP_TEMPLATES: List[str] = [
    # General (original 5)
    "{}",
    "a photo of {}",
    "a video frame of {}",
    "a scene with {}",
    "an image of {}",
    # Action / event specific
    "a person {}",
    "someone is {}",
    "a video of a person {}",
    "security camera footage of {}",
    "surveillance video showing {}",
    # Scene-level
    "a scene showing {}",
    "footage of {}",
]


def _normalize_query(query: str) -> str:
    """
    Strip imperative search prefixes from a user query.

    Examples
    --------
    "find the man waving hand"   → "man waving hand"
    "show me a red car"          → "a red car"
    "search for running person"  → "running person"
    """
    query = query.strip()
    # Apply iteratively — handles "find me the man..." (two prefixes chained)
    for _ in range(3):
        cleaned = _QUERY_PREFIX_RE.sub("", query)
        if cleaned == query:
            break
        query = cleaned
    return query.strip() or query   # never return empty string

@dataclass
class SearchResult:
    score:          float
    cam_id:         str        # camera/source ID (filename stem for pre-recorded)
    video_path:     str
    start_time:     float      # relative_start — seconds from file start (for seeking)
    end_time:       float      # relative_end
    rank:           int
    absolute_start: float = 0.0   # wall-clock unix timestamp (0 for pre-recorded)
    absolute_end:   float = 0.0

    @property
    def video_id(self) -> str:
        """Backward-compatibility alias for cam_id."""
        return self.cam_id

class NLVideoSearcher:
    _DEFAULT: dict = {
        "backend": "pc",
        # Phase 1: default to EVA-CLIP ViT-L/14
        "engine":   {"type":"pc","model_name":"EVA02-L-14","pretrained":"merged2b_s4b_b131k","device":None,"batch_size":16,"frames_per_window":5},
        "pipeline": {"video_backend":"opencv","window_sec":10.0,"overlap_ratio":0.30},
        "index":    {"embed_dim":768,"backend":"qdrant","qdrant_url":"http://localhost:6333","qdrant_collection":"nlvs_segments"},
        "search":   {"top_k":5,"score_threshold":0.10,"nms_iou_threshold":0.30,
                     "adaptive_threshold":True,"translate_vi":True},
    }
    def __init__(self, config=None):
        self._config = {**self._DEFAULT, **(config or {})}
        self._engine = create_engine(self._config)

        # Qdrant is the sole vector store backend
        self._qdrant_client     = None
        self._qdrant_collection: Optional[str] = None
        idx_cfg = self._config.get("index", {})
        self._init_qdrant(idx_cfg)

        p = self._config.get("pipeline", {}); e = self._config.get("engine", {})
        self._window_sec        = p.get("window_sec",       5.0)
        self._overlap_ratio     = p.get("overlap_ratio",    0.5)
        self._frames_per_window = e.get("frames_per_window", 5)
        self._video_backend     = p.get("video_backend",    "opencv")
        s = self._config.get("search", {})
        self._default_top_k      = s.get("top_k",              5)
        self._score_threshold    = s.get("score_threshold",    0.20)
        self._nms_iou            = s.get("nms_iou_threshold",  0.50)
        self._adaptive_threshold = s.get("adaptive_threshold", True)
        self._translate_vi       = s.get("translate_vi",       True)

        # Phase 3: optional BLIP-2 reranker (None by default)
        self._reranker = None

    # ------------------------------------------------------------------
    # Qdrant backend helpers
    # ------------------------------------------------------------------

    def _init_qdrant(self, idx_cfg: dict) -> None:
        """
        Connect to Qdrant and create the collection if it does not exist.
        Raises RuntimeError if Qdrant is unavailable (no Faiss fallback).
        Raises RuntimeError if the collection exists with the wrong embed_dim.
        """
        url  = idx_cfg.get("qdrant_url",        "http://localhost:6333")
        coll = idx_cfg.get("qdrant_collection", "nlvs_segments")
        dim  = idx_cfg.get("embed_dim",         self._engine.embed_dim)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            client = QdrantClient(url=url, timeout=10)
            existing = {c.name for c in client.get_collections().collections}
            if coll not in existing:
                client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            else:
                # Validate that the existing collection has the expected dimension.
                info        = client.get_collection(coll)
                vectors_cfg = info.config.params.vectors
                existing_dim = (
                    vectors_cfg.size
                    if hasattr(vectors_cfg, "size")
                    else next(iter(vectors_cfg.values())).size
                )
                if existing_dim != dim:
                    raise RuntimeError(
                        f"[NLVideoSearcher] Qdrant collection '{coll}' has dim={existing_dim} "
                        f"but engine requires dim={dim}. "
                        f"Delete the collection or use a different qdrant_collection name."
                    )
            self._qdrant_client     = client
            self._qdrant_collection = coll
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"[NLVideoSearcher] Qdrant unavailable at {url}: {exc}. "
                "Start Qdrant before using NLVideoSearcher."
            ) from exc

    def _search_qdrant(self, qvec: np.ndarray, top_k: int) -> list:
        """Query Qdrant. Returns [(score, SegmentMeta), ...] sorted by score desc."""
        # encode_text returns (N, D); squeeze to 1-D so Qdrant doesn't treat it
        # as a multi-vector query and raise "Conversion between multi and regular
        # vectors failed".
        if qvec.ndim > 1:
            qvec = qvec[0]
        response = self._qdrant_client.query_points(
            collection_name=self._qdrant_collection,
            query=qvec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
        results = []
        for hit in hits:
            p = hit.payload or {}
            meta = SegmentMeta(
                cam_id=p.get("cam_id",             ""),
                video_path=p.get("video_path",       ""),
                relative_start=p.get("relative_start",  0.0),
                relative_end=p.get("relative_end",    0.0),
                segment_wall_start=p.get("segment_wall_start", 0.0),
                absolute_start=p.get("absolute_start",   0.0),
                absolute_end=p.get("absolute_end",     0.0),
            )
            results.append((float(hit.score), meta))
        return results

    def _store_window(self, emb_row: np.ndarray, meta: SegmentMeta) -> None:
        """Persist one embedding+metadata to Qdrant."""
        import uuid
        from qdrant_client.models import PointStruct
        self._qdrant_client.upsert(
            collection_name=self._qdrant_collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=emb_row[0].tolist(),
                payload={
                    "cam_id":             meta.cam_id,
                    "video_path":         meta.video_path,
                    "relative_start":     meta.relative_start,
                    "relative_end":       meta.relative_end,
                    "segment_wall_start": meta.segment_wall_start,
                    "absolute_start":     meta.absolute_start,
                    "absolute_end":       meta.absolute_end,
                },
            )],
        )

    @classmethod
    def from_config(cls, path: str):
        with open(path) as f: return cls(yaml.safe_load(f))

    @classmethod
    def from_params(cls, use_sliding_window=True, window_sec=10.0,
                    overlap_ratio=0.30, frames_per_window=5, device=None,
                    model_name="EVA02-L-14", pretrained="merged2b_s4b_b131k",
                    embed_dim=768, qdrant_url="http://localhost:6333",
                    qdrant_collection="nlvs_segments"):
        return cls({
            "backend": "pc",
            "engine":  {"type":"pc","model_name":model_name,"pretrained":pretrained,
                        "device":device,"batch_size":16,"frames_per_window":frames_per_window},
            "pipeline":{"video_backend":"opencv","window_sec":window_sec,
                        "overlap_ratio":overlap_ratio if use_sliding_window else 0.0},
            "index":   {"embed_dim":embed_dim,"backend":"qdrant",
                        "qdrant_url":qdrant_url,"qdrant_collection":qdrant_collection},
            "search":  {"top_k":5,"score_threshold":0.10,"nms_iou_threshold":0.30,
                        "adaptive_threshold":True,"translate_vi":True},
        })

    def index_video(self, video_path: str, use_scene_detection: bool = False) -> int:
        print(f"[Searcher] Indexing: {video_path}")
        cam_id = os.path.splitext(os.path.basename(video_path))[0]
        count  = 0

        def _make_meta(t0: float, t1: float) -> SegmentMeta:
            """Build a v2.0 SegmentMeta for a pre-recorded file (wall_start=0)."""
            return SegmentMeta(
                cam_id=cam_id,
                video_path=video_path,
                relative_start=t0,
                relative_end=t1,
                segment_wall_start=0.0,
                absolute_start=t0,
                absolute_end=t1,
            )

        if use_scene_detection:
            from .scene_segmenter import SceneSegmenter
            from .video_processor import VideoProcessor
            segmenter = SceneSegmenter(
                max_scene_sec=self._window_sec * 2,
                fallback_window_sec=self._window_sec,
                fallback_stride_sec=self._window_sec * (1.0 - self._overlap_ratio),
                use_fallback_if_unavailable=True,
            )
            proc = VideoProcessor(
                fps_mode=None,
                window_sec=self._window_sec,
                stride_sec=self._window_sec * (1.0 - self._overlap_ratio),
                frames_per_window=self._frames_per_window,
            )
            for t0, t1 in segmenter.iter_segments(video_path):
                import cv2
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frames = proc._read_n_frames(cap, t0, t1, fps)
                cap.release()
                if not frames:
                    continue
                emb  = self._engine.encode_segment_frames(frames)
                self._store_window(emb.reshape(1, -1), _make_meta(t0, t1))
                count += 1
        else:
            stride = self._window_sec * (1.0 - self._overlap_ratio)
            pipe   = VideoPipeline(video_path=video_path, window_sec=self._window_sec,
                                   stride_sec=stride, frames_per_window=self._frames_per_window,
                                   backend=self._video_backend)
            for t0, t1, frames in pipe.iter_segments():
                if not frames: continue
                emb = self._engine.encode_segment_frames(frames)
                self._store_window(emb.reshape(1, -1), _make_meta(t0, t1))
                count += 1

        try:
            info = self._qdrant_client.get_collection(self._qdrant_collection)
            backend_total = info.points_count
        except Exception:
            backend_total = "?"
        print(f"[Searcher] +{count} segments | qdrant_total={backend_total}")
        return count

    def index_directory(self, video_dir: str, extensions=(".mp4",".avi",".mov",".mkv")) -> int:
        files = [os.path.join(video_dir,f) for f in sorted(os.listdir(video_dir))
                 if f.lower().endswith(extensions)]
        if not files: print(f"[Searcher] No videos in {video_dir}"); return 0
        total = 0
        for p in tqdm(files, desc="Indexing"): total += self.index_video(p)
        return total

    def search(self, query_text: str, top_k=None, score_threshold=None, nms_iou=None,
               use_templates: bool = True, use_reranker: bool = False) -> List[SearchResult]:
        if self._qdrant_client is None:
            raise RuntimeError("Qdrant unavailable. Start Qdrant and call _init_qdrant().")
        k   = top_k   if top_k   is not None else self._default_top_k
        iou = nms_iou if nms_iou is not None else self._nms_iou

        cleaned = _normalize_query(query_text)

        # Phase 1 §I4: VI→EN translation
        if self._translate_vi:
            cleaned = self._translate_if_vietnamese(cleaned)

        if use_templates:
            qvec = self._encode_with_templates(cleaned)
        else:
            qvec = self._engine.encode_text(cleaned)

        raw = self._search_qdrant(qvec, top_k=k * 4)

        # Phase 1 §I2: adaptive threshold
        if score_threshold is not None:
            thresh = score_threshold
        elif self._adaptive_threshold:
            thresh = self._compute_adaptive_threshold(
                [s for s, _ in raw], self._score_threshold
            )
        else:
            thresh = self._score_threshold

        fil   = [(s, m) for s, m in raw if s >= thresh]
        dedup = temporal_nms(fil, iou_threshold=iou, top_k=k)
        results = [SearchResult(score=sc, cam_id=m.cam_id, video_path=m.video_path,
                                start_time=m.relative_start, end_time=m.relative_end,
                                rank=i+1,
                                absolute_start=m.absolute_start,
                                absolute_end=m.absolute_end)
                   for i, (sc, m) in enumerate(dedup)]

        # Phase 3: two-stage retrieval via reranker (BLIP-2 or BLIP-1 ITM)
        if use_reranker and self._reranker is not None:
            reranked = self._reranker.rerank(results, cleaned, top_k=k)
            results = [
                SearchResult(
                    score=r.score,
                    cam_id=r.video_id,          # RerankResult uses video_id
                    video_path=r.video_path,
                    start_time=r.start_time,
                    end_time=r.end_time,
                    rank=r.rank,
                    absolute_start=getattr(r, "absolute_start", 0.0),
                    absolute_end=getattr(r, "absolute_end", 0.0),
                )
                for r in reranked
            ]

        return results

    def set_reranker(self, reranker) -> None:
        """
        Attach a BLIP-2 reranker to this searcher instance.

        Parameters
        ----------
        reranker : BLIP2Reranker | None
            A ``BLIP2Reranker`` instance (from ``src.reranker``), or None to
            disable reranking.  Passing a reranker does **not** load the model
            immediately when ``lazy_load=True`` (default).

        Example
        -------
            from src.reranker import BLIP2Reranker
            searcher.set_reranker(BLIP2Reranker(device="cpu"))
            results = searcher.search("person climbing fence", use_reranker=True)
        """
        self._reranker = reranker

    def _encode_with_templates(self, cleaned_query: str) -> np.ndarray:
        """
        Encode *cleaned_query* through all CLIP prompt templates and return the
        L2-normalised mean of the resulting embeddings.

        Phase 1: expanded to 12 templates (action/event + surveillance-specific)
        covering a wider variety of visual phrasings to improve recall.
        """
        texts = [t.format(cleaned_query) for t in _CLIP_TEMPLATES]
        embs  = self._engine.encode_text(texts)    # (N_templates, D)
        mean  = embs.mean(axis=0)                   # (D,)
        norm  = float(np.linalg.norm(mean))
        return (mean / norm).astype(np.float32) if norm > 1e-8 else mean

    # ------------------------------------------------------------------
    # Phase 1 helpers
    # ------------------------------------------------------------------

    def _compute_adaptive_threshold(self, raw_scores: List[float],
                                    base_threshold: float = 0.15) -> float:
        """
        Adaptive score threshold based on the distribution of raw Faiss scores.

        - When top-1 score is high (> 0.35): tighten threshold to suppress
          near-duplicate low-confidence results (precision mode).
        - When top-1 score is low  (< 0.25): fall back to base_threshold so
          genuinely hard queries still return candidates (recall mode).
        """
        if not raw_scores:
            return base_threshold
        top1 = max(raw_scores)
        if top1 > 0.35:
            mean_s = float(np.mean(raw_scores))
            std_s  = float(np.std(raw_scores))
            return max(base_threshold, mean_s - 0.5 * std_s)
        return base_threshold

    _VI_DIACRITICS = frozenset(
        "àáảãạăắặẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
        "ÀÁẢÃẠĂẮẶẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
    )

    def _translate_if_vietnamese(self, query: str) -> str:
        """
        Detect Vietnamese queries by diacritic ratio and translate VI→EN.

        Uses `deep_translator.GoogleTranslator` when available; falls back to
        the original query on network error or missing dependency.

        Detection heuristic: if ≥ 5 % of characters are Vietnamese diacritics
        the query is considered Vietnamese.
        """
        if not query:
            return query
        vi_ratio = sum(1 for c in query if c in self._VI_DIACRITICS) / len(query)
        if vi_ratio < 0.05:
            return query   # not Vietnamese
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source="vi", target="en").translate(query)
            if translated and translated.strip():
                return translated.strip()
        except Exception:
            pass   # silently fall through
        return query

    def search_debug(self, query_text: str, top_k: int = 10,
                     score_threshold: float = 0.0) -> dict:
        """
        Return a detailed diagnostic dict for a query.

        Useful for understanding why a query succeeds or fails:
        - original vs cleaned query
        - translated query (Phase 1: VI→EN)
        - adaptive threshold value (Phase 1)
        - raw Qdrant scores before NMS and threshold filtering
        - comparison: template-ensemble vs raw query
        """
        if self._qdrant_client is None:
            raise RuntimeError("Qdrant unavailable.")

        cleaned    = _normalize_query(query_text)
        translated = self._translate_if_vietnamese(cleaned) if self._translate_vi else cleaned
        qvec_raw   = self._engine.encode_text(query_text)
        qvec_clean = self._engine.encode_text(cleaned)
        qvec_tmpl  = self._encode_with_templates(translated)

        def _top(vec, k):
            rows = self._search_qdrant(vec, top_k=k)
            return [{"score": float(s), "cam_id": m.cam_id,
                     "start": m.relative_start, "end": m.relative_end,
                     "absolute_start": m.absolute_start} for s, m in rows]

        _all_k = 200
        raw_scores = [s for s, _ in self._search_qdrant(qvec_tmpl, top_k=_all_k)]
        adaptive_thresh = (
            self._compute_adaptive_threshold(raw_scores, self._score_threshold)
            if self._adaptive_threshold else self._score_threshold
        )

        return {
            "query_original":   query_text,
            "query_cleaned":    cleaned,
            "query_translated": translated,
            "templates_used":   [t.format(translated) for t in _CLIP_TEMPLATES],
            "top_raw_query":    _top(qvec_raw,   top_k),
            "top_clean_query":  _top(qvec_clean, top_k),
            "top_templates":    _top(qvec_tmpl,  top_k),
            "score_stats": {
                "max":              float(max(raw_scores)) if raw_scores else 0,
                "min":              float(min(raw_scores)) if raw_scores else 0,
                "mean":             float(np.mean(raw_scores)) if raw_scores else 0,
                "median":           float(np.median(raw_scores)) if raw_scores else 0,
                "p75":              float(np.percentile(raw_scores, 75)) if raw_scores else 0,
                "adaptive_thresh":  adaptive_thresh,
            },
        }


