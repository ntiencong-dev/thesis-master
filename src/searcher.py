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
from .indexer import SegmentMeta, VideoIndex, temporal_nms, calculate_iou
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
# action and event queries (mirrors the ensemble strategy from the CLIP paper)
_CLIP_TEMPLATES: List[str] = [
    "{}",
    "a photo of {}",
    "a video frame of {}",
    "a scene with {}",
    "an image of {}",
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
    score:      float
    video_id:   str
    video_path: str
    start_time: float
    end_time:   float
    rank:       int

class NLVideoSearcher:
    _DEFAULT: dict = {
        "backend": "pc",
        "engine":   {"type":"pc","model_name":"ViT-B-16","pretrained":"openai","device":None,"batch_size":32,"frames_per_window":5},
        "pipeline": {"video_backend":"opencv","window_sec":5.0,"overlap_ratio":0.50},
        "index":    {"embed_dim":512,"index_dir":None},
        "search":   {"top_k":5,"score_threshold":0.20,"nms_iou_threshold":0.30},
    }
    def __init__(self, config=None):
        self._config = {**self._DEFAULT, **(config or {})}
        self._engine = create_engine(self._config)
        self._index  = VideoIndex(embed_dim=self._engine.embed_dim)
        p = self._config.get("pipeline", {}); e = self._config.get("engine", {})
        self._window_sec        = p.get("window_sec",       5.0)
        self._overlap_ratio     = p.get("overlap_ratio",    0.5)
        self._frames_per_window = e.get("frames_per_window", 5)
        self._video_backend     = p.get("video_backend",    "opencv")
        s = self._config.get("search", {})
        self._default_top_k   = s.get("top_k",             5)
        self._score_threshold = s.get("score_threshold",   0.20)
        self._nms_iou         = s.get("nms_iou_threshold", 0.50)
        self.index_dir        = self._config.get("index", {}).get("index_dir", None)

    @classmethod
    def from_config(cls, path: str):
        with open(path) as f: return cls(yaml.safe_load(f))

    @classmethod
    def from_params(cls, index_dir=None, use_sliding_window=True, window_sec=5.0,
                    overlap_ratio=0.5, frames_per_window=5, device=None):
        return cls({
            "backend": "pc",
            "engine":  {"type":"pc","model_name":"ViT-B-16","pretrained":"openai",
                        "device":device,"batch_size":32,"frames_per_window":frames_per_window},
            "pipeline":{"video_backend":"opencv","window_sec":window_sec,
                        "overlap_ratio":overlap_ratio if use_sliding_window else 0.0},
            "index":   {"embed_dim":512,"index_dir":index_dir},
            "search":  {"top_k":5,"score_threshold":0.20,"nms_iou_threshold":0.30},
        })

    def index_video(self, video_path: str) -> int:
        print(f"[Searcher] Indexing: {video_path}")
        stride = self._window_sec * (1.0 - self._overlap_ratio)
        pipe   = VideoPipeline(video_path=video_path, window_sec=self._window_sec,
                               stride_sec=stride, frames_per_window=self._frames_per_window,
                               backend=self._video_backend)
        vid_id = os.path.splitext(os.path.basename(video_path))[0]
        count  = 0
        for t0, t1, frames in pipe.iter_segments():
            if not frames: continue
            emb  = self._engine.encode_segment_frames(frames)
            meta = SegmentMeta(video_id=vid_id, video_path=video_path, start_time=t0, end_time=t1)
            self._index.add(emb.reshape(1, -1), [meta])
            count += 1
        print(f"[Searcher] +{count} segments | total={self._index.total_vectors()}")
        if self.index_dir: self._index.save(self.index_dir)
        return count

    def index_directory(self, video_dir: str, extensions=(".mp4",".avi",".mov",".mkv")) -> int:
        files = [os.path.join(video_dir,f) for f in sorted(os.listdir(video_dir))
                 if f.lower().endswith(extensions)]
        if not files: print(f"[Searcher] No videos in {video_dir}"); return 0
        total = 0
        for p in tqdm(files, desc="Indexing"): total += self.index_video(p)
        return total

    def search(self, query_text: str, top_k=None, score_threshold=None, nms_iou=None,
               use_templates: bool = True) -> List[SearchResult]:
        if self._index.total_vectors() == 0:
            raise RuntimeError("Index empty. Run index_video() first.")
        k      = top_k           if top_k           is not None else self._default_top_k
        thresh = score_threshold if score_threshold is not None else self._score_threshold
        iou    = nms_iou         if nms_iou         is not None else self._nms_iou

        cleaned = _normalize_query(query_text)
        if use_templates:
            qvec = self._encode_with_templates(cleaned)
        else:
            qvec = self._engine.encode_text(cleaned)

        raw    = self._index.search(qvec, top_k=k * 4)
        fil    = [(s, m) for s, m in raw if s >= thresh]
        dedup  = temporal_nms(fil, iou_threshold=iou, top_k=k)
        return [SearchResult(score=sc, video_id=m.video_id, video_path=m.video_path,
                             start_time=m.start_time, end_time=m.end_time, rank=i+1)
                for i, (sc, m) in enumerate(dedup)]

    def _encode_with_templates(self, cleaned_query: str) -> np.ndarray:
        """
        Encode *cleaned_query* through all CLIP prompt templates and return the
        L2-normalised mean of the resulting embeddings.

        Averaging across multiple phrasings of the same query reduces sensitivity
        to exact wording and improves recall for action/event queries.
        """
        texts = [t.format(cleaned_query) for t in _CLIP_TEMPLATES]
        embs  = self._engine.encode_text(texts)    # (N_templates, D)
        mean  = embs.mean(axis=0)                   # (D,)
        norm  = float(np.linalg.norm(mean))
        return (mean / norm).astype(np.float32) if norm > 1e-8 else mean

    def search_debug(self, query_text: str, top_k: int = 10,
                     score_threshold: float = 0.0) -> dict:
        """
        Return a detailed diagnostic dict for a query.

        Useful for understanding why a query succeeds or fails:
        - original vs cleaned query
        - raw Faiss scores before NMS and threshold filtering
        - comparison: template-ensemble vs raw query
        """
        if self._index.total_vectors() == 0:
            raise RuntimeError("Index empty.")

        cleaned = _normalize_query(query_text)
        qvec_raw  = self._engine.encode_text(query_text)   # (1, D) → flatten
        qvec_clean = self._engine.encode_text(cleaned)
        qvec_tmpl  = self._encode_with_templates(cleaned)

        def _top(vec, k):
            rows = self._index.search(vec, top_k=k)
            return [{"score": float(s), "video_id": m.video_id,
                     "start": m.start_time, "end": m.end_time} for s, m in rows]

        raw_scores = [s for s, _ in self._index.search(qvec_tmpl, top_k=self._index.total_vectors())]

        return {
            "query_original":  query_text,
            "query_cleaned":   cleaned,
            "templates_used":  [t.format(cleaned) for t in _CLIP_TEMPLATES],
            "top_raw_query":   _top(qvec_raw,   top_k),
            "top_clean_query": _top(qvec_clean, top_k),
            "top_templates":   _top(qvec_tmpl,  top_k),
            "score_stats": {
                "max":    float(max(raw_scores)) if raw_scores else 0,
                "min":    float(min(raw_scores)) if raw_scores else 0,
                "mean":   float(np.mean(raw_scores)) if raw_scores else 0,
                "median": float(np.median(raw_scores)) if raw_scores else 0,
                "p75":    float(np.percentile(raw_scores, 75)) if raw_scores else 0,
            },
        }

    def save_index(self, index_dir=None):
        d = index_dir or self.index_dir
        if not d: raise ValueError("No index_dir.")
        self._index.save(d)

    def load_index(self, index_dir=None):
        s = index_dir or self.index_dir
        if not s: raise ValueError("No index_dir.")
        self._index = VideoIndex.load(s)
