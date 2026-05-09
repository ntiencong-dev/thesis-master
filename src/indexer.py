"""
indexer.py
----------
Builds and persists a Faiss index over CLIP video-segment embeddings.

Index type
----------
``IndexFlatIP`` operates on L2-normalised vectors, so the inner-product
score equals the cosine similarity — exactly what CLIP embeddings need.

Persistence
-----------
The Faiss index is saved to ``<index_dir>/faiss.index`` and the
corresponding metadata list is pickled to ``<index_dir>/metadata.pkl``.
Both files must exist together; loading one without the other raises an
error.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Per-segment metadata stored alongside the Faiss index
# ---------------------------------------------------------------------------

@dataclass
class SegmentMeta:
    video_id:   str
    video_path: str
    start_time: float
    end_time:   float


# ---------------------------------------------------------------------------
# Index manager
# ---------------------------------------------------------------------------

class VideoIndex:
    """
    Parameters
    ----------
    embed_dim : int
        Dimensionality of CLIP embeddings (512 for ViT-B/16).
    use_gpu : bool
        Move Faiss index to GPU if CUDA is available.  Falls back to CPU
        silently when no GPU resource is found.
    """

    INDEX_FILE    = "faiss.index"
    METADATA_FILE = "metadata.pkl"

    def __init__(self, embed_dim: int = 512, use_gpu: bool = True) -> None:
        self.embed_dim = embed_dim
        self._meta: List[SegmentMeta] = []
        self._on_gpu: bool = False

        # Build CPU index first, optionally move to GPU later
        self._index: faiss.Index = faiss.IndexFlatIP(embed_dim)
        if use_gpu:
            self._move_to_gpu()

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, metadata: List[SegmentMeta]) -> None:
        """
        Add *embeddings* (float32, shape N×D) together with their
        *metadata* entries to the index.
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"embeddings count ({embeddings.shape[0]}) ≠ "
                f"metadata count ({len(metadata)})"
            )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._meta.extend(metadata)

    def total_vectors(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_dir: str) -> None:
        """Serialise index + metadata to *index_dir*."""
        os.makedirs(index_dir, exist_ok=True)

        # When the index is on GPU, convert back to CPU before serialising.
        # Use the public helper when available; fall back to writing as-is
        # (which works for CPU-only faiss builds).
        if self._on_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            cpu_index = faiss.index_gpu_to_cpu(self._index)
        else:
            cpu_index = self._index
        faiss.write_index(cpu_index, os.path.join(index_dir, self.INDEX_FILE))

        with open(os.path.join(index_dir, self.METADATA_FILE), "wb") as fh:
            pickle.dump(self._meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[VideoIndex] Saved {self._index.ntotal} vectors → {index_dir}")

    @classmethod
    def load(cls, index_dir: str, use_gpu: bool = True) -> "VideoIndex":
        """Load a previously saved index from *index_dir*."""
        idx_path  = os.path.join(index_dir, cls.INDEX_FILE)
        meta_path = os.path.join(index_dir, cls.METADATA_FILE)

        for p in (idx_path, meta_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Index file missing: {p}")

        cpu_index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as fh:
            metadata: List[SegmentMeta] = pickle.load(fh)

        instance = cls.__new__(cls)
        instance.embed_dim = cpu_index.d
        instance._meta     = metadata
        instance._index    = cpu_index
        instance._on_gpu   = False

        if use_gpu:
            instance._move_to_gpu()

        print(f"[VideoIndex] Loaded {cpu_index.ntotal} vectors ← {index_dir}")
        return instance

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[tuple[float, SegmentMeta]]:
        """
        Parameters
        ----------
        query_vector : numpy.ndarray  shape (1, D) or (D,), float32.
        top_k : int

        Returns
        -------
        List of (score, SegmentMeta) sorted by descending cosine similarity.
        """
        q = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:      # Faiss returns -1 for empty slots
                continue
            results.append((float(score), self._meta[idx]))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_to_gpu(self) -> None:
        try:
            if not hasattr(faiss, "StandardGpuResources") or \
               not hasattr(faiss, "index_cpu_to_gpu"):
                return   # faiss CPU-only build
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            self._on_gpu = True
        except Exception:
            pass   # silently keep CPU index


# ---------------------------------------------------------------------------
# Temporal Non-Maximum Suppression (module-level utilities)
# ---------------------------------------------------------------------------

def calculate_iou(seg1: SegmentMeta, seg2: SegmentMeta) -> float:
    """
    Compute the Temporal Intersection over Union (T-IoU) between two segments.

    Parameters
    ----------
    seg1, seg2 : SegmentMeta
        Segment objects with ``start_time`` and ``end_time`` in seconds.

    Returns
    -------
    float
        IoU ∈ [0, 1].  Returns 0.0 for non-overlapping or adjacent segments.

    Formula
    -------
    Let s = start, e = end:

        intersection = max(0, min(e1,e2) - max(s1,s2))
        union        = max(e1,e2) - min(s1,s2)
        IoU          = intersection / union
    """
    inter = max(0.0, min(seg1.end_time, seg2.end_time)
                     - max(seg1.start_time, seg2.start_time))
    if inter == 0.0:
        return 0.0
    union = (max(seg1.end_time, seg2.end_time)
             - min(seg1.start_time, seg2.start_time))
    return inter / union if union > 1e-9 else 0.0


def temporal_nms(
    results: "List[tuple[float, SegmentMeta]]",
    iou_threshold: float = 0.30,
    top_k: "Optional[int]" = None,
) -> "List[tuple[float, SegmentMeta]]":
    """
    Greedy Temporal Non-Maximum Suppression.

    Removes temporally-overlapping segments for the same video, keeping only
    the highest-scoring representative in each overlapping group.

    Algorithm
    ---------
    1. Sort all candidates by cosine score (descending).
    2. Accept the top-scoring candidate into the *kept* list.
    3. For each remaining candidate, compute T-IoU (via ``calculate_iou``)
       against every already-kept segment **of the same video**.
    4. If IoU > ``iou_threshold`` → suppress (discard). Otherwise → keep.
    5. Stop early once ``top_k`` segments are in the kept list.

    Parameters
    ----------
    results       : list of (score, SegmentMeta) — order does not matter on
                    input; output is sorted by score descending.
    iou_threshold : float, default 0.30
        Suppress a candidate when its T-IoU with any kept segment exceeds
        this value.  Use strict ``>`` (equal to threshold is NOT suppressed).
        Typical values: 0.25 (aggressive) – 0.50 (permissive).
    top_k         : int or None
        Stop as soon as this many segments have been accepted.  ``None``
        means no limit (return all non-suppressed candidates).

    Returns
    -------
    List of (score, SegmentMeta) sorted by score descending.
    """
    kept: List[tuple] = []

    for score, meta in sorted(results, key=lambda x: -x[0]):
        if top_k is not None and len(kept) >= top_k:
            break                                  # early exit

        duplicate = False
        for _, kept_meta in kept:
            if kept_meta.video_id != meta.video_id:
                continue
            if calculate_iou(kept_meta, meta) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append((score, meta))

    return kept
