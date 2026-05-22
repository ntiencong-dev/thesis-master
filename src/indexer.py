"""
indexer.py
----------
Metadata types and temporal NMS utilities for NLVS segment search.

This module provides:
  - SegmentMeta dataclass — per-segment metadata stored in Qdrant
  - temporal_nms() — greedy temporal Non-Maximum Suppression
  - calculate_iou() — T-IoU between two SegmentMeta instances

Note: VideoIndex (Faiss storage) was removed in v3.0. Qdrant is the sole
vector store. See src/searcher.py and src/continuous_indexer.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-segment metadata stored alongside the Faiss index
# ---------------------------------------------------------------------------

@dataclass
class SegmentMeta:
    # Identification
    cam_id:             str      # camera/source ID; basename (no ext) for pre-recorded files
    video_path:         str      # absolute path to segment file

    # Position within the file (use for seeking / clip extraction)
    relative_start:     float    # seconds from file start
    relative_end:       float    # seconds from file start

    # Wall-clock timestamps — 0.0 for pre-recorded files with no capture time
    segment_wall_start: float = 0.0
    absolute_start:     float = 0.0   # = segment_wall_start + relative_start
    absolute_end:       float = 0.0   # = segment_wall_start + relative_end

    def __post_init__(self) -> None:
        # Auto-compute absolute timestamps when both are zero and wall_start is set
        if self.absolute_start == 0.0 and self.absolute_end == 0.0:
            self.absolute_start = self.segment_wall_start + self.relative_start
            self.absolute_end   = self.segment_wall_start + self.relative_end


def _migrate_segment_meta(meta: "SegmentMeta") -> "SegmentMeta":
    """Upgrade a pre-v2.0 SegmentMeta (video_id/start_time/end_time) to v2.0 format."""
    d = meta.__dict__
    if "cam_id" in d:
        return meta   # already v2.0 format
    return SegmentMeta(
        cam_id=d.get("video_id", ""),
        video_path=d.get("video_path", ""),
        relative_start=d.get("start_time", 0.0),
        relative_end=d.get("end_time", 0.0),
        segment_wall_start=0.0,
        absolute_start=d.get("start_time", 0.0),
        absolute_end=d.get("end_time", 0.0),
    )




# ---------------------------------------------------------------------------
# Temporal Non-Maximum Suppression (module-level utilities)
# ---------------------------------------------------------------------------

def calculate_iou(seg1: SegmentMeta, seg2: SegmentMeta) -> float:
    """
    Compute the Temporal Intersection over Union (T-IoU) between two segments.

    Uses ``absolute_start`` / ``absolute_end`` so that cross-segment
    deduplication works correctly for the continuous-streaming case
    (where two overlapping files from the same camera contain the same event).
    For pre-recorded files ``absolute_start == relative_start`` (wall_start=0),
    so the behaviour is identical to the original implementation.

    Formula
    -------
        intersection = max(0, min(e1,e2) - max(s1,s2))
        union        = max(e1,e2) - min(s1,s2)
        IoU          = intersection / union
    """
    inter = max(0.0, min(seg1.absolute_end, seg2.absolute_end)
                     - max(seg1.absolute_start, seg2.absolute_start))
    if inter == 0.0:
        return 0.0
    union = (max(seg1.absolute_end, seg2.absolute_end)
             - min(seg1.absolute_start, seg2.absolute_start))
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
            # Only suppress across segments of the same camera/source.
            # Different cameras may legitimately capture the same event.
            if kept_meta.cam_id != meta.cam_id:
                continue
            if calculate_iou(kept_meta, meta) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append((score, meta))

    return kept
