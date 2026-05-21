"""
tests/unit/test_indexer.py
---------------------------
Unit tests for VideoIndex, calculate_iou, and temporal_nms.

Scenarios covered
-----------------
IDX-01  add() increments ntotal correctly
IDX-02  add() raises ValueError when embeddings/metadata count mismatch
IDX-03  total_vectors() returns the count after add
IDX-04  search() on self returns score ≈ 1.0
IDX-05  search() top_k is respected even when index has fewer vectors
IDX-06  search() returns items sorted by score descending
IDX-07  search() filters out index=-1 sentinel rows
IDX-08  search() returns correct SegmentMeta for each hit
IDX-09  save() + load() round-trip: same ntotal, same metadata, same search result
IDX-10  load() raises FileNotFoundError when index files missing
IDX-11  _move_to_gpu: silently skips when faiss has no GPU symbols
IDX-12  multiple add() calls accumulate correctly

IOU-01  Non-overlapping segments → IoU = 0.0
IOU-02  Identical segments → IoU = 1.0
IOU-03  Partial overlap → IoU = intersection / union
IOU-04  One segment contained inside the other
IOU-05  Adjacent (touching) segments → IoU = 0.0
IOU-06  Symmetry: IoU(a, b) == IoU(b, a)

NMS-01  All non-overlapping segments → all kept
NMS-02  Identical segment pair → only highest score kept
NMS-03  100 % IoU → suppressed
NMS-04  IoU below threshold → both kept
NMS-05  Different video_id with same timestamps → both kept
NMS-06  Empty input → empty output
NMS-07  Single-element input → returned as-is
NMS-08  Output sorted by score descending
NMS-09  Correct IoU formula: intersection / union
NMS-10  Partial overlap below threshold → both kept
NMS-11  top_k limits the number of returned segments
NMS-12  IoU exactly == threshold is NOT suppressed (strict '>') 
"""

from __future__ import annotations

import os
import pickle
import shutil
import tempfile

import numpy as np
import pytest

from tests.conftest import EMBED_DIM, make_random_unit_vectors
from src.indexer import VideoIndex, ScalableVideoIndex, SegmentMeta, temporal_nms, calculate_iou

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_meta(n: int, vid: str = "v0", offset: float = 0.0) -> list[SegmentMeta]:
    return [
        SegmentMeta(
            cam_id=vid,
            video_path=f"/fake/{vid}.mp4",
            relative_start=offset + i * 5.0,
            relative_end=offset + i * 5.0 + 5.0,
        )
        for i in range(n)
    ]


def _make_index(n: int = 10) -> tuple[VideoIndex, np.ndarray, list[SegmentMeta]]:
    """Return (index, vectors, metadata) with n random unit vectors."""
    idx   = VideoIndex(embed_dim=EMBED_DIM, use_gpu=False)
    vecs  = make_random_unit_vectors(n)
    metas = _make_meta(n)
    idx.add(vecs, metas)
    return idx, vecs, metas


# ── IDX-01..03  add + total_vectors ──────────────────────────────────────

def test_IDX01_add_increments_ntotal():
    idx = VideoIndex(embed_dim=EMBED_DIM, use_gpu=False)
    assert idx.total_vectors() == 0
    vecs  = make_random_unit_vectors(5)
    metas = _make_meta(5)
    idx.add(vecs, metas)
    assert idx.total_vectors() == 5


def test_IDX02_add_raises_on_count_mismatch():
    idx   = VideoIndex(embed_dim=EMBED_DIM, use_gpu=False)
    vecs  = make_random_unit_vectors(3)
    metas = _make_meta(5)   # 5 ≠ 3
    with pytest.raises(ValueError):
        idx.add(vecs, metas)


def test_IDX03_total_vectors_after_add():
    idx, _, _ = _make_index(7)
    assert idx.total_vectors() == 7


# ── IDX-04..08  search ────────────────────────────────────────────────────

def test_IDX04_search_self_score_near_one():
    idx, vecs, _ = _make_index(5)
    for v in vecs:
        results = idx.search(v, top_k=1)
        assert results, "Expected at least 1 result"
        score, _ = results[0]
        assert score > 0.999, f"Self-similarity should be ≈ 1.0, got {score}"


def test_IDX05_search_top_k_clipped():
    idx, vecs, _ = _make_index(4)
    results = idx.search(vecs[0], top_k=100)
    assert len(results) <= 4, "Should not return more results than index size"


def test_IDX06_search_results_sorted_desc():
    idx, vecs, _ = _make_index(10)
    results = idx.search(vecs[0], top_k=5)
    scores = [r[0] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by score desc"


def test_IDX07_search_filters_minus1_indices():
    """
    Faiss can return -1 as index for unfilled slots.
    VideoIndex must silently skip those.
    """
    idx = VideoIndex(embed_dim=EMBED_DIM, use_gpu=False)
    vecs  = make_random_unit_vectors(2)
    metas = _make_meta(2)
    idx.add(vecs, metas)
    # Requesting top_k=10 with only 2 vectors — no crash expected
    results = idx.search(vecs[0], top_k=10)
    # Should not crash and should return valid SegmentMeta objects only
    for score, meta in results:
        assert isinstance(meta, SegmentMeta), f"Expected SegmentMeta, got {type(meta)}"
        assert isinstance(score, float), f"Expected float score, got {type(score)}"


def test_IDX08_search_returns_correct_metadata():
    idx, vecs, metas = _make_index(5)
    results = idx.search(vecs[2], top_k=1)
    assert results
    _, returned_meta = results[0]
    # The exact match should be for the same relative_start as meta[2]
    assert returned_meta.relative_start == metas[2].relative_start


# ── IDX-09..10  persistence ────────────────────────────────────────────────

def test_IDX09_save_load_round_trip(tmp_index_dir):
    idx, vecs, metas = _make_index(8)
    idx.save(tmp_index_dir)

    loaded = VideoIndex.load(tmp_index_dir, use_gpu=False)
    assert loaded.total_vectors() == 8

    # Metadata preserved
    loaded_metas = loaded._meta
    for orig, reloaded in zip(metas, loaded_metas):
        assert orig.cam_id         == reloaded.cam_id
        assert orig.relative_start == reloaded.relative_start
        assert orig.relative_end   == reloaded.relative_end

    # Search results preserved
    results = loaded.search(vecs[0], top_k=1)
    assert results
    score, _ = results[0]
    assert score > 0.999


def test_IDX10_load_missing_files_raises(tmp_index_dir):
    with pytest.raises(FileNotFoundError):
        VideoIndex.load(tmp_index_dir, use_gpu=False)


# ── IDX-11  GPU move silently skips ──────────────────────────────────────

def test_IDX11_move_to_gpu_skips_gracefully():
    """
    On CPU-only faiss builds there are no GPU symbols.
    _move_to_gpu must not raise; _on_gpu must remain False.
    """
    idx = VideoIndex(embed_dim=EMBED_DIM, use_gpu=True)  # triggers _move_to_gpu()
    import faiss
    if not hasattr(faiss, "StandardGpuResources"):
        assert idx._on_gpu is False, "_on_gpu must stay False on CPU-only faiss"


# ── IDX-12  Multiple add() calls ─────────────────────────────────────────

def test_IDX12_multiple_adds():
    idx = VideoIndex(embed_dim=EMBED_DIM, use_gpu=False)
    for batch in range(4):
        vecs  = make_random_unit_vectors(3)
        metas = _make_meta(3, vid=f"v{batch}", offset=batch * 20.0)
        idx.add(vecs, metas)
    assert idx.total_vectors() == 12
    assert len(idx._meta) == 12


# ==========================================================================
#  temporal_nms tests
# ==========================================================================

def _seg(vid: str, start: float, end: float, score: float) -> tuple:
    m = SegmentMeta(
        cam_id=vid,
        video_path=f"/fake/{vid}.mp4",
        relative_start=start,
        relative_end=end,
    )
    return (score, m)


# ── NMS-01  Non-overlapping → all kept ────────────────────────────────────

def test_NMS01_non_overlapping_all_kept():
    candidates = [
        _seg("v0",  0.0,  5.0, 0.9),
        _seg("v0", 10.0, 15.0, 0.8),
        _seg("v0", 20.0, 25.0, 0.7),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 3


# ── NMS-02  Identical pair → highest score kept ───────────────────────────

def test_NMS02_identical_pair_highest_kept():
    candidates = [
        _seg("v0", 0.0, 5.0, 0.7),
        _seg("v0", 0.0, 5.0, 0.9),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(0.9)


# ── NMS-03  100 % IoU → only top kept ────────────────────────────────────

def test_NMS03_full_overlap_suppressed():
    # [0,10] vs [2,8] → IoU = 6/10 = 0.6 > 0.5 → suppressed
    candidates = [
        _seg("v0",  0.0, 10.0, 0.95),
        _seg("v0",  2.0,  8.0, 0.80),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(0.95)


# ── NMS-04  IoU > threshold → suppressed ─────────────────────────────────

def test_NMS04_iou_above_threshold_suppressed():
    # [0,5] vs [2,7] → intersection=[2,5]=3, union=[0,7]=7, IoU=3/7≈0.43 < 0.5 → both kept
    candidates = [
        _seg("v0", 0.0, 5.0, 0.9),
        _seg("v0", 2.0, 7.0, 0.8),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2, f"IoU=3/7≈0.43 < 0.5, both should be kept, got {len(kept)}"


# ── NMS-05  Different video_id → both kept ───────────────────────────────

def test_NMS05_different_video_id_both_kept():
    candidates = [
        _seg("vid_A", 0.0, 10.0, 0.9),
        _seg("vid_B", 0.0, 10.0, 0.8),   # same timestamps, different video
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2


# ── NMS-06  Empty input ───────────────────────────────────────────────────

def test_NMS06_empty_input():
    assert temporal_nms([], iou_threshold=0.5) == []


# ── NMS-07  Single element ────────────────────────────────────────────────

def test_NMS07_single_element_returned():
    candidates = [_seg("v0", 0.0, 5.0, 0.85)]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(0.85)


# ── NMS-08  Output sorted by score desc ──────────────────────────────────

def test_NMS08_output_sorted_desc():
    candidates = [
        _seg("v0",  0.0,  5.0, 0.5),
        _seg("v0", 10.0, 15.0, 0.9),
        _seg("v0", 20.0, 25.0, 0.7),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    scores = [r[0] for r in kept]
    assert scores == sorted(scores, reverse=True)


# ── NMS-09  Correct IoU formula ───────────────────────────────────────────

def test_NMS09_correct_iou_formula():
    # [0,10] and [6,16]:
    # intersection = [6,10] = 4
    # union        = [0,16] = 16
    # IoU = 4/16 = 0.25  → < 0.5 → both kept
    candidates = [
        _seg("v0",  0.0, 10.0, 0.9),
        _seg("v0",  6.0, 16.0, 0.8),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2, "IoU=0.25 < 0.5, both should be kept"


# ── NMS-10  Partial overlap well below threshold ─────────────────────────

def test_NMS10_partial_overlap_below_threshold_both_kept():
    # [0,5] and [4,9]:
    # intersection = [4,5] = 1, union = [0,9] = 9, IoU ≈ 0.11
    candidates = [
        _seg("v0", 0.0, 5.0, 0.88),
        _seg("v0", 4.0, 9.0, 0.77),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2


# ==========================================================================
#  calculate_iou tests
# ==========================================================================

def _m(start: float, end: float, vid: str = "v") -> SegmentMeta:
    """Shorthand helper for IoU tests."""
    return SegmentMeta(cam_id=vid, video_path=f"/f/{vid}.mp4",
                       relative_start=start, relative_end=end)


# ── IOU-01  Non-overlapping → 0.0 ────────────────────────────────────────

def test_IOU01_non_overlapping():
    # [0,5] and [6,10]: no intersection → IoU = 0.0
    assert calculate_iou(_m(0, 5), _m(6, 10)) == pytest.approx(0.0)


# ── IOU-02  Identical → 1.0 ───────────────────────────────────────────────

def test_IOU02_identical_segments():
    # [2,8] and [2,8]: intersection = union = 6 → IoU = 1.0
    assert calculate_iou(_m(2, 8), _m(2, 8)) == pytest.approx(1.0)


# ── IOU-03  Partial overlap ────────────────────────────────────────────────

def test_IOU03_partial_overlap():
    # [0,6] and [4,10]: inter=[4,6]=2, union=[0,10]=10 → IoU = 0.2
    assert calculate_iou(_m(0, 6), _m(4, 10)) == pytest.approx(0.2)


# ── IOU-04  Contained segment ─────────────────────────────────────────────

def test_IOU04_contained():
    # [0,10] and [3,7]: inter=[3,7]=4, union=[0,10]=10 → IoU = 0.4
    assert calculate_iou(_m(0, 10), _m(3, 7)) == pytest.approx(0.4)


# ── IOU-05  Adjacent / touching → 0.0 ────────────────────────────────────

def test_IOU05_adjacent_touching():
    # [0,5] and [5,10]: intersection = 0 → IoU = 0.0
    assert calculate_iou(_m(0, 5), _m(5, 10)) == pytest.approx(0.0)


# ── IOU-06  Symmetry ─────────────────────────────────────────────────────

def test_IOU06_symmetry():
    a, b = _m(0, 8), _m(5, 15)
    assert calculate_iou(a, b) == pytest.approx(calculate_iou(b, a))


# ==========================================================================
#  Additional NMS tests
# ==========================================================================

# ── NMS-11  top_k limits output ───────────────────────────────────────────

def test_NMS11_top_k_limits_output():
    # 5 non-overlapping segments; top_k=3 → only top-3 by score returned
    candidates = [
        _seg("v0", i * 10.0, i * 10.0 + 5.0, round(0.90 - i * 0.05, 2))
        for i in range(5)
    ]
    kept = temporal_nms(candidates, iou_threshold=0.30, top_k=3)
    assert len(kept) == 3
    assert kept[0][0] == pytest.approx(0.90)
    assert kept[1][0] == pytest.approx(0.85)
    assert kept[2][0] == pytest.approx(0.80)


# ── NMS-12  IoU == threshold is NOT suppressed (strict '>') ──────────────

def test_NMS12_iou_equal_threshold_not_suppressed():
    # [0,3] and [1,4]: inter=[1,3]=2, union=[0,4]=4, IoU = 0.5 exactly
    # Condition: IoU > threshold  (strict) → equal means NOT suppressed
    candidates = [
        _seg("v0", 0.0, 3.0, 0.9),
        _seg("v0", 1.0, 4.0, 0.8),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2, "IoU == threshold should NOT suppress (condition is strict >)"


# ==========================================================================
#  Phase 3: ScalableVideoIndex (IVFFlat) tests
# ==========================================================================

@pytest.mark.unit
def test_SIDX01_scalable_index_importable():
    """ScalableVideoIndex must be importable from src.indexer."""
    from src.indexer import ScalableVideoIndex
    assert hasattr(ScalableVideoIndex, "MIN_TRAIN_RATIO")


@pytest.mark.unit
def test_SIDX02_flat_mode_identical_to_video_index():
    """ScalableVideoIndex(index_type='flat') must behave like VideoIndex."""
    from src.indexer import ScalableVideoIndex
    idx  = ScalableVideoIndex(embed_dim=EMBED_DIM, index_type="flat", use_gpu=False)
    vecs = make_random_unit_vectors(5)
    meta = _make_meta(5)
    idx.add(vecs, meta)
    assert idx.total_vectors() == 5


@pytest.mark.unit
def test_SIDX03_flat_mode_search_returns_correct_hits():
    """Flat ScalableVideoIndex.search() must return correct matches."""
    from src.indexer import ScalableVideoIndex
    idx  = ScalableVideoIndex(embed_dim=EMBED_DIM, index_type="flat", use_gpu=False)
    vecs = make_random_unit_vectors(8)
    meta = _make_meta(8)
    idx.add(vecs, meta)
    results = idx.search(vecs[3], top_k=1)
    assert results
    score, _ = results[0]
    assert score > 0.999


@pytest.mark.unit
def test_SIDX04_ivf_index_fallback_when_too_small():
    """
    When N < nlist * 39, IVFFlat training must fall back to IndexFlatIP
    and still accept add() without raising.
    """
    from src.indexer import ScalableVideoIndex
    nlist = 128
    n     = 10   # far fewer than 128 * 39 = 4992
    idx   = ScalableVideoIndex(
        embed_dim=EMBED_DIM, index_type="ivf", nlist=nlist, use_gpu=False
    )
    vecs = make_random_unit_vectors(n)
    meta = _make_meta(n)
    # add() triggers auto-train → should detect undersized collection → fallback
    idx.add(vecs, meta)
    assert idx.total_vectors() == n, "Fallback flat index must still accept vectors"
    assert idx._index_type == "flat", "After fallback, _index_type must be 'flat'"


@pytest.mark.unit
def test_SIDX05_train_flat_is_noop():
    """Calling train() on flat ScalableVideoIndex sets _is_trained=True."""
    from src.indexer import ScalableVideoIndex
    idx = ScalableVideoIndex(embed_dim=EMBED_DIM, index_type="flat", use_gpu=False)
    vecs = make_random_unit_vectors(20)
    idx.train(vecs)
    assert idx._is_trained is True


@pytest.mark.unit
def test_SIDX06_build_and_train_convenience():
    """build_and_train() must train + add in one shot (flat fallback)."""
    from src.indexer import ScalableVideoIndex
    idx  = ScalableVideoIndex(embed_dim=EMBED_DIM, index_type="ivf", nlist=64, use_gpu=False)
    vecs = make_random_unit_vectors(15)   # <  64 * 39 → fallback
    meta = _make_meta(15)
    idx.build_and_train(vecs, meta)
    assert idx.total_vectors() == 15


@pytest.mark.unit
def test_SIDX07_unknown_index_type_raises():
    """ScalableVideoIndex with unknown index_type must raise ValueError."""
    from src.indexer import ScalableVideoIndex
    idx = ScalableVideoIndex.__new__(ScalableVideoIndex)
    idx.embed_dim   = EMBED_DIM
    idx._index_type = "hnsw"   # unsupported
    with pytest.raises(ValueError, match="Unknown index_type"):
        idx._build_index()


@pytest.mark.unit
def test_SIDX08_flat_save_load_round_trip(tmp_index_dir):
    """ScalableVideoIndex (flat) must survive save → load round-trip."""
    from src.indexer import ScalableVideoIndex
    idx  = ScalableVideoIndex(embed_dim=EMBED_DIM, index_type="flat", use_gpu=False)
    vecs = make_random_unit_vectors(6)
    meta = _make_meta(6)
    idx.add(vecs, meta)
    idx.save(tmp_index_dir)

    loaded = ScalableVideoIndex.load(tmp_index_dir, use_gpu=False)
    assert loaded.total_vectors() == 6
    results = loaded.search(vecs[0], top_k=1)
    assert results[0][0] > 0.999


@pytest.mark.unit
def test_SIDX09_embed_dim_preserved():
    """ScalableVideoIndex.embed_dim must match constructor value."""
    from src.indexer import ScalableVideoIndex
    idx = ScalableVideoIndex(embed_dim=EMBED_DIM, index_type="flat", use_gpu=False)
    assert idx.embed_dim == EMBED_DIM


@pytest.mark.unit
def test_SIDX10_add_count_mismatch_raises():
    """ScalableVideoIndex.add() must raise ValueError on size mismatch."""
    from src.indexer import ScalableVideoIndex
    idx  = ScalableVideoIndex(embed_dim=EMBED_DIM, index_type="flat", use_gpu=False)
    vecs = make_random_unit_vectors(3)
    meta = _make_meta(5)  # mismatch
    with pytest.raises(ValueError):
        idx.add(vecs, meta)
