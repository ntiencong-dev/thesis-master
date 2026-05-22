"""
tests/unit/test_indexer.py
---------------------------
Unit tests for SegmentMeta, calculate_iou, and temporal_nms.
VideoIndex and ScalableVideoIndex removed in v3.0 (Qdrant-only architecture).

Scenarios covered
-----------------
IDX-01  SegmentMeta auto-computes absolute_start / absolute_end from wall start

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

import pytest

from src.indexer import SegmentMeta, temporal_nms, calculate_iou

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────

def _seg(vid: str, start: float, end: float, score: float) -> tuple:
    m = SegmentMeta(
        cam_id=vid,
        video_path=f"/fake/{vid}.mp4",
        relative_start=start,
        relative_end=end,
    )
    return (score, m)


def _m(start: float, end: float, vid: str = "v") -> SegmentMeta:
    """Shorthand helper for IoU tests."""
    return SegmentMeta(cam_id=vid, video_path=f"/f/{vid}.mp4",
                       relative_start=start, relative_end=end)


# ==========================================================================
#  IDX tests — SegmentMeta
# ==========================================================================

def test_IDX01_segment_meta_absolute_timestamps():
    """SegmentMeta must auto-compute absolute timestamps from wall start."""
    meta = SegmentMeta(
        cam_id="cam0",
        video_path="/fake/vid.mp4",
        relative_start=5.0,
        relative_end=15.0,
        segment_wall_start=1000.0,
    )
    assert meta.absolute_start == pytest.approx(1005.0)
    assert meta.absolute_end   == pytest.approx(1015.0)


# ==========================================================================
#  IOU tests
# ==========================================================================

def test_IOU01_non_overlapping():
    assert calculate_iou(_m(0, 5), _m(6, 10)) == pytest.approx(0.0)


def test_IOU02_identical_segments():
    assert calculate_iou(_m(2, 8), _m(2, 8)) == pytest.approx(1.0)


def test_IOU03_partial_overlap():
    assert calculate_iou(_m(0, 6), _m(4, 10)) == pytest.approx(0.2)


def test_IOU04_contained():
    assert calculate_iou(_m(0, 10), _m(3, 7)) == pytest.approx(0.4)


def test_IOU05_adjacent_touching():
    assert calculate_iou(_m(0, 5), _m(5, 10)) == pytest.approx(0.0)


def test_IOU06_symmetry():
    a, b = _m(0, 8), _m(5, 15)
    assert calculate_iou(a, b) == pytest.approx(calculate_iou(b, a))


# ==========================================================================
#  NMS tests
# ==========================================================================

def test_NMS01_non_overlapping_all_kept():
    candidates = [
        _seg("v0",  0.0,  5.0, 0.9),
        _seg("v0", 10.0, 15.0, 0.8),
        _seg("v0", 20.0, 25.0, 0.7),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 3


def test_NMS02_identical_pair_highest_kept():
    candidates = [
        _seg("v0", 0.0, 5.0, 0.7),
        _seg("v0", 0.0, 5.0, 0.9),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(0.9)


def test_NMS03_full_overlap_suppressed():
    candidates = [
        _seg("v0",  0.0, 10.0, 0.95),
        _seg("v0",  2.0,  8.0, 0.80),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(0.95)


def test_NMS04_iou_above_threshold_suppressed():
    # [0,5] vs [2,7]: IoU=3/7≈0.43 < 0.5 → both kept
    candidates = [
        _seg("v0", 0.0, 5.0, 0.9),
        _seg("v0", 2.0, 7.0, 0.8),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2, f"IoU=3/7≈0.43 < 0.5, both should be kept, got {len(kept)}"


def test_NMS05_different_video_id_both_kept():
    candidates = [
        _seg("vid_A", 0.0, 10.0, 0.9),
        _seg("vid_B", 0.0, 10.0, 0.8),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2


def test_NMS06_empty_input():
    assert temporal_nms([], iou_threshold=0.5) == []


def test_NMS07_single_element_returned():
    candidates = [_seg("v0", 0.0, 5.0, 0.85)]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(0.85)


def test_NMS08_output_sorted_desc():
    candidates = [
        _seg("v0",  0.0,  5.0, 0.5),
        _seg("v0", 10.0, 15.0, 0.9),
        _seg("v0", 20.0, 25.0, 0.7),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    scores = [r[0] for r in kept]
    assert scores == sorted(scores, reverse=True)


def test_NMS09_correct_iou_formula():
    # [0,10] and [6,16]: inter=4, union=16 → IoU=0.25 < 0.5 → both kept
    candidates = [
        _seg("v0",  0.0, 10.0, 0.9),
        _seg("v0",  6.0, 16.0, 0.8),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2, "IoU=0.25 < 0.5, both should be kept"


def test_NMS10_partial_overlap_below_threshold_both_kept():
    # [0,5] and [4,9]: inter=1, union=9, IoU≈0.11
    candidates = [
        _seg("v0", 0.0, 5.0, 0.88),
        _seg("v0", 4.0, 9.0, 0.77),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2


def test_NMS11_top_k_limits_output():
    candidates = [
        _seg("v0", i * 10.0, i * 10.0 + 5.0, round(0.90 - i * 0.05, 2))
        for i in range(5)
    ]
    kept = temporal_nms(candidates, iou_threshold=0.30, top_k=3)
    assert len(kept) == 3
    assert kept[0][0] == pytest.approx(0.90)
    assert kept[1][0] == pytest.approx(0.85)
    assert kept[2][0] == pytest.approx(0.80)


def test_NMS12_iou_equal_threshold_not_suppressed():
    # [0,3] and [1,4]: inter=[1,3]=2, union=[0,4]=4, IoU=0.5 exactly
    # Condition: IoU > threshold (strict) → equal means NOT suppressed
    candidates = [
        _seg("v0", 0.0, 3.0, 0.9),
        _seg("v0", 1.0, 4.0, 0.8),
    ]
    kept = temporal_nms(candidates, iou_threshold=0.5)
    assert len(kept) == 2, "IoU == threshold should NOT suppress (condition is strict >)"

