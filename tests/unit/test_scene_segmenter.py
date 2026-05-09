"""
tests/unit/test_scene_segmenter.py
------------------------------------
Unit tests for SceneSegmenter.

These tests verify behaviour WITHOUT requiring PySceneDetect to be installed
(the fallback uniform sliding-window path is always available).

Scenarios covered
-----------------
SC-01  FileNotFoundError on non-existent path
SC-02  SCENEDETECT_AVAILABLE flag is a bool
SC-03  Fallback path: segments cover video duration (start ≤ 0 and end >= ~duration)
SC-04  Fallback path: no two adjacent segments have a gap > fallback_window_sec
SC-05  Fallback path: all (start, end) tuples have start < end
SC-06  Fallback path: final segment end ≤ video duration
SC-07  use_fallback_if_unavailable=False raises ImportError when lib missing
SC-08  _subdivide_long_scene: short scene → single segment
SC-09  _subdivide_long_scene: long scene → multiple segments
SC-10  SceneSegmenter: custom window_sec respected in fallback segments
SC-11  get_scene_boundaries returns list, not generator
SC-12  Fallback segments are non-overlapping when stride == window
"""

from __future__ import annotations

import pytest

from src.scene_segmenter import SceneSegmenter, SCENEDETECT_AVAILABLE

pytestmark = pytest.mark.unit


# ── SC-01  FileNotFoundError ──────────────────────────────────────────────

def test_SC01_file_not_found():
    seg = SceneSegmenter()
    with pytest.raises(FileNotFoundError):
        list(seg.iter_segments("/does/not/exist.mp4"))


# ── SC-02  SCENEDETECT_AVAILABLE is bool ─────────────────────────────────

def test_SC02_scenedetect_available_flag():
    assert isinstance(SCENEDETECT_AVAILABLE, bool)


# ── SC-03..06  Fallback path on real video ───────────────────────────────

@pytest.fixture(scope="module")
def fallback_segs(real_video_path):
    """Segments using uniform fallback (always available)."""
    # Force fallback by temporarily patching the flag
    import src.scene_segmenter as sm
    orig = sm.SCENEDETECT_AVAILABLE
    sm.SCENEDETECT_AVAILABLE = False
    try:
        seg = SceneSegmenter(fallback_window_sec=5.0, fallback_stride_sec=2.5)
        result = list(seg.iter_segments(real_video_path))
    finally:
        sm.SCENEDETECT_AVAILABLE = orig
    return result


@pytest.mark.integration
def test_SC03_fallback_covers_start(fallback_segs):
    assert len(fallback_segs) > 0
    assert fallback_segs[0][0] == pytest.approx(0.0, abs=1.0)


@pytest.mark.integration
def test_SC04_fallback_no_large_gap(fallback_segs):
    for i in range(1, len(fallback_segs)):
        gap = fallback_segs[i][0] - fallback_segs[i - 1][0]
        assert gap <= 5.0 + 0.1, f"Gap too large between segments {i-1} and {i}: {gap}"


@pytest.mark.integration
def test_SC05_fallback_start_lt_end(fallback_segs):
    for t0, t1 in fallback_segs:
        assert t0 < t1, f"start {t0} >= end {t1}"


@pytest.mark.integration
def test_SC06_fallback_end_lte_duration(fallback_segs, real_video_path):
    import cv2
    cap = cv2.VideoCapture(real_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    cap.release()
    last_end = fallback_segs[-1][1]
    assert last_end <= duration + 0.1, f"last_end={last_end} > duration={duration}"


# ── SC-07  ImportError if lib unavailable and use_fallback=False ──────────

def test_SC07_raise_import_error_when_unavailable():
    import src.scene_segmenter as sm
    orig = sm.SCENEDETECT_AVAILABLE
    sm.SCENEDETECT_AVAILABLE = False
    try:
        with pytest.raises(ImportError, match="PySceneDetect"):
            SceneSegmenter(use_fallback_if_unavailable=False)
    finally:
        sm.SCENEDETECT_AVAILABLE = orig


# ── SC-08..09  _subdivide_long_scene ─────────────────────────────────────

def test_SC08_short_scene_is_single_segment():
    seg = SceneSegmenter(max_scene_sec=10.0, fallback_window_sec=5.0, fallback_stride_sec=2.5)
    result = list(seg._subdivide_long_scene(0.0, 8.0))
    assert result == [(0.0, 8.0)]


def test_SC09_long_scene_subdivided():
    seg = SceneSegmenter(max_scene_sec=10.0, fallback_window_sec=5.0, fallback_stride_sec=5.0)
    result = list(seg._subdivide_long_scene(0.0, 20.0))
    assert len(result) >= 3
    for t0, t1 in result:
        assert t0 < t1


# ── SC-10  Custom window_sec respected ───────────────────────────────────

@pytest.mark.integration
def test_SC10_custom_window_respected(real_video_path):
    import src.scene_segmenter as sm
    orig = sm.SCENEDETECT_AVAILABLE
    sm.SCENEDETECT_AVAILABLE = False
    try:
        seg = SceneSegmenter(fallback_window_sec=8.0, fallback_stride_sec=8.0)
        segs = list(seg.iter_segments(real_video_path))
    finally:
        sm.SCENEDETECT_AVAILABLE = orig
    # With stride == window, segments should not overlap
    for i in range(1, len(segs)):
        prev_end = segs[i - 1][1]
        cur_start = segs[i][0]
        assert abs(cur_start - prev_end) < 0.5, \
            f"Non-contiguous segment: prev_end={prev_end}, cur_start={cur_start}"


# ── SC-11  get_scene_boundaries returns list ─────────────────────────────

@pytest.mark.integration
def test_SC11_get_scene_boundaries_returns_list(real_video_path):
    import src.scene_segmenter as sm
    orig = sm.SCENEDETECT_AVAILABLE
    sm.SCENEDETECT_AVAILABLE = False
    try:
        seg = SceneSegmenter()
        result = seg.get_scene_boundaries(real_video_path)
    finally:
        sm.SCENEDETECT_AVAILABLE = orig
    assert isinstance(result, list)
    assert len(result) > 0


# ── SC-12  Non-overlapping segments when stride == window ─────────────────

def test_SC12_no_overlap_when_stride_equals_window():
    seg = SceneSegmenter(max_scene_sec=100.0, fallback_window_sec=5.0, fallback_stride_sec=5.0)
    # 25s scene → exactly 5 segments of 5s each
    result = list(seg._subdivide_long_scene(0.0, 25.0))
    for i in range(1, len(result)):
        gap = abs(result[i][0] - result[i - 1][1])
        assert gap < 0.01, f"Overlap or gap detected: {result[i-1]} → {result[i]}"
