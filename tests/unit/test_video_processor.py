"""
tests/unit/test_video_processor.py
------------------------------------
Unit tests for VideoProcessor and VideoSegment.

Scenarios covered
-----------------
VP-01  FileNotFoundError on non-existent path
VP-02  Keyframe mode: segment count ≈ video duration (1 FPS)
VP-03  Keyframe mode: every segment has correct number of frames
VP-04  Keyframe mode: frames are exactly 224×224×3 BGR uint8
VP-05  Keyframe mode: timestamps are non-negative and monotonically increasing
VP-06  Keyframe mode: end_time never exceeds video duration
VP-07  Sliding-window mode: correct overlap produces expected segment count
VP-08  Sliding-window mode: each segment spans exactly window_sec (except last)
VP-09  Sliding-window mode: consecutive windows overlap by ~50 %
VP-10  Sliding-window mode: boundary segment end_time ≤ video duration
VP-11  Multi-frame count: frames_per_window=1 yields 1 frame per segment
VP-12  Multi-frame count: frames_per_window=5 yields up to 5 frames/segment
VP-13  VideoSegment.representative_frame returns a valid single frame
VP-14  VideoSegment.mid_time = (start + end) / 2
VP-15  frames_per_window=1 in keyframe mode yields exactly 1 frame
VP-16  Custom target_size is respected (not always 224×224)
"""

from __future__ import annotations

import os
import math
import pytest
import numpy as np

from tests.conftest import VIDEO_DURATION, REAL_VIDEO
from src.video_processor import VideoProcessor, VideoSegment


pytestmark = pytest.mark.integration   # need real video file


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def keyframe_segs(real_video_path):
    proc = VideoProcessor(fps_mode=1.0, frames_per_window=3)
    return proc.extract_segments(real_video_path)


@pytest.fixture(scope="module")
def sliding_segs(real_video_path):
    proc = VideoProcessor(fps_mode=None, window_sec=5.0, stride_sec=2.5,
                          frames_per_window=5)
    return proc.extract_segments(real_video_path)


# ── VP-01 ─────────────────────────────────────────────────────────────────

def test_VP01_file_not_found():
    proc = VideoProcessor()
    with pytest.raises(FileNotFoundError):
        proc.extract_segments("/does/not/exist.mp4")


# ── VP-02..06  Keyframe mode ──────────────────────────────────────────────

def test_VP02_keyframe_segment_count(keyframe_segs):
    # At 1 FPS over ~76 s we expect 76 ± 2 segments
    assert 74 <= len(keyframe_segs) <= 78, \
        f"Expected ≈76 segments, got {len(keyframe_segs)}"


def test_VP03_keyframe_frames_count(keyframe_segs):
    for seg in keyframe_segs:
        assert len(seg.frames) >= 1, "Segment must have at least 1 frame"
        assert len(seg.frames) <= 3, "frames_per_window=3 → at most 3 frames"


def test_VP04_keyframe_frame_shape_dtype(keyframe_segs):
    for seg in keyframe_segs:
        for frame in seg.frames:
            assert frame.shape == (224, 224, 3), f"Bad frame shape: {frame.shape}"
            assert frame.dtype == np.uint8, f"Bad dtype: {frame.dtype}"


def test_VP05_keyframe_timestamps_monotonic(keyframe_segs):
    prev_start = -1.0
    for seg in keyframe_segs:
        assert seg.start_time >= 0.0, "start_time must be >= 0"
        assert seg.start_time >= prev_start, "start_time must be monotonically non-decreasing"
        assert seg.end_time > seg.start_time, "end_time must be > start_time"
        prev_start = seg.start_time


def test_VP06_keyframe_end_time_within_duration(keyframe_segs):
    for seg in keyframe_segs:
        assert seg.end_time <= VIDEO_DURATION + 1e-3, \
            f"end_time {seg.end_time:.3f} exceeds video duration {VIDEO_DURATION:.3f}"


# ── VP-07..10  Sliding-window mode ────────────────────────────────────────

def test_VP07_sliding_window_segment_count(sliding_segs):
    # window=5s, stride=2.5s → (76.07 - 5) / 2.5 + 1 ≈ 29-30 segments
    expected_min = math.floor((VIDEO_DURATION - 5.0) / 2.5)
    assert len(sliding_segs) >= expected_min, \
        f"Too few sliding-window segments: {len(sliding_segs)} < {expected_min}"


def test_VP08_sliding_window_spans_correct_duration(sliding_segs):
    for seg in sliding_segs[:-1]:   # skip last (may be shorter)
        span = seg.end_time - seg.start_time
        assert abs(span - 5.0) < 1e-2, f"Window span {span:.4f} s ≠ 5.0 s"


def test_VP09_sliding_window_overlap(sliding_segs):
    # Consecutive windows should overlap by ~50 % (stride=2.5s within window=5s)
    for a, b in zip(sliding_segs, sliding_segs[1:]):
        if a.video_id != b.video_id:
            continue
        stride = b.start_time - a.start_time
        assert abs(stride - 2.5) < 0.1, \
            f"Expected stride ≈ 2.5 s, got {stride:.4f} s"


def test_VP10_sliding_window_end_time_within_duration(sliding_segs):
    for seg in sliding_segs:
        assert seg.end_time <= VIDEO_DURATION + 1e-3


# ── VP-11..15  frames_per_window variants ────────────────────────────────

def test_VP11_frames_per_window_1(real_video_path):
    proc = VideoProcessor(fps_mode=None, window_sec=5.0, stride_sec=5.0,
                          frames_per_window=1)
    segs = proc.extract_segments(real_video_path)
    for seg in segs:
        assert len(seg.frames) == 1, f"Expected 1 frame, got {len(seg.frames)}"


def test_VP12_frames_per_window_5(real_video_path):
    proc = VideoProcessor(fps_mode=None, window_sec=5.0, stride_sec=5.0,
                          frames_per_window=5)
    segs = proc.extract_segments(real_video_path)
    # All segments (except possibly edge cases) should have 5 frames
    for seg in segs:
        assert 1 <= len(seg.frames) <= 5


def test_VP13_representative_frame(sliding_segs):
    for seg in sliding_segs:
        rf = seg.representative_frame
        assert rf.shape == (224, 224, 3)
        assert rf.dtype == np.uint8


def test_VP14_mid_time(sliding_segs):
    for seg in sliding_segs:
        expected = (seg.start_time + seg.end_time) / 2.0
        assert abs(seg.mid_time - expected) < 1e-9


def test_VP15_keyframe_single_frame(real_video_path):
    proc = VideoProcessor(fps_mode=1.0, frames_per_window=1)
    segs = proc.extract_segments(real_video_path)
    for seg in segs:
        assert len(seg.frames) == 1


# ── VP-16  Custom target size ─────────────────────────────────────────────

def test_VP16_custom_target_size(real_video_path):
    proc = VideoProcessor(target_size=(112, 112), fps_mode=1.0,
                          frames_per_window=1)
    segs = proc.extract_segments(real_video_path)
    for seg in segs:
        assert seg.frames[0].shape == (112, 112, 3)
