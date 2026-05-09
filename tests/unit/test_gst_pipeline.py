"""
tests/unit/test_gst_pipeline.py
---------------------------------
Unit tests for VideoPipeline (GStreamer / OpenCV wrapper).

Scenarios covered
-----------------
GST-01  FileNotFoundError on non-existent video path
GST-02  VideoPipeline.create(): stride_sec = window_sec × (1 - overlap)
GST-03  VideoPipeline.create(): reads frames_per_window from engine config
GST-04  iter_segments (opencv): yields 3-tuple (start, end, frames)
GST-05  iter_segments (opencv): each tuple has correct types
GST-06  iter_segments (opencv): frames list length == frames_per_window (mostly)
GST-07  iter_segments (opencv): all frames are 224×224×3 BGR uint8
GST-08  iter_segments (opencv): segment count matches sliding-window formula
GST-09  iter_segments (opencv): start_time monotonically increases
GST-10  iter_segments (opencv): end_time never exceeds video duration
GST-11  GStreamer unavailable → automatic fallback to OpenCV (no exception)
GST-12  Unknown backend → ValueError
GST-13  _read_n_frames_opencv: N=1 returns midpoint frame
GST-14  _read_n_frames_opencv: N=5 returns 5 frames, each 224×224×3
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import VIDEO_DURATION, VIDEO_FPS, REAL_VIDEO

pytestmark = pytest.mark.integration   # need real video


# ── GST-01  FileNotFoundError ─────────────────────────────────────────────

@pytest.mark.unit
def test_GST01_file_not_found():
    from src.gst_pipeline import VideoPipeline
    with pytest.raises(FileNotFoundError):
        VideoPipeline("/does/not/exist.mp4")


# ── GST-02..03  create() factory ─────────────────────────────────────────

@pytest.mark.unit
def test_GST02_create_stride_sec(real_video_path):
    from src.gst_pipeline import VideoPipeline
    config = {
        "pipeline": {"window_sec": 6.0, "overlap_ratio": 0.5, "video_backend": "opencv"},
        "engine":   {"frames_per_window": 3},
    }
    vp = VideoPipeline.create(real_video_path, config)
    assert abs(vp.stride_sec - 3.0) < 1e-9, \
        f"stride_sec should be 6.0 × (1 - 0.5) = 3.0, got {vp.stride_sec}"


@pytest.mark.unit
def test_GST03_create_frames_per_window(real_video_path):
    from src.gst_pipeline import VideoPipeline
    config = {
        "pipeline": {"window_sec": 5.0, "overlap_ratio": 0.5, "video_backend": "opencv"},
        "engine":   {"frames_per_window": 7},
    }
    vp = VideoPipeline.create(real_video_path, config)
    assert vp.frames_per_window == 7


# ── GST-04..10  iter_segments (opencv backend) ───────────────────────────

@pytest.fixture(scope="module")
def opencv_segments(real_video_path):
    from src.gst_pipeline import VideoPipeline
    vp = VideoPipeline(real_video_path, window_sec=5.0, stride_sec=2.5,
                       frames_per_window=3, backend="opencv")
    return list(vp.iter_segments())


def test_GST04_iter_yields_3_tuple(opencv_segments):
    for item in opencv_segments:
        assert len(item) == 3, "iter_segments must yield (start, end, frames)"


def test_GST05_iter_types(opencv_segments):
    for start, end, frames in opencv_segments:
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert isinstance(frames, list)


def test_GST06_iter_frames_count(opencv_segments):
    for _, _, frames in opencv_segments:
        assert 1 <= len(frames) <= 3, \
            f"Expected 1-3 frames (frames_per_window=3), got {len(frames)}"


def test_GST07_iter_frame_shape_dtype(opencv_segments):
    for _, _, frames in opencv_segments:
        for frame in frames:
            assert frame.shape == (224, 224, 3), f"Bad shape: {frame.shape}"
            assert frame.dtype == np.uint8


def test_GST08_iter_segment_count(opencv_segments):
    expected_min = math.floor((VIDEO_DURATION - 5.0) / 2.5)
    assert len(opencv_segments) >= expected_min, \
        f"Too few segments: {len(opencv_segments)} (expected >= {expected_min})"


def test_GST09_iter_start_time_monotonic(opencv_segments):
    starts = [s for s, _, _ in opencv_segments]
    assert starts == sorted(starts), "start_times must be monotonically increasing"


def test_GST10_iter_end_time_within_duration(opencv_segments):
    for _, end, _ in opencv_segments:
        assert end <= VIDEO_DURATION + 1e-3, \
            f"end_time {end:.4f} exceeds duration {VIDEO_DURATION:.4f}"


# ── GST-11  GStreamer fallback to OpenCV ─────────────────────────────────

def test_GST11_gstreamer_backend_fallback(real_video_path, monkeypatch):
    """When _GST_AVAILABLE=False, gstreamer backend silently falls back to OpenCV."""
    import src.gst_pipeline as gst_module
    monkeypatch.setattr(gst_module, "_GST_AVAILABLE", False)

    from src.gst_pipeline import VideoPipeline
    vp = VideoPipeline(real_video_path, window_sec=5.0, stride_sec=5.0,
                       frames_per_window=1, backend="gstreamer")
    segs = list(vp.iter_segments())
    assert len(segs) > 0, "Fallback to OpenCV should still yield segments"


# ── GST-12  Unknown backend ───────────────────────────────────────────────

def test_GST12_unknown_backend_raises(real_video_path):
    from src.gst_pipeline import VideoPipeline
    vp = VideoPipeline(real_video_path, backend="totally_unknown")
    with pytest.raises(ValueError, match="Unknown video_backend"):
        list(vp.iter_segments())


# ── GST-13..14  _read_n_frames_opencv ────────────────────────────────────

def test_GST13_read_n_frames_single(real_video_path):
    import cv2
    from src.gst_pipeline import VideoPipeline
    vp  = VideoPipeline(real_video_path, frames_per_window=1)
    cap = cv2.VideoCapture(real_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        frames = vp._read_n_frames_opencv(cap, 0.0, 5.0, fps)
    finally:
        cap.release()
    assert len(frames) == 1
    assert frames[0].shape == (224, 224, 3)


def test_GST14_read_n_frames_five(real_video_path):
    import cv2
    from src.gst_pipeline import VideoPipeline
    vp  = VideoPipeline(real_video_path, frames_per_window=5)
    cap = cv2.VideoCapture(real_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        frames = vp._read_n_frames_opencv(cap, 0.0, 5.0, fps)
    finally:
        cap.release()
    assert len(frames) == 5
    for f in frames:
        assert f.shape == (224, 224, 3)
        assert f.dtype == np.uint8
