"""
video_processor.py
------------------
Extracts frames / sliding-window clips from a video file and yields
segment metadata (video_id, start_time, end_time, frame).

Strategy
--------
* 1-FPS keyframe mode  : one representative frame per second.
* Sliding-window mode  : one representative frame per window of
  `window_sec` seconds, advanced by `stride_sec` seconds (overlap
  = window_sec - stride_sec).

Both modes resize frames to 224×224 (CLIP input size) and keep the
video in BGR colour space (OpenCV default); the feature extractor
will convert to RGB as needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class VideoSegment:
    video_id: str               # basename of the source file (no extension)
    video_path: str             # absolute path to the source file
    start_time: float           # segment start in seconds
    end_time: float             # segment end   in seconds
    frames: List[np.ndarray]    # N representative frames (H×W×3, BGR, uint8)

    # convenience
    @property
    def mid_time(self) -> float:
        return (self.start_time + self.end_time) / 2.0

    @property
    def representative_frame(self) -> np.ndarray:
        """Single mid frame for thumbnail display."""
        return self.frames[len(self.frames) // 2] if self.frames else np.zeros((224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Parameters
    ----------
    target_size : tuple[int, int]
        (width, height) to resize every frame to.  Defaults to (224, 224).
    fps_mode : float | None
        If given, sample one frame every (1/fps_mode) seconds regardless of
        window/stride.  Set to None to use sliding-window mode instead.
    window_sec : float
        Duration (seconds) of each sliding window.
    stride_sec : float
        Step (seconds) between consecutive windows.
        overlap ratio = (window_sec - stride_sec) / window_sec
    """

    CLIP_SIZE = (224, 224)

    def __init__(
        self,
        target_size: tuple[int, int] = CLIP_SIZE,
        fps_mode: Optional[float] = 1.0,
        window_sec: float = 5.0,
        stride_sec: float = 2.5,   # 50 % overlap
        frames_per_window: int = 5,
    ) -> None:
        self.target_size       = target_size
        self.fps_mode          = fps_mode
        self.window_sec        = window_sec
        self.stride_sec        = stride_sec
        self.frames_per_window = frames_per_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_segments(self, video_path: str) -> List[VideoSegment]:
        """Return all segments for *video_path* as a list."""
        return list(self._generate_segments(video_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_segments(
        self, video_path: str
    ) -> Generator[VideoSegment, None, None]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {video_path}")

        try:
            native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration: float = total_frames / native_fps

            video_id = os.path.splitext(os.path.basename(video_path))[0]

            if self.fps_mode is not None:
                yield from self._keyframe_mode(
                    cap, video_id, video_path, native_fps, duration
                )
            else:
                yield from self._sliding_window_mode(
                    cap, video_id, video_path, native_fps, duration
                )
        finally:
            cap.release()

    # ---- 1-FPS keyframe mode ------------------------------------------

    def _keyframe_mode(
        self,
        cap: cv2.VideoCapture,
        video_id: str,
        video_path: str,
        native_fps: float,
        duration: float,
    ) -> Generator[VideoSegment, None, None]:
        interval_sec = 1.0 / self.fps_mode  # type: ignore[operator]
        half         = interval_sec / 2.0
        t = 0.0
        while t < duration:
            t_start = max(0.0, t - half)
            t_end   = min(duration, t + half)
            frames  = self._read_n_frames(cap, t_start, t_end, native_fps)
            if not frames:
                break
            yield VideoSegment(
                video_id=video_id,
                video_path=video_path,
                start_time=t_start,
                end_time=t_end,
                frames=frames,
            )
            t += interval_sec

    # ---- Sliding-window mode ------------------------------------------

    def _sliding_window_mode(
        self,
        cap: cv2.VideoCapture,
        video_id: str,
        video_path: str,
        native_fps: float,
        duration: float,
    ) -> Generator[VideoSegment, None, None]:
        t_start = 0.0
        while t_start < duration:
            t_end  = min(t_start + self.window_sec, duration)
            frames = self._read_n_frames(cap, t_start, t_end, native_fps)
            if frames:
                yield VideoSegment(
                    video_id=video_id,
                    video_path=video_path,
                    start_time=t_start,
                    end_time=t_end,
                    frames=frames,
                )
            if t_end >= duration:
                break
            t_start += self.stride_sec

    # ---- Low-level frame readers ----------------------------------------

    def _read_n_frames(
        self,
        cap: cv2.VideoCapture,
        t_start: float,
        t_end: float,
        native_fps: float,
    ) -> List[np.ndarray]:
        """
        Extract frames_per_window evenly-spaced frames from [t_start, t_end].
        Returns a list of resized BGR uint8 arrays (empty on failure).
        """
        n = self.frames_per_window
        if n <= 1:
            timestamps = [(t_start + t_end) / 2.0]
        else:
            step       = (t_end - t_start) / (n - 1)
            timestamps = [t_start + i * step for i in range(n)]

        frames: List[np.ndarray] = []
        for ts in timestamps:
            frame = self._read_frame_at(cap, ts, native_fps)
            if frame is not None:
                frames.append(frame)
        return frames

    def _read_frame_at(
        self,
        cap: cv2.VideoCapture,
        timestamp_sec: float,
        native_fps: float,
    ) -> Optional[np.ndarray]:
        """Seek to *timestamp_sec* and return the resized frame (BGR)."""
        frame_idx = int(timestamp_sec * native_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
