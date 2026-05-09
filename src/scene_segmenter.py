"""
scene_segmenter.py
------------------
Phase 2: Scene-Aware Adaptive Segmentation (RESEARCH_SOTA_NLVS.md §II1).

Replaces naive fixed-window segmentation with content-aware boundaries using
PySceneDetect's ``ContentDetector`` (HSV histogram difference).  Falls back to
regular sliding-window when PySceneDetect is unavailable.

Algorithm
---------
1. Run ``ContentDetector`` over the video → list of scene cut timestamps.
2. For each detected scene:
   - If scene duration ≤ ``max_scene_sec``: emit as one segment.
   - If scene duration > ``max_scene_sec``: subdivide with sliding-window
     (window=``fallback_window_sec``, stride=``fallback_stride_sec``).
3. Yield ``(start_time, end_time)`` tuples consumed by the caller.

Benefits vs fixed window
------------------------
* Eliminates chimera embeddings — segments never span a hard scene cut.
* Reduces index size for static/surveillance footage (long static scenes
  → one segment instead of many overlapping windows).
* Automatically handles montage videos (many short scenes).

Usage
-----
    from src.scene_segmenter import SceneSegmenter

    seg = SceneSegmenter(threshold=27.0)
    for t_start, t_end in seg.iter_segments(video_path):
        ...

    # Check whether PySceneDetect is available
    from src.scene_segmenter import SCENEDETECT_AVAILABLE
"""

from __future__ import annotations

import os
from typing import Generator, List, Tuple

# ---------------------------------------------------------------------------
# Optional PySceneDetect import
# ---------------------------------------------------------------------------
SCENEDETECT_AVAILABLE = False
try:
    from scenedetect import open_video, ContentDetector, SceneManager  # type: ignore
    SCENEDETECT_AVAILABLE = True
except ImportError:
    pass


class SceneSegmenter:
    """
    Parameters
    ----------
    threshold : float
        ContentDetector sensitivity.  Lower → more cuts detected.
        Default 27.0 is the PySceneDetect recommended value for general video.
        Use ~18–22 for surveillance / low-motion video.
    min_scene_frames : int
        Minimum number of frames per detected scene (filters micro-cuts).
    max_scene_sec : float
        Scenes longer than this are sub-divided by sliding window.
    fallback_window_sec : float
        Sliding-window duration used for over-long scenes.
    fallback_stride_sec : float | None
        Stride for over-long scene subdivision.  Defaults to
        ``fallback_window_sec * 0.5`` (50 % overlap).
    use_fallback_if_unavailable : bool
        If True and PySceneDetect is not installed, silently falls back to
        uniform sliding-window segmentation.  If False, raises ImportError.
    """

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_frames: int = 15,
        max_scene_sec: float = 10.0,
        fallback_window_sec: float = 5.0,
        fallback_stride_sec: float | None = None,
        use_fallback_if_unavailable: bool = True,
    ) -> None:
        self.threshold = threshold
        self.min_scene_frames = min_scene_frames
        self.max_scene_sec = max_scene_sec
        self.fallback_window_sec = fallback_window_sec
        self.fallback_stride_sec = (
            fallback_stride_sec
            if fallback_stride_sec is not None
            else fallback_window_sec * 0.5
        )
        self.use_fallback_if_unavailable = use_fallback_if_unavailable

        if not SCENEDETECT_AVAILABLE and not use_fallback_if_unavailable:
            raise ImportError(
                "PySceneDetect is required but not installed. "
                "Run: pip install scenedetect[opencv]"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_segments(
        self, video_path: str
    ) -> Generator[Tuple[float, float], None, None]:
        """
        Yield (start_time, end_time) pairs in chronological order.

        Uses scene detection when PySceneDetect is available; falls back
        to uniform sliding-window otherwise.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if SCENEDETECT_AVAILABLE:
            yield from self._scene_aware_segments(video_path)
        else:
            import warnings
            warnings.warn(
                "[SceneSegmenter] PySceneDetect not installed — "
                "falling back to uniform sliding-window. "
                "Install with: pip install scenedetect[opencv]",
                stacklevel=2,
            )
            yield from self._fallback_segments(video_path)

    def get_scene_boundaries(self, video_path: str) -> List[Tuple[float, float]]:
        """Return a list of all (start, end) segments (not a generator)."""
        return list(self.iter_segments(video_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scene_aware_segments(
        self, video_path: str
    ) -> Generator[Tuple[float, float], None, None]:
        """Detect scene boundaries and yield natural segments."""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_frames,
            )
        )
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            # No cuts detected — treat entire video as one scene
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = n_frames / fps
            cap.release()
            yield from self._subdivide_long_scene(0.0, duration)
            return

        for start_tc, end_tc in scene_list:
            t_start = start_tc.get_seconds()
            t_end = end_tc.get_seconds()
            yield from self._subdivide_long_scene(t_start, t_end)

    def _subdivide_long_scene(
        self, t_start: float, t_end: float
    ) -> Generator[Tuple[float, float], None, None]:
        """
        Emit the scene as a single segment if short enough; otherwise
        sub-divide with sliding window.
        """
        duration = t_end - t_start
        if duration <= self.max_scene_sec:
            yield (t_start, t_end)
        else:
            t = t_start
            while t < t_end:
                seg_end = min(t + self.fallback_window_sec, t_end)
                yield (t, seg_end)
                if seg_end >= t_end:
                    break
                t += self.fallback_stride_sec

    def _fallback_segments(
        self, video_path: str
    ) -> Generator[Tuple[float, float], None, None]:
        """Uniform sliding-window — used when PySceneDetect is unavailable."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps
        cap.release()

        t = 0.0
        while t < duration:
            t_end = min(t + self.fallback_window_sec, duration)
            yield (t, t_end)
            if t_end >= duration:
                break
            t += self.fallback_stride_sec
