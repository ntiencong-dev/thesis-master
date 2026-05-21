"""
gst_pipeline.py
---------------
GStreamer-based video ingestion layer with OpenCV fallback.

Architecture
------------
                  ┌─────────────────────────────────────┐
PC (OpenCV)       │ VideoCapture → seek → frame at t    │
                  └─────────────────────────────────────┘
                  ┌─────────────────────────────────────┐
PC (GStreamer)    │ filesrc → decodebin → videoconvert  │
                  │ → videoscale → BGRx caps → appsink  │
                  └─────────────────────────────────────┘
                  ┌─────────────────────────────────────┐
Kria (VVAS)       │ vvas_xmultisrc → vvas_xdec          │
                  │ → vvas_xfilter (scaler) → appsink   │
                  └─────────────────────────────────────┘

The public interface is identical regardless of backend:

    pipeline = VideoPipeline.create(video_path, config)
    for segment_frames in pipeline.iter_segments():
        # segment_frames: List[np.ndarray] — N BGR (224×224) frames
        ...

Switching backend: change config['pipeline']['video_backend']
  'opencv'            → OpenCV seek (default, always available)
  'gstreamer'         → GStreamer appsink (PC standard plugins)
  'gstreamer_vvas'    → VVAS plugins (Kria KV260 only)
"""

from __future__ import annotations

import os
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

# GStreamer Python bindings (optional)
_GST_AVAILABLE = False
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    Gst.init(None)
    _GST_AVAILABLE = True
except Exception:
    pass

TARGET_SIZE = (224, 224)   # CLIP input resolution


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

class VideoPipeline:
    """
    Unified video-ingestion wrapper.

    Parameters
    ----------
    video_path       : absolute path to the video file.
    window_sec       : sliding-window duration in seconds.
    stride_sec       : step between windows (= window_sec × (1 − overlap)).
    frames_per_window: how many evenly-spaced frames to extract per window.
    backend          : 'opencv' | 'gstreamer' | 'gstreamer_vvas'
    gst_overrides    : optional VVAS element name overrides (Kria only).
    """

    def __init__(
        self,
        video_path: str,
        window_sec: float = 10.0,
        stride_sec: float = 7.0,
        frames_per_window: int = 5,
        backend: str = "opencv",
        gst_overrides: Optional[dict] = None,
    ) -> None:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path        = video_path
        self.window_sec        = window_sec
        self.stride_sec        = stride_sec
        self.frames_per_window = frames_per_window
        self.backend           = backend
        self.gst_overrides     = gst_overrides or {}

    @classmethod
    def create(cls, video_path: str, config: dict) -> "VideoPipeline":
        """Convenience factory that reads parameters from the loaded config dict."""
        pipe_cfg = config.get("pipeline", {})
        eng_cfg  = config.get("engine", {})

        window_sec  = pipe_cfg.get("window_sec", 10.0)
        overlap     = pipe_cfg.get("overlap_ratio", 0.30)
        stride_sec  = window_sec * (1.0 - overlap)
        fpw         = eng_cfg.get("frames_per_window", 5)
        backend     = pipe_cfg.get("video_backend", "opencv")

        gst_overrides = {
            k: v for k, v in pipe_cfg.items()
            if k.startswith("gst_") and k != "gst_overrides"
        }

        return cls(
            video_path=video_path,
            window_sec=window_sec,
            stride_sec=stride_sec,
            frames_per_window=fpw,
            backend=backend,
            gst_overrides=gst_overrides,
        )

    # ------------------------------------------------------------------
    # Public iteration interface
    # ------------------------------------------------------------------

    def iter_segments(
        self,
    ) -> Generator[Tuple[float, float, List[np.ndarray]], None, None]:
        """
        Yield (start_time, end_time, frames) for every sliding window.

        frames : List[np.ndarray]  — exactly frames_per_window BGR uint8
                                     arrays, each 224×224×3.
        """
        if self.backend == "opencv":
            yield from self._opencv_segments()
        elif self.backend in ("gstreamer", "gstreamer_vvas"):
            if not _GST_AVAILABLE:
                print("[VideoPipeline] GStreamer not available — falling back to OpenCV.")
                yield from self._opencv_segments()
            else:
                yield from self._gstreamer_segments()
        else:
            raise ValueError(f"Unknown video_backend: '{self.backend}'")

    # ------------------------------------------------------------------
    # OpenCV backend (always available)
    # ------------------------------------------------------------------

    def _opencv_segments(
        self,
    ) -> Generator[Tuple[float, float, List[np.ndarray]], None, None]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open: {self.video_path}")

        try:
            native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n_frames: int     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration: float   = n_frames / native_fps

            t_start = 0.0
            while t_start < duration:
                t_end = min(t_start + self.window_sec, duration)
                frames = self._read_n_frames_opencv(
                    cap, t_start, t_end, native_fps
                )
                if frames:
                    yield t_start, t_end, frames
                if t_end >= duration:
                    break
                t_start += self.stride_sec
        finally:
            cap.release()

    def _read_n_frames_opencv(
        self,
        cap: cv2.VideoCapture,
        t_start: float,
        t_end: float,
        fps: float,
    ) -> List[np.ndarray]:
        """Read frames_per_window evenly-spaced frames from [t_start, t_end]."""
        n = self.frames_per_window
        if n == 1:
            timestamps = [(t_start + t_end) / 2.0]
        else:
            step = (t_end - t_start) / (n - 1)
            timestamps = [t_start + i * step for i in range(n)]

        frames: List[np.ndarray] = []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR))
        return frames

    # ------------------------------------------------------------------
    # GStreamer backend (PC: standard plugins, Kria: VVAS plugins)
    # ------------------------------------------------------------------

    def _gstreamer_segments(
        self,
    ) -> Generator[Tuple[float, float, List[np.ndarray]], None, None]:
        """
        Pull all frames into a list via appsink, then apply sliding window
        exactly like the OpenCV backend.  GStreamer handles hardware-
        accelerated decode (nvdec on PC, vvas_xdec on Kria).
        """
        all_frames, native_fps = self._gst_decode_all_frames()
        if not all_frames:
            return

        duration = len(all_frames) / native_fps
        t_start  = 0.0

        while t_start < duration:
            t_end   = min(t_start + self.window_sec, duration)
            frames  = self._slice_frames(all_frames, t_start, t_end, native_fps)
            if frames:
                yield t_start, t_end, frames
            if t_end >= duration:
                break
            t_start += self.stride_sec

    def _gst_decode_all_frames(self) -> Tuple[List[np.ndarray], float]:
        """
        Build and run a GStreamer pipeline that decodes the video to raw
        224×224 BGRx frames and collects them via appsink.

        Pipeline (PC):
          filesrc → decodebin → videoconvert → videoscale →
          video/x-raw,format=BGRx,width=224,height=224 → appsink

        Pipeline (Kria/VVAS) — element names overridden by config:
          vvas_xmultisrc → vvas_xdec → vvas_xfilter → appsink
        """
        collected: List[np.ndarray] = []

        # Choose decoder and scaler elements
        if self.backend == "gstreamer_vvas":
            src_elem   = self.gst_overrides.get("gst_source_element",  "filesrc")
            dec_elem   = self.gst_overrides.get("gst_decoder_element", "decodebin")
            scale_elem = self.gst_overrides.get("gst_scaler_element",  "videoscale")
            pipe_str   = (
                f'{src_elem} location="{self.video_path}" ! '
                f'{dec_elem} ! videoconvert ! {scale_elem} ! '
                f'video/x-raw,format=BGRx,width=224,height=224 ! '
                f'appsink name=sink emit-signals=false sync=false max-buffers=0 drop=false'
            )
        else:
            # Standard PC pipeline
            pipe_str = (
                f'filesrc location="{self.video_path}" ! '
                f'decodebin ! videoconvert ! videoscale ! '
                f'video/x-raw,format=BGRx,width=224,height=224 ! '
                f'appsink name=sink emit-signals=false sync=false max-buffers=0 drop=false'
            )

        pipeline = Gst.parse_launch(pipe_str)
        sink     = pipeline.get_by_name("sink")

        pipeline.set_state(Gst.State.PLAYING)
        bus = pipeline.get_bus()

        while True:
            sample = sink.emit("pull-sample")
            if sample is None:
                # Check bus for EOS or errors
                msg = bus.timed_pop_filtered(
                    100 * Gst.MSECOND,
                    Gst.MessageType.EOS | Gst.MessageType.ERROR,
                )
                if msg is not None:
                    break
                continue

            buf  = sample.get_buffer()
            data = buf.extract_dup(0, buf.get_size())
            arr  = np.frombuffer(data, dtype=np.uint8).reshape(224, 224, 4)
            collected.append(arr[:, :, :3].copy())   # drop X channel → BGR

        pipeline.set_state(Gst.State.NULL)

        # Retrieve native FPS from the stream
        native_fps = 25.0
        try:
            pad  = pipeline.get_by_name("sink").get_static_pad("sink")
            caps = pad.get_current_caps()
            if caps:
                s   = caps.get_structure(0)
                fps_frac = s.get_fraction("framerate")
                if fps_frac[0]:
                    native_fps = fps_frac[1] / max(fps_frac[2], 1)
        except Exception:
            pass

        return collected, native_fps

    def _slice_frames(
        self,
        all_frames: List[np.ndarray],
        t_start: float,
        t_end: float,
        fps: float,
    ) -> List[np.ndarray]:
        """Pick frames_per_window evenly-spaced frames from a decoded list."""
        idx_start = int(t_start * fps)
        idx_end   = min(int(t_end * fps), len(all_frames) - 1)
        n         = self.frames_per_window

        if idx_end <= idx_start:
            return [all_frames[idx_start]] if idx_start < len(all_frames) else []

        if n == 1:
            indices = [(idx_start + idx_end) // 2]
        else:
            step    = (idx_end - idx_start) / (n - 1)
            indices = [int(idx_start + i * step) for i in range(n)]

        return [all_frames[min(i, len(all_frames) - 1)] for i in indices]
