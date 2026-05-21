"""
windows_capture_server.py — Windows-side capture server for WSL2 bridge

WHY THIS EXISTS
---------------
WSL2's vhci_hcd (USB/IP virtual host controller) does NOT implement isochronous
USB transfers (vhci_get_frame_number returns ENOSYS).  All consumer webcams use
UVC with isochronous endpoints — so V4L2 capture from WSL2 always fails with
ECONNRESET.  This script runs on the Windows host, captures via DirectShow
(cv2.VideoCapture works natively on Windows), and writes segment files to a
shared folder accessible from WSL2 at /mnt/c/...

USAGE (Windows PowerShell or CMD)
----------------------------------
  # Install dependencies once:
  pip install opencv-python

  # Run the server (writes to C:\segments\cam01\ by default):
  python windows_capture_server.py

  # Custom output folder, camera index, and segment length:
  python windows_capture_server.py --output C:\Users\tienc\Prototype\segments --cam 0 --seg-sec 60

  # WSL2 accesses the folder at /mnt/c/Users/tienc/Prototype/segments
  # Start WSL2 app.py in "Windows bridge" mode pointing to the same folder.

OUTPUT FORMAT
-------------
  {output_dir}\\cam01_{wall_ts:010d}_{seg_id:05d}.mp4

  Same filename convention as OverlapCaptureDaemon so existing indexer
  and queue logic work without changes.

ARCHITECTURE
------------
  Windows host:
    windows_capture_server.py
      → cv2.VideoCapture(0) via DirectShow
      → writes  C:\...\segments\cam01_*.mp4

  WSL2:
    app.py  (Start Pipeline → Windows bridge mode)
      → FolderWatcherDaemon watches /mnt/c/.../segments/
      → PersistentJobQueue  ← enqueued when file is stable
      → ContinuousIndexer   ← indexes each segment
      → Qdrant / Faiss      ← searchable vectors
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (must match capture_daemon.py)
# ---------------------------------------------------------------------------

SEGMENT_SEC: int = 60
STRIDE_SEC: int  = 50
CAM_ID: str      = "cam01"
DEFAULT_FPS: float = 25.0
DEFAULT_WIDTH: int  = 640
DEFAULT_HEIGHT: int = 480

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("win_capture")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_path(output_dir: Path, cam_id: str, wall_ts: float, seg_id: int) -> Path:
    name = f"{cam_id}_{int(wall_ts):010d}_{seg_id:05d}.mp4"
    return output_dir / name


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def capture_loop(
    output_dir: Path,
    cam_index: int,
    seg_sec: int,
    stride_sec: int,
    cam_id: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Continuously capture video in segments and write to output_dir."""
    try:
        import cv2
    except ImportError:
        logger.error("cv2 not found.  Run:  pip install opencv-python")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir : %s", output_dir)
    logger.info("Camera     : index=%d", cam_index)
    logger.info("Segment    : %d s  (stride %d s)", seg_sec, stride_sec)
    logger.info("Press Ctrl-C to stop.\n")

    seg_id = 0
    loop_start = time.monotonic()
    stop = False

    def _sigint(_sig, _frame):
        nonlocal stop
        logger.info("Ctrl-C received — finishing current segment then stopping.")
        stop = True

    signal.signal(signal.SIGINT, _sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sigint)

    while not stop:
        # Wait until the next stride boundary
        stride_boundary = loop_start + seg_id * stride_sec
        now = time.monotonic()
        if now < stride_boundary:
            time.sleep(stride_boundary - now)
            if stop:
                break

        # Open camera
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error("Cannot open camera index %d — retry in 5 s", cam_index)
            time.sleep(5)
            continue

        # Configure resolution & fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = fps
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or width
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

        wall_ts  = time.time()
        out_path = _output_path(output_dir, cam_id, wall_ts, seg_id)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (actual_w, actual_h))
        if not writer.isOpened():
            logger.error("Cannot open VideoWriter for %s", out_path)
            cap.release()
            seg_id += 1
            continue

        logger.info(
            "seg=%05d  wall=%d  fps=%.1f  size=%dx%d  → %s",
            seg_id, int(wall_ts), actual_fps, actual_w, actual_h, out_path.name,
        )

        deadline = time.monotonic() + seg_sec
        frames_written = 0
        while time.monotonic() < deadline and not stop:
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                frames_written += 1
            else:
                time.sleep(0.005)

        cap.release()
        writer.release()

        file_size = out_path.stat().st_size if out_path.exists() else 0
        if file_size > 4096:
            logger.info(
                "  → OK  frames=%d  size=%s",
                frames_written,
                f"{file_size / 1024:.1f} KB",
            )
        else:
            logger.warning("  → EMPTY or MISSING: %s (%d bytes)", out_path.name, file_size)

        seg_id += 1

    logger.info("Capture server stopped after %d segments.", seg_id)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Windows-side capture server for WSL2 bridge (NLVS project)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=r"C:\segments\cam01",
        help="Output directory for segment .mp4 files",
    )
    parser.add_argument(
        "--cam", "-c",
        type=int, default=0,
        help="Camera index (0 = first camera, 1 = second, …)",
    )
    parser.add_argument(
        "--seg-sec", type=int, default=SEGMENT_SEC,
        help="Segment duration in seconds",
    )
    parser.add_argument(
        "--stride-sec", type=int, default=STRIDE_SEC,
        help="Stride between segment starts in seconds",
    )
    parser.add_argument(
        "--cam-id", default=CAM_ID,
        help="Logical camera ID (filename prefix)",
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="Target frame rate",
    )
    parser.add_argument(
        "--width",  type=int, default=DEFAULT_WIDTH,
        help="Capture width in pixels",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help="Capture height in pixels",
    )
    args = parser.parse_args()

    capture_loop(
        output_dir  = Path(args.output),
        cam_index   = args.cam,
        seg_sec     = args.seg_sec,
        stride_sec  = args.stride_sec,
        cam_id      = args.cam_id,
        fps         = args.fps,
        width       = args.width,
        height      = args.height,
    )


if __name__ == "__main__":
    main()