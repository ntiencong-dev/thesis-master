"""
capture_daemon.py — Tail-only overlap segment capture daemon (v2.0)

Architecture ref: CONTINUOUS_STREAM_RESEARCH.md §3.3 Tail-Only 10s Overlap

Strategy:
  Every STRIDE_SEC seconds launch an ffmpeg subprocess that records for
  SEGMENT_SEC seconds using '-c copy' (no transcoding).  The last OVERLAP_SEC
  of every segment will therefore be captured again at the start of the *next*
  segment launched STRIDE_SEC later — hence "tail-only 10s overlap".

  SEGMENT_SEC = 60
  STRIDE_SEC  = 50
  OVERLAP_SEC = 10   (= SEGMENT_SEC - STRIDE_SEC)

  Timeline (cam_id="cam01"):
    t=0  → launch ffmpeg  → writes cam01_0000000000_00000.mp4  (t=0..60)
    t=50 → launch ffmpeg  → writes cam01_0000000050_00001.mp4  (t=50..110)
    t=100→ launch ffmpeg  → writes cam01_0000000100_00002.mp4  (t=100..160)
    ...

  Each finished file is enqueued into PersistentJobQueue for asynchronous
  indexing by ContinuousIndexer.

Output filename format:
  {cam_id}_{wall_ts:010d}_{seg_id:05d}.mp4

Usage::

    from src.capture_daemon import OverlapCaptureDaemon
    from src.job_queue import PersistentJobQueue

    queue  = PersistentJobQueue("/opt/nlvs/job_queue.db")
    daemon = OverlapCaptureDaemon(
        cam_id="cam01",
        rtsp_url="rtsp://192.168.1.100:554/stream",
        output_dir="/storage/cam01",
        queue=queue,
    )
    daemon.start()   # non-blocking; use daemon.stop() to gracefully shut down
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from .job_queue import CircuitBreaker, IndexJob, PersistentJobQueue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (CONTINUOUS_STREAM_RESEARCH.md §3.3)
# ---------------------------------------------------------------------------

SEGMENT_SEC = 60    # duration of each captured segment file
STRIDE_SEC  = 50    # gap between segment start times (= SEGMENT_SEC - OVERLAP_SEC)
OVERLAP_SEC = 10    # tail overlap re-captured in the next segment

# ffmpeg/device errors that are unrecoverable and should stop the capture loop
_FATAL_PATTERNS = (
    "Permission denied",
    "No such file or directory",
    "No such device",
    "Input/output error",
)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

import shutil as _shutil


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return _shutil.which("ffmpeg") is not None


def is_wsl2() -> bool:
    """Return True when running inside WSL2 (Microsoft kernel)."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Utility: discover local webcam / V4L2 devices
# ---------------------------------------------------------------------------

def list_webcams() -> list:
    """
    Return a list of available video capture device paths on this machine.

    Suppresses all OpenCV V4L2 / obsensor stderr noise during probing.
    Returns device strings like ['/dev/video0', '0', ...] for devices that
    respond to cv2.VideoCapture.  Returns an empty list on WSL2 without
    USB passthrough (see usbipd-win setup guide).
    """
    import glob as _glob
    import os as _os
    import cv2 as _cv2

    devices: list = []

    # Redirect stderr to /dev/null to silence V4L2 / obsensor errors
    saved_stderr = _os.dup(2)
    try:
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        _os.dup2(devnull_fd, 2)
        _os.close(devnull_fd)

        # 1. Try /dev/video* paths (Linux native)
        for dev in sorted(_glob.glob("/dev/video*")):
            try:
                cap = _cv2.VideoCapture(dev)
                if cap.isOpened():
                    devices.append(dev)
                cap.release()
            except Exception:
                pass

        # 2. Fallback: integer indices 0..7
        if not devices:
            for idx in range(8):
                try:
                    cap = _cv2.VideoCapture(idx)
                    if cap.isOpened():
                        devices.append(str(idx))
                    cap.release()
                except Exception:
                    pass
    finally:
        _os.dup2(saved_stderr, 2)
        _os.close(saved_stderr)

    return devices


# ---------------------------------------------------------------------------
# OverlapCaptureDaemon
# ---------------------------------------------------------------------------

class OverlapCaptureDaemon:
    """
    Launches ffmpeg subprocesses every STRIDE_SEC to capture SEGMENT_SEC
    of video from an RTSP stream using '-c copy' (no transcoding).

    Parameters
    ----------
    cam_id     : Logical camera identifier, used as filename prefix.
    rtsp_url   : Full RTSP URL (e.g. rtsp://user:pass@host/stream).
    output_dir : Directory to write segment .mp4 files into.
    queue      : PersistentJobQueue instance.  Each finished file is
                 enqueued immediately after the ffmpeg process exits.
    circuit_breaker : Optional CircuitBreaker.  When the circuit is open
                 the new capture still proceeds (we don't skip capture to
                 avoid data loss) but enqueue is skipped.
    """

    def __init__(
        self,
        cam_id:          str,
        rtsp_url:        str,
        output_dir:      str,
        queue:           PersistentJobQueue,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._cam_id          = cam_id
        self._rtsp_url        = rtsp_url
        self._output_dir      = Path(output_dir)
        self._queue           = queue
        self._circuit_breaker = circuit_breaker
        self._seg_id          = 0
        self._stop_event      = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def last_error(self) -> Optional[str]:
        """Fatal error message set when the daemon stops due to an unrecoverable error."""
        return self._last_error

    def start(self) -> None:
        """Start the capture loop in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("[CaptureD:%s] Already running.", self._cam_id)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"CaptureD-{self._cam_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[CaptureD:%s] Started. output_dir=%s", self._cam_id, self._output_dir)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the capture loop to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("[CaptureD:%s] Stopped.", self._cam_id)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _output_path(self, wall_ts: float) -> Path:
        """Generate the output file path for a segment starting at wall_ts."""
        name = f"{self._cam_id}_{int(wall_ts):010d}_{self._seg_id:05d}.mp4"
        return self._output_dir / name

    def _is_network_source(self) -> bool:
        """Return True if the source is a network stream (RTSP/RTMP/HTTP)."""
        return self._rtsp_url.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    def _launch_ffmpeg(self, out_path: Path, wall_ts: float) -> subprocess.Popen:
        """
        Spawn ffmpeg to capture SEGMENT_SEC seconds.

        Network streams (RTSP/RTMP):
          -rtsp_transport tcp · -c copy  (passthrough H.264, zero CPU)

        Local V4L2 / webcam devices (/dev/videoN or integer "N"):
          -f v4l2 · -c:v libx264 -preset ultrafast  (encode raw frames)
        """
        if self._is_network_source():
            cmd = [
                "ffmpeg",
                "-loglevel", "warning",
                "-rtsp_transport", "tcp",
                "-i", self._rtsp_url,
                "-t", str(SEGMENT_SEC),
                "-c", "copy",
                "-y",
                str(out_path),
            ]
        else:
            # V4L2 device: source is /dev/videoN or a bare integer "0"
            device = self._rtsp_url
            if device.isdigit():
                device = f"/dev/video{device}"
            cmd = [
                "ffmpeg",
                "-loglevel", "warning",
                "-f", "v4l2",
                "-i", device,
                "-t", str(SEGMENT_SEC),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-y",
                str(out_path),
            ]
        logger.debug("[CaptureD:%s] ffmpeg cmd: %s", self._cam_id, " ".join(cmd))
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _enqueue_segment(self, out_path: Path, wall_ts: float) -> None:
        """
        Enqueue the finished segment into the job queue.
        Respects the CircuitBreaker: when open, log and skip enqueue.
        """
        if self._circuit_breaker and self._circuit_breaker.is_open():
            logger.warning(
                "[CaptureD:%s] CircuitBreaker OPEN — skipping enqueue for %s",
                self._cam_id, out_path.name,
            )
            return
        job = IndexJob(
            cam_id=self._cam_id,
            segment_path=str(out_path),
            capture_timestamp=wall_ts,
        )
        ok = self._queue.enqueue(job)
        if ok:
            logger.info("[CaptureD:%s] Enqueued %s", self._cam_id, out_path.name)
        else:
            logger.debug("[CaptureD:%s] Duplicate skipped: %s", self._cam_id, out_path.name)

    def _run_loop(self) -> None:
        """
        Main capture loop.

        Every STRIDE_SEC seconds:
          1. Record the current wall timestamp.
          2. Spawn an ffmpeg process for SEGMENT_SEC seconds.
          3. Wait for it to finish (blocking inside this thread).
          4. Enqueue the resulting file.
          5. Increment seg_id and sleep until the next stride boundary.
        """
        logger.info("[CaptureD:%s] Capture loop started.", self._cam_id)
        loop_start = time.monotonic()

        while not self._stop_event.is_set():
            # ------- capture timing -------
            stride_boundary = loop_start + self._seg_id * STRIDE_SEC
            now             = time.monotonic()
            if now < stride_boundary:
                # Wait until the next stride window starts
                wait = stride_boundary - now
                self._stop_event.wait(timeout=wait)
                if self._stop_event.is_set():
                    break

            wall_ts  = time.time()
            out_path = self._output_path(wall_ts)

            # ------- spawn ffmpeg -------
            logger.info(
                "[CaptureD:%s] seg=%05d  wall=%d  out=%s",
                self._cam_id, self._seg_id, int(wall_ts), out_path.name,
            )
            proc = self._launch_ffmpeg(out_path, wall_ts)

            # ------- wait for completion -------
            # Poll so we can honour stop_event
            while proc.poll() is None:
                if self._stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    logger.info("[CaptureD:%s] ffmpeg terminated on stop.", self._cam_id)
                    return
                time.sleep(1.0)

            rc = proc.returncode
            if rc != 0:
                stderr = (proc.stderr.read() or b"").decode(errors="replace").strip()
                logger.error(
                    "[CaptureD:%s] ffmpeg exited %d for %s — %s",
                    self._cam_id, rc, out_path.name, stderr,
                )
                if any(p in stderr for p in _FATAL_PATTERNS):
                    source = self._rtsp_url
                    self._last_error = (
                        f"Fatal error opening {source!r}: {stderr}. "
                        "If this is a V4L2 device, add the user to the 'video' group: "
                        "sudo usermod -aG video $USER  (then log out and log back in)."
                    )
                    logger.error(
                        "[CaptureD:%s] FATAL — stopping capture loop. %s",
                        self._cam_id, self._last_error,
                    )
                    break
            elif out_path.exists() and out_path.stat().st_size > 0:
                self._enqueue_segment(out_path, wall_ts)
            else:
                logger.warning("[CaptureD:%s] Output missing or empty: %s", self._cam_id, out_path)

            self._seg_id += 1

        logger.info("[CaptureD:%s] Capture loop exited.", self._cam_id)


# ---------------------------------------------------------------------------
# OpenCVCaptureDaemon — fallback when ffmpeg is unavailable
# ---------------------------------------------------------------------------

class OpenCVCaptureDaemon:
    """
    Fallback capture daemon using cv2.VideoCapture (no ffmpeg required).

    Records SEGMENT_SEC seconds per segment using OpenCV VideoWriter.
    Segments are written sequentially (no overlap in this fallback mode).
    Works with WSL2 after USB passthrough (usbipd-win) or any OpenCV-
    accessible device.

    Parameters
    ----------
    cam_id      : Logical camera identifier.
    device      : OpenCV device — integer string ("0") or /dev/videoN path.
    output_dir  : Directory to write segment .mp4 files.
    queue       : PersistentJobQueue for enqueuing finished segments.
    circuit_breaker : Optional CircuitBreaker.
    """

    def __init__(
        self,
        cam_id: str,
        device: str,
        output_dir: str,
        queue: PersistentJobQueue,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._cam_id          = cam_id
        self._device: object  = int(device) if str(device).isdigit() else device
        self._output_dir      = Path(output_dir)
        self._queue           = queue
        self._circuit_breaker = circuit_breaker
        self._seg_id          = 0
        self._stop_event      = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_error(self) -> Optional[str]:
        """Fatal error message set when the daemon stops due to an unrecoverable error."""
        return self._last_error

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"CVCapture-{self._cam_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[CVCapture:%s] Started. device=%s output_dir=%s",
                    self._cam_id, self._device, self._output_dir)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("[CVCapture:%s] Stopped.", self._cam_id)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _output_path(self, wall_ts: float) -> Path:
        name = f"{self._cam_id}_{int(wall_ts):010d}_{self._seg_id:05d}.mp4"
        return self._output_dir / name

    def _enqueue_segment(self, out_path: Path, wall_ts: float) -> None:
        if self._circuit_breaker and self._circuit_breaker.is_open():
            logger.warning("[CVCapture:%s] CircuitBreaker OPEN — skip %s",
                           self._cam_id, out_path.name)
            return
        job = IndexJob(
            cam_id=self._cam_id,
            segment_path=str(out_path),
            capture_timestamp=wall_ts,
        )
        if self._queue.enqueue(job):
            logger.info("[CVCapture:%s] Enqueued %s", self._cam_id, out_path.name)

    def _run_loop(self) -> None:
        logger.info("[CVCapture:%s] Capture loop started.", self._cam_id)

        while not self._stop_event.is_set():
            wall_ts  = time.time()
            out_path = self._output_path(wall_ts)

            # Open device — check OS-level permission before trying OpenCV
            device_path = (
                f"/dev/video{self._device}"
                if isinstance(self._device, int)
                else str(self._device)
            )
            if device_path.startswith("/dev/") and os.path.exists(device_path):
                if not os.access(device_path, os.R_OK):
                    self._last_error = (
                        f"Permission denied: cannot open {device_path!r}. "
                        "Add the user to the 'video' group: "
                        "sudo usermod -aG video $USER  (then log out and log back in)."
                    )
                    logger.error(
                        "[CVCapture:%s] FATAL — stopping capture loop. %s",
                        self._cam_id, self._last_error,
                    )
                    break

            cap = cv2.VideoCapture(self._device)
            if not cap.isOpened():
                logger.error(
                    "[CVCapture:%s] Cannot open device %s — retry in 5 s",
                    self._cam_id, self._device,
                )
                self._stop_event.wait(5.0)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            logger.info(
                "[CVCapture:%s] seg=%05d  wall=%d  fps=%.1f  size=%dx%d  out=%s",
                self._cam_id, self._seg_id, int(wall_ts), fps, w, h, out_path.name,
            )

            deadline = time.monotonic() + SEGMENT_SEC
            while time.monotonic() < deadline and not self._stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    writer.write(frame)
                else:
                    time.sleep(0.01)

            cap.release()
            writer.release()

            if out_path.exists() and out_path.stat().st_size > 4096:
                self._enqueue_segment(out_path, wall_ts)
            else:
                logger.warning("[CVCapture:%s] Output empty or missing: %s",
                               self._cam_id, out_path)

            self._seg_id += 1

        logger.info("[CVCapture:%s] Capture loop exited.", self._cam_id)


# ---------------------------------------------------------------------------
# FolderWatcherDaemon — Windows capture bridge for WSL2
# ---------------------------------------------------------------------------

class FolderWatcherDaemon:
    """
    Watches a directory for new, complete .mp4 files and enqueues them.

    **WSL2 isochronous USB workaround**: WSL2's vhci_hcd does not implement
    isochronous USB transfers (vhci_get_frame_number returns -ENOSYS), so
    UVC webcams cannot stream via V4L2.  The recommended workaround is to run
    ``windows_capture_server.py`` on the Windows host (using DirectShow /
    cv2.VideoCapture) which writes segment files to a path accessible from
    both Windows and WSL2 (e.g. ``C:\\Users\\<user>\\segments\\`` ↔
    ``/mnt/c/Users/<user>/segments/``).  This daemon monitors that directory
    and enqueues each completed segment for indexing.

    A file is considered *complete* when:
      - Its size is non-zero AND
      - Its size has not changed for ``stable_sec`` seconds (default 2 s).

    Parameters
    ----------
    cam_id      : Logical camera identifier (prefix filter: ``{cam_id}_*.mp4``).
    watch_dir   : Directory to monitor.  Can be a WSL2 mount point like
                  ``/mnt/c/Users/<user>/segments``.
    queue       : PersistentJobQueue instance.
    poll_sec    : Polling interval in seconds (default 2.0).
    stable_sec  : Seconds of stable file size before a file is considered
                  complete (default 2.0).
    circuit_breaker : Optional CircuitBreaker.
    """

    def __init__(
        self,
        cam_id: str,
        watch_dir: str,
        queue: PersistentJobQueue,
        poll_sec: float = 2.0,
        stable_sec: float = 2.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._cam_id          = cam_id
        self._watch_dir       = Path(watch_dir)
        self._queue           = queue
        self._poll_sec        = poll_sec
        self._stable_sec      = stable_sec
        self._circuit_breaker = circuit_breaker
        self._stop_event      = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        # Maps path → (last_size, first_seen_stable_ts)
        self._size_cache: dict[Path, tuple[int, float]] = {}
        # Set of already-enqueued paths to avoid duplicates
        self._seen: set[Path] = set()

        self._watch_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("[FolderWatch:%s] Already running.", self._cam_id)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"FolderWatch-{self._cam_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[FolderWatch:%s] Started. watch_dir=%s poll=%.1fs stable=%.1fs",
            self._cam_id, self._watch_dir, self._poll_sec, self._stable_sec,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("[FolderWatch:%s] Stopped.", self._cam_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue_segment(self, path: Path) -> None:
        if self._circuit_breaker and self._circuit_breaker.is_open():
            logger.warning("[FolderWatch:%s] CircuitBreaker OPEN — skip %s",
                           self._cam_id, path.name)
            return
        job = IndexJob(
            cam_id=self._cam_id,
            segment_path=str(path),
            capture_timestamp=path.stat().st_mtime,
        )
        if self._queue.enqueue(job):
            logger.info("[FolderWatch:%s] Enqueued %s", self._cam_id, path.name)
        else:
            logger.debug("[FolderWatch:%s] Duplicate skipped: %s",
                         self._cam_id, path.name)

    def _run_loop(self) -> None:
        logger.info("[FolderWatch:%s] Watch loop started.", self._cam_id)
        prefix = f"{self._cam_id}_"

        while not self._stop_event.is_set():
            try:
                candidates = [
                    p for p in self._watch_dir.glob("*.mp4")
                    if p.name.startswith(prefix) and p not in self._seen
                ]
            except OSError as exc:
                logger.error("[FolderWatch:%s] Cannot read watch_dir: %s", self._cam_id, exc)
                self._last_error = str(exc)
                self._stop_event.wait(self._poll_sec)
                continue

            now = time.monotonic()
            for path in candidates:
                try:
                    size = path.stat().st_size
                except OSError:
                    self._size_cache.pop(path, None)
                    continue

                if size == 0:
                    continue

                prev_size, stable_since = self._size_cache.get(path, (None, now))
                if prev_size == size:
                    # Size unchanged — check if stable long enough
                    if (now - stable_since) >= self._stable_sec:
                        self._seen.add(path)
                        self._size_cache.pop(path, None)
                        self._enqueue_segment(path)
                else:
                    # Size changed (file still being written)
                    self._size_cache[path] = (size, now)

            self._stop_event.wait(timeout=self._poll_sec)

        logger.info("[FolderWatch:%s] Watch loop exited.", self._cam_id)
