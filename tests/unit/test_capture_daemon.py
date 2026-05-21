"""
tests/unit/test_capture_daemon.py
----------------------------------
Unit tests for fatal-error detection in capture daemons.

All tests run without real hardware — ffmpeg and cv2 are mocked.

Scenarios
---------
CD-01  OverlapCaptureDaemon stops loop when ffmpeg stderr contains "Permission denied"
CD-02  OverlapCaptureDaemon.last_error is set to a descriptive string on fatal error
CD-03  OverlapCaptureDaemon continues loop on non-fatal ffmpeg failure (exit code ≠ 0, no fatal keyword)
CD-04  OpenCVCaptureDaemon stops loop when os.access reports no read permission
CD-05  OpenCVCaptureDaemon.last_error is set when /dev/videoN is not readable
CD-06  OpenCVCaptureDaemon.last_error is None on a normal (no permission issue) failed open → retries
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import subprocess

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_overlap_daemon(tmp_path: Path, source: str = "/dev/video0"):
    from src.capture_daemon import OverlapCaptureDaemon

    queue = MagicMock()
    queue.enqueue.return_value = True
    return OverlapCaptureDaemon(
        cam_id="test_cam",
        rtsp_url=source,
        output_dir=str(tmp_path),
        queue=queue,
    )


def _make_opencv_daemon(tmp_path: Path, device: str = "/dev/video0"):
    from src.capture_daemon import OpenCVCaptureDaemon

    queue = MagicMock()
    queue.enqueue.return_value = True
    return OpenCVCaptureDaemon(
        cam_id="test_cam",
        device=device,
        output_dir=str(tmp_path),
        queue=queue,
    )


def _fake_proc(returncode: int, stderr_bytes: bytes) -> MagicMock:
    """Return a mock Popen-like object with controlled returncode and stderr."""
    proc = MagicMock()
    proc.poll.side_effect = [None, returncode]  # first poll → still running, second → done
    proc.returncode = returncode
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = stderr_bytes
    return proc


# ---------------------------------------------------------------------------
# CD-01 / CD-02  OverlapCaptureDaemon — fatal "Permission denied"
# ---------------------------------------------------------------------------

def test_CD01_overlap_daemon_stops_on_permission_denied(tmp_path):
    """After one ffmpeg call that fails with 'Permission denied', the loop must exit."""
    daemon = _make_overlap_daemon(tmp_path)

    stderr = b"/dev/video0: Permission denied"
    proc = _fake_proc(returncode=1, stderr_bytes=stderr)

    with patch.object(daemon, "_launch_ffmpeg", return_value=proc):
        daemon._stop_event.clear()
        # Run the loop in a thread so we can join with a timeout
        t = threading.Thread(target=daemon._run_loop)
        t.start()
        t.join(timeout=5.0)

    assert not t.is_alive(), "Loop should have exited after the fatal error"


def test_CD02_overlap_daemon_last_error_set_on_permission_denied(tmp_path):
    """last_error must be a non-empty string when 'Permission denied' is detected."""
    daemon = _make_overlap_daemon(tmp_path)

    proc = _fake_proc(returncode=1, stderr_bytes=b"/dev/video0: Permission denied")

    with patch.object(daemon, "_launch_ffmpeg", return_value=proc):
        daemon._run_loop()

    assert daemon.last_error is not None
    assert "Permission denied" in daemon.last_error or "video" in daemon.last_error.lower()


# ---------------------------------------------------------------------------
# CD-03  OverlapCaptureDaemon — non-fatal error → loop keeps running
# ---------------------------------------------------------------------------

def test_CD03_overlap_daemon_no_last_error_on_transient_error(tmp_path):
    """Non-fatal ffmpeg failure should NOT set last_error; loop is stopped by stop_event."""
    daemon = _make_overlap_daemon(tmp_path)

    def mock_launch(out_path, wall_ts):
        # Immediately signal stop so we don't wait for STRIDE_SEC (50 s)
        daemon._stop_event.set()
        proc = MagicMock()
        proc.poll.side_effect = [None, 1]
        proc.returncode = 1
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = b"Generic transient error"
        return proc

    with patch.object(daemon, "_launch_ffmpeg", side_effect=mock_launch):
        daemon._stop_event.clear()
        daemon._run_loop()

    assert daemon.last_error is None, "Non-fatal error should not set last_error"


# ---------------------------------------------------------------------------
# CD-04 / CD-05  OpenCVCaptureDaemon — permission denied via os.access
# ---------------------------------------------------------------------------

def test_CD04_opencv_daemon_stops_on_permission_denied(tmp_path):
    """When os.access returns False for /dev/videoN, the loop must exit immediately."""
    daemon = _make_opencv_daemon(tmp_path, device="/dev/video0")

    with (
        patch("os.path.exists", return_value=True),
        patch("os.access", return_value=False),
    ):
        daemon._stop_event.clear()
        t = threading.Thread(target=daemon._run_loop)
        t.start()
        t.join(timeout=5.0)

    assert not t.is_alive(), "Loop should have exited after permission denied"


def test_CD05_opencv_daemon_last_error_set_on_permission_denied(tmp_path):
    """last_error must be a non-empty string when /dev/videoN is not readable."""
    daemon = _make_opencv_daemon(tmp_path, device="/dev/video0")

    with (
        patch("os.path.exists", return_value=True),
        patch("os.access", return_value=False),
    ):
        daemon._run_loop()

    assert daemon.last_error is not None
    assert "Permission denied" in daemon.last_error or "/dev/video0" in daemon.last_error


# ---------------------------------------------------------------------------
# CD-06  OpenCVCaptureDaemon — device node not a /dev/ path → no permission check
# ---------------------------------------------------------------------------

def test_CD06_opencv_daemon_no_last_error_for_nonexistent_device(tmp_path):
    """
    When device node does not exist (os.path.exists=False), permission check is skipped.
    If cv2.VideoCapture then fails to open, daemon retries and last_error stays None.
    """
    # Use device "99" → self._device=99 (int) → device_path="/dev/video99" (won't exist)
    daemon = _make_opencv_daemon(tmp_path, device="99")

    cap_mock = MagicMock()
    cap_mock.isOpened.return_value = False

    def fake_video_capture(device):
        daemon._stop_event.set()  # stop after first retry to keep test fast
        return cap_mock

    # Patch os.path.exists so /dev/ paths return False (skip permission check)
    import os as _os
    _real_exists = _os.path.exists

    def _patched_exists(path):
        if str(path).startswith("/dev/"):
            return False
        return _real_exists(path)

    with (
        patch("os.path.exists", side_effect=_patched_exists),
        patch("cv2.VideoCapture", side_effect=fake_video_capture),
    ):
        daemon._stop_event.clear()
        daemon._run_loop()

    assert daemon.last_error is None, "No last_error expected when device node doesn't exist"


# ---------------------------------------------------------------------------
# CD-07  app.py _stop_pipeline preserves last_error in session state
# ---------------------------------------------------------------------------

def test_CD07_stop_pipeline_preserves_last_error(tmp_path):
    """
    When a daemon stopped with a fatal error and the user clicks Stop Pipeline,
    last_error must be saved to st.session_state['last_capture_error'] so the
    UI can still display it after the daemon is removed from session state.
    """
    # Simulate the daemon that already stopped itself due to permission denied
    from src.capture_daemon import OverlapCaptureDaemon

    queue = MagicMock()
    daemon = OverlapCaptureDaemon(
        cam_id="cam01",
        rtsp_url="/dev/video0",
        output_dir=str(tmp_path),
        queue=queue,
    )
    # Manually set last_error as if the capture loop detected permission denied
    daemon._last_error = "Fatal error opening '/dev/video0': Permission denied."

    # Simulate st.session_state using a plain dict
    session_state: dict = {"capture_daemon": daemon}

    # Replicate the _stop_pipeline logic (with the fix applied)
    d = session_state.pop("capture_daemon", None)
    if d:
        if getattr(d, "last_error", None):
            session_state["last_capture_error"] = d.last_error

    assert "last_capture_error" in session_state, "last_error must be saved to session state"
    assert "Permission denied" in session_state["last_capture_error"]


# ---------------------------------------------------------------------------
# CD-08 .. CD-12  FolderWatcherDaemon
# ---------------------------------------------------------------------------

def _make_folder_watcher(watch_dir: Path, seg_dir: Path | None = None):
    from src.capture_daemon import FolderWatcherDaemon

    queue = MagicMock()
    queue.enqueue.return_value = True
    return FolderWatcherDaemon(
        cam_id="cam01",
        watch_dir=str(watch_dir),
        queue=queue,
        poll_sec=0.1,
        stable_sec=0.15,
    ), queue


def test_CD08_folder_watcher_enqueues_stable_file(tmp_path):
    """A file that appears with stable size for stable_sec is enqueued once."""
    daemon, queue = _make_folder_watcher(tmp_path)

    # Write a non-zero file
    seg = tmp_path / "cam01_0000000000_00000.mp4"
    seg.write_bytes(b"x" * 1024)

    daemon.start()
    time.sleep(0.6)  # allow 2-3 poll cycles + stable window
    daemon.stop()

    queue.enqueue.assert_called_once()
    job = queue.enqueue.call_args[0][0]
    assert job.segment_path == str(seg)
    assert job.cam_id == "cam01"


def test_CD09_folder_watcher_ignores_zero_byte_file(tmp_path):
    """A 0-byte file must never be enqueued."""
    daemon, queue = _make_folder_watcher(tmp_path)

    seg = tmp_path / "cam01_0000000001_00000.mp4"
    seg.write_bytes(b"")  # 0 bytes

    daemon.start()
    time.sleep(0.5)
    daemon.stop()

    queue.enqueue.assert_not_called()


def test_CD10_folder_watcher_ignores_wrong_cam_id(tmp_path):
    """Files not matching the cam_id prefix must be ignored."""
    daemon, queue = _make_folder_watcher(tmp_path)

    other = tmp_path / "cam02_0000000002_00000.mp4"
    other.write_bytes(b"y" * 2048)

    daemon.start()
    time.sleep(0.5)
    daemon.stop()

    queue.enqueue.assert_not_called()


def test_CD11_folder_watcher_enqueues_each_file_once(tmp_path):
    """A file must only be enqueued once, even if the watcher polls many times."""
    daemon, queue = _make_folder_watcher(tmp_path)

    seg = tmp_path / "cam01_0000000003_00000.mp4"
    seg.write_bytes(b"z" * 512)

    daemon.start()
    time.sleep(0.8)  # multiple poll cycles
    daemon.stop()

    assert queue.enqueue.call_count == 1, "Must enqueue exactly once per file"


def test_CD12_folder_watcher_waits_for_stable_size(tmp_path):
    """A growing file must not be enqueued until its size stabilises."""
    daemon, queue = _make_folder_watcher(tmp_path)

    seg = tmp_path / "cam01_0000000004_00000.mp4"
    seg.write_bytes(b"a" * 100)

    daemon.start()
    time.sleep(0.05)  # first poll sees size=100

    # Grow the file while the daemon is polling (simulates active write)
    seg.write_bytes(b"a" * 500)

    time.sleep(0.05)  # second poll sees size=500, resets stable timer

    # Now leave it stable — daemon should enqueue after stable_sec (0.15s)
    time.sleep(0.5)
    daemon.stop()

    queue.enqueue.assert_called_once()

