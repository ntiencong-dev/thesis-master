"""
app.py
------
Streamlit web demo for the Natural Language Video Search (NLVS) system.

Implements the continuous-stream surveillance model described in
CONTINUOUS_STREAM_RESEARCH.md v2.3:

  Camera (webcam / RTSP)
    → OverlapCaptureDaemon  (60 s segments, 10 s tail overlap, stride 50 s)
    → PersistentJobQueue    (SQLite, crash-safe)
    → ContinuousIndexer     (watchdog + encoder + Qdrant)
    → NLVideoSearcher       (EVA-CLIP ViT-L/14)

Confirmed parameters (CONTINUOUS_STREAM_RESEARCH.md §2.1):
  window=10 s · overlap=30% · stride=7.0 s · frames=5 · min_score=0.20

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import datetime
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
import yaml

from src.capture_daemon import (
    FolderWatcherDaemon,
    OpenCVCaptureDaemon,
    OverlapCaptureDaemon,
    check_ffmpeg,
    is_wsl2,
    list_webcams,
)
from src.continuous_indexer import ContinuousIndexer
from src.job_queue import PersistentJobQueue
from src.searcher import NLVideoSearcher, SearchResult, _normalize_query

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Confirmed operational parameters (CONTINUOUS_STREAM_RESEARCH.md v2.1)
# ---------------------------------------------------------------------------

_WINDOW_SEC          = 10.0
_OVERLAP_RATIO       = 0.30
_FRAMES_PER_WINDOW   = 5
_SCORE_THRESHOLD     = 0.20
_DEFAULT_SEGMENT_DIR = os.path.join(os.path.dirname(__file__), "segments")
_DEFAULT_DB_PATH     = os.path.join(_DEFAULT_SEGMENT_DIR, "job_queue.db")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NLVS — Natural Language Video Search",
    page_icon="📹",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _fmt_wall_time(ts: float) -> str:
    if ts <= 0:
        return ""
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _extract_thumbnail(video_path: str, timestamp_sec: float,
                       width: int = 320, height: int = 180) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    finally:
        cap.release()


def _export_clip(video_path: str, start: float, end: float) -> bytes:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    end_frame = int(end * fps)
    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos > end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    with open(tmp_path, "rb") as fh:
        data = fh.read()
    os.unlink(tmp_path)
    return data


def _grab_webcam_frame(source: str) -> Optional[np.ndarray]:
    """Grab a single frame from a webcam/RTSP source for live preview.

    Returns None silently for folder-watcher sources (no live stream to show).
    """
    # Folder-watcher mode — no live stream
    if os.path.isdir(source) or source.startswith("watch://"):
        return None
    device: object = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return None
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Pipeline config builder
# ---------------------------------------------------------------------------

def _make_indexer_config(segment_dir: str) -> dict:
    """Build a pc.yaml-equivalent config dict pointing watchdog at segment_dir."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "pc.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    if "capture" not in cfg or cfg["capture"] is None:
        cfg["capture"] = {}
    cfg["capture"]["storage_dirs"] = [segment_dir]
    if "index" not in cfg or cfg["index"] is None:
        cfg["index"] = {}
    # Note: backend is driven by pc.yaml — qdrant only
    cfg["index"].setdefault("embed_dim", 768)
    return cfg


# ---------------------------------------------------------------------------
# Session-state: daemon lifecycle
# ---------------------------------------------------------------------------

def _get_job_queue() -> PersistentJobQueue:
    if "job_queue" not in st.session_state:
        os.makedirs(_DEFAULT_SEGMENT_DIR, exist_ok=True)
        st.session_state["job_queue"] = PersistentJobQueue(db_path=_DEFAULT_DB_PATH)
    return st.session_state["job_queue"]


def _start_pipeline(source: str, cam_id: str, segment_dir: str) -> str:
    """
    Start capture daemon + ContinuousIndexer.
    Auto-selects backend:
      - Folder path (watch:// prefix or existing dir) → FolderWatcherDaemon
        (WSL2 bridge: windows_capture_server.py writes to a shared folder)
      - Network source (rtsp://) → OverlapCaptureDaemon (ffmpeg -c copy)
      - Local device             → OverlapCaptureDaemon if ffmpeg available,
                                   OpenCVCaptureDaemon otherwise
    Returns error string on failure, '' on success.
    """
    os.makedirs(segment_dir, exist_ok=True)
    # Remove 0-byte segment files left by previous failed capture attempts
    # and purge their stale job queue entries so the indexer doesn't retry them
    queue = _get_job_queue()
    for _f in Path(segment_dir).glob("*.mp4"):
        if _f.stat().st_size == 0:
            _f.unlink(missing_ok=True)
    queue.purge_missing_files()

    if "capture_daemon" not in st.session_state:
        # Clear any stale error from a previous pipeline run
        st.session_state.pop("last_capture_error", None)

        # Detect "watch folder" mode: explicit watch:// prefix OR a directory path
        _is_watch_folder = source.startswith("watch://") or (
            not source.lower().startswith(("rtsp://", "rtmp://", "http://"))
            and not source.startswith("/dev/")
            and not source.isdigit()
            and (Path(source).is_dir() or source.startswith("/mnt/"))
        )
        is_network = source.lower().startswith(("rtsp://", "rtmp://", "http://"))
        use_ffmpeg = check_ffmpeg()

        if is_network and not use_ffmpeg:
            return "ffmpeg is required for RTSP capture. Run: sudo apt install ffmpeg"

        try:
            if _is_watch_folder:
                watch_dir = source.removeprefix("watch://")
                daemon = FolderWatcherDaemon(
                    cam_id=cam_id,
                    watch_dir=watch_dir,
                    queue=queue,
                )
                st.session_state["capture_backend"] = "folder_watcher"
            elif use_ffmpeg:
                daemon = OverlapCaptureDaemon(
                    cam_id=cam_id,
                    rtsp_url=source,
                    output_dir=segment_dir,
                    queue=queue,
                )
                st.session_state["capture_backend"] = "ffmpeg"
            else:
                # OpenCV fallback (local devices only, no ffmpeg needed)
                daemon = OpenCVCaptureDaemon(
                    cam_id=cam_id,
                    device=source,
                    output_dir=segment_dir,
                    queue=queue,
                )
                st.session_state["capture_backend"] = "opencv"
            daemon.start()
            st.session_state["capture_daemon"]  = daemon
            st.session_state["capture_source"]  = source
            st.session_state["capture_cam_id"]  = cam_id
            st.session_state["capture_seg_dir"] = segment_dir
        except Exception as exc:
            return f"Failed to start capture daemon: {exc}"

    if "indexer" not in st.session_state:
        try:
            cfg     = _make_indexer_config(segment_dir)
            db_path = os.path.join(segment_dir, "job_queue.db")
            indexer = ContinuousIndexer(cfg, db_path=db_path)
            indexer.start()
            st.session_state["indexer"]        = indexer
            st.session_state["indexer_config"] = cfg
        except Exception as exc:
            return f"Failed to start indexer: {exc}"

    return ""


def _stop_pipeline() -> None:
    daemon = st.session_state.pop("capture_daemon", None)
    if daemon:
        # Preserve last_error so the UI can show it after the pipeline is stopped
        if getattr(daemon, "last_error", None):
            st.session_state["last_capture_error"] = daemon.last_error
        try:
            daemon.stop(timeout=5.0)
        except Exception:
            pass
    indexer = st.session_state.pop("indexer", None)
    if indexer:
        try:
            indexer.stop(timeout=10.0)
        except Exception:
            pass
    st.session_state.pop("searcher", None)


def _pipeline_running() -> bool:
    d = st.session_state.get("capture_daemon")
    return bool(d and d.is_running)


def _pipeline_status() -> dict:
    status: dict = {
        "capture_running": False,
        "indexer_running": False,
        "segments_captured": 0,
        "queue_pending": 0,
        "queue_done": 0,
        "vectors_indexed": 0,
        "source": st.session_state.get("capture_source", "—"),
        "cam_id": st.session_state.get("capture_cam_id", "—"),
        "seg_dir": st.session_state.get("capture_seg_dir", "—"),
    }
    daemon = st.session_state.get("capture_daemon")
    if daemon and daemon.is_running:
        status["capture_running"] = True
        seg_dir = Path(st.session_state.get("capture_seg_dir", ""))
        if seg_dir.exists():
            status["segments_captured"] = len(list(seg_dir.glob("*.mp4")))
    # last_error: prefer live daemon, fall back to preserved error after stop
    live_error = getattr(daemon, "last_error", None) if daemon else None
    status["capture_error"] = live_error or st.session_state.get("last_capture_error")

    indexer = st.session_state.get("indexer")
    if indexer is not None:
        status["indexer_running"] = True
        try:
            stats = indexer._queue.stats()
            status["queue_pending"] = stats.get("pending", 0)
            status["queue_done"]    = stats.get("done", 0)
        except Exception:
            pass
        try:
            if indexer._qdrant_client is not None:
                coll_info = indexer._qdrant_client.count(
                    collection_name=indexer._qdrant_collection,
                    exact=False,
                )
                status["vectors_indexed"] = coll_info.count
        except Exception:
            pass
    return status


# ---------------------------------------------------------------------------
# Session-state: searcher
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_searcher() -> NLVideoSearcher:
    """Load NLVideoSearcher once per process lifetime (survives Streamlit reruns)."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "pc.yaml")
    if os.path.isfile(cfg_path):
        return NLVideoSearcher.from_config(cfg_path)
    return NLVideoSearcher.from_params(
        window_sec=_WINDOW_SEC,
        overlap_ratio=_OVERLAP_RATIO,
        frames_per_window=_FRAMES_PER_WINDOW,
    )


def _get_searcher() -> NLVideoSearcher:
    return _load_searcher()


def _reset_searcher() -> None:
    """Reset Qdrant connection without destroying the cached model.

    Re-initialises Qdrant connection (useful after Qdrant restarts).
    The heavy EVA-CLIP model stays cached so there is no 25-second reload,
    and no Streamlit 1.32 ``expire_cache`` coroutine warning.
    """
    s = _load_searcher()
    idx_cfg = s._config.get("index", {})
    s._qdrant_client = None
    s._qdrant_collection = None
    s._init_qdrant(idx_cfg)
    for k in ("_results", "_query", "_elapsed", "_norm_note"):
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Sidebar — search / debug controls only
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("NLVS Settings")

    with st.expander("⚙️ Operational parameters", expanded=False):
        st.markdown(
            f"| Parameter | Value |\n"
            f"|---|---|\n"
            f"| `window_sec` | **{_WINDOW_SEC} s** |\n"
            f"| `overlap_ratio` | **{int(_OVERLAP_RATIO*100)}%** (stride=7.0 s) |\n"
            f"| `frames_per_window` | **{_FRAMES_PER_WINDOW}** |\n"
            f"| `min_score` | **{_SCORE_THRESHOLD}** |\n"
            f"| Engine | EVA-CLIP ViT-L/14 · embed=768 |\n"
            f"| Capture stride | 50 s (segment=60 s, overlap=10 s) |\n"
        )

    st.divider()
    st.subheader("Search Parameters")
    top_k = st.number_input("Top-K results", min_value=1, max_value=20, value=5)
    score_threshold = st.slider(
        "Min similarity score", 0.00, 0.60, _SCORE_THRESHOLD, 0.01,
    )
    use_templates = st.toggle("CLIP prompt templates", value=True)

    st.divider()
    st.subheader("Debug")
    show_debug = st.toggle("Show vector diagnostics", value=False)

    st.divider()
    with st.expander("📁 Index video file (forensics)", expanded=False):
        st.caption("For manual / one-off forensic analysis.")
        uploaded_files = st.file_uploader(
            "Select video files",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
        )
        build_btn = st.button("Build Index", type="primary", use_container_width=True)

    load_btn = st.button("🔄 Reconnect Qdrant", use_container_width=True)
    if load_btn:
        try:
            _reset_searcher()
            s = _get_searcher()
            info = s._qdrant_client.get_collection(s._qdrant_collection)
            n = info.points_count
            st.success(f"Qdrant reconnected. {n:,} vectors.")
        except Exception as exc:
            st.error(str(exc))

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📹 Natural Language Video Search")
st.caption(
    "Powered by **EVA-CLIP ViT-L/14** + Qdrant · NLVS v3.0  \n"
    "window=10 s · overlap=30% · stride=7.0 s · min\_score=0.10"
)

tab_camera, tab_search = st.tabs(["📷 Camera & Indexing", "🔍 Search"])

# ===========================================================================
# TAB 1 — Camera & Indexing
# ===========================================================================

with tab_camera:
    st.subheader("Camera Configuration")

    if "webcam_list" not in st.session_state:
        with st.spinner("Scanning for cameras…"):
            st.session_state["webcam_list"] = list_webcams()
    webcams = st.session_state["webcam_list"]

    # -----------------------------------------------------------------------
    # Environment warnings / guidance
    # -----------------------------------------------------------------------
    _in_wsl2 = is_wsl2()
    _has_ffmpeg = check_ffmpeg()

    if not _has_ffmpeg:
        st.warning(
            "**ffmpeg not found.** "
            "RTSP capture and segment-overlap mode require ffmpeg.  \n"
            "Install with: `sudo apt install ffmpeg`  \n"
            "Local webcam capture via OpenCV fallback is still available.",
            icon="⚠️",
        )

    if _in_wsl2 and not webcams:
        with st.expander("🔧 WSL2 Camera Setup — no /dev/video* devices detected", expanded=True):
            st.markdown(
                """
**WSL2 does not expose USB cameras by default.** You need to attach the
camera from the Windows host using **usbipd-win**.

#### Steps (run as Administrator in Windows PowerShell)

```powershell
# 1. Install usbipd-win (one-time)
winget install --interactive --exact dorssel.usbipd-win

# 2. List available USB devices and find your camera's Bus ID
usbipd list

# 3. Bind the camera (one-time, requires Admin)
usbipd bind --busid <BusID>

# 4. Attach to WSL2 (run each session, or set auto-attach)
usbipd attach --wsl --busid <BusID>
```

#### Verify in WSL2

```bash
ls /dev/video*     # should show /dev/video0 or similar
```

Then click **🔄 Refresh Camera List** below to detect the camera.
                """
            )
            if st.button("🔄 Refresh Camera List"):
                st.session_state.pop("webcam_list", None)
                st.rerun()

    # WSL2 isochronous USB warning (shown even when /dev/video0 exists)
    if _in_wsl2 and webcams:
        with st.expander(
            "⚠️ WSL2 isochronous USB limitation — V4L2 capture will fail",
            expanded=True,
        ):
            st.warning(
                "**WSL2's USB/IP virtual HCD does not support isochronous transfers.**  \n"
                "All consumer webcams (UVC) use isochronous endpoints for video — so "
                "`/dev/video0` is visible but **no frames can be read** (kernel returns "
                "`ECONNRESET -104` on every URB).  \n\n"
                "**Workaround — Windows capture bridge:**  \n"
                "1. Copy `windows_capture_server.py` to your Windows user folder.  \n"
                "2. In a Windows CMD/PowerShell (no Admin needed):  \n"
                "   ```\n"
                "   pip install opencv-python\n"
                "   python windows_capture_server.py --output C:\\Users\\<you>\\segments\n"
                "   ```\n"
                "3. In the **Camera source** box below, enter the WSL2 path to that folder, "
                "e.g. `/mnt/c/Users/<you>/segments`  \n"
                "4. Click ▶ Start Pipeline — the indexer will auto-detect completed segments.",
                icon="⚠️",
            )

    running = _pipeline_running()

    col_src, col_id = st.columns([3, 1])
    with col_src:
        if webcams:
            source_options = (
                webcams
                + ["Watch folder (Windows bridge)…"]
                + ["Custom RTSP / device…"]
            )
            sel = st.selectbox(
                "Camera source",
                options=source_options,
                disabled=running,
                help=(
                    "Local webcam devices detected on this machine.  "
                    "In WSL2 use 'Watch folder' for the Windows capture bridge."
                ),
            )
            if sel == "Custom RTSP / device…":
                camera_source = st.text_input(
                    "Enter RTSP URL or device path",
                    placeholder="rtsp://192.168.1.100:554/stream  or  /dev/video2",
                    disabled=running,
                )
            elif sel == "Watch folder (Windows bridge)…":
                camera_source = st.text_input(
                    "Enter folder path to watch",
                    placeholder="/mnt/c/Users/<you>/segments  (written by windows_capture_server.py)",
                    disabled=running,
                )
            else:
                camera_source = sel
        else:
            camera_source = st.text_input(
                "Camera source (no webcam auto-detected)",
                placeholder=(
                    "rtsp://192.168.1.100:554/stream  or  /dev/video0  or  "
                    "/mnt/c/Users/<you>/segments"
                ),
                disabled=running,
            )

    with col_id:
        cam_id = st.text_input(
            "Camera ID",
            value=st.session_state.get("capture_cam_id", "cam01"),
            disabled=running,
        )

    segment_dir = st.text_input(
        "Segment output directory",
        value=st.session_state.get("capture_seg_dir", _DEFAULT_SEGMENT_DIR),
        disabled=running,
        help="Captured .mp4 segments are written here; the indexer watches this directory.",
    )

    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 4])
    with btn_col1:
        start_btn = st.button(
            "▶ Start Pipeline",
            type="primary",
            use_container_width=True,
            disabled=running or not camera_source,
        )
    with btn_col2:
        stop_btn = st.button(
            "⏹ Stop Pipeline",
            use_container_width=True,
            disabled=not running,
        )
    with btn_col3:
        refresh_btn = st.button("🔄 Refresh Status", use_container_width=True)
        auto_refresh = st.toggle("Auto-refresh (5 s)", value=False)

    if start_btn and camera_source:
        err = _start_pipeline(camera_source.strip(), cam_id.strip(), segment_dir.strip())
        if err:
            st.error(err)
        else:
            st.success(
                f"Pipeline started — capturing from **{camera_source}** as `{cam_id}`.  \n"
                f"Segments → `{segment_dir}`"
            )
            st.rerun()

    if stop_btn:
        _stop_pipeline()
        st.info("Pipeline stopped. Index data is retained.")
        st.rerun()

    st.divider()

    # ── Live status ────────────────────────────────────────────────────────
    st.subheader("Pipeline Status")
    status = _pipeline_status()

    m1, m2, m3, m4 = st.columns(4)
    _backend = st.session_state.get("capture_backend", "ffmpeg")
    _backend_label = "ffmpeg" if _backend == "ffmpeg" else "OpenCV (no ffmpeg)"
    m1.metric(
        "Capture",
        "🟢 Running" if status["capture_running"] else "🔴 Stopped",
        delta=f"{_backend_label} · {status['source']}",
    )
    m2.metric(
        "Indexer",
        "🟢 Running" if status["indexer_running"] else "🔴 Stopped",
    )
    m3.metric("Segments captured", status["segments_captured"])
    m4.metric(
        "Queue",
        f"{status['queue_pending']} pending",
        delta=f"{status['queue_done']} done",
    )

    capture_err = status.get("capture_error")
    if capture_err:
        if "Permission denied" in capture_err or "permission denied" in capture_err:
            st.error(
                f"**Camera error:** {capture_err}\n\n"
                "**Fix:** Run in a terminal and then **log out and log back in**:\n"
                "```bash\nsudo usermod -aG video $USER\n```"
            )
        else:
            st.error(f"**Capture error:** {capture_err}")

    n_live = status["vectors_indexed"]
    if n_live > 0:
        st.success(f"**{n_live:,} vectors** in live index — search is available.")
    elif running:
        st.info(
            "Indexer is running. First segment will be searchable "
            "~60 s after capture starts (segment duration + encode time)."
        )
    else:
        st.warning(
            "Pipeline is not running.  \n"
            "Press **▶ Start Pipeline** to begin live indexing, "
            "or use **📁 Index video file** in the sidebar for a demo."
        )

    # ── Live camera preview ────────────────────────────────────────────────
    if status["capture_running"] and camera_source:
        st.divider()
        st.subheader("Live Preview")
        frame = _grab_webcam_frame(camera_source)
        if frame is not None:
            st.image(frame, caption="Live frame (refreshed on reload)", use_column_width=False, width=480)
        else:
            st.info("Preview unavailable — camera is in use by the capture process.")

    # ── Latest segment thumbnail ───────────────────────────────────────────
    seg_dir_path = Path(segment_dir)
    if seg_dir_path.exists():
        mp4s = sorted(seg_dir_path.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s:
            latest = mp4s[0]
            st.divider()
            st.subheader("Latest Captured Segment")
            c1, c2 = st.columns([1, 3])
            with c1:
                thumb = _extract_thumbnail(str(latest), 5.0)
                if thumb is not None:
                    st.image(thumb, caption=latest.name, use_column_width=True)
            with c2:
                size_mb = latest.stat().st_size / 1e6
                mtime   = datetime.datetime.fromtimestamp(latest.stat().st_mtime).strftime("%H:%M:%S")
                st.markdown(
                    f"**File:** `{latest.name}`  \n"
                    f"**Size:** {size_mb:.1f} MB  \n"
                    f"**Modified:** {mtime}"
                )

    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ===========================================================================
# TAB 2 — Search
# ===========================================================================

with tab_search:

    # ── Determine active index ─────────────────────────────────────────────
    n_vectors = 0
    cam_ids: List[str] = []

    indexer_ref = st.session_state.get("indexer")
    try:
        s = _get_searcher()
        # Qdrant backend: count from Qdrant directly
        if s._qdrant_client is not None:
                try:
                    col_info = s._qdrant_client.get_collection(s._qdrant_collection)
                    n_vectors = col_info.points_count or 0
                    # cam_ids from a scroll sample (up to 100 points, payload only)
                    scroll_result, _ = s._qdrant_client.scroll(
                        collection_name=s._qdrant_collection,
                        limit=100,
                        with_payload=True,
                        with_vectors=False,
                    )
                    cam_ids = sorted({
                        p.payload.get("cam_id", "unknown")
                        for p in scroll_result
                        if p.payload
                    })
                except Exception:
                    n_vectors = 0
        else:
            n_vectors = 0
            cam_ids   = []
    except Exception:
        pass

    if n_vectors > 0:
        cam_str = ", ".join(f"`{c}`" for c in cam_ids) if cam_ids else "unknown"
        st.success(
            f"**Index loaded** · {n_vectors:,} vectors · "
            f"{len(cam_ids)} camera(s): {cam_str}"
        )
    else:
        st.warning(
            "Index is empty. Start the camera pipeline in **📷 Camera & Indexing**, "
            "or use **📁 Index video file** in the sidebar."
        )

    indexed = n_vectors > 0

    # ── Forensic index build ───────────────────────────────────────────────
    if build_btn:
        if not uploaded_files:
            st.warning("Please upload at least one video file first.")
        else:
            _reset_searcher()
            s = _get_searcher()
            progress_bar = st.progress(0, text="Indexing …")
            for i, uf in enumerate(uploaded_files):
                suffix = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                with st.spinner(f"Processing **{uf.name}** …"):
                    s.index_video(tmp_path)
                os.unlink(tmp_path)
                progress_bar.progress(
                    (i + 1) / len(uploaded_files),
                    text=f"Indexed {uf.name}",
                )
            try:
                info = s._qdrant_client.get_collection(s._qdrant_collection)
                total = info.points_count or 0
            except Exception:
                total = 0
            st.success(f"Index updated — {total:,} total vectors from {len(uploaded_files)} file(s).")
            st.rerun()

    # ── Search ─────────────────────────────────────────────────────────────
    query = st.text_input(
        "Enter your search query",
        placeholder='"man climbing fence"  /  "người leo rào"  /  "red truck at gate"',
        disabled=not indexed,
    )
    search_btn = st.button("🔍 Search", type="primary", disabled=not indexed or not query)

    if search_btn and query:
        s = _get_searcher()

        cleaned = _normalize_query(query)
        st.session_state["_norm_note"] = (
            f'Query normalised: **"{query}"** → **"{cleaned}"**'
            if cleaned.lower() != query.strip().lower() else ""
        )
        with st.spinner("Searching …"):
            t0 = time.perf_counter()
            results: List[SearchResult] = s.search(
                query,
                top_k=int(top_k),
                score_threshold=score_threshold,
                use_templates=use_templates,
            )
            elapsed = time.perf_counter() - t0
        st.session_state["_results"] = results
        st.session_state["_query"]   = query
        st.session_state["_elapsed"] = elapsed

    # ── Results ────────────────────────────────────────────────────────────
    if "_results" in st.session_state:
        results    = st.session_state["_results"]
        elapsed    = st.session_state["_elapsed"]
        last_query = st.session_state["_query"]
        norm_note  = st.session_state.get("_norm_note", "")
        if norm_note:
            st.caption(norm_note)

        if show_debug and indexed:
            with st.expander("Vector diagnostics", expanded=True):
                try:
                    s = _get_searcher()
                    dbg = s.search_debug(last_query, top_k=10, score_threshold=0.0)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Score statistics**")
                        stats = dbg["score_stats"]
                        st.dataframe({
                            "Metric": ["max","p75","median","mean","min","adaptive_thresh"],
                            "Value":  [f"{stats[k]:.4f}"
                                       for k in ["max","p75","median","mean","min","adaptive_thresh"]],
                        }, hide_index=True)
                    with col_b:
                        st.markdown("**Top-5 raw (no NMS)**")
                        for row in dbg["top_raw_query"][:5]:
                            wall = _fmt_wall_time(row.get("absolute_start", 0))
                            st.write(
                                f"`{row['score']:.4f}` cam=`{row['cam_id']}` "
                                f"[{row['start']:.1f}s–{row['end']:.1f}s]"
                                + (f"  @{wall}" if wall else "")
                            )
                except Exception as exc:
                    st.warning(f"Debug unavailable: {exc}")

        if not results:
            st.info(
                f"No results above threshold **{score_threshold:.2f}**. "
                "Try lowering Min similarity score or rephrase the query."
            )
        else:
            st.write(f"**{len(results)} results** — {elapsed*1000:.1f} ms")
            st.divider()

            for res in results:
                col1, col2 = st.columns([1, 3])
                mid_sec       = (res.start_time + res.end_time) / 2.0
                video_available = os.path.isfile(res.video_path)

                with col1:
                    if video_available:
                        thumb = _extract_thumbnail(res.video_path, mid_sec)
                        if thumb is not None:
                            st.image(thumb, use_column_width=True, caption=f"{mid_sec:.1f}s")
                        else:
                            st.warning("Thumbnail unavailable")
                    else:
                        wall = _fmt_wall_time(res.absolute_start)
                        st.info(
                            (f"Video not in retention window.  \nEvent at: **{wall}**"
                             if wall else "Video unavailable.")
                        )

                with col2:
                    bar_pct = min(int(res.score / 0.5 * 100), 100)
                    st.progress(bar_pct / 100, text=f"Score: {res.score:.4f}")

                    wall_str = ""
                    if res.absolute_start > 0:
                        wall_str = (
                            f"  \n🕐 Wall-clock: "
                            f"`{_fmt_wall_time(res.absolute_start)}` → "
                            f"`{_fmt_wall_time(res.absolute_end)}`"
                        )

                    st.markdown(
                        f"**Rank {res.rank}** · Camera: `{res.cam_id}`  \n"
                        f"⏱ `{res.start_time:.2f}s` → `{res.end_time:.2f}s`"
                        f"  ({res.end_time - res.start_time:.1f}s)"
                        f"{wall_str}"
                    )
                    st.caption(f"File: `{res.video_path}`")

                    if show_debug:
                        n_frames = 4
                        frame_cols = st.columns(n_frames)
                        for fi, fc in enumerate(frame_cols):
                            t = res.start_time + (
                                (res.end_time - res.start_time) * fi / max(n_frames - 1, 1)
                            )
                            fr = _extract_thumbnail(res.video_path, t, 160, 90)
                            if fr is not None:
                                fc.image(fr, caption=f"{t:.1f}s", use_column_width=True)

                    if video_available:
                        try:
                            clip_bytes = _export_clip(res.video_path, res.start_time, res.end_time)
                            st.download_button(
                                label="⬇ Download clip",
                                data=clip_bytes,
                                file_name=f"{res.cam_id}_{res.start_time:.1f}-{res.end_time:.1f}.mp4",
                                mime="video/mp4",
                                key=f"dl_{res.rank}_{res.cam_id}_{res.start_time}",
                            )
                        except Exception as exc:
                            st.caption(f"Clip export unavailable: {exc}")
                    else:
                        st.caption("⚠️ Video outside retention window — download unavailable.")

                st.divider()
