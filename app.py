"""
app.py
------
Streamlit web demo for the Natural Language Video Search (NLVS) system.

Usage
-----
    streamlit run app.py

Workflow
--------
1. Upload video files OR load a pre-built index from disk (sidebar).
2. Type a natural language query (e.g. "person climbing a fence").
3. Browse ranked result clips with thumbnails and similarity scores.

Query tips
----------
* DO:   "man waving hand"  /  "red car turning"  /  "people sitting"
* SKIP: "find the …"  "show me …"  (stripped automatically)
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import List

import cv2
import numpy as np
import streamlit as st

from src.searcher import NLVideoSearcher, SearchResult, _normalize_query

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Natural Language Video Search",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Utility helpers  (defined FIRST so they're available everywhere below)
# ---------------------------------------------------------------------------

def _extract_thumbnail(video_path: str, timestamp_sec: float,
                       width: int = 320, height: int = 180):
    """
    Seek to *timestamp_sec* and return an RGB numpy array for st.image,
    or None if the seek / read fails.

    Uses MSEC seeking which is more reliable than frame-index seeking for
    videos with variable frame rate or large GOP size.
    """
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
    """
    Extract [start, end] seconds from *video_path* and return raw MP4 bytes
    using OpenCV (re-encode).  Uses MSEC seeking for reliability.
    """
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


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "index_store")
INDEX_DIR = os.path.join(tempfile.gettempdir(), "nlvs_index")


def _get_searcher() -> NLVideoSearcher:
    if "searcher" not in st.session_state:
        st.session_state["searcher"] = NLVideoSearcher.from_params(
            index_dir=st.session_state.get("index_dir", _DEFAULT_INDEX_DIR),
            use_sliding_window=st.session_state.get("use_sliding_window", True),
            window_sec=st.session_state.get("window_sec", 5.0),
            overlap_ratio=st.session_state.get("overlap_ratio", 0.5),
            frames_per_window=st.session_state.get("frames_per_window", 5),
        )
    return st.session_state["searcher"]


def _reset_searcher() -> None:
    """Force a new searcher instance (called when settings change)."""
    st.session_state.pop("searcher", None)
    st.session_state["indexed"] = False


# ---------------------------------------------------------------------------
# Sidebar – configuration & video upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("NLVS Settings")

    st.subheader("Indexing Strategy")
    use_sw = st.toggle(
        "Sliding Window",
        value=True,
        help="Recommended for action search (e.g. 'person climbing fence').",
        key="use_sliding_window",
        on_change=_reset_searcher,
    )

    window_sec = st.slider(
        "Window size (s)", 1.0, 15.0, 5.0, 0.5,
        key="window_sec", on_change=_reset_searcher,
    )
    overlap_pct = st.slider(
        "Overlap (%)", 0, 75, 50, 5,
        key="overlap_pct",
        help="25–50 % is recommended to avoid missing boundary events.",
        on_change=_reset_searcher,
    )
    st.session_state["overlap_ratio"] = overlap_pct / 100.0

    frames_per_window = st.slider(
        "Frames per window", 1, 10, 5, 1,
        key="frames_per_window",
        help="Multi-frame averaging: more frames = better accuracy, slower indexing.",
        on_change=_reset_searcher,
    )

    st.divider()

    # ── Search parameters ─────────────────────────────────────────────────
    st.subheader("Search Parameters")
    top_k = st.number_input("Top-K results", min_value=1, max_value=20, value=5)
    score_threshold = st.slider(
        "Min similarity score", 0.00, 0.60, 0.15, 0.01,
        help="Lower = more results. Recommended 0.15–0.25 for action queries.",
    )
    use_templates = st.toggle(
        "CLIP prompt templates",
        value=True,
        help="Encodes query through 5 CLIP-style templates and averages — "
             "improves recall for action / event queries.",
    )

    st.divider()

    st.subheader("Upload & Index Videos")
    uploaded_files = st.file_uploader(
        "Select video files",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
    )
    build_btn = st.button("Build Index", type="primary", use_container_width=True)

    st.divider()

    # ── Load pre-built index ───────────────────────────────────────────────
    st.subheader("Load Existing Index")
    load_index_dir = st.text_input(
        "Index directory",
        value=_DEFAULT_INDEX_DIR,
        help="Path to a folder containing faiss.index + metadata.pkl",
    )
    load_btn = st.button("Load Index from Disk", use_container_width=True)

    if load_btn:
        try:
            _reset_searcher()
            st.session_state["index_dir"] = load_index_dir
            searcher = _get_searcher()
            searcher.load_index(load_index_dir)
            n = searcher._index.total_vectors()
            if n == 0:
                st.warning("Index loaded but contains 0 vectors.")
            else:
                st.session_state["indexed"] = True
                st.success(f"Loaded {n} segments from disk.")
        except FileNotFoundError:
            st.error(f"No index found in:\n`{load_index_dir}`\nIndex the video first.")
        except Exception as exc:
            st.error(f"Load failed: {exc}")

    st.divider()
    st.subheader("Debug")
    show_debug = st.toggle(
        "Show vector diagnostics", value=False,
        help="Show query normalisation, score distribution, and top-N raw scores.",
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Natural Language Video Search")
st.caption("Powered by CLIP (ViT-B/16) + Faiss — open-vocabulary, any language")

# ---- Indexing -------------------------------------------------------------

if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one video file first.")
    else:
        _reset_searcher()
        searcher = _get_searcher()
        progress_bar = st.progress(0, text="Indexing …")
        tmp_paths: List[str] = []

        for i, uf in enumerate(uploaded_files):
            # Write upload to a temp file (Streamlit provides a BytesIO)
            suffix = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_paths.append(tmp.name)

            with st.spinner(f"Processing **{uf.name}** …"):
                n = searcher.index_video(tmp_paths[-1])

            progress_bar.progress(
                (i + 1) / len(uploaded_files),
                text=f"Indexed {uf.name} → {n} segments",
            )

        st.session_state["tmp_paths"] = tmp_paths
        st.session_state["indexed"]   = True
        st.success(
            f"Index ready — {searcher._index.total_vectors()} total segments "
            f"from {len(uploaded_files)} video(s)."
        )

# ── Search ─────────────────────────────────────────────────────────────────

indexed = st.session_state.get("indexed", False)

if not indexed:
    st.info(
        "No index loaded.  \n"
        "Upload a video and click **Build Index**, "
        "or use **Load Existing Index** in the sidebar."
    )

query = st.text_input(
    "Enter your search query",
    placeholder='e.g. "man waving hand"  /  "người leo rào"  /  "red truck"',
    disabled=not indexed,
    help='Prefix words like "find", "show me", "search for" are stripped automatically.',
)

search_btn = st.button("Search", type="primary", disabled=not indexed or not query)

if search_btn and query:
    searcher = _get_searcher()

    # Show normalised query so the user can see what CLIP actually encodes
    cleaned = _normalize_query(query)
    if cleaned.lower() != query.strip().lower():
        st.caption(f'Query normalised: **"{query}"** → **"{cleaned}"**')

    with st.spinner("Searching …"):
        t0 = time.perf_counter()
        results: List[SearchResult] = searcher.search(
            query,
            top_k=int(top_k),
            score_threshold=score_threshold,
            use_templates=use_templates,
        )
        elapsed = time.perf_counter() - t0

    # ── Debug panel ────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("Vector diagnostics", expanded=True):
            try:
                dbg = searcher.search_debug(query, top_k=10, score_threshold=0.0)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Score statistics (all segments)**")
                    stats = dbg["score_stats"]
                    st.dataframe({
                        "Metric": ["max", "p75", "median", "mean", "min"],
                        "Value":  [f"{stats[k]:.4f}"
                                   for k in ["max","p75","median","mean","min"]],
                    }, hide_index=True)
                    st.markdown("**Templates used**")
                    for t in dbg["templates_used"]:
                        st.code(t, language=None)
                with col_b:
                    st.markdown("**Top-5 raw scores (template ensemble, no NMS)**")
                    for row in dbg["top_templates"][:5]:
                        st.write(
                            f"`{row['score']:.4f}` — {row['video_id']} "
                            f"[{row['start']:.1f}s – {row['end']:.1f}s]"
                        )
                    st.markdown("**Top-5 raw scores (original query, no NMS)**")
                    for row in dbg["top_raw_query"][:5]:
                        st.write(
                            f"`{row['score']:.4f}` — {row['video_id']} "
                            f"[{row['start']:.1f}s – {row['end']:.1f}s]"
                        )
            except Exception as exc:
                st.warning(f"Debug info unavailable: {exc}")

    # ── Results ────────────────────────────────────────────────────────────
    if not results:
        st.info(
            f"No results above threshold **{score_threshold:.2f}**.  \n"
            "Try lowering **Min similarity score** in the sidebar, "
            "or rephrase as a scene description "
            "(e.g. *'man waving hand'* instead of *'find the man waving hand'*)."
        )
    else:
        st.write(f"**{len(results)} results** — query time: {elapsed*1000:.1f} ms")
        st.divider()

        for res in results:
            col1, col2 = st.columns([1, 3])

            # ── Thumbnail ──────────────────────────────────────────────────
            with col1:
                mid_sec = (res.start_time + res.end_time) / 2.0
                thumb = _extract_thumbnail(res.video_path, mid_sec)
                if thumb is not None:
                    st.image(thumb, use_column_width=True,
                             caption=f"{mid_sec:.1f}s")
                else:
                    st.warning(
                        f"Cannot read thumbnail  \n`{res.video_path}`  \n"
                        "(file moved or temp expired)"
                    )

            # ── Info + clip ────────────────────────────────────────────────
            with col2:
                bar_pct = min(int(res.score / 0.5 * 100), 100)
                st.progress(bar_pct / 100, text=f"Score: {res.score:.4f}")
                st.markdown(
                    f"**Rank {res.rank}** · Video: `{res.video_id}`  \n"
                    f"⏱ `{res.start_time:.2f}s` → `{res.end_time:.2f}s`"
                    f"  (duration: {res.end_time - res.start_time:.1f}s)"
                )

                # Frame strip in debug mode
                if show_debug:
                    n_frames = 4
                    frame_cols = st.columns(n_frames)
                    for fi, fc in enumerate(frame_cols):
                        t = res.start_time + (
                            (res.end_time - res.start_time) * fi / max(n_frames - 1, 1)
                        )
                        fr = _extract_thumbnail(res.video_path, t, 160, 90)
                        if fr is not None:
                            fc.image(fr, caption=f"{t:.1f}s",
                                     use_column_width=True)

                try:
                    clip_bytes = _export_clip(
                        res.video_path, res.start_time, res.end_time
                    )
                    st.download_button(
                        label="⬇ Download clip",
                        data=clip_bytes,
                        file_name=(
                            f"{res.video_id}_{res.start_time:.1f}"
                            f"-{res.end_time:.1f}.mp4"
                        ),
                        mime="video/mp4",
                        key=f"dl_{res.rank}_{res.video_id}_{res.start_time}",
                    )
                except Exception as exc:
                    st.caption(f"Clip export unavailable: {exc}")

            st.divider()
