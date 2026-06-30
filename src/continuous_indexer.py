"""
continuous_indexer.py — Continuous Indexing Daemon v3.0

Architecture ref: CONTINUOUS_STREAM_RESEARCH.md §4.1–§4.6

Thread model (2 threads + main):
  1. watchdog  — inotify (watchdog lib) or polling: new .mp4 files → enqueue
  2. indexer   — dequeue → encode → store in Qdrant

Vector store: Qdrant (sole backend since v3.0). Faiss fallback removed.
If Qdrant is unavailable at startup, ContinuousIndexer raises RuntimeError.

Startup recovery:
  ContinuousIndexer.__init__ calls queue.replay_from_storage() with
  last_indexed_ts = now − 7200 (re-index last 2 h to survive short outages).

Usage::

    import yaml
    from src.continuous_indexer import ContinuousIndexer

    with open("config/kria.yaml") as f:
        cfg = yaml.safe_load(f)

    ci = ContinuousIndexer(cfg)
    ci.start()                   # non-blocking
    # ... application runs ...
    ci.stop()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional watchdog import (file-system event monitoring)
# ---------------------------------------------------------------------------
try:
    from watchdog.events import FileSystemEventHandler, FileClosedEvent
    from watchdog.observers import Observer as WatchdogObserver
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False
    logger.info("[ContinuousIndexer] watchdog not installed; falling back to polling.")

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from .indexer import SegmentMeta
from .job_queue import CircuitBreaker, IndexJob, PersistentJobQueue
from .engines.factory import create_engine

_DEFAULT_DB_PATH    = "/opt/nlvs/job_queue.db"
_POLL_INTERVAL      = 5.0    # seconds between polling cycles (fallback watchdog)
_DEQUEUE_SLEEP      = 2.0    # seconds to sleep when queue is empty


# ---------------------------------------------------------------------------
# ContinuousIndexer
# ---------------------------------------------------------------------------

class ContinuousIndexer:
    """
    Continuous segment indexing daemon.

    Parameters
    ----------
    config  : Dict loaded from kria.yaml / pc.yaml (or equivalent dict).
    db_path : Override SQLite DB path (default /opt/nlvs/job_queue.db).
    """

    def __init__(self, config: Dict[str, Any], db_path: str = _DEFAULT_DB_PATH) -> None:
        self._config    = config
        self._stop      = threading.Event()

        # --- Job queue + circuit breaker ---
        self._queue   = PersistentJobQueue(db_path=db_path)
        self._breaker = CircuitBreaker(self._queue)

        # --- Embedding engine ---
        self._engine = create_engine(config)

        # --- Vector store ---
        self._qdrant_client     = None
        self._qdrant_collection = None
        self._init_vector_store()

        # --- Startup recovery ---
        storage_dirs = (config.get("capture") or {}).get("storage_dirs", [])
        if storage_dirs:
            self._queue.replay_from_storage(
                storage_dirs,
                last_indexed_ts=time.time() - 7200,
            )

        # --- Background threads ---
        self._thread_watchdog = threading.Thread(
            target=self._watchdog_loop, name="CI-Watchdog", daemon=True
        )
        self._thread_indexer  = threading.Thread(
            target=self._indexer_loop, name="CI-Indexer", daemon=True
        )

        logger.info("[ContinuousIndexer] Initialized.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all background threads (non-blocking)."""
        self._stop.clear()
        self._thread_watchdog.start()
        self._thread_indexer.start()
        logger.info("[ContinuousIndexer] All threads started.")

    def stop(self, timeout: float = 15.0) -> None:
        """Signal all threads to stop and wait for them."""
        self._stop.set()
        for t in (self._thread_watchdog, self._thread_indexer):
            t.join(timeout=timeout)
        self._queue.close()
        logger.info("[ContinuousIndexer] Stopped.")

    # ------------------------------------------------------------------
    # Vector store initialization
    # ------------------------------------------------------------------

    def _init_vector_store(self) -> None:
        """
        Connect to Qdrant and create the collection if absent.
        Raises RuntimeError if Qdrant is unavailable (no Faiss fallback).
        """
        idx_cfg         = self._config.get("index", {})
        embed_dim       = idx_cfg.get("embed_dim", 256)
        qdrant_path     = idx_cfg.get("qdrant_path",       "./local_qdrant_db")
        collection_name = idx_cfg.get("qdrant_collection", "nlvs_segments_blip1")
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(path=qdrant_path)
            existing = {c.name for c in client.get_collections().collections}
            if collection_name not in existing:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
                )
                logger.info("[ContinuousIndexer] Created Qdrant collection '%s'.", collection_name)
            else:
                logger.info("[ContinuousIndexer] Using existing Qdrant collection '%s'.", collection_name)

            self._qdrant_client     = client
            self._qdrant_collection = collection_name
            logger.info("[ContinuousIndexer] Qdrant backend ready at %s.", qdrant_path)
        except Exception as exc:
            raise RuntimeError(
                f"[ContinuousIndexer] Qdrant unavailable at {qdrant_path}: {exc}. "
                "Check folder permissions."
            ) from exc

    # ------------------------------------------------------------------
    # Watchdog thread
    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        """Monitor storage directories for new .mp4 files and enqueue them."""
        storage_dirs: List[str] = (
            (self._config.get("capture") or {}).get("storage_dirs", [])
        )
        if not storage_dirs:
            logger.warning("[CI-Watchdog] No storage_dirs configured.")
            return

        if _HAS_WATCHDOG:
            self._run_inotify_watchdog(storage_dirs)
        else:
            self._run_polling_watchdog(storage_dirs)

    def _run_inotify_watchdog(self, storage_dirs: List[str]) -> None:
        """Use watchdog library (inotify on Linux) to watch for new files.

        Uses IN_CLOSE_WRITE (on_closed) so the file is guaranteed to be fully
        written before being enqueued — avoids 'moov atom not found' errors
        that occur when on_created fires while ffmpeg is still writing.
        """
        indexer_ref = self   # closure

        class _Handler(FileSystemEventHandler):
            def on_closed(self, event: FileClosedEvent) -> None:
                if not isinstance(event, FileClosedEvent):
                    return
                path = Path(event.src_path)
                if path.suffix.lower() != ".mp4":
                    return
                if indexer_ref._breaker.is_open():
                    logger.warning("[CI-Watchdog] CircuitBreaker OPEN — drop %s", path.name)
                    return
                cam_id = path.stem.split("_")[0]
                indexer_ref._queue.enqueue(IndexJob(
                    cam_id=cam_id,
                    segment_path=str(path),
                    capture_timestamp=path.stat().st_mtime,
                ))
                logger.info("[CI-Watchdog] Enqueued (closed) %s", path.name)

        observer = WatchdogObserver()
        for d in storage_dirs:
            observer.schedule(_Handler(), path=d, recursive=False)
        observer.start()
        logger.info("[CI-Watchdog] inotify watching: %s", storage_dirs)

        while not self._stop.is_set():
            self._stop.wait(timeout=1.0)

        observer.stop()
        observer.join()
        logger.info("[CI-Watchdog] inotify stopped.")

    def _run_polling_watchdog(self, storage_dirs: List[str]) -> None:
        """Fallback polling watchdog when watchdog library is not installed.

        Waits for file size to stabilise for two consecutive poll cycles before
        enqueuing, so partially-written MP4s are never indexed.
        """
        # Maps path → last observed size (for stability check)
        size_cache: dict[str, int] = {}
        seen: set[str] = set()
        logger.info("[CI-Watchdog] polling %s every %.1fs", storage_dirs, _POLL_INTERVAL)

        while not self._stop.is_set():
            for d in storage_dirs:
                p = Path(d)
                if not p.exists():
                    continue
                for f in p.glob("*.mp4"):
                    key = str(f)
                    if key in seen:
                        continue
                    try:
                        size = f.stat().st_size
                    except OSError:
                        size_cache.pop(key, None)
                        continue
                    if size == 0:
                        continue
                    prev = size_cache.get(key, -1)
                    if prev == size:
                        # Size stable since last poll — file is complete
                        seen.add(key)
                        size_cache.pop(key, None)
                        if self._breaker.is_open():
                            logger.warning("[CI-Watchdog] CircuitBreaker OPEN — drop %s", f.name)
                            continue
                        cam_id = f.stem.split("_")[0]
                        self._queue.enqueue(IndexJob(
                            cam_id=cam_id,
                            segment_path=key,
                            capture_timestamp=f.stat().st_mtime,
                        ))
                        logger.info("[CI-Watchdog] Enqueued (stable) %s", f.name)
                    else:
                        size_cache[key] = size

            self._stop.wait(timeout=_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Indexer thread
    # ------------------------------------------------------------------

    def _indexer_loop(self) -> None:
        """Dequeue jobs and index each segment."""
        logger.info("[CI-Indexer] Started.")
        while not self._stop.is_set():
            job = self._queue.dequeue()
            if job is None:
                self._stop.wait(timeout=_DEQUEUE_SLEEP)
                continue
            try:
                self._process_segment(job)
                self._queue.mark_done(job.id)
            except FileNotFoundError:
                # Segment file was deleted after enqueue — no point retrying
                logger.warning(
                    "[CI-Indexer] Segment file missing, skipping (no retry): %s",
                    job.segment_path,
                )
                self._queue.mark_dead(job.id, reason="FileNotFoundError")
            except Exception as exc:
                logger.exception("[CI-Indexer] Error processing %s: %s", job.segment_path, exc)
                self._queue.mark_failed(job.id)
        logger.info("[CI-Indexer] Stopped.")

    def _process_segment(self, job: IndexJob) -> None:
        """
        Index an entire 60s segment file.

        Extracts sliding windows at (window=10s, overlap=30%) and stores
        each window embedding with a SegmentMeta v2.0 payload.
        """
        seg_path  = job.segment_path
        wall_ts   = job.capture_timestamp

        pipeline_cfg = self._config.get("pipeline", {})
        window_sec   = pipeline_cfg.get("window_sec",   10.0)
        overlap_ratio= pipeline_cfg.get("overlap_ratio",  0.30)
        frames_pw    = (self._config.get("engine") or {}).get("frames_per_window", 5)
        stride_sec   = window_sec * (1.0 - overlap_ratio)

        from .gst_pipeline import VideoPipeline
        pipe = VideoPipeline(
            video_path=seg_path,
            window_sec=window_sec,
            stride_sec=stride_sec,
            frames_per_window=frames_pw,
            backend=pipeline_cfg.get("video_backend", "opencv"),
        )

        count = 0
        for t0, t1, frames in pipe.iter_segments():
            if not frames:
                continue
            emb  = self._engine.encode_segment_frames(frames)
            meta = SegmentMeta(
                cam_id=job.cam_id,
                video_path=seg_path,
                relative_start=t0,
                relative_end=t1,
                segment_wall_start=wall_ts,
                absolute_start=wall_ts + t0,
                absolute_end=wall_ts + t1,
            )
            self._store_vector(emb, meta)
            count += 1

        logger.info("[CI-Indexer] %s → %d windows indexed.", Path(seg_path).name, count)

    def _store_vector(self, embedding, meta: SegmentMeta) -> None:
        """Store a single embedding+meta in Qdrant."""
        import numpy as np
        vec = embedding.astype(np.float32).reshape(1, -1)

        from qdrant_client.models import PointStruct
        import uuid
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vec[0].tolist(),
            payload={
                "cam_id":             meta.cam_id,
                "video_path":         meta.video_path,
                "relative_start":     meta.relative_start,
                "relative_end":       meta.relative_end,
                "segment_wall_start": meta.segment_wall_start,
                "absolute_start":     meta.absolute_start,
                "absolute_end":       meta.absolute_end,
            },
        )
        self._qdrant_client.upsert(
            collection_name=self._qdrant_collection,
            points=[point],
        )


