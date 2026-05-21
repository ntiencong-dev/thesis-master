"""
continuous_indexer.py — Continuous Indexing Daemon v2.0

Architecture ref: CONTINUOUS_STREAM_RESEARCH.md §4.1–§4.6

Thread model (3 threads + main):
  1. watchdog  — inotify (watchdog lib) or polling: new .mp4 files → enqueue
  2. indexer   — dequeue → encode → store in Qdrant/Faiss
  3. persist   — periodic Faiss save to disk (Qdrant is persistent by design)

Vector store selection (CONTINUOUS_STREAM_RESEARCH.md §5):
  - config["index"]["backend"] == "qdrant" → Qdrant on-disk HNSW
  - anything else (or Qdrant connection failure) → Faiss VideoIndex fallback

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
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    from watchdog.observers import Observer as WatchdogObserver
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False
    logger.info("[ContinuousIndexer] watchdog not installed; falling back to polling.")

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from .indexer import SegmentMeta, VideoIndex
from .job_queue import CircuitBreaker, IndexJob, PersistentJobQueue
from .engines.factory import create_engine

_DEFAULT_DB_PATH    = "/opt/nlvs/job_queue.db"
_PERSIST_INTERVAL   = 300    # seconds between Faiss index saves
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
        self._faiss_index: Optional[VideoIndex] = None
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
        self._thread_persist  = threading.Thread(
            target=self._persist_loop, name="CI-Persist", daemon=True
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
        self._thread_persist.start()
        logger.info("[ContinuousIndexer] All threads started.")

    def stop(self, timeout: float = 15.0) -> None:
        """Signal all threads to stop and wait for them."""
        self._stop.set()
        for t in (self._thread_watchdog, self._thread_indexer, self._thread_persist):
            t.join(timeout=timeout)
        # Flush Faiss index one last time
        if self._faiss_index is not None:
            self._save_faiss()
        self._queue.close()
        logger.info("[ContinuousIndexer] Stopped.")

    # ------------------------------------------------------------------
    # Vector store initialization
    # ------------------------------------------------------------------

    def _init_vector_store(self) -> None:
        """
        Try Qdrant first; fall back to Faiss if unavailable or not configured.
        """
        idx_cfg   = self._config.get("index", {})
        backend   = idx_cfg.get("backend", "faiss").lower()
        embed_dim = idx_cfg.get("embed_dim", 768)
        index_dir = idx_cfg.get("index_dir", "./index_store")

        if backend == "qdrant":
            qdrant_url        = idx_cfg.get("qdrant_url", "http://localhost:6333")
            collection_name   = idx_cfg.get("qdrant_collection", "nlvs_segments")
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                client = QdrantClient(url=qdrant_url, timeout=10)
                # Create collection if it does not exist
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
                logger.info("[ContinuousIndexer] Qdrant backend ready at %s.", qdrant_url)
                return
            except Exception as exc:
                logger.warning(
                    "[ContinuousIndexer] Qdrant unavailable (%s). Falling back to Faiss.", exc
                )

        # Faiss fallback
        self._faiss_index = VideoIndex.load_or_create(
            index_dir=index_dir,
            embed_dim=embed_dim,
            use_gpu=False,   # background daemon — don't monopolise GPU
        )
        logger.info("[ContinuousIndexer] Faiss backend ready. index_dir=%s", index_dir)

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
        """Use watchdog library (inotify on Linux) to watch for new files."""
        indexer_ref = self   # closure

        class _Handler(FileSystemEventHandler):
            def on_created(self, event: FileCreatedEvent) -> None:
                if not isinstance(event, FileCreatedEvent):
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
                    capture_timestamp=time.time(),
                ))
                logger.info("[CI-Watchdog] Enqueued %s", path.name)

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
        """Fallback polling watchdog when watchdog library is not installed."""
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
                    seen.add(key)
                    if self._breaker.is_open():
                        logger.warning("[CI-Watchdog] CircuitBreaker OPEN — drop %s", f.name)
                        continue
                    cam_id = f.stem.split("_")[0]
                    self._queue.enqueue(IndexJob(
                        cam_id=cam_id,
                        segment_path=key,
                        capture_timestamp=f.stat().st_mtime,
                    ))
                    logger.info("[CI-Watchdog] Enqueued %s", f.name)

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
        """Store a single embedding+meta in Qdrant or Faiss."""
        import numpy as np
        vec = embedding.astype(np.float32).reshape(1, -1)

        if self._qdrant_client is not None:
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
        elif self._faiss_index is not None:
            self._faiss_index.add(vec, [meta])

    # ------------------------------------------------------------------
    # Persist thread (Faiss only)
    # ------------------------------------------------------------------

    def _persist_loop(self) -> None:
        """Periodically flush Faiss index to disk."""
        logger.info("[CI-Persist] Started (interval=%ds).", _PERSIST_INTERVAL)
        while not self._stop.is_set():
            self._stop.wait(timeout=_PERSIST_INTERVAL)
            if self._faiss_index is not None:
                self._save_faiss()
        logger.info("[CI-Persist] Stopped.")

    def _save_faiss(self) -> None:
        idx_dir = (self._config.get("index") or {}).get("index_dir", "./index_store")
        try:
            self._faiss_index.save(idx_dir)
            logger.info("[CI-Persist] Faiss index saved to %s.", idx_dir)
        except Exception as exc:
            logger.error("[CI-Persist] Faiss save failed: %s", exc)
