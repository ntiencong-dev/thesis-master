"""
job_queue.py — Persistent SQLite-backed job queue + CircuitBreaker (v2.0)

Architecture ref: CONTINUOUS_STREAM_RESEARCH.md §4.5 PersistentJobQueue, §4.6 CircuitBreaker

Design goals:
  - Survive process crashes: all state lives in SQLite (WAL mode)
  - Idempotent enqueue: UNIQUE on segment_path, duplicate inserts are silently ignored
  - Startup recovery: jobs stuck in 'processing' > PROCESSING_TIMEOUT seconds are
    automatically reset to 'pending' on __init__
  - Exponential backoff: retry_delay = BACKOFF_BASE × 2^(retry_count - 1)
  - Dead-letter: jobs that exceed MAX_RETRIES transition to status='dead'
  - CircuitBreaker: open (stop accepting jobs) when queue depth ≥ OPEN_THRESHOLD;
    close (resume) when depth ≤ CLOSE_THRESHOLD
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH      = "/opt/nlvs/job_queue.db"
MAX_RETRIES           = 3          # jobs exceeding this become 'dead'
PROCESSING_TIMEOUT    = 300        # seconds before stuck-processing job is reset
BACKOFF_BASE          = 30         # seconds; delay = BACKOFF_BASE × 2^(retry-1)

# CircuitBreaker thresholds
OPEN_THRESHOLD        = 20         # queue depth at which circuit opens (drops new enqueues)
CLOSE_THRESHOLD       = 5          # queue depth at which circuit closes (resumes enqueues)


# ---------------------------------------------------------------------------
# IndexJob dataclass
# ---------------------------------------------------------------------------

@dataclass
class IndexJob:
    """Represents a single segment-indexing work item."""
    cam_id:            str
    segment_path:      str
    capture_timestamp: float             # wall-clock unix time when segment was written
    id:                Optional[int] = None
    retry_count:       int = 0
    status:            str = "pending"   # pending | processing | done | failed | dead
    created_at:        float = field(default_factory=time.time)
    next_retry_after:  float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# PersistentJobQueue
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    cam_id             TEXT    NOT NULL,
    segment_path       TEXT    NOT NULL UNIQUE,
    capture_timestamp  REAL    NOT NULL,
    status             TEXT    NOT NULL DEFAULT 'pending',
    retry_count        INTEGER NOT NULL DEFAULT 0,
    created_at         REAL    NOT NULL,
    next_retry_after   REAL    NOT NULL,
    updated_at         REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_retry   ON jobs(status, next_retry_after);
"""


class PersistentJobQueue:
    """
    Thread-safe, SQLite-backed persistent job queue.

    All public methods acquire ``_lock`` so callers on different threads
    (watchdog, indexer, persist) can share a single queue instance safely.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._lock    = threading.Lock()
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_connection()
        self._init_db()
        self._recover_stuck_jobs()
        logger.info("[JobQueue] Initialized. DB: %s  pending=%d", db_path, self.depth())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _recover_stuck_jobs(self) -> None:
        """Reset jobs stuck in 'processing' longer than PROCESSING_TIMEOUT."""
        cutoff = time.time() - PROCESSING_TIMEOUT
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='pending', updated_at=? "
                "WHERE status='processing' AND updated_at < ?",
                (time.time(), cutoff),
            )
            self._conn.commit()
            if cur.rowcount:
                logger.warning("[JobQueue] Recovered %d stuck jobs.", cur.rowcount)

    def _backoff_delay(self, retry_count: int) -> float:
        """Return backoff delay in seconds: BACKOFF_BASE × 2^(retry-1) (capped at 3600s)."""
        if retry_count <= 0:
            return 0.0
        return min(BACKOFF_BASE * (2 ** (retry_count - 1)), 3600.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, job: IndexJob) -> bool:
        """
        Insert a new job.  Returns True if inserted, False if segment_path
        already exists (idempotent — duplicate silently ignored).
        """
        now = time.time()
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO jobs "
                    "(cam_id, segment_path, capture_timestamp, status, "
                    " retry_count, created_at, next_retry_after, updated_at) "
                    "VALUES (?, ?, ?, 'pending', 0, ?, ?, ?)",
                    (job.cam_id, job.segment_path, job.capture_timestamp, now, now, now),
                )
                self._conn.commit()
                return True
        except sqlite3.Error as exc:
            logger.error("[JobQueue] enqueue error: %s", exc)
            return False

    def dequeue(self) -> Optional[IndexJob]:
        """
        Atomically claim the oldest pending job whose next_retry_after ≤ now.
        Returns None if the queue is empty or no job is ready yet.
        """
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs "
                "WHERE status='pending' AND next_retry_after <= ? "
                "ORDER BY created_at ASC LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE jobs SET status='processing', updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            self._conn.commit()
            return IndexJob(
                id=row["id"],
                cam_id=row["cam_id"],
                segment_path=row["segment_path"],
                capture_timestamp=row["capture_timestamp"],
                retry_count=row["retry_count"],
                status="processing",
                created_at=row["created_at"],
                next_retry_after=row["next_retry_after"],
            )

    def mark_done(self, job_id: int) -> None:
        """Mark a job as successfully completed."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='done', updated_at=? WHERE id=?",
                (time.time(), job_id),
            )
            self._conn.commit()

    def mark_dead(self, job_id: int, reason: str = "") -> None:
        """Immediately move a job to dead-letter without consuming a retry slot."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='dead', updated_at=? WHERE id=?",
                (now, job_id),
            )
            self._conn.commit()
        logger.warning("[JobQueue] Job %d marked dead. %s", job_id, reason)

    def purge_missing_files(self) -> int:
        """
        Mark as 'dead' any pending/processing jobs whose segment_path no longer
        exists on disk.  Returns the number of jobs purged.
        Called on pipeline start-up after 0-byte files are removed.
        """
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, segment_path FROM jobs "
                "WHERE status IN ('pending', 'processing')"
            ).fetchall()
            purged = 0
            for row in rows:
                if not os.path.exists(row["segment_path"]):
                    self._conn.execute(
                        "UPDATE jobs SET status='dead', updated_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                    logger.info(
                        "[JobQueue] Purged missing-file job: %s", row["segment_path"]
                    )
                    purged += 1
            self._conn.commit()
        return purged

    def mark_failed(self, job_id: int) -> None:
        """
        Increment retry_count and either schedule a retry (status='pending')
        or transition to 'dead' if MAX_RETRIES is exceeded.
        """
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT retry_count FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                return
            new_retry = row["retry_count"] + 1
            if new_retry > MAX_RETRIES:
                self._conn.execute(
                    "UPDATE jobs SET status='dead', retry_count=?, updated_at=? WHERE id=?",
                    (new_retry, now, job_id),
                )
                logger.warning("[JobQueue] Job %d moved to dead-letter (retries=%d).", job_id, new_retry)
            else:
                delay = self._backoff_delay(new_retry)
                self._conn.execute(
                    "UPDATE jobs SET status='pending', retry_count=?, "
                    "next_retry_after=?, updated_at=? WHERE id=?",
                    (new_retry, now + delay, now, job_id),
                )
                logger.info("[JobQueue] Job %d retry %d/%d in %.0fs.", job_id, new_retry, MAX_RETRIES, delay)
            self._conn.commit()

    def depth(self) -> int:
        """Return number of jobs in 'pending' or 'processing' status."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('pending','processing')"
        ).fetchone()
        return row["n"] if row else 0

    def replay_from_storage(
        self,
        storage_dirs: list[str],
        last_indexed_ts: float = 0.0,
        glob_pattern: str = "*.mp4",
    ) -> int:
        """
        Scan ``storage_dirs`` for segment files newer than ``last_indexed_ts``
        (default: re-index everything from the past 2 h) and enqueue any that
        are not already in the DB.  Returns the number of newly-enqueued jobs.
        """
        count = 0
        for storage_dir in storage_dirs:
            p = Path(storage_dir)
            if not p.exists():
                logger.warning("[JobQueue] replay_from_storage: dir not found: %s", storage_dir)
                continue
            for seg_file in sorted(p.glob(glob_pattern)):
                mtime = seg_file.stat().st_mtime
                if mtime < last_indexed_ts:
                    continue
                # Derive cam_id from the first token before '_' in the filename
                cam_id = seg_file.stem.split("_")[0]
                job = IndexJob(
                    cam_id=cam_id,
                    segment_path=str(seg_file),
                    capture_timestamp=mtime,
                )
                if self.enqueue(job):
                    count += 1
        logger.info("[JobQueue] replay_from_storage: enqueued %d new jobs.", count)
        return count

    def stats(self) -> dict:
        """
        Return a dict of job counts by status:
        ``{"pending": N, "processing": N, "done": N, "failed": N, "dead": N}``
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0, "dead": 0}
        for row in rows:
            counts[row["status"]] = row["n"]
        return counts

    def close(self) -> None:
        """Close the SQLite connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Simple queue-depth-based circuit breaker.

    State machine:
      CLOSED  →  (depth ≥ OPEN_THRESHOLD)  →  OPEN
      OPEN    →  (depth ≤ CLOSE_THRESHOLD) →  CLOSED

    Usage::

        cb = CircuitBreaker(queue)
        if cb.is_open():
            # skip enqueue — indexer is falling behind
            pass
    """

    def __init__(
        self,
        queue: PersistentJobQueue,
        open_threshold:  int = OPEN_THRESHOLD,
        close_threshold: int = CLOSE_THRESHOLD,
    ) -> None:
        self._queue           = queue
        self._open_threshold  = open_threshold
        self._close_threshold = close_threshold
        self._open            = False

    # Public -----------------------------------------------------------------

    def is_open(self) -> bool:
        """
        Returns True if the circuit is open (new jobs should NOT be enqueued).
        Updates state based on current queue depth.
        """
        depth = self._queue.depth()
        if not self._open and depth >= self._open_threshold:
            self._open = True
            logger.warning(
                "[CircuitBreaker] OPEN — queue depth %d ≥ %d. "
                "New segments will be dropped until depth ≤ %d.",
                depth, self._open_threshold, self._close_threshold,
            )
        elif self._open and depth <= self._close_threshold:
            self._open = False
            logger.info(
                "[CircuitBreaker] CLOSED — queue depth %d ≤ %d. Resuming.",
                depth, self._close_threshold,
            )
        return self._open

    def is_closed(self) -> bool:
        return not self.is_open()
