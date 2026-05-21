"""
tests/unit/test_continuous_indexer.py
--------------------------------------
Unit tests for ContinuousIndexer vector-store selection and _store_vector().

All tests mock external dependencies (Qdrant, CLIP engine) so they run in
milliseconds without a GPU or a running Qdrant instance.

Scenarios
---------
CI-01  backend=qdrant, Qdrant reachable  → _qdrant_client is set, _faiss_index is None
CI-02  backend=qdrant, Qdrant unreachable → fallback to Faiss (_qdrant_client is None)
CI-03  backend=faiss (explicit)          → _qdrant_client is None, _faiss_index is set
CI-04  _store_vector() dispatches to qdrant_client.upsert() when Qdrant configured
CI-05  _store_vector() dispatches to faiss_index.add()     when Qdrant is None
CI-06  _init_vector_store() creates Qdrant collection if it doesn't exist
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_config(backend: str = "qdrant") -> dict:
    return {
        "capture": {
            "storage_dirs": ["/tmp/segments"],
        },
        "index": {
            "backend": backend,
            "embed_dim": 768,
            "index_dir": "/tmp/idx",
            "qdrant_url": "http://localhost:6333",
            "qdrant_collection": "nlvs_segments",
        },
        "job_queue": {"db_path": "/tmp/jq.db"},
        "engine": {"type": "pc"},
    }


def _make_mock_engine() -> MagicMock:
    eng = MagicMock()
    eng.embed_dim = 768
    eng.encode_segment_frames.return_value = np.zeros((1, 768), dtype=np.float32)
    return eng


# ---------------------------------------------------------------------------
# CI-01  Qdrant reachable → _qdrant_client set, _faiss_index None
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI01_qdrant_backend_sets_client_when_reachable():
    """When backend=qdrant and Qdrant is reachable, _qdrant_client must be set."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    with patch("src.continuous_indexer.create_engine", return_value=_make_mock_engine()), \
         patch("src.continuous_indexer.PersistentJobQueue"), \
         patch("qdrant_client.QdrantClient", return_value=mock_client):

        from src.continuous_indexer import ContinuousIndexer
        ci = ContinuousIndexer.__new__(ContinuousIndexer)
        ci._config  = _make_base_config("qdrant")
        ci._engine  = _make_mock_engine()
        ci._qdrant_client     = None
        ci._qdrant_collection = None
        ci._faiss_index       = None
        ci._init_vector_store()

    assert ci._qdrant_client is not None
    assert ci._faiss_index is None


# ---------------------------------------------------------------------------
# CI-02  Qdrant unreachable → fallback Faiss
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI02_qdrant_backend_fallback_when_unreachable():
    """When Qdrant is unreachable, ContinuousIndexer must fall back to Faiss."""
    with patch("src.continuous_indexer.create_engine", return_value=_make_mock_engine()), \
         patch("src.continuous_indexer.PersistentJobQueue"), \
         patch("qdrant_client.QdrantClient", side_effect=Exception("connection refused")):

        from src.continuous_indexer import ContinuousIndexer, VideoIndex
        ci = ContinuousIndexer.__new__(ContinuousIndexer)
        ci._config  = _make_base_config("qdrant")
        ci._engine  = _make_mock_engine()
        ci._qdrant_client     = None
        ci._qdrant_collection = None
        ci._faiss_index       = None
        ci._init_vector_store()

    assert ci._qdrant_client is None
    assert ci._faiss_index is not None


# ---------------------------------------------------------------------------
# CI-03  backend=faiss → always uses Faiss, never tries Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI03_faiss_backend_skips_qdrant():
    """When backend=faiss, _init_vector_store() must not attempt a Qdrant connection."""
    qdrant_constructor = MagicMock()

    with patch("src.continuous_indexer.create_engine", return_value=_make_mock_engine()), \
         patch("src.continuous_indexer.PersistentJobQueue"), \
         patch("qdrant_client.QdrantClient", qdrant_constructor):

        from src.continuous_indexer import ContinuousIndexer
        ci = ContinuousIndexer.__new__(ContinuousIndexer)
        ci._config  = _make_base_config("faiss")
        ci._engine  = _make_mock_engine()
        ci._qdrant_client     = None
        ci._qdrant_collection = None
        ci._faiss_index       = None
        ci._init_vector_store()

    qdrant_constructor.assert_not_called()
    assert ci._qdrant_client is None
    assert ci._faiss_index is not None


# ---------------------------------------------------------------------------
# CI-04  _store_vector dispatches to qdrant_client.upsert()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI04_store_vector_calls_qdrant_upsert():
    """_store_vector() must call qdrant_client.upsert() when Qdrant is configured."""
    from src.continuous_indexer import ContinuousIndexer
    from src.indexer import SegmentMeta

    mock_client = MagicMock()
    embedding   = np.zeros((768,), dtype=np.float32)
    meta = SegmentMeta(
        cam_id="cam0",
        video_path="/tmp/seg.mp4",
        relative_start=0.0,
        relative_end=10.0,
        segment_wall_start=0.0,
        absolute_start=0.0,
        absolute_end=10.0,
    )

    ci = ContinuousIndexer.__new__(ContinuousIndexer)
    ci._qdrant_client     = mock_client
    ci._qdrant_collection = "nlvs_segments"
    ci._faiss_index       = None
    ci._store_vector(embedding, meta)

    mock_client.upsert.assert_called_once()
    call_kwargs = mock_client.upsert.call_args
    assert call_kwargs.kwargs.get("collection_name") == "nlvs_segments" or \
           (call_kwargs.args and call_kwargs.args[0] == "nlvs_segments")


# ---------------------------------------------------------------------------
# CI-05  _store_vector dispatches to faiss_index.add()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI05_store_vector_calls_faiss_add():
    """_store_vector() must call faiss_index.add() when Qdrant is not configured."""
    from src.continuous_indexer import ContinuousIndexer
    from src.indexer import SegmentMeta

    mock_faiss = MagicMock()
    embedding  = np.zeros((768,), dtype=np.float32)
    meta = SegmentMeta(
        cam_id="cam0",
        video_path="/tmp/seg.mp4",
        relative_start=0.0,
        relative_end=10.0,
        segment_wall_start=0.0,
        absolute_start=0.0,
        absolute_end=10.0,
    )

    ci = ContinuousIndexer.__new__(ContinuousIndexer)
    ci._qdrant_client     = None
    ci._qdrant_collection = None
    ci._faiss_index       = mock_faiss
    ci._store_vector(embedding, meta)

    mock_faiss.add.assert_called_once()


# ---------------------------------------------------------------------------
# CI-06  _init_vector_store creates Qdrant collection when absent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI06_creates_qdrant_collection_when_absent():
    """_init_vector_store() must create the collection if it does not exist."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []  # empty

    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        from src.continuous_indexer import ContinuousIndexer
        ci = ContinuousIndexer.__new__(ContinuousIndexer)
        ci._config  = _make_base_config("qdrant")
        ci._engine  = _make_mock_engine()
        ci._qdrant_client     = None
        ci._qdrant_collection = None
        ci._faiss_index       = None
        ci._init_vector_store()

    mock_client.create_collection.assert_called_once()


# ---------------------------------------------------------------------------
# CI-07  _indexer_loop: FileNotFoundError → mark_dead (no retry)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI07_file_not_found_marks_dead_not_failed():
    """
    When _process_segment raises FileNotFoundError, _indexer_loop must call
    mark_dead() (not mark_failed()), so the job is not retried.
    """
    import threading
    from src.continuous_indexer import ContinuousIndexer
    from src.job_queue import IndexJob

    mock_job = IndexJob(
        id=99,
        cam_id="cam0",
        segment_path="/tmp/does_not_exist.mp4",
        capture_timestamp=0.0,
    )

    mock_queue = MagicMock()
    # dequeue returns job once, then None to break the loop
    mock_queue.dequeue.side_effect = [mock_job, None]

    ci = ContinuousIndexer.__new__(ContinuousIndexer)
    ci._queue  = mock_queue
    ci._stop   = threading.Event()
    ci._config = {}
    ci._engine = _make_mock_engine()
    ci._qdrant_client = None
    ci._faiss_index   = None

    with patch.object(ci, "_process_segment", side_effect=FileNotFoundError("missing")):
        # Run loop once — the second dequeue returns None and _stop is not set,
        # so we set the stop event after the job is processed
        stop_after = threading.Event()

        original_dequeue = mock_queue.dequeue.side_effect

        call_count = 0
        def dequeue_and_maybe_stop():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_job
            ci._stop.set()
            return None

        mock_queue.dequeue.side_effect = dequeue_and_maybe_stop
        ci._indexer_loop()

    mock_queue.mark_dead.assert_called_once_with(99, reason="FileNotFoundError")
    mock_queue.mark_failed.assert_not_called()
    mock_queue.mark_done.assert_not_called()


# ---------------------------------------------------------------------------
# CI-08  purge_missing_files: stale pending jobs → status='dead'
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_CI08_purge_missing_files_marks_stale_jobs_dead(tmp_path):
    """
    purge_missing_files() must set status='dead' for any pending job whose
    segment_path no longer exists on disk, and leave existing files untouched.
    """
    from src.job_queue import PersistentJobQueue, IndexJob

    db_path = str(tmp_path / "test_jq.db")
    seg_dir = str(tmp_path / "segs")
    import os; os.makedirs(seg_dir)

    # Create a real file and a ghost path
    real_file = tmp_path / "segs" / "real.mp4"
    real_file.write_bytes(b"\x00" * 100)
    ghost_path = str(tmp_path / "segs" / "ghost.mp4")  # never created on disk

    queue = PersistentJobQueue(db_path=db_path)

    queue.enqueue(IndexJob(cam_id="cam0", segment_path=str(real_file), capture_timestamp=1.0))
    queue.enqueue(IndexJob(cam_id="cam0", segment_path=ghost_path, capture_timestamp=2.0))

    purged = queue.purge_missing_files()

    assert purged == 1, "exactly one job should be purged"

    # Verify DB states
    import sqlite3
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = {r["segment_path"]: r["status"] for r in
            con.execute("SELECT segment_path, status FROM jobs").fetchall()}
    con.close()

    assert rows[str(real_file)] == "pending", "existing file job must remain pending"
    assert rows[ghost_path] == "dead", "missing file job must be dead"

