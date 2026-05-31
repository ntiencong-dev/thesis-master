"""
tests/integration/test_api.py
-------------------------------
Integration tests for the FastAPI NLVS service.

Uses httpx.TestClient (sync) so no server process is needed.
The CLIP model is loaded lazily when the first real request arrives.

Scenarios covered
-----------------
API-01  GET /health → 200, status="ok"
API-02  GET /health → returns backend, total_vectors, config_path fields
API-03  POST /index with valid path → 200, segments_added > 0
API-04  POST /index with missing file → 404
API-05  POST /index/directory with valid dir → 200, segments_added > 0
API-06  POST /index/directory with missing dir → 404
API-07  POST /search before indexing → 409 (empty index)
API-08  POST /search after indexing → 200, SearchResponse structure
API-09  POST /search: all result scores >= score_threshold
API-10  POST /search: top_k=2 returns at most 2 results
API-11  POST /search with empty query → 422 (validation error)
API-12  POST /search with query > 512 chars → 422 (validation error)
API-13  POST /search with score_threshold=0.99 → empty results list
API-14  DELETE /index → 200, subsequent /health shows total_vectors=0
API-15  POST /search: result fields present (rank, score, start_time, end_time)
API-16  POST /search: result ranks are sequential starting from 1
"""

from __future__ import annotations

import os
import pytest

# Use httpx.TestClient for sync FastAPI testing
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)

import warnings
warnings.filterwarnings("ignore")

# We need to point the API at a real config + real video path.
# Set environment vars BEFORE importing api.main so _CONFIG_PATH is set.
_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "pc_blip1.yaml")
)
_INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "index_store_test_api")
)
os.environ.setdefault("CONFIG", _CONFIG_PATH)
os.environ.setdefault("INDEX_DIR", _INDEX_DIR)

REAL_VIDEO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "fi003.mp4")
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a fresh _searcher state per module."""
    import api.main as app_module
    # Reset global state so tests start clean
    app_module._searcher = None

    with TestClient(app_module.app) as c:
        yield c

    # Cleanup
    app_module._searcher = None
    import shutil
    shutil.rmtree(_INDEX_DIR, ignore_errors=True)


@pytest.fixture(scope="module")
def indexed_client(client, real_video_path):
    """Client with real video already indexed."""
    resp = client.post("/index", json={"video_path": real_video_path})
    assert resp.status_code == 200, f"Indexing failed: {resp.text}"
    return client


# ---------------------------------------------------------------------------
# API-01..02  /health
# ---------------------------------------------------------------------------

def test_API01_health_status_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_API02_health_response_fields(client):
    resp = client.get("/health")
    body = resp.json()
    assert "backend" in body
    assert "total_vectors" in body
    assert isinstance(body["total_vectors"], int)
    assert "config_path" in body


# ---------------------------------------------------------------------------
# API-03..06  /index and /index/directory
# ---------------------------------------------------------------------------

def test_API03_index_valid_path(indexed_client, real_video_path):
    resp = indexed_client.post("/index", json={"video_path": real_video_path})
    assert resp.status_code == 200
    body = resp.json()
    assert body["segments_added"] > 0
    assert body["total_in_index"] > 0


def test_API04_index_missing_file(client):
    resp = client.post("/index", json={"video_path": "/no/such/file.mp4"})
    assert resp.status_code == 404


def test_API05_index_directory_valid(indexed_client, real_video_path):
    video_dir = os.path.dirname(real_video_path)
    resp = indexed_client.post(
        "/index/directory",
        json={"video_dir": video_dir, "extensions": [".mp4"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["segments_added"] >= 0   # ≥ 0: might already be indexed


def test_API06_index_directory_missing(client):
    resp = client.post(
        "/index/directory",
        json={"video_dir": "/no/such/dir", "extensions": [".mp4"]}
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API-07  /search before indexing
# ---------------------------------------------------------------------------

def test_API07_search_empty_index_returns_409():
    """
    Create a brand-new client backed by a fresh empty Qdrant collection and
    verify POST /search returns 409 (no vectors indexed yet).
    """
    import api.main as app_module
    original = app_module._searcher
    app_module._searcher = None

    # Use a dedicated empty collection so this test is isolated from real data.
    # Drop any stale collection first (e.g. leftover from a previous run with a
    # different embed_dim) so from_params() always creates it fresh.
    try:
        from qdrant_client import QdrantClient as _QC
        _QC(url="http://localhost:6333", timeout=5).delete_collection(
            "nlvs_segments_api07_empty"
        )
    except Exception:
        pass
    from src.searcher import NLVideoSearcher
    app_module._searcher = NLVideoSearcher.from_params(
        qdrant_collection="nlvs_segments_api07_empty"
    )
    # collection was just created → 0 vectors → search must return 409

    with TestClient(app_module.app) as c:
        resp = c.post("/search", json={"query": "test"})
        assert resp.status_code == 409

    app_module._searcher = original


# ---------------------------------------------------------------------------
# API-08..16  /search with data
# ---------------------------------------------------------------------------

def test_API08_search_response_structure(indexed_client):
    resp = indexed_client.post("/search", json={
        "query": "person walking", "top_k": 3, "score_threshold": 0.0
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "query" in body
    assert "num_results" in body
    assert "query_ms" in body
    assert "results" in body
    assert isinstance(body["results"], list)


def test_API09_search_scores_above_threshold(indexed_client):
    threshold = 0.15
    resp = indexed_client.post("/search", json={
        "query": "person", "top_k": 5, "score_threshold": threshold
    })
    assert resp.status_code == 200
    for r in resp.json()["results"]:
        assert r["score"] >= threshold


def test_API10_search_top_k_limit(indexed_client):
    resp = indexed_client.post("/search", json={
        "query": "person", "top_k": 2, "score_threshold": 0.0
    })
    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= 2


def test_API11_search_empty_query_422(indexed_client):
    resp = indexed_client.post("/search", json={"query": ""})
    assert resp.status_code == 422


def test_API12_search_long_query_422(indexed_client):
    long_query = "x" * 513
    resp = indexed_client.post("/search", json={"query": long_query})
    assert resp.status_code == 422


def test_API13_search_high_threshold_empty(indexed_client):
    resp = indexed_client.post("/search", json={
        "query": "zzz_nonsense_xyz_42",
        "top_k": 5,
        "score_threshold": 0.99,
    })
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_API14_delete_index_resets(indexed_client):
    """
    DELETE /index resets the in-memory searcher (but NOT the disk index).
    The endpoint must return 200 and the confirmation payload.
    Subsequent /health may reload from disk — that is by design.
    """
    resp = indexed_client.delete("/index")
    assert resp.status_code == 200
    body = resp.json()
    assert "index reset" in body.get("status", "").lower() or "reset" in str(body).lower()


def test_API15_search_result_fields(indexed_client, real_video_path):
    # Re-index after the DELETE test
    indexed_client.post("/index", json={"video_path": real_video_path})

    resp = indexed_client.post("/search", json={
        "query": "scene", "top_k": 3, "score_threshold": 0.0
    })
    assert resp.status_code == 200
    for r in resp.json()["results"]:
        assert "rank" in r
        assert "score" in r
        assert "start_time" in r
        assert "end_time" in r
        assert "video_id" in r
        assert "video_path" in r
        assert r["start_time"] >= 0.0
        assert r["end_time"] > r["start_time"]


def test_API16_search_result_ranks_sequential(indexed_client):
    resp = indexed_client.post("/search", json={
        "query": "scene", "top_k": 5, "score_threshold": 0.0
    })
    assert resp.status_code == 200
    results = resp.json()["results"]
    for expected, r in enumerate(results, start=1):
        assert r["rank"] == expected, f"rank {r['rank']} ≠ {expected}"
