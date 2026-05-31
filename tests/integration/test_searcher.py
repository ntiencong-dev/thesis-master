"""
tests/integration/test_searcher.py
------------------------------------
Integration tests for NLVideoSearcher end-to-end pipeline.

These tests load the CLIP model and use the real video fi003.mp4.
They are intentionally slower but exercise the full stack.

Scenarios covered
-----------------
SR-01  from_config(pc_blip1.yaml): loads without error, engine_type=blip1
SR-02  from_params(): correct window_sec, overlap, frames_per_window
SR-03  index_video: returns segment count > 0
SR-04  index_video: Qdrant vector count matches returned segment count
SR-05  index_video: segment count matches sliding-window formula
SR-06  search with Qdrant unavailable → RuntimeError
SR-07  search returns non-empty results for valid query
SR-08  all result scores >= score_threshold
SR-09  top_k is respected in returned result count
SR-10  result rank=1 has highest score
SR-11  result timestamps within video duration
SR-12  no two results from same video overlap > nms_iou threshold
SR-13  score_threshold=0.0 returns more results than default 0.20
SR-14  score_threshold=0.99 returns empty list
SR-15  (removed in v3.0 — Faiss save/load)
SR-16  (removed in v3.0 — Faiss save/load)
SR-17  (removed in v3.0 — Faiss save/load)
SR-18  index_video on non-existent path → FileNotFoundError
SR-19  SearchResult.rank is sequential (1, 2, 3, ...)
SR-20  from_config: config path not found → FileNotFoundError
SR-21  search_debug returns expected dict keys; query_cleaned differs from query_original
SR-22  search_debug query_cleaned matches _normalize_query() output
SR-23  index_video with use_scene_detection=True returns positive count (Phase 2)
SR-24  index_video scene detection: result count is positive integer
SR-25  index_video scene detection: all indexed segments have valid timestamps
SR-26  search(use_reranker=False) without attached reranker gives same results (Phase 3)
SR-27  set_reranker() attached; use_reranker=False bypasses reranker (Phase 3)
SR-28  search(use_reranker=True) with mock reranker: reranker.rerank() called (Phase 3)
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from tests.conftest import VIDEO_DURATION

pytestmark = pytest.mark.integration


# ── SR-01  from_config ────────────────────────────────────────────────────

def test_SR01_from_config_loads(real_video_path):
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "pc_blip1.yaml")
    if not os.path.isfile(config_path):
        pytest.skip("config/pc_blip1.yaml not found")
    s = NLVideoSearcher.from_config(config_path)
    assert s is not None
    assert s._engine is not None


# ── SR-02  from_params config structure ──────────────────────────────────

def test_SR02_from_params_config(real_video_path):
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(
        use_sliding_window=True,
        window_sec=8.0,
        overlap_ratio=0.25,
        frames_per_window=4,
    )
    assert abs(s._window_sec - 8.0) < 1e-9
    assert abs(s._overlap_ratio - 0.25) < 1e-9
    assert s._frames_per_window == 4


# ── SR-03..05  index_video ────────────────────────────────────────────────

def test_SR03_index_video_returns_positive(searcher_with_real_video, real_video_path):
    # Check Qdrant vector count via the searcher's client
    s = searcher_with_real_video
    if s._qdrant_client is not None:
        info = s._qdrant_client.get_collection(s._qdrant_collection)
        count = info.points_count or 0
    else:
        pytest.skip("Qdrant not available")
    assert count > 0, "index_video must add at least one segment"


def test_SR04_index_video_total_vectors(searcher_with_real_video):
    s = searcher_with_real_video
    if s._qdrant_client is None:
        pytest.skip("Qdrant not available")
    info = s._qdrant_client.get_collection(s._qdrant_collection)
    total = info.points_count or 0
    # fi003.mp4 is 76.07 s; window=5s, stride=2.5s → ~29 segments
    assert 25 <= total <= 45, f"Unexpected segment count: {total}"


def test_SR05_sliding_window_count_formula(real_video_path):
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    import math
    s = NLVideoSearcher.from_params(
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=2,
    )
    n = s.index_video(real_video_path)
    expected_min = math.floor((VIDEO_DURATION - 5.0) / 2.5)
    assert n >= expected_min, f"count={n} < formula_min={expected_min}"


# ── SR-06  empty index raises ─────────────────────────────────────────────

def test_SR06_search_empty_raises():
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    from unittest.mock import patch
    # Create a searcher but patch Qdrant to be unavailable after init
    with patch("qdrant_client.QdrantClient", side_effect=Exception("refused")):
        with pytest.raises(RuntimeError):
            NLVideoSearcher.from_params()


# ── SR-07..14  search behaviour ──────────────────────────────────────────

def test_SR07_search_returns_results(searcher_with_real_video):
    results = searcher_with_real_video.search("person", top_k=5, score_threshold=0.0)
    assert len(results) > 0


def test_SR08_all_scores_above_threshold(searcher_with_real_video):
    threshold = 0.15
    results = searcher_with_real_video.search("person walking", score_threshold=threshold)
    for r in results:
        assert r.score >= threshold, f"Score {r.score} < threshold {threshold}"


def test_SR09_top_k_respected(searcher_with_real_video):
    results = searcher_with_real_video.search("scene", top_k=3, score_threshold=0.0)
    assert len(results) <= 3


def test_SR10_rank1_highest_score(searcher_with_real_video):
    results = searcher_with_real_video.search("scene", top_k=5, score_threshold=0.0)
    if len(results) >= 2:
        assert results[0].score >= results[1].score


def test_SR11_timestamps_within_duration(searcher_with_real_video):
    results = searcher_with_real_video.search("scene", top_k=5, score_threshold=0.0)
    for r in results:
        assert r.start_time >= 0.0
        assert r.end_time <= VIDEO_DURATION + 1e-2
        assert r.end_time > r.start_time


def test_SR12_nms_no_overlapping_results(searcher_with_real_video):
    """No two results from same video should have temporal IoU > nms_iou_threshold."""
    nms_iou = 0.5
    results = searcher_with_real_video.search("scene", top_k=10,
                                              score_threshold=0.0, nms_iou=nms_iou)
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            if a.video_id != b.video_id:
                continue
            inter = max(0.0, min(a.end_time, b.end_time) - max(a.start_time, b.start_time))
            union = (a.end_time - a.start_time) + (b.end_time - b.start_time) - inter
            iou   = inter / union if union > 1e-9 else 0.0
            assert iou <= nms_iou + 1e-6, \
                f"Overlapping results: [{a.start_time:.1f}-{a.end_time:.1f}] " \
                f"and [{b.start_time:.1f}-{b.end_time:.1f}], IoU={iou:.3f}"


def test_SR13_zero_threshold_returns_more(searcher_with_real_video):
    r_default = searcher_with_real_video.search("test", top_k=50, score_threshold=0.25)
    r_zero    = searcher_with_real_video.search("test", top_k=50, score_threshold=0.0)
    # Equal or more results with lower threshold
    assert len(r_zero) >= len(r_default)


def test_SR14_high_threshold_empty(searcher_with_real_video):
    results = searcher_with_real_video.search("zzz nonsense query xkcd", score_threshold=0.99)
    assert results == [], "score_threshold=0.99 should return no results"


# ── SR-15..17 removed (Faiss save/load, v3.0) ────────────────────────────
# SR-15, SR-16, SR-17 removed: save_index/load_index not present in Qdrant-only v3.0

# ── SR-18  error cases ────────────────────────────────────────────────────

def test_SR18_index_video_missing_file():
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        s.index_video("/path/that/does/not/exist.mp4")


# ── SR-19  sequential ranks ───────────────────────────────────────────────

def test_SR19_sequential_ranks(searcher_with_real_video):
    results = searcher_with_real_video.search("scene", top_k=5, score_threshold=0.0)
    for expected_rank, r in enumerate(results, start=1):
        assert r.rank == expected_rank, f"rank {r.rank} ≠ {expected_rank}"


# ── SR-20  from_config bad path ───────────────────────────────────────────

def test_SR20_from_config_bad_path():
    from src.searcher import NLVideoSearcher
    with pytest.raises((FileNotFoundError, OSError)):
        NLVideoSearcher.from_config("/no/such/config.yaml")


# ── SR-21  search_debug dict structure ───────────────────────────────────

def test_SR21_search_debug_structure(searcher_with_real_video):
    """search_debug() must return the expected keys and strip the query prefix."""
    dbg = searcher_with_real_video.search_debug(
        "find the man waving hand", top_k=5
    )
    required_keys = {
        "query_original", "query_cleaned", "templates_used",
        "top_raw_query", "top_templates", "score_stats",
    }
    assert required_keys.issubset(dbg.keys()), (
        f"Missing keys: {required_keys - dbg.keys()}"
    )
    # "find the" prefix should be stripped
    assert dbg["query_cleaned"] != dbg["query_original"]
    assert "find" not in dbg["query_cleaned"].lower()
    # score_stats must have all five sub-keys
    assert {"max", "min", "mean", "median", "p75"}.issubset(dbg["score_stats"])
    # templates_used must be a non-empty list of strings
    assert isinstance(dbg["templates_used"], list)
    assert len(dbg["templates_used"]) == 12


# ── SR-22  search_debug cleaned query == _normalize_query ────────────────

def test_SR22_search_debug_cleaned_matches_normalize(searcher_with_real_video):
    from src.searcher import _normalize_query
    q = "show me the person running"
    dbg = searcher_with_real_video.search_debug(q, top_k=3)
    assert dbg["query_cleaned"] == _normalize_query(q)


# ── SR-23..25  Phase 2: scene-detection indexing ─────────────────────────

def test_SR23_scene_detection_index_video_positive(real_video_path):
    """index_video with use_scene_detection=True must index at least one segment."""
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=3,
    )
    count = s.index_video(real_video_path, use_scene_detection=True)
    assert count > 0, f"Expected > 0 segments, got {count}"


def test_SR24_scene_detection_count_is_int(real_video_path):
    """index_video with use_scene_detection=True returns an integer."""
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(frames_per_window=3)
    count = s.index_video(real_video_path, use_scene_detection=True)
    assert isinstance(count, int)


def test_SR25_scene_detection_valid_timestamps(real_video_path):
    """All scene-segmented segments must have 0 <= start_time < end_time."""
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(frames_per_window=3)
    s.index_video(real_video_path, use_scene_detection=True)
    results = s.search("scene", top_k=50, score_threshold=0.0)
    for r in results:
        assert r.start_time >= 0.0, f"start_time {r.start_time} < 0"
        assert r.end_time > r.start_time, \
            f"end_time {r.end_time} <= start_time {r.start_time}"
        assert r.end_time <= VIDEO_DURATION + 1.0, \
            f"end_time {r.end_time} > video duration {VIDEO_DURATION}"


# ── SR-26..28  Phase 3: two-stage reranking integration ──────────────────

def test_SR26_search_use_reranker_false_no_error(searcher_with_real_video):
    """
    search(use_reranker=False) must work correctly without any reranker attached.
    Results must be identical to default search().
    """
    r1 = searcher_with_real_video.search("person walking", top_k=5,
                                          score_threshold=0.0, use_reranker=False)
    r2 = searcher_with_real_video.search("person walking", top_k=5,
                                          score_threshold=0.0)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert abs(a.score - b.score) < 1e-6
        assert a.video_id == b.video_id
        assert a.start_time == b.start_time


def test_SR27_set_reranker_then_search_without_flag(searcher_with_real_video):
    """
    Even after attaching a mock reranker, search(use_reranker=False) must
    bypass it and return the same results as baseline.
    """
    from unittest.mock import MagicMock
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = AssertionError("rerank must not be called")

    searcher_with_real_video.set_reranker(mock_reranker)
    # Should NOT call reranker when use_reranker=False
    results = searcher_with_real_video.search(
        "person walking", top_k=5, score_threshold=0.0, use_reranker=False
    )
    mock_reranker.rerank.assert_not_called()
    # Restore
    searcher_with_real_video.set_reranker(None)
    assert len(results) >= 0   # basic sanity


def test_SR28_mock_reranker_called_when_flag_true(searcher_with_real_video):
    """
    search(use_reranker=True) with a mock reranker must call reranker.rerank()
    and return its output as SearchResult objects.
    """
    from unittest.mock import MagicMock
    from src.searcher import SearchResult

    # Build a mock reranker that returns reversed-rank results
    class _FakeRerankResult:
        def __init__(self, score, cam_id, vid_path, start, end, rank):
            self.score          = score
            self.cam_id         = cam_id
            self.video_id       = cam_id   # backward-compat alias
            self.video_path     = vid_path
            self.start_time     = start
            self.end_time       = end
            self.rank           = rank
            self.absolute_start = 0.0
            self.absolute_end   = 0.0

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        _FakeRerankResult(0.99, "vid0", "/fake/vid0.mp4", 0.0, 5.0, 1),
    ]

    searcher_with_real_video.set_reranker(mock_reranker)
    results = searcher_with_real_video.search(
        "person", top_k=5, score_threshold=0.0, use_reranker=True
    )
    mock_reranker.rerank.assert_called_once()
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.99)
    assert isinstance(results[0], SearchResult)
    # Restore
    searcher_with_real_video.set_reranker(None)


# ── SR-29  from_config(pc_blip1.yaml) loads with qdrant backend attributes ──

def test_SR29_from_config_has_qdrant_attributes():
    """
    from_config(pc_blip1.yaml) must populate _qdrant_client/_qdrant_collection
    attributes on NLVideoSearcher (either set or None, but always present).
    Skip if config file is missing.
    """
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "pc_blip1.yaml"
    )
    if not os.path.isfile(config_path):
        pytest.skip("config/pc_blip1.yaml not found")

    s = NLVideoSearcher.from_config(config_path)
    assert hasattr(s, "_qdrant_client"), \
        "NLVideoSearcher must have _qdrant_client attribute"
    assert hasattr(s, "_qdrant_collection"), \
        "NLVideoSearcher must have _qdrant_collection attribute"


# ── SR-30  search on mock-empty Qdrant returns [] (no RuntimeError) ───────

def test_SR30_qdrant_client_wired_returns_empty_list():
    """
    search() on an NLVideoSearcher backed by a Qdrant client that returns no
    results must return [] (not raise RuntimeError).  The RuntimeError guard
    only fires when both Qdrant is absent AND the Faiss index is empty.
    """
    import warnings; warnings.filterwarnings("ignore")
    from unittest.mock import MagicMock
    from src.searcher import NLVideoSearcher

    s = NLVideoSearcher.from_params(
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=2,
    )
    # Wire up a mock Qdrant that returns zero hits
    mock_client = MagicMock()
    mock_client.search.return_value = []
    s._qdrant_client     = mock_client
    s._qdrant_collection = "nlvs_segments_blip1"

    results = s.search("person", top_k=5, score_threshold=0.0)
    assert results == [], "Wired-but-empty Qdrant should return [] not raise"
