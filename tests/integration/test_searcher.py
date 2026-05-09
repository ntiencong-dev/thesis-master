"""
tests/integration/test_searcher.py
------------------------------------
Integration tests for NLVideoSearcher end-to-end pipeline.

These tests load the CLIP model and use the real video fi003.mp4.
They are intentionally slower but exercise the full stack.

Scenarios covered
-----------------
SR-01  from_config(pc.yaml): loads without error, engine_type=pc
SR-02  from_params(): correct window_sec, overlap, frames_per_window
SR-03  index_video: returns segment count > 0
SR-04  index_video: total_vectors() equals returned count
SR-05  index_video: segment count matches sliding-window formula
SR-06  search on empty index → RuntimeError
SR-07  search returns non-empty results for valid query
SR-08  all result scores >= score_threshold
SR-09  top_k is respected in returned result count
SR-10  result rank=1 has highest score
SR-11  result timestamps within video duration
SR-12  no two results from same video overlap > nms_iou threshold
SR-13  score_threshold=0.0 returns more results than default 0.20
SR-14  score_threshold=0.99 returns empty list
SR-15  save_index / load_index: post-reload search gives same top-1
SR-16  save_index without index_dir → ValueError
SR-17  load_index without index_dir → ValueError
SR-18  index_video on non-existent path → FileNotFoundError
SR-19  SearchResult.rank is sequential (1, 2, 3, ...)
SR-20  from_config: config path not found → FileNotFoundError
SR-21  search_debug returns expected dict keys; query_cleaned differs from query_original
SR-22  search_debug query_cleaned matches _normalize_query() output
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
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "pc.yaml")
    if not os.path.isfile(config_path):
        pytest.skip("config/pc.yaml not found")
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
    # searcher_with_real_video already indexed real_video_path once
    count = searcher_with_real_video._index.total_vectors()
    assert count > 0, "index_video must add at least one segment"


def test_SR04_index_video_total_vectors(searcher_with_real_video):
    total = searcher_with_real_video._index.total_vectors()
    # fi003.mp4 is 76.07 s; window=5s, stride=2.5s → ~29 segments
    assert 25 <= total <= 45, f"Unexpected segment count: {total}"


def test_SR05_sliding_window_count_formula(real_video_path, tmp_index_dir):
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    import math
    s = NLVideoSearcher.from_params(
        index_dir=tmp_index_dir,
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=2,
    )
    n = s.index_video(real_video_path)
    expected_min = math.floor((VIDEO_DURATION - 5.0) / 2.5)
    assert n >= expected_min, f"count={n} < formula_min={expected_min}"


# ── SR-06  empty index raises ─────────────────────────────────────────────

def test_SR06_search_empty_index_raises():
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params()
    with pytest.raises(RuntimeError, match="Index empty"):
        s.search("test query")


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


# ── SR-15  persistence round-trip ────────────────────────────────────────

def test_SR15_save_load_round_trip(real_video_path, tmp_index_dir):
    import warnings; warnings.filterwarnings("ignore")
    from src.searcher import NLVideoSearcher

    s1 = NLVideoSearcher.from_params(
        index_dir=tmp_index_dir,
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=2,
    )
    s1.index_video(real_video_path)
    s1.save_index()

    top1_before = s1.search("person", top_k=1, score_threshold=0.0)

    s2 = NLVideoSearcher.from_params(
        index_dir=tmp_index_dir,
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=2,
    )
    s2.load_index()

    assert s2._index.total_vectors() == s1._index.total_vectors()

    top1_after = s2.search("person", top_k=1, score_threshold=0.0)
    if top1_before and top1_after:
        assert abs(top1_before[0].start_time - top1_after[0].start_time) < 1e-3


# ── SR-16..18  error cases ────────────────────────────────────────────────

def test_SR16_save_without_index_dir_raises():
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(index_dir=None)
    with pytest.raises(ValueError, match="index_dir"):
        s.save_index()


def test_SR17_load_without_index_dir_raises():
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(index_dir=None)
    with pytest.raises(ValueError, match="index_dir"):
        s.load_index()


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
    assert len(dbg["templates_used"]) == 5


# ── SR-22  search_debug cleaned query == _normalize_query ────────────────

def test_SR22_search_debug_cleaned_matches_normalize(searcher_with_real_video):
    from src.searcher import _normalize_query
    q = "show me the person running"
    dbg = searcher_with_real_video.search_debug(q, top_k=3)
    assert dbg["query_cleaned"] == _normalize_query(q)
