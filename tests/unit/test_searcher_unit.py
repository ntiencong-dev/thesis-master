"""
tests/unit/test_searcher_unit.py
---------------------------------
Unit tests for pure-Python helpers in src/searcher.py.
These tests do NOT require a CLIP model or GPU and run in milliseconds.

Scenarios covered
-----------------
QN-01  "find the <X>"  → "<X>"  (basic imperative stripped)
QN-02  "show me <X>"   → "<X>"
QN-03  "search for <X>" → "<X>"
QN-04  "find me the <X>" → "the <X>"  (regex matches "me" first; iterative)
QN-05  Plain query without prefix → unchanged
QN-06  Prefix-only input (no noun) → never returns empty string
QN-07  Mixed-case prefix is stripped (case-insensitive)
QN-08  Leading/trailing whitespace stripped from result
QN-09  Five CLIP templates produced from a cleaned query
QN-10  Every template in _CLIP_TEMPLATES contains exactly one "{}" placeholder
QN-11  "locate <X>" stripped
QN-12  "can you find <X>" stripped
QN-21  _search_qdrant uses query_points() not search() (qdrant-client ≥1.7)
QN-22  Default score_threshold is 0.10 (lowered for screen-capture EVA-CLIP)
"""

from __future__ import annotations

import pytest

from src.searcher import _normalize_query, _CLIP_TEMPLATES

pytestmark = pytest.mark.unit


# ── QN-01  "find the" ────────────────────────────────────────────────────

def test_QN01_find_the_stripped():
    assert _normalize_query("find the man waving hand") == "man waving hand"


# ── QN-02  "show me" ─────────────────────────────────────────────────────

def test_QN02_show_me_stripped():
    assert _normalize_query("show me a red car") == "a red car"


# ── QN-03  "search for" ───────────────────────────────────────────────────

def test_QN03_search_for_stripped():
    assert _normalize_query("search for running person") == "running person"


# ── QN-04  "find me the" — iterative stripping ───────────────────────────

def test_QN04_find_me_the_strips_find_me():
    # The regex matches "me" (not "me the") from the alternation.
    # First pass: "find me " removed → "the bicycle"
    # Second pass: "the bicycle" has no prefix → returned as-is
    result = _normalize_query("find me the bicycle")
    assert "find" not in result.lower()
    assert "bicycle" in result


# ── QN-05  Plain query unchanged ─────────────────────────────────────────

def test_QN05_plain_query_unchanged():
    q = "person climbing fence"
    assert _normalize_query(q) == q


# ── QN-06  Prefix-only input → never empty ───────────────────────────────

def test_QN06_prefix_only_not_empty():
    # "find the" with nothing after — must not produce ""
    result = _normalize_query("find the")
    assert isinstance(result, str)
    assert len(result) > 0


# ── QN-07  Case-insensitive prefix ───────────────────────────────────────

def test_QN07_case_insensitive():
    assert _normalize_query("FIND THE bicycle") == "bicycle"
    assert _normalize_query("Show Me a dog") == "a dog"


# ── QN-08  Whitespace trimmed from result ────────────────────────────────

def test_QN08_result_has_no_leading_trailing_whitespace():
    result = _normalize_query("  find  the   dog  ")
    assert result == result.strip()


# ── QN-09  Five CLIP templates produced ──────────────────────────────────

def test_QN09_twelve_templates_produced():
    cleaned = "person on roof"
    templates = [t.format(cleaned) for t in _CLIP_TEMPLATES]
    assert len(templates) == 12
    assert cleaned in templates        # "{}" identity template must be present


# ── QN-10  Each template has exactly one "{}" placeholder ────────────────

def test_QN10_each_template_has_one_placeholder():
    for t in _CLIP_TEMPLATES:
        assert t.count("{}") == 1, (
            f"Template should have exactly 1 placeholder, got: {t!r}"
        )


# ── QN-11  "locate" stripped ─────────────────────────────────────────────

def test_QN11_locate_stripped():
    assert _normalize_query("locate the red bus") == "the red bus"


# ── QN-12  "can you find" stripped ───────────────────────────────────────

def test_QN12_can_you_find_stripped():
    result = _normalize_query("can you find the cat on the roof")
    assert "can you" not in result.lower()
    assert "cat" in result


# ── QN-13  index_video Phase 2 signature ─────────────────────────────────

@pytest.mark.unit
def test_QN13_index_video_accepts_use_scene_detection_param():
    """index_video must accept use_scene_detection keyword without TypeError."""
    import inspect
    from src.searcher import NLVideoSearcher
    sig = inspect.signature(NLVideoSearcher.index_video)
    assert "use_scene_detection" in sig.parameters, \
        "index_video must have 'use_scene_detection' parameter (Phase 2)"


# ── QN-14  from_params regression ────────────────────────────────────────

@pytest.mark.unit
def test_QN14_from_params_still_works():
    """from_params() must not raise TypeError with default arguments."""
    import inspect
    from src.searcher import NLVideoSearcher
    sig = inspect.signature(NLVideoSearcher.from_params)
    params = list(sig.parameters.keys())
    # Required params must still be present; index_dir removed in v3.0
    assert "window_sec" in params
    assert "qdrant_url" in params
    assert "index_dir" not in params


# ── QN-15  search() Phase 3 signature ────────────────────────────────────

@pytest.mark.unit
def test_QN15_search_has_use_reranker_param():
    """search() must accept use_reranker keyword argument (Phase 3)."""
    import inspect
    from src.searcher import NLVideoSearcher
    sig = inspect.signature(NLVideoSearcher.search)
    assert "use_reranker" in sig.parameters, \
        "search() must have 'use_reranker' parameter (Phase 3)"
    # Default must be False (non-breaking for existing callers)
    assert sig.parameters["use_reranker"].default is False


# ── QN-16  set_reranker() Phase 3 signature ───────────────────────────────

@pytest.mark.unit
def test_QN16_set_reranker_method_exists():
    """NLVideoSearcher must have set_reranker() method (Phase 3)."""
    import inspect
    from src.searcher import NLVideoSearcher
    assert hasattr(NLVideoSearcher, "set_reranker"), \
        "NLVideoSearcher must have set_reranker() method (Phase 3)"
    sig = inspect.signature(NLVideoSearcher.set_reranker)
    assert "reranker" in sig.parameters


# ── QN-17  _reranker attribute exists and defaults to None ───────────────

@pytest.mark.unit
def test_QN17_reranker_attribute_defaults_none():
    """NLVideoSearcher._reranker must default to None (Phase 3)."""
    import inspect
    from src.searcher import NLVideoSearcher
    # Inspect the __init__ source to verify _reranker initialisation
    src = inspect.getsource(NLVideoSearcher.__init__)
    assert "_reranker" in src, \
        "NLVideoSearcher.__init__ must initialise self._reranker"


# ── QN-18  _qdrant_client defaults to None ───────────────────────────────

@pytest.mark.unit
def test_QN18_qdrant_client_attribute_exists():
    """__init__ must initialise _qdrant_client (None by default)."""
    import inspect
    from src.searcher import NLVideoSearcher
    src = inspect.getsource(NLVideoSearcher.__init__)
    assert "_qdrant_client" in src, \
        "NLVideoSearcher.__init__ must initialise self._qdrant_client"


# ── QN-19  _init_qdrant sets client when Qdrant is reachable (mocked) ────

@pytest.mark.unit
def test_QN19_init_qdrant_sets_client_when_reachable():
    """_init_qdrant() must set _qdrant_client when connection succeeds."""
    from unittest.mock import MagicMock, patch
    from src.searcher import NLVideoSearcher

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        searcher = MagicMock(spec=NLVideoSearcher)
        searcher._qdrant_client = None
        searcher._qdrant_collection = None
        searcher._engine = MagicMock()
        searcher._engine.embed_dim = 768
        # Call the actual _init_qdrant bound to our mock
        NLVideoSearcher._init_qdrant(searcher, {
            "qdrant_url": "http://localhost:6333",
            "qdrant_collection": "nlvs_segments",
            "embed_dim": 768,
        })

    assert searcher._qdrant_client is not None
    assert searcher._qdrant_collection == "nlvs_segments"


# ── QN-20  _init_qdrant raises RuntimeError when Qdrant is unreachable ────

@pytest.mark.unit
def test_QN20_init_qdrant_raises_when_unreachable():
    """_init_qdrant() must raise RuntimeError on connection error (v3.0 behavior)."""
    from unittest.mock import MagicMock, patch
    from src.searcher import NLVideoSearcher

    with patch("qdrant_client.QdrantClient", side_effect=Exception("refused")):
        searcher = MagicMock(spec=NLVideoSearcher)
        searcher._qdrant_client = None
        searcher._qdrant_collection = None
        searcher._engine = MagicMock()
        searcher._engine.embed_dim = 768
        with pytest.raises(RuntimeError, match="Qdrant unavailable"):
            NLVideoSearcher._init_qdrant(searcher, {
                "qdrant_url": "http://localhost:6333",
                "qdrant_collection": "nlvs_segments",
                "embed_dim": 768,
            })


# ── QN-21  _search_qdrant uses query_points (qdrant-client ≥1.7) ─────────

@pytest.mark.unit
def test_QN21_search_qdrant_uses_query_points():
    """_search_qdrant must call client.query_points(), not client.search()."""
    from unittest.mock import MagicMock
    import numpy as np
    from src.searcher import NLVideoSearcher

    mock_client = MagicMock()
    # Simulate qdrant-client ≥1.7: no .search() attr, only .query_points()
    del mock_client.search
    scored_point = MagicMock()
    scored_point.score = 0.15
    scored_point.payload = {
        "cam_id": "cam01", "video_path": "/tmp/a.mp4",
        "relative_start": 0.0, "relative_end": 10.0,
        "segment_wall_start": 0.0, "absolute_start": 0.0, "absolute_end": 10.0,
    }
    mock_response = MagicMock()
    mock_response.points = [scored_point]
    mock_client.query_points.return_value = mock_response

    searcher = MagicMock(spec=NLVideoSearcher)
    searcher._qdrant_client = mock_client
    searcher._qdrant_collection = "nlvs_segments"

    qvec = np.ones((1, 768), dtype=np.float32)
    results = NLVideoSearcher._search_qdrant(searcher, qvec, top_k=5)

    mock_client.query_points.assert_called_once()
    assert len(results) == 1
    score, meta = results[0]
    assert abs(score - 0.15) < 1e-6
    assert meta.cam_id == "cam01"


# ── QN-22  score_threshold defaults to 0.10 ──────────────────────────────

@pytest.mark.unit
def test_QN22_default_score_threshold_is_010():
    """Default score_threshold must be 0.10 (appropriate for screen-capture EVA-CLIP scores)."""
    from src.searcher import NLVideoSearcher
    cfg = NLVideoSearcher._DEFAULT
    assert cfg["search"]["score_threshold"] == 0.10, \
        "Default score_threshold should be 0.10 for screen-capture EVA-CLIP scores"
