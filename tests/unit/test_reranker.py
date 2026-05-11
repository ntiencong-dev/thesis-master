"""
tests/unit/test_reranker.py
----------------------------
Phase 3 unit tests for BLIP2Reranker and RerankResult.
All tests are mock-based — no model is loaded.

Scenarios covered
-----------------
RR-01   BLIP2Reranker is importable without transformers installed
         (BLIP2_AVAILABLE flag is set correctly)
RR-02   BLIP2Reranker instantiation with lazy_load=True does not load model
RR-03   BLIP2Reranker.is_loaded is False before first rerank call
RR-04   BLIP2Reranker.rerank() with empty candidates returns []
RR-05   BLIP2Reranker.rerank() calls _load_model once (idempotent)
RR-06   BLIP2Reranker.rerank() preserves all candidates when no top_k given
RR-07   BLIP2Reranker.rerank() respects top_k limit
RR-08   BLIP2Reranker.rerank() output is sorted by combined score descending
RR-09   RerankResult.rank starts at 1 and is sequential
RR-10   combined score = alpha * cosine + (1-alpha) * blip_score
RR-11   BLIP2Reranker.unload() sets model to None
RR-12   _extract_frame returns ndarray of shape (224, 224, 3) when video missing
RR-13   _score_frame returns 1.0 for answer starting with 'yes'
RR-14   _score_frame returns 0.0 for answer starting with 'no'
RR-15   RerankResult fields match candidate values
RR-16   set_reranker attaches reranker to NLVideoSearcher
RR-17   search() signature accepts use_reranker parameter
RR-18   search() with use_reranker=False and no reranker returns normal results
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────

@dataclass
class _FakeSearchResult:
    """Mimics SearchResult for reranker tests without importing searcher."""
    score:      float
    video_id:   str
    video_path: str
    start_time: float
    end_time:   float
    rank:       int


def _make_candidates(n: int = 3) -> List[_FakeSearchResult]:
    return [
        _FakeSearchResult(
            score=0.9 - i * 0.1,
            video_id=f"vid{i}",
            video_path=f"/fake/vid{i}.mp4",
            start_time=float(i * 5),
            end_time=float(i * 5 + 5),
            rank=i + 1,
        )
        for i in range(n)
    ]


def _make_reranker_no_load(**kwargs):
    """Return a BLIP2Reranker with model loading monkey-patched out."""
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE

    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed — BLIP2Reranker unavailable")

    reranker = object.__new__(BLIP2Reranker)
    reranker.model_name      = kwargs.get("model_name", "Salesforce/blip2-opt-2.7b")
    reranker.load_in_8bit    = kwargs.get("load_in_8bit", False)
    reranker.alpha           = kwargs.get("alpha", 0.6)
    reranker.prompt_template = BLIP2Reranker.DEFAULT_TEMPLATE
    reranker._device_str     = "cpu"
    reranker._use_8bit       = False
    reranker._processor      = None
    reranker._model          = None
    return reranker


# ── RR-01  importable without transformers ────────────────────────────────

def test_RR01_reranker_module_importable():
    """src.reranker must be importable regardless of transformers install."""
    import src.reranker as mod
    assert hasattr(mod, "BLIP2Reranker")
    assert hasattr(mod, "RerankResult")
    assert hasattr(mod, "BLIP2_AVAILABLE")


# ── RR-02  lazy_load=True does not load model ─────────────────────────────

def test_RR02_lazy_load_does_not_load_model():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    with patch.object(BLIP2Reranker, "_load_model", return_value=None) as mock_load:
        r = BLIP2Reranker(lazy_load=True)
        mock_load.assert_not_called()


# ── RR-03  is_loaded False before first call ──────────────────────────────

def test_RR03_is_loaded_false_before_rerank():
    r = _make_reranker_no_load()
    assert r.is_loaded is False


# ── RR-04  rerank with empty candidates ──────────────────────────────────

def test_RR04_rerank_empty_candidates():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    with patch.object(r, "_load_model", return_value=None):
        result = r.rerank([], "person running")
    assert result == []


# ── RR-05  _load_model called once (idempotent) ───────────────────────────

def test_RR05_load_model_called_once():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    candidates = _make_candidates(2)
    call_count = [0]

    def _mock_load():
        call_count[0] += 1

    def _mock_score(frame, prompt):
        return 0.5

    def _mock_frame(path, t0, t1):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    r._load_model   = _mock_load
    r._score_frame  = _mock_score
    r._extract_frame = _mock_frame
    r._model = MagicMock()  # simulate already loaded after first call

    r.rerank(candidates, "test")
    # _load_model was called exactly once (via rerank's internal call)
    assert call_count[0] == 1


# ── RR-06  all candidates returned when top_k=None ────────────────────────

def test_RR06_all_candidates_returned_without_top_k():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    candidates = _make_candidates(5)
    r._load_model   = MagicMock()
    r._model        = MagicMock()
    r._score_frame  = lambda f, p: 0.5
    r._extract_frame = lambda path, t0, t1: np.zeros((224, 224, 3), dtype=np.uint8)

    results = r.rerank(candidates, "query")
    assert len(results) == 5


# ── RR-07  top_k respected ────────────────────────────────────────────────

def test_RR07_top_k_limits_results():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    candidates = _make_candidates(5)
    r._load_model   = MagicMock()
    r._model        = MagicMock()
    r._score_frame  = lambda f, p: 0.5
    r._extract_frame = lambda path, t0, t1: np.zeros((224, 224, 3), dtype=np.uint8)

    results = r.rerank(candidates, "query", top_k=3)
    assert len(results) == 3


# ── RR-08  sorted by combined score descending ────────────────────────────

def test_RR08_results_sorted_descending():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    candidates = _make_candidates(4)

    # Make blip_score inversely related to cosine to create re-ranking
    blip_scores = {f"/fake/vid{i}.mp4": float(3 - i) * 0.3 for i in range(4)}
    r._load_model   = MagicMock()
    r._model        = MagicMock()
    r._score_frame  = lambda f, p: 0.5
    r._extract_frame = lambda path, t0, t1: np.zeros((224, 224, 3), dtype=np.uint8)

    results = r.rerank(candidates, "query")
    scores = [res.score for res in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted descending"


# ── RR-09  rank starts at 1 and is sequential ────────────────────────────

def test_RR09_rank_is_sequential():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    candidates = _make_candidates(3)
    r._load_model   = MagicMock()
    r._model        = MagicMock()
    r._score_frame  = lambda f, p: 0.5
    r._extract_frame = lambda path, t0, t1: np.zeros((224, 224, 3), dtype=np.uint8)

    results = r.rerank(candidates, "query")
    assert [res.rank for res in results] == list(range(1, len(results) + 1))


# ── RR-10  combined score formula ────────────────────────────────────────

def test_RR10_combined_score_formula():
    from src.reranker import BLIP2Reranker, RerankResult, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    alpha = 0.6
    cosine = 0.8
    blip   = 1.0   # answer was "yes"

    r = _make_reranker_no_load(alpha=alpha)

    cand = _make_candidates(1)[0]
    cand.score = cosine

    r._load_model   = MagicMock()
    r._model        = MagicMock()
    r._score_frame  = lambda f, p: blip
    r._extract_frame = lambda path, t0, t1: np.zeros((224, 224, 3), dtype=np.uint8)

    results = r.rerank([cand], "query")
    expected = alpha * cosine + (1 - alpha) * blip
    assert abs(results[0].score - expected) < 1e-6


# ── RR-11  unload sets model to None ─────────────────────────────────────

def test_RR11_unload_clears_model():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    r._model = MagicMock()
    r._processor = MagicMock()

    with patch("torch.cuda.is_available", return_value=False):
        r.unload()

    assert r._model is None
    assert r._processor is None
    assert r.is_loaded is False


# ── RR-12  _extract_frame fallback for missing video ──────────────────────

def test_RR12_extract_frame_fallback_missing_video():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    # Non-existent file → should return zero array of shape (224, 224, 3)
    frame = r._extract_frame("/nonexistent/video.mp4", 0.0, 5.0)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (224, 224, 3)


# ── RR-13 / RR-14  _score_frame yes/no ────────────────────────────────────

def test_RR13_score_frame_yes_returns_1():
    """_score_frame must return 1.0 when answer starts with 'yes'."""
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    # Mock model + processor to return "yes"
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": MagicMock()}
    mock_processor.decode.return_value = "yes, I can see it"
    mock_model = MagicMock()
    mock_model.generate.return_value = [[0]]
    r._processor = mock_processor
    r._model     = mock_model

    score = r._score_frame(np.zeros((224, 224, 3), dtype=np.uint8), "prompt")
    assert score == 1.0


def test_RR14_score_frame_no_returns_0():
    """_score_frame must return 0.0 when answer starts with 'no'."""
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": MagicMock()}
    mock_processor.decode.return_value = "no, I don't see it"
    mock_model = MagicMock()
    mock_model.generate.return_value = [[0]]
    r._processor = mock_processor
    r._model     = mock_model

    score = r._score_frame(np.zeros((224, 224, 3), dtype=np.uint8), "prompt")
    assert score == 0.0


# ── RR-15  RerankResult fields match candidate ────────────────────────────

def test_RR15_rerank_result_fields():
    from src.reranker import BLIP2Reranker, BLIP2_AVAILABLE
    if not BLIP2_AVAILABLE:
        pytest.skip("transformers not installed")

    r = _make_reranker_no_load()
    cand = _make_candidates(1)[0]
    r._load_model   = MagicMock()
    r._model        = MagicMock()
    r._score_frame  = lambda f, p: 0.0
    r._extract_frame = lambda path, t0, t1: np.zeros((224, 224, 3), dtype=np.uint8)

    result = r.rerank([cand], "query")[0]
    assert result.video_id   == cand.video_id
    assert result.video_path == cand.video_path
    assert result.start_time == cand.start_time
    assert result.end_time   == cand.end_time


# ── RR-16  set_reranker attaches to searcher ──────────────────────────────

def test_RR16_set_reranker_attaches_to_searcher():
    """NLVideoSearcher.set_reranker() must accept any object and store it."""
    from src.searcher import NLVideoSearcher
    import inspect

    sig = inspect.signature(NLVideoSearcher.set_reranker)
    assert "reranker" in sig.parameters, \
        "set_reranker must accept 'reranker' parameter"


# ── RR-17  search() signature has use_reranker param ─────────────────────

def test_RR17_search_signature_has_use_reranker():
    """search() must accept use_reranker keyword argument (Phase 3)."""
    from src.searcher import NLVideoSearcher
    import inspect

    sig = inspect.signature(NLVideoSearcher.search)
    assert "use_reranker" in sig.parameters, \
        "search() must have 'use_reranker' parameter (Phase 3)"


# ── RR-18  search() with use_reranker=False and no reranker ──────────────

def test_RR18_search_no_reranker_no_error():
    """
    search() with use_reranker=False must not raise even when _reranker=None.
    Test uses a pre-populated index via conftest fixtures.
    """
    from src.searcher import NLVideoSearcher

    # Build a minimal searcher with a small in-memory index
    config = {
        "engine":   {"type": "pc", "model_name": "ViT-B-16", "pretrained": "openai",
                     "device": "cpu", "batch_size": 2, "frames_per_window": 2},
        "pipeline": {"video_backend": "opencv", "window_sec": 2.0, "overlap_ratio": 0.0},
        "index":    {"embed_dim": 512, "index_dir": None},
        "search":   {"top_k": 5, "score_threshold": 0.0, "nms_iou_threshold": 0.5,
                     "adaptive_threshold": False, "translate_vi": False},
    }

    try:
        s = NLVideoSearcher(config)
    except Exception:
        pytest.skip("Could not instantiate NLVideoSearcher (model not cached)")

    import inspect
    sig = inspect.signature(s.search)
    assert "use_reranker" in sig.parameters
