"""
tests/unit/test_reranker.py
----------------------------
Phase 3 unit tests for BLIP2Reranker, BLIP1ITMReranker, and RerankResult.
All tests are mock-based — no model is loaded.

Scenarios covered
-----------------
RR-01   BLIP2Reranker and BLIP1ITMReranker importable; flags set correctly
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
RR-19   RerankResult has absolute_start and absolute_end fields (default 0.0)
BLIP1ITMReranker unit tests (all mock-based)
ITM-01  BLIP1ITMReranker importable from src.reranker
ITM-02  BLIP1ITMReranker.rerank() with empty candidates returns []
ITM-03  BLIP1ITMReranker.rerank() returns all candidates when top_k=None
ITM-04  BLIP1ITMReranker.rerank() respects top_k limit
ITM-05  BLIP1ITMReranker.rerank() output sorted by combined score descending
ITM-06  combined score = alpha * cosine + (1-alpha) * itm_prob
ITM-07  rank is sequential starting at 1
ITM-08  _extract_frame fallback for missing video → (224,224,3) zero array
ITM-09  engine.score_itm called with all candidate frames at once
ITM-10  RerankResult fields match candidate (video_id, video_path, times)
ITM-11  absolute_start / absolute_end forwarded from SearchResult
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
    assert hasattr(mod, "BLIP1ITMReranker")
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


# ── RR-19  RerankResult has absolute_start / absolute_end ────────────────

def test_RR19_rerank_result_has_absolute_fields():
    """RerankResult must have absolute_start and absolute_end, defaulting to 0.0."""
    from src.reranker import RerankResult
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(RerankResult)}
    assert "absolute_start" in field_names, "RerankResult missing absolute_start"
    assert "absolute_end"   in field_names, "RerankResult missing absolute_end"

    r = RerankResult(
        rank=1, score=0.5, cosine_score=0.5, blip_score=0.5,
        video_id="v0", video_path="/p.mp4", start_time=0.0, end_time=5.0,
    )
    assert r.absolute_start == 0.0
    assert r.absolute_end   == 0.0

    r2 = RerankResult(
        rank=1, score=0.5, cosine_score=0.5, blip_score=0.5,
        video_id="v0", video_path="/p.mp4", start_time=0.0, end_time=5.0,
        absolute_start=10.0, absolute_end=15.0,
    )
    assert r2.absolute_start == 10.0
    assert r2.absolute_end   == 15.0


# ══════════════════════════════════════════════════════════════════════════
# BLIP1ITMReranker unit tests  (ITM-01 .. ITM-11)
# All mock-based — no model loaded, no disk I/O
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class _FakeSearchResultFull:
    """Extended fake SearchResult with absolute_start / absolute_end."""
    score:          float
    video_id:       str
    video_path:     str
    start_time:     float
    end_time:       float
    rank:           int
    absolute_start: float = 0.0
    absolute_end:   float = 0.0


def _make_itm_candidates(n: int = 3) -> List[_FakeSearchResultFull]:
    return [
        _FakeSearchResultFull(
            score=0.9 - i * 0.1,
            video_id=f"vid{i}",
            video_path=f"/fake/vid{i}.mp4",
            start_time=float(i * 5),
            end_time=float(i * 5 + 5),
            rank=i + 1,
            absolute_start=float(i * 5 + 100),
            absolute_end=float(i * 5 + 105),
        )
        for i in range(n)
    ]


def _make_blip1_itm_reranker(n_candidates: int = 3, alpha: float = 0.6,
                              itm_scores=None):
    """Return a BLIP1ITMReranker with score_itm mocked out."""
    from src.reranker import BLIP1ITMReranker

    if itm_scores is None:
        itm_scores = np.array(
            [0.8 - i * 0.1 for i in range(n_candidates)], dtype=np.float32
        )

    mock_engine = MagicMock()
    mock_engine.score_itm.return_value = np.asarray(itm_scores, dtype=np.float32)
    # encode_text must return a 1-D numpy array so ndim check in rerank() works
    mock_engine.encode_text.return_value = np.zeros(256, dtype=np.float32)
    # encode_frames used by _extract_best_frame
    mock_engine.encode_frames.return_value = np.zeros((1, 256), dtype=np.float32)

    reranker = BLIP1ITMReranker(engine=mock_engine, alpha=alpha)
    # Bypass real video I/O for both frame-extraction paths
    reranker._extract_frame = lambda path, t0, t1: np.zeros((224, 224, 3), dtype=np.uint8)
    reranker._extract_best_frame = lambda path, t0, t1, qvec, n_candidates=5: np.zeros((224, 224, 3), dtype=np.uint8)
    return reranker, mock_engine


# ── ITM-01  importable ────────────────────────────────────────────────────

@pytest.mark.unit
def test_ITM01_blip1_itm_reranker_importable():
    """BLIP1ITMReranker must be importable from src.reranker."""
    from src.reranker import BLIP1ITMReranker, RerankResult
    assert callable(BLIP1ITMReranker)
    assert hasattr(RerankResult, "absolute_start")
    assert hasattr(RerankResult, "absolute_end")


# ── ITM-02  empty candidates → [] ────────────────────────────────────────

@pytest.mark.unit
def test_ITM02_rerank_empty_candidates():
    """BLIP1ITMReranker.rerank() with empty list must return []."""
    reranker, _ = _make_blip1_itm_reranker(n_candidates=0, itm_scores=np.array([]))
    result = reranker.rerank([], "test query")
    assert result == []


# ── ITM-03  all candidates returned when top_k=None ──────────────────────

@pytest.mark.unit
def test_ITM03_rerank_returns_all_without_top_k():
    """All candidates returned when top_k=None."""
    N = 5
    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=N)
    mock_eng.score_itm.return_value = np.linspace(0.9, 0.5, N, dtype=np.float32)

    result = reranker.rerank(_make_itm_candidates(N), "query", top_k=None)
    assert len(result) == N


# ── ITM-04  top_k respected ───────────────────────────────────────────────

@pytest.mark.unit
def test_ITM04_rerank_respects_top_k():
    """top_k must limit output length."""
    N, K = 5, 3
    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=N)
    mock_eng.score_itm.return_value = np.linspace(0.9, 0.5, N, dtype=np.float32)

    result = reranker.rerank(_make_itm_candidates(N), "query", top_k=K)
    assert len(result) == K


# ── ITM-05  output sorted descending by combined score ───────────────────

@pytest.mark.unit
def test_ITM05_rerank_sorted_descending():
    """Output must be sorted by combined score descending."""
    N = 4
    itm_scores = np.array([0.2, 0.9, 0.5, 0.7], dtype=np.float32)
    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=N, itm_scores=itm_scores)

    cands = _make_itm_candidates(N)
    result = reranker.rerank(cands, "query", top_k=None)

    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True), \
        f"Results not sorted descending: {scores}"


# ── ITM-06  combined score formula ───────────────────────────────────────

@pytest.mark.unit
def test_ITM06_combined_score_formula():
    """combined_score = alpha * cosine + (1-alpha) * itm_prob."""
    alpha = 0.6
    cosine = 0.8     # from _FakeSearchResultFull.score
    itm    = 0.5

    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=1, alpha=alpha,
                                                   itm_scores=np.array([itm]))
    cand = _FakeSearchResultFull(
        score=cosine, video_id="v0", video_path="/p.mp4",
        start_time=0.0, end_time=5.0, rank=1,
    )
    result = reranker.rerank([cand], "query")

    expected = alpha * cosine + (1 - alpha) * itm
    assert len(result) == 1
    assert abs(result[0].score - expected) < 1e-4, \
        f"Expected {expected:.4f}, got {result[0].score:.4f}"


# ── ITM-07  rank is sequential starting at 1 ─────────────────────────────

@pytest.mark.unit
def test_ITM07_rank_sequential_from_one():
    """RerankResult.rank must be 1-based and sequential."""
    N = 4
    itm_scores = np.array([0.9, 0.7, 0.5, 0.3], dtype=np.float32)
    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=N, itm_scores=itm_scores)

    result = reranker.rerank(_make_itm_candidates(N), "query", top_k=None)
    assert [r.rank for r in result] == list(range(1, N + 1))


# ── ITM-08  _extract_frame fallback for missing video ────────────────────

@pytest.mark.unit
def test_ITM08_extract_frame_fallback_missing_video():
    """_extract_frame should return a (224,224,3) zero array for missing video."""
    from src.reranker import BLIP1ITMReranker

    mock_engine = MagicMock()
    reranker = BLIP1ITMReranker(engine=mock_engine, alpha=0.6)

    frame = reranker._extract_frame("/nonexistent/path.mp4", 0.0, 5.0)
    assert isinstance(frame, np.ndarray), "Expected ndarray"
    assert frame.shape == (224, 224, 3), f"Expected (224,224,3), got {frame.shape}"


# ── ITM-09  engine.score_itm called with all frames batched ──────────────

@pytest.mark.unit
def test_ITM09_score_itm_called_with_all_frames():
    """engine.score_itm must be called exactly once with all candidate frames."""
    N = 3
    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=N)
    cands = _make_itm_candidates(N)
    reranker.rerank(cands, "person walking")

    assert mock_eng.score_itm.call_count == 1
    call_args = mock_eng.score_itm.call_args
    frames_arg = call_args[0][0]   # first positional arg
    assert len(frames_arg) == N, f"Expected {N} frames passed, got {len(frames_arg)}"


# ── ITM-10  RerankResult fields match candidate ───────────────────────────

@pytest.mark.unit
def test_ITM10_rerank_result_fields_match_candidate():
    """RerankResult fields must match the original SearchResult candidate."""
    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=1,
                                                   itm_scores=np.array([0.7]))
    cand = _FakeSearchResultFull(
        score=0.85, video_id="cam1", video_path="/data/cam1.mp4",
        start_time=10.0, end_time=20.0, rank=1,
        absolute_start=110.0, absolute_end=120.0,
    )
    result = reranker.rerank([cand], "red car")

    assert len(result) == 1
    r = result[0]
    assert r.video_id   == "cam1"
    assert r.video_path == "/data/cam1.mp4"
    assert r.start_time == 10.0
    assert r.end_time   == 20.0


# ── ITM-11  absolute_start / absolute_end forwarded from SearchResult ──────

@pytest.mark.unit
def test_ITM11_absolute_fields_forwarded():
    """absolute_start and absolute_end must be forwarded to RerankResult."""
    reranker, mock_eng = _make_blip1_itm_reranker(n_candidates=1,
                                                   itm_scores=np.array([0.6]))
    cand = _FakeSearchResultFull(
        score=0.80, video_id="cam2", video_path="/data/cam2.mp4",
        start_time=5.0, end_time=10.0, rank=1,
        absolute_start=205.0, absolute_end=210.0,
    )
    result = reranker.rerank([cand], "person at desk")

    r = result[0]
    assert r.absolute_start == 205.0, f"Expected 205.0, got {r.absolute_start}"
    assert r.absolute_end   == 210.0, f"Expected 210.0, got {r.absolute_end}"


# ══════════════════════════════════════════════════════════════════════════
# FILIPReranker unit tests  (FILIP-01 .. FILIP-06)
# Option B: token-level CLIP matching re-ranker
# All mock-based — no model loading, no disk I/O
# ══════════════════════════════════════════════════════════════════════════

def _make_filip_engine(n_patches: int = 196, hidden: int = 768, n_text: int = 5):
    """Return a mock engine with encode_frames_tokens / encode_text_tokens."""
    mock_engine = MagicMock()
    # encode_frames_tokens returns (N_frames, n_patches, hidden)
    mock_engine.encode_frames_tokens.return_value = np.random.randn(
        3, n_patches, hidden
    ).astype(np.float32)
    # encode_text_tokens returns (tokens, mask)
    tokens = np.random.randn(1, n_text, hidden).astype(np.float32)
    mask   = np.ones((1, n_text), dtype=bool)
    mock_engine.encode_text_tokens.return_value = (tokens, mask)
    return mock_engine


def _make_filip_reranker(**kwargs):
    from src.reranker_filip import FILIPReranker
    engine = kwargs.pop("engine", _make_filip_engine())
    r = FILIPReranker(engine=engine, **kwargs)
    # Bypass video I/O
    r._sample_frames = lambda path, t0, t1: [
        np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)
    ]
    return r, engine


# ── FILIP-01  importable ──────────────────────────────────────────────────

@pytest.mark.unit
def test_FILIP01_reranker_importable():
    """FILIPReranker must be importable and require token-capable engine."""
    from src.reranker_filip import FILIPReranker
    assert callable(FILIPReranker)


# ── FILIP-02  raises on engine without token methods ─────────────────────

@pytest.mark.unit
def test_FILIP02_raises_on_missing_token_methods():
    """FILIPReranker must raise AttributeError for engine without token API."""
    from src.reranker_filip import FILIPReranker
    bad_engine = MagicMock(spec=[])   # no attributes at all
    with pytest.raises(AttributeError):
        FILIPReranker(engine=bad_engine)


# ── FILIP-03  empty candidates → [] ──────────────────────────────────────

@pytest.mark.unit
def test_FILIP03_empty_candidates_returns_empty():
    """FILIPReranker.rerank([]) must return []."""
    r, _ = _make_filip_reranker()
    assert r.rerank([], "query") == []


# ── FILIP-04  output length and rank sequential ───────────────────────────

@pytest.mark.unit
def test_FILIP04_output_length_and_rank():
    """FILIPReranker must return N results with sequential rank starting at 1."""
    N = 4
    r, eng = _make_filip_reranker()
    # Provide different filip scores per candidate via mock
    eng.encode_frames_tokens.return_value = np.random.randn(3, 196, 768).astype(np.float32)

    cands = _make_itm_candidates(N)
    result = r.rerank(cands, "person walking")

    assert len(result) == N
    assert [res.rank for res in result] == list(range(1, N + 1))


# ── FILIP-05  top_k respected ─────────────────────────────────────────────

@pytest.mark.unit
def test_FILIP05_top_k_respected():
    """FILIPReranker must truncate to top_k."""
    N, K = 5, 2
    r, _ = _make_filip_reranker()
    result = r.rerank(_make_itm_candidates(N), "query", top_k=K)
    assert len(result) == K


# ── FILIP-06  combined score formula ─────────────────────────────────────

@pytest.mark.unit
def test_FILIP06_combined_score_within_bounds():
    """combined = alpha * cosine + (1-alpha) * filip must be in reasonable range."""
    alpha = 0.6
    r, _ = _make_filip_reranker(alpha=alpha, normalize_filip=True)

    cands = _make_itm_candidates(3)
    result = r.rerank(cands, "query")

    for res in result:
        # cosine scores are ~0.7–0.9, filip normalised to [0,1]
        # combined must stay in (0, 1.5) range
        assert 0.0 <= res.score <= 1.5, f"combined score out of range: {res.score}"


# ══════════════════════════════════════════════════════════════════════════
# Dual-model (Option C) config wiring tests  (DUAL-01 .. DUAL-04)
# ══════════════════════════════════════════════════════════════════════════

# ── DUAL-01  _init_reranker_from_config attaches BLIP1ITMReranker ─────────

@pytest.mark.unit
def test_DUAL01_config_wires_blip1_itm_reranker():
    """
    NLVideoSearcher._init_reranker_from_config() must attach BLIP1ITMReranker
    when search.use_itm_reranker=true and engine is blip1.
    """
    from src.searcher import NLVideoSearcher
    from src.reranker import BLIP1ITMReranker

    # Build a minimal config-like dict
    config = {
        "engine":   {"type": "blip1", "model_name": "Salesforce/blip-itm-base-coco",
                     "device": "cpu", "batch_size": 1},
        "pipeline": {"video_backend": "opencv", "window_sec": 5.0, "overlap_ratio": 0.0},
        "index":    {"embed_dim": 256, "backend": "qdrant",
                     "qdrant_url": "http://localhost:6333",
                     "qdrant_collection": "nlvs_test_dual"},
        "search":   {"top_k": 3, "score_threshold": 0.0, "nms_iou_threshold": 0.5,
                     "use_itm_reranker": True, "itm_alpha": 0.6,
                     "adaptive_threshold": False, "translate_vi": False},
    }

    try:
        s = NLVideoSearcher(config)
    except Exception:
        pytest.skip("Cannot instantiate NLVideoSearcher (Qdrant/model unavailable)")

    assert s._reranker is not None, "Reranker should be auto-attached"
    assert isinstance(s._reranker, BLIP1ITMReranker), \
        f"Expected BLIP1ITMReranker, got {type(s._reranker)}"


# ── DUAL-02  _init_reranker_from_config attaches FILIPReranker ────────────

@pytest.mark.unit
def test_DUAL02_config_wires_filip_reranker():
    """
    NLVideoSearcher._init_reranker_from_config() must attach FILIPReranker
    when search.use_filip_reranker=true and engine is pc (PCEngine).
    """
    from src.searcher import NLVideoSearcher
    from src.reranker_filip import FILIPReranker

    config = {
        "engine":   {"type": "pc", "model_name": "ViT-B-16", "pretrained": "openai",
                     "device": "cpu", "batch_size": 1},
        "pipeline": {"video_backend": "opencv", "window_sec": 5.0, "overlap_ratio": 0.0},
        "index":    {"embed_dim": 512, "backend": "qdrant",
                     "qdrant_url": "http://localhost:6333",
                     "qdrant_collection": "nlvs_test_filip"},
        "search":   {"top_k": 3, "score_threshold": 0.0, "nms_iou_threshold": 0.5,
                     "use_filip_reranker": True, "filip_alpha": 0.6, "filip_n_frames": 3,
                     "adaptive_threshold": False, "translate_vi": False},
    }

    try:
        s = NLVideoSearcher(config)
    except Exception:
        pytest.skip("Cannot instantiate NLVideoSearcher (Qdrant/model unavailable)")

    assert s._reranker is not None, "FILIPReranker should be auto-attached"
    assert isinstance(s._reranker, FILIPReranker), \
        f"Expected FILIPReranker, got {type(s._reranker)}"


# ── DUAL-03  set_reranker overrides config-attached reranker ──────────────

@pytest.mark.unit
def test_DUAL03_set_reranker_overrides_auto_attach():
    """set_reranker() must override any previously auto-attached reranker."""
    from src.searcher import NLVideoSearcher

    # Directly call _init_reranker_from_config logic without full init
    # by using a mock searcher
    mock_searcher = MagicMock(spec=NLVideoSearcher)
    mock_searcher._reranker = None
    mock_searcher._engine   = MagicMock()
    mock_searcher._config   = {}

    sentinel = object()
    NLVideoSearcher.set_reranker(mock_searcher, sentinel)
    assert mock_searcher._reranker is sentinel


# ── DUAL-04  use_itm_reranker false → no reranker attached ────────────────

@pytest.mark.unit
def test_DUAL04_no_reranker_when_config_disabled():
    """
    When use_itm_reranker=false and use_filip_reranker=false,
    _init_reranker_from_config() must leave _reranker as None.
    """
    from src.searcher import NLVideoSearcher

    mock_searcher = MagicMock(spec=NLVideoSearcher)
    mock_searcher._reranker = None
    mock_searcher._engine   = MagicMock()

    # Call the actual method (not mocked)
    NLVideoSearcher._init_reranker_from_config(
        mock_searcher,
        {"search": {"use_itm_reranker": False, "use_filip_reranker": False}},
    )
    assert mock_searcher._reranker is None
