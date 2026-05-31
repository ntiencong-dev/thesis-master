"""
conftest.py
-----------
Shared fixtures for the entire NLVS test suite.

Fixture scopes
--------------
session  : CLIP engine (loaded once, expensive ~10 s)
session  : real video path guard (skip if file missing)
function : temporary index directory (cleaned up after each test)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REAL_VIDEO = os.path.join(os.path.dirname(__file__), "..", "data", "fi003.mp4")
REAL_VIDEO = os.path.abspath(REAL_VIDEO)

VIDEO_FPS      = 60.0
VIDEO_FRAMES   = 4564
VIDEO_DURATION = VIDEO_FRAMES / VIDEO_FPS   # ≈ 76.07 s
VIDEO_WIDTH    = 1920
VIDEO_HEIGHT   = 1080
EMBED_DIM      = 768   # EVA-CLIP ViT-L/14 (Phase 1 upgrade)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_unit_vectors(n: int, dim: int = EMBED_DIM) -> np.ndarray:
    """Return (n, dim) float32 matrix, each row L2-normalised."""
    rng  = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-8)
    return vecs / norms


def make_dummy_frames(n: int = 5, h: int = 224, w: int = 224) -> List[np.ndarray]:
    """Return n random BGR uint8 224×224 frames."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def make_black_frames(n: int = 5) -> List[np.ndarray]:
    return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_video_path():
    """Skip test if the real video file is not present."""
    if not os.path.isfile(REAL_VIDEO):
        pytest.skip(f"Real video not found: {REAL_VIDEO}")
    return REAL_VIDEO


@pytest.fixture(scope="function")
def tmp_index_dir():
    """Provide a clean temporary directory; remove it after the test."""
    d = tempfile.mkdtemp(prefix="nlvs_test_index_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def clip_engine():
    """
    Load the PC (EVA-CLIP) engine once for the whole test session.
    Used by ENG-02..12 and feature-extractor unit tests.
    Note: this is the legacy PCEngine (768-dim); the primary pipeline now
    uses BLIP1Engine via searcher_with_real_video / NLVideoSearcher.from_params().
    """
    import warnings
    warnings.filterwarnings("ignore")
    from src.engines.pc_engine import PCEngine
    engine = PCEngine({
        "model_name": "EVA02-L-14",
        "pretrained": "merged2b_s4b_b131k",
        "device": None,      # auto-detect CUDA
        "batch_size": 16,
    })
    return engine


@pytest.fixture(scope="session")
def searcher_with_real_video(real_video_path, tmp_path_factory):
    """
    NLVideoSearcher with the real video already indexed.
    Heavy fixture — created once per session.
    Uses a unique Qdrant collection so tests don't pollute nlvs_segments.
    The collection is deleted before indexing so each test session starts
    from exactly 0 points (prevents vector count from drifting across runs).
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Wipe the test collection so previous runs don't accumulate.
    try:
        from qdrant_client import QdrantClient
        _c = QdrantClient(url="http://localhost:6333")
        _c.delete_collection("nlvs_segments_test")
    except Exception:
        pass   # collection may not exist yet; that's fine

    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=3,    # fewer frames → faster session fixture
        model_name="Salesforce/blip-itm-base-coco",  # BLIP-1 ITC+ITM
        embed_dim=256,
        qdrant_collection="nlvs_segments_test",
    )
    s.index_video(real_video_path)
    return s
