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
    Load the PC CLIP engine once for the whole test session.
    Tests that need CLIP should request this fixture.
    Phase 1: uses EVA02-L-14 (embed_dim=768) with graceful fallback to ViT-B-16.
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
def video_index_with_data():
    """
    A VideoIndex pre-populated with 10 synthetic unit vectors.
    Shared across all tests in the session.
    """
    from src.indexer import VideoIndex, SegmentMeta

    idx = VideoIndex(embed_dim=EMBED_DIM, use_gpu=False)
    vecs = make_random_unit_vectors(10)
    metas = [
        SegmentMeta(
            video_id=f"vid{i}",
            video_path=f"/fake/vid{i}.mp4",
            start_time=float(i * 5),
            end_time=float(i * 5 + 5),
        )
        for i in range(10)
    ]
    idx.add(vecs, metas)
    return idx, vecs, metas


@pytest.fixture(scope="session")
def searcher_with_real_video(real_video_path, tmp_path_factory):
    """
    NLVideoSearcher with the real video already indexed.
    Heavy fixture — created once per session.
    """
    import warnings
    warnings.filterwarnings("ignore")

    index_dir = str(tmp_path_factory.mktemp("searcher_index"))
    from src.searcher import NLVideoSearcher
    s = NLVideoSearcher.from_params(
        index_dir=index_dir,
        use_sliding_window=True,
        window_sec=5.0,
        overlap_ratio=0.5,
        frames_per_window=3,    # fewer frames → faster session fixture
    )
    s.index_video(real_video_path)
    return s
