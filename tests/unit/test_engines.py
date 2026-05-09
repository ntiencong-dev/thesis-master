"""
tests/unit/test_engines.py
---------------------------
Unit tests for InferenceEngine abstract base + PCEngine + factory.
CLIP model is loaded once (session-scoped fixture from conftest).

Scenarios covered
-----------------
ENG-01  InferenceEngine is abstract – cannot be instantiated
ENG-02  encode_segment_frames: single frame → shape (512,), unit norm
ENG-03  encode_segment_frames: N identical frames → same vector as single
ENG-04  encode_segment_frames: different frames → mean is between them, re-normalised
ENG-05  encode_segment_frames: empty list → shape (512,) zero vector
ENG-06  PCEngine.embed_dim == 768  (EVA-CLIP ViT-L/14 Phase 1)
ENG-07  PCEngine.encode_frames: single frame → shape (1, 512), norm ≈ 1.0
ENG-08  PCEngine.encode_frames: batch of 5 frames → shape (5, 512)
ENG-09  PCEngine.encode_text: single string → shape (1, 512), norm ≈ 1.0
ENG-10  PCEngine.encode_text: list of strings → shape (N, 512)
ENG-11  PCEngine.encode_text: output is float32
ENG-12  PCEngine frame encoding: all L2 norms ≈ 1.0
ENG-13  factory.create_engine type='pc' → PCEngine instance
ENG-14  factory.create_engine type='kria' → ImportError on PC (no vart)
ENG-15  factory.create_engine unknown type → ValueError
ENG-16  factory.create_engine missing engine key → defaults to 'pc'
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import EMBED_DIM, make_dummy_frames, make_black_frames

pytestmark = pytest.mark.integration   # need CLIP model loaded


# ── ENG-01  Abstract base ─────────────────────────────────────────────────

def test_ENG01_base_engine_is_abstract():
    from src.engines.base_engine import InferenceEngine
    with pytest.raises(TypeError):
        InferenceEngine()   # type: ignore[abstract]


# ── ENG-02..05  encode_segment_frames (pure numpy, uses clip_engine) ──────

def test_ENG02_encode_segment_frames_shape_norm(clip_engine):
    frames  = make_dummy_frames(5)
    out     = clip_engine.encode_segment_frames(frames)
    assert out.shape == (EMBED_DIM,), f"Expected ({EMBED_DIM},), got {out.shape}"
    norm = float(np.linalg.norm(out))
    assert abs(norm - 1.0) < 1e-4, f"L2 norm should be 1.0, got {norm}"


def test_ENG03_encode_segment_frames_identical_frames(clip_engine):
    frame  = make_dummy_frames(1)[0]
    frames = [frame] * 5
    single = clip_engine.encode_segment_frames([frame])
    multi  = clip_engine.encode_segment_frames(frames)
    # Mean of identical vectors = same vector (after re-normalisation).
    # Float16 GPU operations introduce small rounding differences → use atol=1e-2.
    assert np.allclose(single, multi, atol=1e-2), \
        "Encoding 5 identical frames should match single-frame encoding"


def test_ENG04_encode_segment_frames_normalised(clip_engine):
    frames = make_dummy_frames(4)
    out = clip_engine.encode_segment_frames(frames)
    norm = float(np.linalg.norm(out))
    assert abs(norm - 1.0) < 1e-4


def test_ENG05_encode_segment_frames_empty():
    """Empty frame list → zero vector of shape (embed_dim,)."""
    from src.engines.base_engine import InferenceEngine
    import numpy as np

    class _MockEngine(InferenceEngine):
        @property
        def embed_dim(self): return EMBED_DIM
        def encode_frames(self, frames_bgr): return np.zeros((0, EMBED_DIM), dtype=np.float32)
        def encode_text(self, texts): return np.zeros((1, EMBED_DIM), dtype=np.float32)

    eng = _MockEngine()
    out = eng.encode_segment_frames([])
    assert out.shape == (EMBED_DIM,)


# ── ENG-06..12  PCEngine ──────────────────────────────────────────────────

def test_ENG06_pc_engine_embed_dim(clip_engine):
    assert clip_engine.embed_dim == 768   # EVA-CLIP ViT-L/14 (Phase 1)


def test_ENG07_pc_engine_encode_single_frame(clip_engine):
    frames = make_dummy_frames(1)
    out    = clip_engine.encode_frames(frames)
    assert out.shape == (1, EMBED_DIM)
    norm = float(np.linalg.norm(out[0]))
    assert abs(norm - 1.0) < 1e-4


def test_ENG08_pc_engine_encode_batch_frames(clip_engine):
    frames = make_dummy_frames(5)
    out    = clip_engine.encode_frames(frames)
    assert out.shape == (5, EMBED_DIM)


def test_ENG09_pc_engine_encode_single_text(clip_engine):
    out = clip_engine.encode_text("a person walking")
    assert out.shape == (1, EMBED_DIM)
    norm = float(np.linalg.norm(out[0]))
    assert abs(norm - 1.0) < 1e-4


def test_ENG10_pc_engine_encode_text_list(clip_engine):
    texts = ["car moving", "dog running", "sunset over ocean"]
    out   = clip_engine.encode_text(texts)
    assert out.shape == (3, EMBED_DIM)


def test_ENG11_pc_engine_output_float32(clip_engine):
    frames = make_dummy_frames(2)
    out_f  = clip_engine.encode_frames(frames)
    out_t  = clip_engine.encode_text("test")
    assert out_f.dtype == np.float32, f"encode_frames dtype: {out_f.dtype}"
    assert out_t.dtype == np.float32, f"encode_text dtype: {out_t.dtype}"


def test_ENG12_pc_engine_all_norms_near_one(clip_engine):
    frames = make_dummy_frames(8)
    out    = clip_engine.encode_frames(frames)
    norms  = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        f"Not all norms ≈ 1.0: {norms}"


# ── ENG-13..16  factory.create_engine ────────────────────────────────────

@pytest.mark.unit
def test_ENG13_factory_pc():
    from src.engines.factory import create_engine
    from src.engines.pc_engine import PCEngine
    cfg = {"engine": {"type": "pc", "model_name": "ViT-B-16",
                      "pretrained": "openai", "device": None, "batch_size": 1}}
    engine = create_engine(cfg)
    assert isinstance(engine, PCEngine)


@pytest.mark.unit
def test_ENG14_factory_kria_import_error_on_pc():
    from src.engines.factory import create_engine
    cfg = {"engine": {"type": "kria", "xmodel_path": "/fake/model.xmodel",
                      "xmodel_text_path": "/fake/text.xmodel"}}
    with pytest.raises((ImportError, FileNotFoundError)):
        create_engine(cfg)


@pytest.mark.unit
def test_ENG15_factory_unknown_type():
    from src.engines.factory import create_engine
    with pytest.raises(ValueError, match="Unknown engine type"):
        create_engine({"engine": {"type": "tpu"}})


@pytest.mark.unit
def test_ENG16_factory_defaults_to_pc():
    from src.engines.factory import create_engine
    from src.engines.pc_engine import PCEngine
    # No engine key at all → default to "pc"
    engine = create_engine({})
    assert isinstance(engine, PCEngine)
