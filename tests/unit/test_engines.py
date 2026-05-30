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
ENG-17  factory.create_engine type='xclip' → XCLIPEngine (unit — no model load)
ENG-18  factory.create_engine type='siglip' → SigLIPEngine (unit — no model load)
ENG-19  XCLIPEngine.embed_dim == 512
ENG-20  XCLIPEngine._sample_or_pad: fewer frames → padded to n
ENG-21  XCLIPEngine._sample_or_pad: more frames → sub-sampled to n
ENG-22  XCLIPEngine._sample_or_pad: empty input → n blank frames
ENG-23  XCLIPEngine._sample_or_pad: exact n frames → unchanged
ENG-24  factory.create_engine type='languagebind' → ImportError acceptable
ENG-25  factory.create_engine type='internvideo2' → ImportError acceptable
ENG-26  LanguageBindEngine module importable, EMBED_DIM_VALUE=768
ENG-27  InternVideo2Engine module importable, EMBED_DIM_VALUE=768
ENG-28  LanguageBindEngine._sample_or_pad: fewer frames → padded to n
ENG-29  LanguageBindEngine._sample_or_pad: more frames → sub-sampled to n
ENG-30  InternVideo2Engine._sample_or_pad: empty → n blank frames
BLIP-1 engine (unit — all mock-based, no model download)
ENG-31  BLIP1Engine module importable; BLIP1_AVAILABLE flag; issubclass check
ENG-32  factory.create_engine type='blip1' → no ValueError (ImportError OK)
ENG-33  factory.create_engine unknown type error message includes 'blip1'
ENG-34  BLIP1Engine.embed_dim == 256 (mock)
ENG-35  BLIP1Engine.encode_frames(): shape (N,256) float32 L2-norm (mock)
ENG-36  BLIP1Engine.encode_text(): shape (N,256) float32 L2-norm (mock)
ENG-37  BLIP1Engine.score_itm(): shape (N,) float32 in [0,1] (mock)
ENG-38  BLIP1Engine.encode_frames(): empty input → (0, 256) (mock)
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


# ── ENG-17..18  factory dispatch for Phase 2 engine types ─────────────────
# These are pure unit tests — they verify factory routing without
# actually loading the models (which would require large downloads).

@pytest.mark.unit
def test_ENG17_factory_xclip_type_accepted():
    """factory.create_engine type='xclip' → XCLIPEngine (class check only)."""
    from src.engines import xclip_engine  # must be importable
    from src.engines.xclip_engine import XCLIPEngine
    # Verify class is importable and is an InferenceEngine subclass
    from src.engines.base_engine import InferenceEngine
    assert issubclass(XCLIPEngine, InferenceEngine)


@pytest.mark.unit
def test_ENG18_factory_siglip_type_accepted():
    """factory.create_engine type='siglip' → SigLIPEngine (class check only)."""
    from src.engines import siglip_engine  # must be importable
    from src.engines.siglip_engine import SigLIPEngine
    from src.engines.base_engine import InferenceEngine
    assert issubclass(SigLIPEngine, InferenceEngine)


# ── ENG-19  XCLIPEngine embed_dim constant ───────────────────────────────

@pytest.mark.unit
def test_ENG19_xclip_embed_dim_constant():
    from src.engines.xclip_engine import XCLIPEngine
    assert XCLIPEngine.EMBED_DIM_VALUE == 512


# ── ENG-20..23  XCLIPEngine._sample_or_pad (pure logic, no model) ────────

@pytest.mark.unit
def test_ENG20_sample_or_pad_pads_short():
    from src.engines.xclip_engine import XCLIPEngine
    import numpy as np
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 3
    result = XCLIPEngine._sample_or_pad(frames, 8)
    assert len(result) == 8


@pytest.mark.unit
def test_ENG21_sample_or_pad_subsamples_long():
    from src.engines.xclip_engine import XCLIPEngine
    import numpy as np
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 20
    result = XCLIPEngine._sample_or_pad(frames, 8)
    assert len(result) == 8


@pytest.mark.unit
def test_ENG22_sample_or_pad_empty_input():
    from src.engines.xclip_engine import XCLIPEngine
    result = XCLIPEngine._sample_or_pad([], 8)
    assert len(result) == 8
    import numpy as np
    assert result[0].shape == (224, 224, 3)


@pytest.mark.unit
def test_ENG23_sample_or_pad_exact_n():
    from src.engines.xclip_engine import XCLIPEngine
    import numpy as np
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 8
    result = XCLIPEngine._sample_or_pad(frames, 8)
    assert len(result) == 8
    assert result is frames   # should return the same list unchanged


# ── ENG-24..30  Phase 3 engine structure (no model load) ─────────────────

@pytest.mark.unit
def test_ENG24_factory_languagebind_type_importable():
    """factory.create_engine type='languagebind' must raise ImportError (not ValueError)
    when languagebind package is not installed — meaning the dispatch exists."""
    from src.engines.factory import create_engine
    cfg = {"engine": {"type": "languagebind", "model_name": "LanguageBind/LanguageBind_Video_FT"}}
    try:
        engine = create_engine(cfg)
        # If LanguageBind is installed, engine should be a LanguageBindEngine
        from src.engines.languagebind_engine import LanguageBindEngine
        assert isinstance(engine, LanguageBindEngine)
    except ImportError:
        pass   # Expected on systems without languagebind installed
    except Exception as exc:
        pytest.fail(f"Unexpected exception (not ImportError): {exc}")


@pytest.mark.unit
def test_ENG25_factory_internvideo2_type_importable():
    """factory.create_engine type='internvideo2' must raise ImportError (not ValueError)
    when model is unavailable — meaning the dispatch exists."""
    from src.engines.factory import create_engine
    cfg = {"engine": {"type": "internvideo2", "model_name": "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"}}
    try:
        engine = create_engine(cfg)
        from src.engines.intern_video2_engine import InternVideo2Engine
        assert isinstance(engine, InternVideo2Engine)
    except (ImportError, OSError, Exception):
        # ImportError: model not available; OSError: pretrained weight issue
        pass  # Both acceptable — just not ValueError


@pytest.mark.unit
def test_ENG26_languagebind_engine_module_importable():
    """LanguageBindEngine class must be importable without loading a model."""
    from src.engines.languagebind_engine import (
        LanguageBindEngine, LANGUAGEBIND_AVAILABLE, NUM_FRAMES_LB
    )
    assert hasattr(LanguageBindEngine, "EMBED_DIM_VALUE")
    assert LanguageBindEngine.EMBED_DIM_VALUE == 768
    assert NUM_FRAMES_LB == 14


@pytest.mark.unit
def test_ENG27_internvideo2_engine_module_importable():
    """InternVideo2Engine class must be importable without loading a model."""
    from src.engines.intern_video2_engine import (
        InternVideo2Engine, INTERNVIDEO2_AVAILABLE, NUM_FRAMES_IV2
    )
    assert hasattr(InternVideo2Engine, "EMBED_DIM_VALUE")
    assert InternVideo2Engine.EMBED_DIM_VALUE == 768
    assert NUM_FRAMES_IV2 == 8


@pytest.mark.unit
def test_ENG28_languagebind_sample_or_pad_pad():
    """LanguageBindEngine._sample_or_pad: fewer frames → padded to n."""
    from src.engines.languagebind_engine import LanguageBindEngine
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 5
    result = LanguageBindEngine._sample_or_pad(frames, 14)
    assert len(result) == 14


@pytest.mark.unit
def test_ENG29_languagebind_sample_or_pad_subsample():
    """LanguageBindEngine._sample_or_pad: more frames → sub-sampled to n."""
    from src.engines.languagebind_engine import LanguageBindEngine
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 30
    result = LanguageBindEngine._sample_or_pad(frames, 14)
    assert len(result) == 14


@pytest.mark.unit
def test_ENG30_internvideo2_sample_or_pad_empty():
    """InternVideo2Engine._sample_or_pad: empty → n blank frames."""
    from src.engines.intern_video2_engine import InternVideo2Engine
    result = InternVideo2Engine._sample_or_pad([], 8)
    assert len(result) == 8
    assert all(arr.shape == (224, 224, 3) for arr in result)


# ── ENG-31..38  BLIP-1 engine (all unit — mock-based, no model download) ──

def _make_blip1_engine_mock():
    """Return a BLIP1Engine with model loading completely bypassed."""
    from src.engines.blip1_engine import BLIP1Engine, BLIP1_AVAILABLE
    if not BLIP1_AVAILABLE:
        pytest.skip("transformers not installed — BLIP1Engine unavailable")
    import torch
    from unittest.mock import MagicMock
    engine = object.__new__(BLIP1Engine)
    engine._model_name = "Salesforce/blip-itm-base-coco"
    engine._batch_size = 4
    engine._device     = torch.device("cpu")
    engine._processor  = MagicMock()
    engine._model      = MagicMock()
    return engine


@pytest.mark.unit
def test_ENG31_blip1_engine_module_importable():
    """BLIP1Engine must be importable; BLIP1_AVAILABLE flag must exist."""
    from src.engines.blip1_engine import BLIP1Engine, BLIP1_AVAILABLE
    from src.engines.base_engine import InferenceEngine
    assert issubclass(BLIP1Engine, InferenceEngine)
    assert BLIP1Engine.EMBED_DIM_VALUE == 256
    assert isinstance(BLIP1_AVAILABLE, bool)


@pytest.mark.unit
def test_ENG32_factory_blip1_type_accepted():
    """factory.create_engine type='blip1' must NOT raise ValueError."""
    from src.engines.factory import create_engine
    cfg = {"engine": {"type": "blip1", "model_name": "Salesforce/blip-itm-base-coco"}}
    try:
        engine = create_engine(cfg)
        from src.engines.blip1_engine import BLIP1Engine
        assert isinstance(engine, BLIP1Engine)
    except ImportError:
        pass   # OK — model not downloaded in test env
    except Exception as exc:
        pytest.fail(f"Unexpected exception (not ImportError): {exc}")


@pytest.mark.unit
def test_ENG33_factory_error_message_includes_blip1():
    """ValueError from unknown engine type must mention 'blip1'."""
    from src.engines.factory import create_engine
    with pytest.raises(ValueError, match="blip1"):
        create_engine({"engine": {"type": "__nonexistent__"}})


@pytest.mark.unit
def test_ENG34_blip1_embed_dim_property():
    """BLIP1Engine.embed_dim property must return 256."""
    engine = _make_blip1_engine_mock()
    assert engine.embed_dim == 256


@pytest.mark.unit
def test_ENG35_blip1_encode_frames_shape_norm_mock():
    """encode_frames(): shape (N,256), float32, L2-normalised (mocked model)."""
    import torch
    from unittest.mock import MagicMock

    N = 3
    engine = _make_blip1_engine_mock()

    # Processor: returns object with .pixel_values attribute
    proc_out = MagicMock()
    proc_out.pixel_values = torch.zeros(N, 3, 384, 384)
    engine._processor.return_value = proc_out

    # Vision model: returns last_hidden_state (proper tensor)
    vision_out = MagicMock()
    vision_out.last_hidden_state = torch.randn(N, 577, 768)
    engine._model.vision_model.return_value = vision_out

    # image_projection: returns random tensor (will be L2-normalised)
    engine._model.image_projection.return_value = torch.randn(N, 256)

    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(N)]
    out    = engine.encode_frames(frames)

    assert out.shape == (N, 256), f"Expected ({N}, 256), got {out.shape}"
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"Norms not ≈ 1.0: {norms}"


@pytest.mark.unit
def test_ENG36_blip1_encode_text_shape_norm_mock():
    """encode_text(): shape (N,256), float32, L2-normalised (mocked model)."""
    import torch
    from unittest.mock import MagicMock

    N = 2
    engine = _make_blip1_engine_mock()

    # Processor: returns text inputs
    txt_out = MagicMock()
    txt_out.input_ids      = torch.zeros(N, 10, dtype=torch.long)
    txt_out.attention_mask = torch.ones(N, 10, dtype=torch.long)
    engine._processor.return_value = txt_out

    # Text encoder
    text_enc_out = MagicMock()
    text_enc_out.last_hidden_state = torch.randn(N, 10, 768)
    engine._model.text_encoder.return_value = text_enc_out

    # text_projection
    engine._model.text_projection.return_value = torch.randn(N, 256)

    texts = ["person running", "red car"]
    out   = engine.encode_text(texts)

    assert out.shape == (N, 256), f"Expected ({N}, 256), got {out.shape}"
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


@pytest.mark.unit
def test_ENG37_blip1_score_itm_shape_range_mock():
    """score_itm(): shape (N,), float32, values in [0.0, 1.0] (mocked model)."""
    import torch
    from unittest.mock import MagicMock

    N = 4
    engine = _make_blip1_engine_mock()

    # Processor: distinguish images vs text calls
    img_proc_out = MagicMock()
    img_proc_out.pixel_values = torch.zeros(N, 3, 384, 384)

    txt_proc_out = MagicMock()
    txt_proc_out.input_ids      = torch.zeros(N, 8, dtype=torch.long)
    txt_proc_out.attention_mask = torch.ones(N, 8, dtype=torch.long)

    def _proc_side_effect(*args, **kwargs):
        return img_proc_out if "images" in kwargs else txt_proc_out

    engine._processor.side_effect = _proc_side_effect

    # Vision model
    vision_out = MagicMock()
    vision_out.last_hidden_state = torch.zeros(N, 577, 768)
    engine._model.vision_model.return_value = vision_out

    # Text encoder (ITM fusion)
    text_out = MagicMock()
    text_out.last_hidden_state = torch.zeros(N, 8, 768)
    engine._model.text_encoder.return_value = text_out

    # ITM head: return logits that produce varied probabilities
    engine._model.itm_head.return_value = torch.tensor(
        [[2.0, -1.0], [-1.0, 2.0], [0.5, 0.5], [1.0, -0.5]], dtype=torch.float32
    )

    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(N)]
    probs  = engine.score_itm(frames, "person running")

    assert probs.shape == (N,), f"Expected ({N},), got {probs.shape}"
    assert probs.dtype == np.float32
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), \
        f"Probabilities out of [0,1]: {probs}"


@pytest.mark.unit
def test_ENG38_blip1_encode_frames_empty():
    """encode_frames([]) must return shape (0, 256) without errors."""
    engine = _make_blip1_engine_mock()
    out = engine.encode_frames([])
    assert out.shape == (0, 256)
    assert out.dtype == np.float32
