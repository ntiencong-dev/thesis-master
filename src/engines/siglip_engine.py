"""
engines/siglip_engine.py
------------------------
Phase 2: SigLIP multilingual engine (RESEARCH_SOTA_NLVS.md §II3).

Uses ``SigLIPFeatureExtractor`` (ViT-L-16-SigLIP-256 / webli) with EMBED_DIM=1024.
Natively supports 100+ languages including Vietnamese — no translation layer needed.

Usage (via factory)
-------------------
    config = {
        "engine": {
            "type":        "siglip",
            "model_name":  "ViT-L-16-SigLIP-256",
            "pretrained":  "webli",
            "device":      "cuda",
            "batch_size":  16,
        },
        "index": {"embed_dim": 1024},
    }
    engine = create_engine(config)
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .base_engine import InferenceEngine


class SigLIPEngine(InferenceEngine):
    """
    InferenceEngine backed by SigLIPFeatureExtractor.

    Parameters
    ----------
    engine_cfg : dict
        Sub-dict from config['engine']. Recognised keys:
        - model_name   (str, default "ViT-L-16-SigLIP-256")
        - pretrained   (str, default "webli")
        - device       (str | None)
        - batch_size   (int, default 16)
    """

    def __init__(self, engine_cfg: dict) -> None:
        from src.feature_extractor import SigLIPFeatureExtractor
        self._extractor = SigLIPFeatureExtractor(
            model_name=engine_cfg.get("model_name", SigLIPFeatureExtractor.MODEL_NAME),
            pretrained=engine_cfg.get("pretrained", SigLIPFeatureExtractor.PRETRAINED),
            device=engine_cfg.get("device", None),
            batch_size=int(engine_cfg.get("batch_size", 16)),
        )

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._extractor.EMBED_DIM

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        return self._extractor.encode_frames(frames_bgr)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self._extractor.encode_text(texts)

    def encode_segment_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """Average-pool frame embeddings → single (D,) segment vector."""
        if not frames_bgr:
            return np.zeros(self.embed_dim, dtype=np.float32)
        frame_embs = self._extractor.encode_frames(frames_bgr)  # (N, D)
        mean = frame_embs.mean(axis=0)                           # (D,)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-8:
            mean = mean / norm
        return mean.astype(np.float32)
