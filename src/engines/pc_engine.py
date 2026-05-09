"""
engines/pc_engine.py
---------------------
PC backend: CLIP ViT-B/16 via open_clip (PyTorch + CUDA).

Delegates actual CLIP calls to the existing CLIPFeatureExtractor so
no logic is duplicated.  This class only adapts the interface to
InferenceEngine's contract.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .base_engine import InferenceEngine
from ..feature_extractor import CLIPFeatureExtractor


class PCEngine(InferenceEngine):
    """
    Parameters (from config['engine'] dict)
    ----------------------------------------
    model_name : str   default "ViT-B-16"
    pretrained : str   default "openai"
    device     : str   default None (auto-detect CUDA)
    batch_size : int   default 32
    """

    def __init__(self, engine_cfg: dict) -> None:
        self._extractor = CLIPFeatureExtractor(
            model_name=engine_cfg.get("model_name", "ViT-B-16"),
            pretrained=engine_cfg.get("pretrained", "openai"),
            device=engine_cfg.get("device", None),
            batch_size=engine_cfg.get("batch_size", 32),
        )

    # ------------------------------------------------------------------
    # InferenceEngine contract
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._extractor.EMBED_DIM

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        return self._extractor.encode_frames(frames_bgr)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self._extractor.encode_text(texts)
