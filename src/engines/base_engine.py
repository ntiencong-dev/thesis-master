"""
engines/base_engine.py
-----------------------
Abstract interface that every inference backend must implement.

Design contract
---------------
* ``encode_frames(frames_bgr)`` — batch-encode BGR video frames.
* ``encode_text(texts)``        — batch-encode free-text queries.
* ``encode_segment_frames(frames_bgr)`` — built-in multi-frame averaging:
  encodes N frames from one segment and returns their mean L2-normalised
  embedding.  Both PC and Kria engines inherit this without override.

All outputs are float32 numpy arrays, L2-normalised, shape (N, embed_dim)
or (embed_dim,) for the averaged variant.

Switching from PC → Kria requires only a config change; the rest of the
pipeline (VideoProcessor, VideoIndex, Searcher, API) is untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class InferenceEngine(ABC):
    """Unified interface for all inference backends (PC/PyTorch, Kria/VART)."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of output embeddings (e.g. 512 for ViT-B/16)."""
        ...

    # ------------------------------------------------------------------
    # Core abstract methods (must be implemented per backend)
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Encode a batch of BGR uint8 frames into L2-normalised embeddings.

        Parameters
        ----------
        frames_bgr : List of H×W×3 numpy uint8 arrays (BGR colour order).
                     All frames should already be resized to the model's
                     input resolution (224×224 for ViT-B/16).

        Returns
        -------
        np.ndarray  shape (N, embed_dim), dtype float32, L2-normalised.
        """
        ...

    @abstractmethod
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode free-text strings into L2-normalised embeddings.

        Parameters
        ----------
        texts : str or List[str]

        Returns
        -------
        np.ndarray  shape (N, embed_dim), dtype float32, L2-normalised.
        """
        ...

    # ------------------------------------------------------------------
    # Shared implementation — multi-frame averaging
    # ------------------------------------------------------------------

    def encode_segment_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Multi-frame averaging for one video segment.

        Encodes all N frames from a single window, then returns their
        element-wise mean re-normalised to the unit sphere.  This gives
        a single representative embedding that captures temporal context
        better than any single frame.

        Parameters
        ----------
        frames_bgr : List of N BGR frames (all from the same segment).

        Returns
        -------
        np.ndarray  shape (embed_dim,), dtype float32, L2-normalised.
        """
        if not frames_bgr:
            return np.zeros(self.embed_dim, dtype=np.float32)

        embs     = self.encode_frames(frames_bgr)   # (N, D)
        mean_emb = embs.mean(axis=0)                 # (D,)
        norm     = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        return mean_emb.astype(np.float32)
