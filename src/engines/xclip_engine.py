"""
engines/xclip_engine.py
------------------------
Phase 2: X-CLIP video-language engine (RESEARCH_SOTA_NLVS.md §II2).

X-CLIP (microsoft/xclip-base-patch16) extends CLIP with temporal attention
across frames and a cross-frame communication mechanism, giving it strong
performance on action / event retrieval.

Design choices for GTX 1650 Ti (4 GB VRAM)
-------------------------------------------
* Uses ``microsoft/xclip-base-patch16`` (embedding dim = 512) — fits in < 2 GB.
* Video encoder expects exactly NUM_FRAMES=8 frames per clip.
  → Frames are padded (by repeating the last frame) or uniformly sub-sampled
    to always produce exactly 8 frames.
* FP16 on CUDA, FP32 on CPU.
* Text encoder identical to standard CLIP ViT-B/16 (dim=512, context=77).

Usage (via factory)
-------------------
    config = {
        "engine": {
            "type":        "xclip",
            "model_name":  "microsoft/xclip-base-patch16",
            "device":      "cuda",
            "batch_size":  8,
            "frames_per_window": 8,
        }
    }
    engine = create_engine(config)
    seg_emb = engine.encode_segment_frames(frames)  # (512,)
    txt_emb = engine.encode_text(["a person running"])  # (1, 512)
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base_engine import InferenceEngine

NUM_FRAMES = 8   # X-CLIP fixed temporal dimension


def _get_device(device_str=None):
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XCLIPEngine(InferenceEngine):
    """
    InferenceEngine backed by X-CLIP (microsoft/xclip-base-patch16).

    Parameters
    ----------
    engine_cfg : dict
        Sub-dict from config['engine']. Recognised keys:
        - model_name   (str, default "microsoft/xclip-base-patch16")
        - device       (str | None)
        - batch_size   (int, default 8)
    """

    EMBED_DIM_VALUE = 512

    def __init__(self, engine_cfg: dict) -> None:
        model_name = engine_cfg.get("model_name", "microsoft/xclip-base-patch16")
        device_str = engine_cfg.get("device", None)
        self._batch_size = int(engine_cfg.get("batch_size", 8))
        self._device = _get_device(device_str)

        from transformers import XCLIPProcessor, XCLIPModel  # type: ignore
        self._processor = XCLIPProcessor.from_pretrained(model_name)
        self._model = XCLIPModel.from_pretrained(model_name)
        self._model.to(self._device).eval()
        if self._device.type == "cuda":
            self._model = self._model.half()

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM_VALUE

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Encode individual frames in isolation (single-frame CLIP mode).

        This falls back to the CLIP visual encoder applied per-frame, then
        L2-normalises. Returns shape (N, 512).

        Note: For best X-CLIP performance use ``encode_segment_frames``
        which leverages the temporal attention mechanism.
        """
        if not frames_bgr:
            return np.zeros((0, self.EMBED_DIM_VALUE), dtype=np.float32)

        results = []
        for i in range(0, len(frames_bgr), self._batch_size):
            batch = frames_bgr[i : i + self._batch_size]
            clip_input = self._frames_to_xclip_input(batch, n_frames=1)
            with torch.no_grad():
                out = self._model.get_video_features(**clip_input)  # (B, 512)
            vecs = F.normalize(out.float(), dim=-1).cpu().numpy()
            results.append(vecs)
        return np.concatenate(results, axis=0).astype(np.float32)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text query/queries. Returns (N, 512) float32 L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)
        if self._device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)  # (N, 512)
        vecs = F.normalize(out.float(), dim=-1).cpu().numpy()
        return vecs.astype(np.float32)

    def encode_segment_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Encode a video segment using X-CLIP temporal attention.

        Pads or uniformly sub-samples ``frames_bgr`` to exactly NUM_FRAMES=8.
        Returns shape (512,) — a single L2-normalised segment embedding.
        """
        frames = self._sample_or_pad(frames_bgr, NUM_FRAMES)
        clip_input = self._frames_to_xclip_input(frames, n_frames=NUM_FRAMES)
        # processor returns pixel_values of shape (1, T, C, H, W)
        with torch.no_grad():
            out = self._model.get_video_features(**clip_input)  # (1, 512)
        vec = F.normalize(out.float(), dim=-1).squeeze(0).cpu().numpy()
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_or_pad(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
        """Uniformly sub-sample or repeat-pad to exactly n frames."""
        if not frames:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            return [blank] * n
        if len(frames) == n:
            return frames
        if len(frames) > n:
            # Uniform sub-sample
            indices = [int(i * len(frames) / n) for i in range(n)]
            return [frames[idx] for idx in indices]
        # Repeat last frame to reach n
        out = list(frames)
        while len(out) < n:
            out.append(frames[-1])
        return out

    def _frames_to_xclip_input(
        self, frames_bgr: List[np.ndarray], n_frames: int
    ) -> dict:
        """
        Convert a list of BGR uint8 numpy arrays to the dict format expected
        by XCLIPProcessor (pixel_values as tensor).
        """
        rgb_images = [
            Image.fromarray(f[:, :, ::-1]) for f in frames_bgr  # BGR→RGB
        ]
        # XCLIPProcessor expects a list of lists of PIL images:
        # [[frame0, frame1, ...]] for batch_size=1
        inputs = self._processor(
            images=[rgb_images],
            return_tensors="pt",
        )
        # Move to device and cast
        out = {}
        for k, v in inputs.items():
            v = v.to(self._device)
            if self._device.type == "cuda" and v.dtype == torch.float32:
                v = v.half()
            out[k] = v
        return out
