"""
engines/languagebind_engine.py
-------------------------------
Phase 3: LanguageBind video-language engine (RESEARCH_SOTA_NLVS.md §III1).

LanguageBind (Zhu et al., NeurIPS 2023) trains a ViT-L/14 video encoder
bound to language via cross-modal contrastive learning on VIDAL-10M (10M
video-text pairs).  It achieves 55.6% MSR-VTT R@1 vs 43.1% for CLIP4Clip.

Key differences from standard CLIP:
- Video-native: temporal attention over 14 frames per clip.
- Trained on real video-text pairs → better action & event understanding.
- Embedding dim: 768 (same as EVA-CLIP ViT-L/14).
- Compatible with the same Faiss index as Phase 1 EVA-CLIP backbone.

VRAM budget (GTX 1650 Ti, 4 GB):
- LanguageBind ViT-L/14 FP16 ≈ 1.7 GB  → within 4 GB limit.

Usage (via factory)
-------------------
    config = {
        "engine": {
            "type":        "languagebind",
            "model_name":  "LanguageBind/LanguageBind_Video_FT",
            "device":      "cuda",
            "batch_size":  4,
            "frames_per_window": 14,   # LanguageBind expects 14 frames
        },
        "index": {"embed_dim": 768},
    }
    engine = create_engine(config)
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base_engine import InferenceEngine

LANGUAGEBIND_AVAILABLE = False
try:
    from languagebind import LanguageBind, to_device, transform_dict  # type: ignore
    from languagebind import LanguageBindImageTokenizer                # type: ignore
    LANGUAGEBIND_AVAILABLE = True
except ImportError:
    pass

NUM_FRAMES_LB = 14   # LanguageBind video encoder fixed temporal dimension


def _get_device(device_str=None):
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageBindEngine(InferenceEngine):
    """
    InferenceEngine backed by LanguageBind_Video_FT.

    Parameters
    ----------
    engine_cfg : dict
        Sub-dict from config['engine']. Recognised keys:
        - model_name   (str, default "LanguageBind/LanguageBind_Video_FT")
        - device       (str | None)
        - batch_size   (int, default 4)
    """

    EMBED_DIM_VALUE = 768

    def __init__(self, engine_cfg: dict) -> None:
        if not LANGUAGEBIND_AVAILABLE:
            raise ImportError(
                "LanguageBind is not installed. "
                "Install with: pip install languagebind"
            )
        model_name = engine_cfg.get("model_name", "LanguageBind/LanguageBind_Video_FT")
        device_str = engine_cfg.get("device", None)
        self._batch_size = int(engine_cfg.get("batch_size", 4))
        self._device = _get_device(device_str)

        clip_type = {"video": model_name}
        self._model = LanguageBind(clip_type=clip_type, cache_dir=None)
        self._model = self._model.to(self._device)
        if self._device.type == "cuda":
            self._model = self._model.half()
        self._model.eval()

        # Tokenizer: LanguageBind reuses the CLIP image tokenizer for text
        tokenizer_name = engine_cfg.get(
            "tokenizer_name", "LanguageBind/LanguageBind_Image"
        )
        self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
            tokenizer_name, cache_dir=None
        )
        # Video transform (resize, normalize to LanguageBind spec)
        self._video_transform = transform_dict["video"]

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM_VALUE

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Per-frame encoding (single frame at a time, no temporal context).
        Useful for building per-frame index when LanguageBind is used as
        a visual feature extractor.

        Returns shape (N, 768).
        """
        if not frames_bgr:
            return np.zeros((0, self.EMBED_DIM_VALUE), dtype=np.float32)

        results = []
        for frame in frames_bgr:
            # Wrap single frame in a temporal dim
            seg_emb = self.encode_segment_frames([frame])
            results.append(seg_emb)
        return np.stack(results, axis=0).astype(np.float32)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text query/queries. Returns (N, 768) float32 L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self._tokenizer(
            texts, max_length=77, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        inputs = to_device(inputs, self._device)
        if self._device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        with torch.no_grad():
            text_out = self._model(language=inputs)
        vecs = F.normalize(text_out["language"].float(), dim=-1)
        return vecs.cpu().numpy().astype(np.float32)

    def encode_segment_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Encode a video segment using LanguageBind's temporal attention.

        Pads or uniformly sub-samples to exactly NUM_FRAMES_LB=14 frames.
        Returns shape (768,) — a single L2-normalised segment embedding.
        """
        frames = self._sample_or_pad(frames_bgr, NUM_FRAMES_LB)
        video_tensor = self._frames_to_tensor(frames)   # (1, C, T, H, W)

        with torch.no_grad():
            video_out = self._model(video={"pixel_values": video_tensor})
        vec = F.normalize(video_out["video"].float(), dim=-1).squeeze(0)
        return vec.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_or_pad(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
        """Uniformly sub-sample or repeat-pad frames to exactly n."""
        if not frames:
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * n
        if len(frames) == n:
            return frames
        if len(frames) > n:
            idx = [int(i * len(frames) / n) for i in range(n)]
            return [frames[i] for i in idx]
        out = list(frames)
        while len(out) < n:
            out.append(frames[-1])
        return out

    def _frames_to_tensor(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """
        Convert list of BGR frames to LanguageBind video tensor.
        Shape: (1, C, T, H, W) — batch=1, temporal=T frames.
        """
        pil_frames = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]
        # transform_dict["video"] returns a tensor of shape (C, T, H, W)
        video_tensor = self._video_transform(pil_frames)
        video_tensor = video_tensor.unsqueeze(0).to(self._device)   # add batch dim
        if self._device.type == "cuda":
            video_tensor = video_tensor.half()
        return video_tensor
