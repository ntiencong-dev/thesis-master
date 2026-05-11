"""
engines/intern_video2_engine.py
--------------------------------
Phase 3: InternVideo2-1B video-language engine (RESEARCH_SOTA_NLVS.md §III2).

InternVideo2 (Wang et al., 2024) is a 1B-parameter video foundation model
trained on 12M video-text pairs with masked video modelling + contrastive
objectives.  The 1B CLIP stage-2 model achieves:
- MSR-VTT R@1  = 57.2%   (best among feasible models)
- DiDeMo  R@1  = 60.5%
- MSVD    R@1  = 90.0%

VRAM budget (GTX 1650 Ti, 4 GB):
- InternVideo2-CLIP-1B FP16 ≈ 2.2 GB → fits within 4 GB limit.
- Cannot coexist with EVA-CLIP at full FP16; use INT8 or swap engines.

HuggingFace model: ``OpenGVLab/InternVideo2-CLIP-1B-224p-f8``
- Input: 8 frames per clip, 224×224
- Embedding dim: 768

Usage (via factory)
-------------------
    config = {
        "engine": {
            "type":        "internvideo2",
            "model_name":  "OpenGVLab/InternVideo2-CLIP-1B-224p-f8",
            "device":      "cuda",
            "batch_size":  4,
            "frames_per_window": 8,
            "load_in_8bit": False,
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

INTERNVIDEO2_AVAILABLE = False
try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    INTERNVIDEO2_AVAILABLE = True
except ImportError:
    pass

NUM_FRAMES_IV2 = 8   # InternVideo2 1B-224p-f8 → 8 frames


def _get_device(device_str=None):
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# InternVideo2 video preprocessing (same as standard ViT-L CLIP preprocessing)
def _build_video_transform(target_size: int = 224):
    """Build torchvision transforms for InternVideo2 video frames."""
    try:
        from torchvision import transforms as T
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    except ImportError:
        return None


class InternVideo2Engine(InferenceEngine):
    """
    InferenceEngine backed by InternVideo2-CLIP-1B (HuggingFace Transformers).

    The model is loaded via ``AutoModel.from_pretrained`` using the
    ``trust_remote_code=True`` flag required by OpenGVLab models.

    Parameters
    ----------
    engine_cfg : dict
        Sub-dict from config['engine']. Recognised keys:
        - model_name   (str, default "OpenGVLab/InternVideo2-CLIP-1B-224p-f8")
        - device       (str | None)
        - batch_size   (int, default 4)
        - load_in_8bit (bool, default False) — INT8 quantization (slower)
    """

    EMBED_DIM_VALUE = 768
    DEFAULT_MODEL   = "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"

    def __init__(self, engine_cfg: dict) -> None:
        if not INTERNVIDEO2_AVAILABLE:
            raise ImportError(
                "InternVideo2 requires the 'transformers' package (>=4.35). "
                "Install with: pip install transformers>=4.35.0"
            )

        model_name    = engine_cfg.get("model_name", self.DEFAULT_MODEL)
        device_str    = engine_cfg.get("device", None)
        load_in_8bit  = engine_cfg.get("load_in_8bit", False)
        self._batch_size = int(engine_cfg.get("batch_size", 4))
        self._device = _get_device(device_str)

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )

        kwargs: dict = {"trust_remote_code": True}
        if load_in_8bit and self._device.type == "cuda":
            kwargs["load_in_8bit"] = True
            kwargs["device_map"]   = "auto"
        else:
            kwargs["torch_dtype"] = (
                torch.float16 if self._device.type == "cuda" else torch.float32
            )

        self._model = AutoModel.from_pretrained(model_name, **kwargs)
        if not load_in_8bit:
            self._model = self._model.to(self._device)
        self._model.eval()

        self._transform = _build_video_transform(224)

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM_VALUE

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Per-frame encoding — each frame is treated as a 1-frame video clip.
        Returns shape (N, 768) float32 L2-normalised.
        """
        if not frames_bgr:
            return np.zeros((0, self.EMBED_DIM_VALUE), dtype=np.float32)

        results = []
        for frame in frames_bgr:
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
        ).to(self._device)

        with torch.no_grad():
            text_emb = self._model.encode_text(inputs)  # (N, 768)
        vecs = F.normalize(text_emb.float(), dim=-1)
        return vecs.cpu().numpy().astype(np.float32)

    def encode_segment_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Encode a video segment using InternVideo2's temporal transformer.

        Pads or sub-samples to exactly NUM_FRAMES_IV2=8 frames.
        Returns shape (768,) — a single L2-normalised segment embedding.
        """
        frames = self._sample_or_pad(frames_bgr, NUM_FRAMES_IV2)
        video_tensor = self._frames_to_tensor(frames)  # (1, T, C, H, W)

        with torch.no_grad():
            video_emb = self._model.encode_video(video_tensor)  # (1, 768)
        vec = F.normalize(video_emb.float(), dim=-1).squeeze(0)
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
        Convert list of BGR frames to InternVideo2 video tensor.
        Shape: (1, T, C, H, W) — batch=1, T=num_frames.
        """
        tensors = []
        for frame in frames_bgr:
            rgb = frame[:, :, ::-1].copy()
            pil = Image.fromarray(rgb.astype(np.uint8))
            if self._transform is not None:
                t = self._transform(pil)
            else:
                # Fallback: naive numpy → tensor
                arr = np.array(pil, dtype=np.float32) / 255.0
                t = torch.tensor(arr).permute(2, 0, 1)
            tensors.append(t)

        video = torch.stack(tensors, dim=0)            # (T, C, H, W)
        video = video.unsqueeze(0).to(self._device)    # (1, T, C, H, W)
        if self._device.type == "cuda":
            video = video.half()
        return video
