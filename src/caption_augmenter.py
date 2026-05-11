"""
caption_augmenter.py
----------------------
Phase 3: Dense Caption Augmentation for hybrid visual + textual embeddings.
(RESEARCH_SOTA_NLVS.md §III4 — Recall +8–12% estimated)

Architecture
------------
Given a video segment (N frames):

1. **Visual embedding**: `visual_extractor.encode_segment_frames(frames)` → v ∈ ℝ^D
2. **Caption generation**: A vision-language model (LLaVA-1.5 or BLIP-2) reads
   the representative frame and generates a dense natural-language description.
3. **Caption embedding**: The CLIP/EVA-CLIP text encoder encodes the caption → c ∈ ℝ^D
4. **Hybrid embedding**: h = α·v + (1-α)·c,  then L2-normalised.
   - Default α = 0.7 (70 % visual, 30 % semantic caption signal)

This hybrid embedding improves recall for abstract / action-centric queries
because text features capture semantic context that pure visual features miss.

VRAM note (GTX 1650 Ti, 4 GB)
------------------------------
LLaVA-1.5-7B requires ~6 GB VRAM → too large for simultaneous use with CLIP.
Use ``device="cpu"`` for caption generation (slower, ~2–5 s/frame) or a
smaller model such as ``Salesforce/blip2-opt-2.7b`` (can share GPU in 8-bit).
A lighter captioner is ``Salesforce/blip-image-captioning-base`` (~330 MB).

Usage
-----
    from src.caption_augmenter import CaptionAugmenter

    augmenter = CaptionAugmenter(
        visual_extractor=clip_extractor,   # CLIPFeatureExtractor or any engine
        caption_model="Salesforce/blip-image-captioning-base",
        device="cpu",
        visual_weight=0.7,
    )
    hybrid_emb = augmenter.encode_segment_frames_augmented(frames_bgr)
    # → shape (D,) float32 L2-normalised
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Optional imports — BLIP (lighter, always available) or LLaVA
BLIP_CAPTION_AVAILABLE = False
try:
    from transformers import (  # type: ignore
        BlipForConditionalGeneration,
        BlipProcessor,
    )
    BLIP_CAPTION_AVAILABLE = True
except ImportError:
    pass

LLAVA_AVAILABLE = False
try:
    from transformers import (  # type: ignore
        LlavaForConditionalGeneration,
        AutoProcessor as LlavaProcessor,
    )
    LLAVA_AVAILABLE = True
except ImportError:
    pass

# ── Caption prompt ─────────────────────────────────────────────────────────
DEFAULT_CAPTION_PROMPT = "Describe this image in detail:"


class CaptionAugmenter:
    """
    Augments visual segment embeddings with auto-generated dense captions.

    Parameters
    ----------
    visual_extractor
        Any object with:
        - ``encode_segment_frames(frames_bgr) → np.ndarray (D,)``
        - ``encode_text(texts) → np.ndarray (N, D)``
    caption_model : str
        HuggingFace model id for the captioner.
        - Lightweight: ``"Salesforce/blip-image-captioning-base"`` (~330 MB)
        - Higher quality: ``"Salesforce/blip2-opt-2.7b"`` (8-bit ≈ 2.8 GB)
        - SOTA (requires ≥6 GB VRAM): ``"llava-hf/llava-1.5-7b-hf"``
    device : str | None
        Device for caption model. Defaults to CUDA if available.
    visual_weight : float
        α in h = α·visual + (1-α)·caption_text.  Range [0, 1].
    lazy_load : bool
        Defer caption model loading until first call.
    """

    DEFAULT_MODEL  = "Salesforce/blip-image-captioning-base"

    def __init__(
        self,
        visual_extractor,
        caption_model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        visual_weight: float = 0.7,
        lazy_load: bool = True,
    ) -> None:
        self._extractor    = visual_extractor
        self._model_name   = caption_model
        self._visual_weight = float(visual_weight)
        self._lazy_load    = lazy_load

        import torch as _torch
        if device:
            self._device = _torch.device(device)
        else:
            self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

        self._caption_processor = None
        self._caption_model_obj = None

        if not lazy_load:
            self._load_caption_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_segment_frames_augmented(
        self, frames_bgr: List[np.ndarray]
    ) -> np.ndarray:
        """
        Produce a hybrid embedding for a video segment.

        1. Visual embedding from ``visual_extractor.encode_segment_frames``.
        2. Dense caption of the midpoint frame via the caption model.
        3. Caption text embedding via ``visual_extractor.encode_text``.
        4. Weighted interpolation + L2-normalisation.

        Parameters
        ----------
        frames_bgr : List of BGR uint8 numpy arrays.

        Returns
        -------
        np.ndarray, shape (D,), float32, L2-normalised.
        """
        if not frames_bgr:
            raise ValueError("frames_bgr must be non-empty")

        # Stage 1: visual embedding
        visual_emb = self._extractor.encode_segment_frames(frames_bgr)  # (D,)
        visual_emb = visual_emb.astype(np.float32)

        # Stage 2: generate caption from representative frame
        mid_idx = len(frames_bgr) // 2
        representative_frame = frames_bgr[mid_idx]
        caption = self.generate_caption(representative_frame)

        # Stage 3: encode caption text with the same visual extractor
        caption_emb_batch = self._extractor.encode_text([caption])  # (1, D)
        caption_emb = caption_emb_batch[0]                          # (D,)

        # Stage 4: weighted combination + L2 normalise
        alpha = self._visual_weight
        hybrid = alpha * visual_emb + (1.0 - alpha) * caption_emb
        norm = np.linalg.norm(hybrid)
        if norm > 1e-9:
            hybrid /= norm
        return hybrid.astype(np.float32)

    def generate_caption(self, frame_bgr: np.ndarray) -> str:
        """
        Generate a dense caption for a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray, shape (H, W, 3), dtype uint8, BGR.

        Returns
        -------
        str — generated caption.
        """
        self._load_caption_model()

        rgb = frame_bgr[:, :, ::-1].copy()
        pil_image = Image.fromarray(rgb.astype(np.uint8))

        inputs = self._caption_processor(pil_image, return_tensors="pt")
        inputs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            out = self._caption_model_obj.generate(**inputs, max_new_tokens=50)
        caption = self._caption_processor.decode(
            out[0], skip_special_tokens=True
        ).strip()
        return caption

    def unload(self) -> None:
        """Release caption model from GPU/CPU memory."""
        self._caption_model_obj = None
        self._caption_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._caption_model_obj is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_caption_model(self) -> None:
        """Load BLIP or LLaVA caption model (idempotent)."""
        if self._caption_model_obj is not None:
            return

        if not BLIP_CAPTION_AVAILABLE:
            raise ImportError(
                "Caption augmentation requires 'transformers' (>=4.27). "
                "Install: pip install transformers"
            )

        self._caption_processor = BlipProcessor.from_pretrained(self._model_name)
        self._caption_model_obj = BlipForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=(
                torch.float16 if self._device.type == "cuda" else torch.float32
            ),
        ).to(self._device)
        self._caption_model_obj.eval()
