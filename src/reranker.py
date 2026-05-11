"""
reranker.py
-----------
Phase 3: Two-stage retrieval with BLIP-2 visual QA reranking.
(RESEARCH_SOTA_NLVS.md §II3 / §III — Precision@5 +15–20% estimated)

Architecture
------------
Stage 1 (fast, already in searcher.py):
    CLIP/EVA-CLIP → Faiss → Top-50 candidates  (cosine similarity)

Stage 2 (this module):
    For each candidate: extract representative frame → BLIP-2 VQA →
    "Does this scene contain <query>?" → yes/no log-prob → rerank score

Combined score:
    combined = alpha × cosine_score + (1 - alpha) × blip_score
    where alpha = 0.6 by default

VRAM note (GTX 1650 Ti, 4 GB)
------------------------------
BLIP-2 OPT-2.7B with 8-bit quantization ≈ 2.8 GB → cannot coexist with
EVA-CLIP (1.4 GB). The Reranker therefore uses lazy loading: model is loaded
on the first call to ``rerank()`` and released via ``unload()`` when done.
As an alternative, use ``device="cpu"`` to avoid VRAM conflict (slower ~10×).

Usage
-----
    from src.reranker import BLIP2Reranker

    reranker = BLIP2Reranker(device="cpu")   # or "cuda" if CLIP unloaded first
    results = reranker.rerank(candidates, query="person climbing fence")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

# BLIP2Reranker.AVAILABLE is set at import time
BLIP2_AVAILABLE = False
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration  # type: ignore
    BLIP2_AVAILABLE = True
except (ImportError, RuntimeError, AttributeError):
    # RuntimeError can arise from incompatible deepspeed/accelerate versions;
    # AttributeError from pydantic version mismatches in transformers internals.
    pass


@dataclass
class RerankResult:
    """Reranked search result with separate cosine and BLIP scores."""
    score:        float   # combined score
    cosine_score: float   # Stage-1 cosine similarity
    blip_score:   float   # Stage-2 BLIP-2 yes/no probability
    video_id:     str
    video_path:   str
    start_time:   float
    end_time:     float
    rank:         int


class BLIP2Reranker:
    """
    Two-stage reranker using BLIP-2 (Salesforce/blip2-opt-2.7b or
    Salesforce/blip2-flan-t5-xxl for smaller VRAM footprint).

    Parameters
    ----------
    model_name : str
        HuggingFace model id.  Defaults to ``"Salesforce/blip2-opt-2.7b"``.
        Lighter alternative: ``"Salesforce/blip2-flan-t5-xl"`` (~1.9 GB 8-bit).
    device : str | None
        ``"cuda"``, ``"cpu"``, or None (auto-detect).
        On CPU inference is ~10× slower but avoids VRAM conflict with CLIP.
    load_in_8bit : bool
        Quantize to 8-bit (requires bitsandbytes).  Reduces VRAM from ~5.4 GB
        to ~2.8 GB for OPT-2.7B.  Ignored when device="cpu".
    alpha : float
        Weight for Stage-1 cosine score in the combined score.
        combined = alpha × cosine + (1 - alpha) × blip_score.
    lazy_load : bool
        If True, defer model loading until the first call to ``rerank()``.
    prompt_template : str
        Question template.  Must contain ``{}`` which is replaced by the query.
    """

    DEFAULT_MODEL    = "Salesforce/blip2-opt-2.7b"
    DEFAULT_ALPHA    = 0.6
    DEFAULT_TEMPLATE = "Question: Does this scene contain {}? Answer:"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        load_in_8bit: bool = True,
        alpha: float = DEFAULT_ALPHA,
        lazy_load: bool = True,
        prompt_template: str = DEFAULT_TEMPLATE,
    ) -> None:
        if not BLIP2_AVAILABLE:
            raise ImportError(
                "BLIP-2 requires the 'transformers' package (>=4.27). "
                "Install with: pip install transformers accelerate bitsandbytes"
            )

        self.model_name      = model_name
        self.load_in_8bit    = load_in_8bit
        self.alpha           = alpha
        self.prompt_template = prompt_template

        import torch
        if device is not None:
            self._device_str = device
        else:
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Will not quantize on CPU
        self._use_8bit = load_in_8bit and (self._device_str == "cuda")

        self._processor: Optional[Blip2Processor] = None
        self._model: Optional[Blip2ForConditionalGeneration] = None

        if not lazy_load:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list,   # List[SearchResult] from searcher
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank *candidates* using BLIP-2 visual QA.

        Parameters
        ----------
        candidates : list of SearchResult (from NLVideoSearcher.search)
        query : str — original natural-language query
        top_k : int | None — limit output; None returns all candidates

        Returns
        -------
        List[RerankResult] sorted by combined score descending, with .rank set.
        """
        if not candidates:
            return []

        self._load_model()   # no-op if already loaded

        prompt = self.prompt_template.format(query)
        scored: List[tuple[float, float, float, object]] = []  # (combined, cosine, blip, cand)

        for cand in candidates:
            frame = self._extract_frame(cand.video_path, cand.start_time, cand.end_time)
            blip_score = self._score_frame(frame, prompt)
            combined = self.alpha * cand.score + (1.0 - self.alpha) * blip_score
            scored.append((combined, cand.score, blip_score, cand))

        scored.sort(key=lambda x: -x[0])
        if top_k is not None:
            scored = scored[:top_k]

        return [
            RerankResult(
                score=combined,
                cosine_score=cosine,
                blip_score=blip,
                video_id=cand.video_id,
                video_path=cand.video_path,
                start_time=cand.start_time,
                end_time=cand.end_time,
                rank=i + 1,
            )
            for i, (combined, cosine, blip, cand) in enumerate(scored)
        ]

    def unload(self) -> None:
        """Release GPU/CPU memory occupied by the BLIP-2 model."""
        import torch
        self._model    = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load BLIP-2 model + processor (idempotent)."""
        if self._model is not None:
            return
        import torch

        kwargs: dict = {"torch_dtype": torch.float16 if self._device_str == "cuda" else torch.float32}
        if self._use_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"]   = "auto"
        else:
            kwargs["device_map"] = "auto" if self._device_str == "cuda" else None

        self._processor = Blip2Processor.from_pretrained(self.model_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        if not self._use_8bit and self._device_str == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()

    def _extract_frame(
        self, video_path: str, start_time: float, end_time: float
    ) -> np.ndarray:
        """Extract the representative (midpoint) frame from a video segment."""
        from PIL import Image
        mid = (start_time + end_time) / 2.0
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        # Resize to 224×224 BGR
        return cv2.resize(frame, (224, 224))

    def _score_frame(self, frame_bgr: np.ndarray, prompt: str) -> float:
        """
        Ask BLIP-2 the yes/no question and return 1.0 if answer starts with
        "yes", 0.0 otherwise.

        For a richer score, the log-probability of generating "yes" could be
        used; however that requires access to model logits which is complex
        with 8-bit quantized models. Binary scoring is simpler and still
        provides a useful re-ranking signal.
        """
        import torch
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]   # BGR → RGB
        pil_image = Image.fromarray(rgb.astype(np.uint8))

        inputs = self._processor(pil_image, prompt, return_tensors="pt")
        if self._device_str == "cuda" and not self._use_8bit:
            inputs = inputs.to("cuda")
        elif self._device_str == "cuda" and self._use_8bit:
            # 8-bit model uses device_map="auto"; inputs need to match
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                      for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=5, min_new_tokens=1
            )
        answer = self._processor.decode(out[0], skip_special_tokens=True).strip().lower()
        return 1.0 if answer.startswith("yes") else 0.0
