"""
feature_extractor.py
--------------------
Wraps OpenAI CLIP (ViT-B/16) via the `open_clip` library to produce
L2-normalised float32 embeddings for both video frames and text queries.

Design choices
--------------
* Uses `torch.no_grad()` + `float16` to keep GTX 1650 Ti (4 GB VRAM)
  well within budget.
* Processes frames in batches to amortise CUDA kernel launch overhead.
* Automatically falls back to CPU if CUDA is unavailable.
* Returns numpy float32 arrays so that Faiss can consume them directly.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import open_clip
    _BACKEND = "open_clip"
except ImportError:                          # fallback to HuggingFace CLIP
    from transformers import CLIPModel, CLIPProcessor
    _BACKEND = "transformers"


# ---------------------------------------------------------------------------
# Helper: device selection
# ---------------------------------------------------------------------------

def _get_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class CLIPFeatureExtractor:
    """
    Parameters
    ----------
    model_name : str
        open_clip model name.  Default ``"ViT-B-16"``.
    pretrained : str
        open_clip pretrained weights tag.  Default ``"openai"``.
    device : str | None
        ``"cuda"``, ``"cpu"``, or None (auto-detect).
    batch_size : int
        Number of frames processed per forward pass.
    """

    # EVA-CLIP ViT-L/14 — Phase 1 upgrade (see RESEARCH_SOTA_NLVS.md §I1)
    # ImageNet zero-shot 79.8% vs 68.3% (ViT-B/16); VRAM ~1.4 GB FP16.
    MODEL_NAME  = "EVA02-L-14"
    PRETRAINED  = "merged2b_s4b_b131k"
    EMBED_DIM   = 768          # ViT-L/14 joint embedding dimension (up from 512)

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: str = PRETRAINED,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.device     = _get_device(device)
        self.batch_size = batch_size

        if _BACKEND == "open_clip":
            try:
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
                self._tokenizer = open_clip.get_tokenizer(model_name)
            except Exception as exc:
                # Graceful fallback to ViT-B-16 if EVA weights not downloaded yet
                import warnings
                warnings.warn(
                    f"[CLIPFeatureExtractor] Failed to load {model_name!r} "
                    f"({pretrained!r}): {exc}. "
                    "Falling back to ViT-B-16 / openai."
                )
                model_name = "ViT-B-16"
                pretrained = "openai"
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
                self._tokenizer = open_clip.get_tokenizer(model_name)
                self.EMBED_DIM = 512   # update instance EMBED_DIM on fallback
        else:
            # HuggingFace fallback
            self._hf_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self._hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self._model = self._hf_model

        # Move to device and cast to float16 to save VRAM
        self._model.to(self.device).eval()
        if self.device.type == "cuda":
            self._model = self._model.half()   # float16 on GPU

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        frames_bgr : list of H×W×3 uint8 BGR numpy arrays (224×224).

        Returns
        -------
        numpy.ndarray  shape (N, EMBED_DIM), dtype float32, L2-normalised.
        """
        if not frames_bgr:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self.batch_size):
            batch = frames_bgr[i : i + self.batch_size]
            tensor = self._prepare_image_batch(batch)
            with torch.no_grad():
                if _BACKEND == "open_clip":
                    feats = self._model.encode_image(tensor)
                else:
                    feats = self._hf_model.get_image_features(pixel_values=tensor)
            feats = F.normalize(feats.float(), dim=-1)
            all_embeddings.append(feats.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Parameters
        ----------
        texts : single string or list of strings.

        Returns
        -------
        numpy.ndarray  shape (N, EMBED_DIM), dtype float32, L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            if _BACKEND == "open_clip":
                tokens = self._tokenizer(texts).to(self.device)
                feats  = self._model.encode_text(tokens)
            else:
                inputs = self._hf_processor(
                    text=texts, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats  = self._hf_model.get_text_features(**inputs)

        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_image_batch(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """Convert BGR uint8 numpy arrays → preprocessed float tensor on device."""
        pil_images = [
            Image.fromarray(frame[:, :, ::-1])   # BGR → RGB
            for frame in frames_bgr
        ]

        if _BACKEND == "open_clip":
            tensors = torch.stack([self._preprocess(img) for img in pil_images])
        else:
            inputs   = self._hf_processor(images=pil_images, return_tensors="pt")
            tensors  = inputs["pixel_values"]

        tensors = tensors.to(self.device)
        if self.device.type == "cuda":
            tensors = tensors.half()   # float16 to match model
        return tensors


# ---------------------------------------------------------------------------
# Phase 2 — mSigLIP Multilingual Feature Extractor
# ---------------------------------------------------------------------------

class SigLIPFeatureExtractor:
    """
    Multilingual CLIP variant using ``ViT-L-16-SigLIP-256`` / ``webli`` weights
    (Google SigLIP trained on Web Images with Language pairs).

    Benefits over standard CLIP for VI→EN use-case
    ------------------------------------------------
    * Natively supports 100+ languages (including Vietnamese) — no translation.
    * SigLIP's sigmoid loss (vs softmax) improves retrieval recall at low
      similarity scores.
    * 256×256 input resolution (vs 224×224) — better fine-grained detail.

    Hardware budget (GTX 1650 Ti, 4 GB)
    ------------------------------------
    * EMBED_DIM = 1024 (ViT-L SigLIP)
    * VRAM ~1.6 GB FP16 — within budget.

    Parameters
    ----------
    model_name : str
        open_clip model identifier. Default ``"ViT-L-16-SigLIP-256"``.
    pretrained : str
        open_clip pretrained tag. Default ``"webli"``.
    device : str | None
        ``"cuda"``, ``"cpu"``, or None (auto-detect).
    batch_size : int
        Frames per GPU forward pass.
    """

    MODEL_NAME  = "ViT-L-16-SigLIP-256"
    PRETRAINED  = "webli"
    EMBED_DIM   = 1024

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: str = PRETRAINED,
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        if _BACKEND != "open_clip":
            raise ImportError("SigLIPFeatureExtractor requires open_clip (pip install open-clip-torch)")

        self.device = _get_device(device)
        self.batch_size = batch_size
        self.EMBED_DIM = self.__class__.EMBED_DIM  # instance copy for fallback override

        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"[SigLIPFeatureExtractor] Failed to load {model_name!r} "
                f"({pretrained!r}): {exc}. "
                "Falling back to ViT-B-16 / openai (dim=512)."
            )
            model_name = "ViT-B-16"
            pretrained = "openai"
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self.EMBED_DIM = 512

        self._model.to(self.device).eval()
        if self.device.type == "cuda":
            self._model = self._model.half()

    # ------------------------------------------------------------------
    # Public API — identical interface to CLIPFeatureExtractor
    # ------------------------------------------------------------------

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Returns (N, EMBED_DIM) float32 L2-normalised frame embeddings.
        """
        if not frames_bgr:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        results: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self.batch_size):
            batch = frames_bgr[i : i + self.batch_size]
            tensor = self._prepare_image_batch(batch)
            with torch.no_grad():
                feats = self._model.encode_image(tensor)
            feats = F.normalize(feats.float(), dim=-1)
            results.append(feats.cpu().numpy())
        return np.vstack(results).astype(np.float32)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Returns (N, EMBED_DIM) float32 L2-normalised text embeddings.
        SigLIP tokenizer handles non-ASCII characters (multilingual).
        """
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self.device)
            feats  = self._model.encode_text(tokens)
        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_image_batch(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """BGR uint8 → preprocessed float tensor on device."""
        pil_images = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]
        tensors = torch.stack([self._preprocess(img) for img in pil_images])
        tensors = tensors.to(self.device)
        if self.device.type == "cuda":
            tensors = tensors.half()
        return tensors
