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

    MODEL_NAME  = "ViT-B-16"
    PRETRAINED  = "openai"
    EMBED_DIM   = 512          # ViT-B/16 joint embedding dimension

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
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
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
