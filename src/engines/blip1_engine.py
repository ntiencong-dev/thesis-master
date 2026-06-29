"""
engines/blip1_engine.py
------------------------
BLIP-1 ITC (Image-Text Contrastive) bi-encoder engine.

Model  : Salesforce/blip-itm-base-coco (BlipForImageTextRetrieval)
Backend: HuggingFace transformers

Two operating modes
-------------------
ITC (bi-encoder) — used for indexing & stage-1 search:
    encode_frames() → 256d L2-normalised visual embeddings  (pre-computable)
    encode_text()   → 256d L2-normalised text embeddings

ITM (cross-encoder) — used by BLIP1ITMReranker for stage-2 reranking:
    score_itm(frames, text) → match probability per frame  (NOT pre-computable)
    Text encoder attends over image patches via cross-attention.
    ARM Cortex-A53 latency: ~46 ms/candidate → 50 candidates ≈ 2.3 s (viable).

Why BLIP-1 instead of EVA-CLIP for Kria KV260?
    * ViT-B/16 visual encoder (12 layers × 768-dim) can be quantised to INT8
      for Kria DPU B4096 — same architecture as CLIP ViT-B/16.
    * 256d ITC outperforms CLIP 512d on COCO R@1 (65.4 % vs 37.3 %) due to
      CapFilt data cleaning + momentum encoder.
    * ITM stage-2 uses BERT + cross-attention only (~110 M params, no LLM).
      EVA-CLIP / BLIP-2 OPT-2.7 B are NOT viable on ARM Cortex-A53.

Internal architecture
---------------------
    BlipForImageTextRetrieval
        .vision_model         — BlipVisionModel (ViT-B/16)
        .text_encoder         — BlipTextModel (BERT-base, with cross-attn)
        .image_projection     — Linear(768 → 256)  [ITC visual head]
        .text_projection      — Linear(768 → 256)  [ITC text head]
        .itm_head             — Linear(768 → 2)    [ITM binary head]

Input  : BGR uint8 frames (any size — BlipProcessor auto-resizes to 384×384)
Output : float32 np.ndarray, L2-normalised, shape (N, 256)
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

BLIP1_AVAILABLE = False
try:
    from transformers import BlipForImageTextRetrieval, BlipProcessor  # type: ignore
    BLIP1_AVAILABLE = True
except (ImportError, RuntimeError, AttributeError):
    pass

from .base_engine import InferenceEngine


class BLIP1Engine(InferenceEngine):
    """
    BLIP-1 ITC bi-encoder + ITM cross-encoder.

    Embed-dim 256 is set at the class level so callers (factory, tests) can
    read it without instantiating a model.
    """

    EMBED_DIM_VALUE: int = 256
    DEFAULT_MODEL:   str = "Salesforce/blip-itm-base-coco"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        if not BLIP1_AVAILABLE:
            raise ImportError(
                "BLIP-1 requires the 'transformers' package (>=4.27). "
                "Install with: pip install transformers"
            )

        import torch

        self._model_name: str = config.get("model_name", self.DEFAULT_MODEL)
        self._batch_size: int = int(config.get("batch_size", 16))

        device_str: str = config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._device = torch.device(device_str)

        logger.info("[BLIP1Engine] Loading %s on %s", self._model_name, self._device)

        self._processor: BlipProcessor = BlipProcessor.from_pretrained(
            self._model_name
        )
        dtype = torch.float16 if str(self._device).startswith("cuda") else torch.float32
        self._dtype = dtype
        self._model: BlipForImageTextRetrieval = (
            BlipForImageTextRetrieval.from_pretrained(
                self._model_name, torch_dtype=dtype
            ).to(self._device)
        )
        self._model.eval()

        # ── AWQ INT4 text encoder (optional) ─────────────────────────────
        # When awq_bert_path is set (Kria deployment), replace the FP32 text
        # encoder with the AWQ INT4 variant to reduce ARM memory footprint
        # from ~440 MB → ~110 MB.  Visual encoder and ITM head are unchanged.
        self._awq_bert_path: str = config.get("awq_bert_path", "")
        if self._awq_bert_path:
            self._load_awq_bert(self._awq_bert_path)

        logger.info(
            "[BLIP1Engine] Ready — embed_dim=%d device=%s awq_bert=%s",
            self.EMBED_DIM_VALUE, self._device,
            "enabled" if self._awq_bert_path else "disabled",
        )

    def _load_awq_bert(self, awq_path: str) -> None:
        """
        Replace BLIP-1 text_encoder weights with AWQ INT4 quantized version.

        The checkpoint saved by scripts/awq_quantize_bert.py stores either:
          a) AutoAWQ format: full quantized model (auto-detected)
          b) Custom format: {"text_encoder_state_dict": ..., "text_proj_state_dict": ...}

        Only the text encoder weights are swapped — vision_model, itm_head,
        and the processor remain the FP32 originals.
        """
        import os
        import torch

        pt_file = os.path.join(awq_path, "blip1_text_awq.pt")
        if os.path.isfile(pt_file):
            # Custom AWQ format (from _run_custom_awq in awq_quantize_bert.py)
            logger.info("[BLIP1Engine] Loading AWQ BERT from: %s", pt_file)
            ckpt = torch.load(pt_file, map_location=self._device, weights_only=False)
            try:
                from .quantized_linear import replace_with_quantized_linear
                replace_with_quantized_linear(self._model.text_encoder, ckpt["text_encoder_state_dict"])
                self._model.text_encoder.load_state_dict(
                    ckpt["text_encoder_state_dict"], strict=False
                )
                if "text_proj_state_dict" in ckpt and ckpt["text_proj_state_dict"] is not None:
                    self._model.text_proj.load_state_dict(
                        ckpt["text_proj_state_dict"], strict=False
                    )
                logger.info("[BLIP1Engine] AWQ BERT loaded — text encoder quantized INT4")
            except Exception as exc:
                logger.warning(
                    "[BLIP1Engine] AWQ BERT load failed (%s), using FP32 text encoder", exc
                )
        else:
            # AutoAWQ format: full quantized model directory
            try:
                from awq import AutoAWQForCausalLM  # type: ignore
                from transformers import AutoTokenizer
                logger.info("[BLIP1Engine] Loading AutoAWQ BERT from: %s", awq_path)
                awq_model = AutoAWQForCausalLM.from_quantized(awq_path, fuse_layers=False)
                # Replace only text encoder, keeping vision model + ITM head intact
                if hasattr(awq_model, "model") and hasattr(awq_model.model, "encoder"):
                    self._model.text_encoder = awq_model.model.to(self._device)
                logger.info("[BLIP1Engine] AutoAWQ BERT loaded successfully")
            except Exception as exc:
                logger.warning(
                    "[BLIP1Engine] AutoAWQ BERT load failed (%s), using FP32 fallback", exc
                )

    # ------------------------------------------------------------------
    # InferenceEngine ABC
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM_VALUE

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        ITC mode: encode BGR frames into 256d L2-normalised visual embeddings.

        BlipProcessor converts BGR→RGB and resizes to the model's expected
        input size (384×384 for blip-itm-base-coco).  Frames are batched for
        efficient GPU inference.

        Parameters
        ----------
        frames_bgr : List[np.ndarray]
            BGR uint8 HxWx3 arrays.  Any spatial resolution is accepted.

        Returns
        -------
        np.ndarray  shape (N, 256), dtype float32, L2-normalised.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if not frames_bgr:
            return np.zeros((0, self.EMBED_DIM_VALUE), dtype=np.float32)

        all_embeds: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self._batch_size):
            batch = frames_bgr[i : i + self._batch_size]

            # BGR → RGB → PIL
            pil_images = [Image.fromarray(f[:, :, ::-1]) for f in batch]

            img_inputs   = self._processor(images=pil_images, return_tensors="pt")
            pixel_values = img_inputs.pixel_values.to(self._device, dtype=self._dtype)

            with torch.no_grad():
                vision_out = self._model.vision_model(
                    pixel_values=pixel_values, return_dict=True
                )
                cls_feat = vision_out.last_hidden_state[:, 0, :]  # (B, hidden)
                embed    = self._model.vision_proj(cls_feat)  # (B, 256)
                embed    = F.normalize(embed, dim=-1)

            all_embeds.append(embed.float().cpu().numpy())

        return np.concatenate(all_embeds, axis=0)  # (N, 256)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        ITC mode: encode text strings into 256d L2-normalised embeddings.

        The BERT text encoder runs WITHOUT cross-attention (no image features)
        to produce query-independent text embeddings.

        Returns
        -------
        np.ndarray  shape (N, 256), dtype float32, L2-normalised.
        """
        import torch
        import torch.nn.functional as F

        if isinstance(texts, str):
            texts = [texts]

        all_embeds: List[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            txt_inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids      = txt_inputs.input_ids.to(self._device)
            attention_mask = txt_inputs.attention_mask.to(self._device)

            with torch.no_grad():
                # Pure text encoding — no encoder_hidden_states → self-attention only
                text_out = self._model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                cls_feat = text_out.last_hidden_state[:, 0, :]  # (B, hidden)
                embed    = self._model.text_proj(cls_feat)  # (B, 256)
                embed    = F.normalize(embed, dim=-1)

            all_embeds.append(embed.float().cpu().numpy())

        return np.concatenate(all_embeds, axis=0)  # (N, 256)

    # ------------------------------------------------------------------
    # ITM cross-encoder — called exclusively by BLIP1ITMReranker
    # ------------------------------------------------------------------

    def score_itm(self, frames_bgr: List[np.ndarray], text: str) -> np.ndarray:
        """
        ITM (cross-encoder) mode: score N frames against a single text query.

        Runs the image-grounded text encoder where BERT text tokens attend
        over image patch features via cross-attention.  Fine-grained alignment
        catches colour, count, and spatial-relation differences that the
        bi-encoder (ITC cosine) often misses.

        This method is NOT pre-computable because the text encoding depends
        on the specific image it is being matched against.  Use only for
        stage-2 reranking on a small top-K candidate set.

        Kria KV260 deployment note
        --------------------------
        ViT-B/16 patch features can be computed on the DPU INT8 quantised
        model.  The BERT cross-attention forward pass runs on ARM Cortex-A53
        (~46 ms/candidate, acceptable for top-50 reranking ≈ 2.3 s total).

        Parameters
        ----------
        frames_bgr : List[np.ndarray]
            One representative BGR frame per candidate (any spatial size).
        text : str
            The search query; repeated internally for each frame.

        Returns
        -------
        np.ndarray  shape (N,), dtype float32, values in [0.0, 1.0].
                    Probability that frame i matches the query.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if not frames_bgr:
            return np.zeros(0, dtype=np.float32)

        all_probs: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self._batch_size):
            batch = frames_bgr[i : i + self._batch_size]
            B     = len(batch)

            # ── Vision encoder ──────────────────────────────────────────
            pil_images   = [Image.fromarray(f[:, :, ::-1]) for f in batch]
            img_inputs   = self._processor(images=pil_images, return_tensors="pt")
            pixel_values = img_inputs.pixel_values.to(self._device, dtype=self._dtype)

            # ── Text inputs (query repeated B times) ────────────────────
            txt_inputs = self._processor(
                text=[text] * B,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids      = txt_inputs.input_ids.to(self._device)
            attention_mask = txt_inputs.attention_mask.to(self._device)

            with torch.no_grad():
                # 1. ViT → patch feature sequence
                vision_out  = self._model.vision_model(
                    pixel_values=pixel_values, return_dict=True
                )
                image_feats = vision_out.last_hidden_state        # (B, img_seq, hidden)
                img_att     = torch.ones(
                    image_feats.shape[:2], device=self._device, dtype=torch.long
                )

                # 2. Image-grounded text encoder: BERT + cross-attention to patches
                text_out = self._model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_feats,
                    encoder_attention_mask=img_att,
                    return_dict=True,
                )
                cls_feat   = text_out.last_hidden_state[:, 0, :]  # (B, hidden)

                # 3. ITM binary head → match probability
                itm_logits = self._model.itm_head(cls_feat)       # (B, 2)
                probs      = F.softmax(itm_logits.float(), dim=-1)[:, 1]  # (B,)

            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs).astype(np.float32)       # (N,)
