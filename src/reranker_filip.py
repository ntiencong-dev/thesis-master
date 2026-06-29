"""
reranker_filip.py
-----------------
Option B stage-2 re-ranker: FILIPReranker (Fine-grained Interactive Language-
Image Pre-Training token matching).

Architecture
------------
Stage-1 (unchanged, in searcher.py):
    CLIP ViT-B/16 → global [CLS] embedding (512d) → Qdrant cosine → top-K

Stage-2 — this module:
    For each top-K candidate:
      1. Sample N representative frames (default N=5) from the segment.
      2. engine.encode_frames_tokens(frames) → (N, 196, 768) ViT patch tokens
         via forward hook on the last ViT attention block.
      3. engine.encode_text_tokens(query)   → (1, T, 768) word-level tokens
         via forward hook on the last text transformer block.
      4. Compute FILIP score:
             filip = mean_{text_tokens t_i}( max_{patches v_j}( dot(t_i, v_j) ) )
         This measures "for each word, how well does the best-matching image
         patch represent it?" — true fine-grained spatial alignment.
      5. combined = alpha × cosine_score + (1-alpha) × filip_score
    Return top-K results sorted by combined score.

AWQ INT4 and ARM CPU (Kria KV260) notes
-----------------------------------------
The ViT patch tokens (step 2) come from the DPU-side inference (PTQ INT8
xmodel), which is the standard visual encoder.  A separate CLIP FILIP xmodel
that exposes patch tokens is required for the Kria deployment; on PC, the
forward hook extracts them directly.

The text token extraction (step 3) runs the CLIP text transformer on ARM
Cortex-A53 CPU.  If `awq_text_path` is set in the engine config, the
AWQ INT4-quantized text encoder (produced by
scripts/awq_quantize_clip_text.py) is loaded instead of the FP32 model.
AWQ INT4 reduces the text encoder footprint from ~250 MB FP32 → ~63 MB,
making it viable alongside the visual DPU runner on Kria 4 GB LPDDR4.

MARCO persistent runner compatibility
---------------------------------------
FILIPReranker reuses the same engine instance as stage-1 — no extra DPU
runner is loaded.  MACRO (persistent runner pool) is not required for Option B
unless a separate CLIP FILIP xmodel is used for patch extraction on Kria.

Usage
-----
    from src.reranker_filip import FILIPReranker
    from src.engines.pc_engine import PCEngine

    engine   = PCEngine(engine_cfg)
    reranker = FILIPReranker(engine, alpha=0.6, n_frames=5)
    searcher.set_reranker(reranker)

    results  = searcher.search("a person climbing a fence", top_k=10)
    # results is List[RerankResult] if reranker is attached,
    # or List[SearchResult] otherwise.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from .reranker import RerankResult  # reuse the shared dataclass

logger = logging.getLogger(__name__)


class FILIPReranker:
    """
    Stage-2 re-ranker using FILIP token-level CLIP matching.

    All inference reuses the *same* engine already loaded for stage-1 —
    no additional model is loaded into memory.

    Parameters
    ----------
    engine : PCEngine (or any engine that implements encode_frames_tokens
             and encode_text_tokens — see src/engines/pc_engine.py)
        The stage-1 CLIP engine instance.
    alpha : float
        Stage-1 cosine score weight.
        combined = alpha × cosine + (1 - alpha) × filip_score.
        Default 0.6.
    n_frames : int
        Number of frames sampled uniformly from each candidate segment.
        Default 5.  More frames improve recall at the cost of latency.
    normalize_filip : bool
        If True, FILIP scores are normalized to [0, 1] across all
        candidates before combining with cosine scores.  This prevents
        scale mismatch between cosine similarity and raw FILIP scores.
        Default True.
    """

    DEFAULT_ALPHA: float = 0.6

    def __init__(
        self,
        engine,
        alpha: float = DEFAULT_ALPHA,
        n_frames: int = 5,
        normalize_filip: bool = True,
    ) -> None:
        self._engine          = engine
        self.alpha            = alpha
        self.n_frames         = n_frames
        self.normalize_filip  = normalize_filip

        # Verify the engine exposes the FILIP token API
        if not (hasattr(engine, "encode_frames_tokens") and
                hasattr(engine, "encode_text_tokens")):
            raise AttributeError(
                "FILIPReranker requires an engine with encode_frames_tokens() "
                "and encode_text_tokens() methods.  Use PCEngine (pc_engine.py) "
                "which delegates to CLIPFeatureExtractor."
            )

        logger.info(
            "[FILIPReranker] Initialized — alpha=%.2f n_frames=%d normalize=%s",
            alpha, n_frames, normalize_filip,
        )

    # ------------------------------------------------------------------
    # Public API (mirrors BLIP1ITMReranker interface)
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list,           # List[SearchResult] from NLVideoSearcher
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank *candidates* using FILIP token-level CLIP scores.

        Parameters
        ----------
        candidates : list of SearchResult (from searcher.NLVideoSearcher.search)
        query      : str — search query (after _normalize_query preprocessing)
        top_k      : int | None — limit output; None returns all candidates

        Returns
        -------
        List[RerankResult] sorted by combined score descending, .rank set.
        """
        if not candidates:
            return []

        # ── Step 1: encode text tokens once (shared across all candidates) ──
        # encode_text_tokens returns (tokens, mask):
        #   tokens : (1, T, hidden)   — word-level embeddings
        #   mask   : (1, T)           — True for valid tokens
        text_tokens_batch, text_mask_batch = self._engine.encode_text_tokens(query)
        # Squeeze batch dimension → (T, hidden) and (T,)
        text_tokens = text_tokens_batch[0]    # (T, hidden)
        text_mask   = text_mask_batch[0]      # (T,)
        # Keep only valid (non-padding) text tokens
        valid_text  = text_tokens[text_mask]  # (T_valid, hidden)
        if valid_text.shape[0] == 0:
            logger.warning("[FILIPReranker] No valid text tokens for query: %r", query)
            valid_text = text_tokens[:1]      # fallback: keep at least one

        # ── Step 2: compute FILIP score per candidate ────────────────────────
        filip_scores: List[float] = []
        for cand in candidates:
            score = self._score_candidate(cand, valid_text)
            filip_scores.append(score)

        filip_arr = np.array(filip_scores, dtype=np.float32)

        # ── Step 3: optional normalization to [0, 1] ─────────────────────────
        if self.normalize_filip and filip_arr.ptp() > 1e-8:
            filip_arr = (filip_arr - filip_arr.min()) / filip_arr.ptp()

        # ── Step 4: combine scores and sort ──────────────────────────────────
        scored: list = []
        for cand, filip_s in zip(candidates, filip_arr.tolist()):
            combined = self.alpha * cand.score + (1.0 - self.alpha) * filip_s
            scored.append((combined, cand.score, filip_s, cand))

        scored.sort(key=lambda x: -x[0])
        if top_k is not None:
            scored = scored[:top_k]

        return [
            RerankResult(
                score=combined,
                cosine_score=cosine,
                blip_score=filip_s,         # blip_score field reused for FILIP
                video_id=cand.video_id,
                video_path=cand.video_path,
                start_time=cand.start_time,
                end_time=cand.end_time,
                rank=i + 1,
                absolute_start=getattr(cand, "absolute_start", 0.0),
                absolute_end=getattr(cand, "absolute_end", 0.0),
            )
            for i, (combined, cosine, filip_s, cand) in enumerate(scored)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_candidate(
        self, cand, valid_text_tokens: np.ndarray
    ) -> float:
        """
        Compute the FILIP score for a single candidate.

        1. Sample n_frames uniformly from the segment.
        2. encode_frames_tokens(frames) → (N, 196, hidden) patch embeddings.
        3. L2-normalize patch embeddings per-token.
        4. For each text token t_i, compute max cosine over all patches and
           all frames, then average over text tokens.

        Parameters
        ----------
        cand : SearchResult
        valid_text_tokens : np.ndarray  shape (T_valid, hidden), float32, L2-norm.

        Returns
        -------
        float  FILIP score in (-1, 1) before normalization.
        """
        frames = self._sample_frames(cand.video_path, cand.start_time, cand.end_time)
        if not frames:
            return 0.0

        # (N_frames, num_patches, hidden)  — raw ViT hidden states, NOT projected
        patch_tokens = self._engine.encode_frames_tokens(frames)

        # Flatten frames × patches → all_patches for vectorised max operation
        # Shape: (N_frames × 196, hidden)
        N, P, H = patch_tokens.shape
        all_patches = patch_tokens.reshape(N * P, H).astype(np.float32)

        # L2-normalise both sides (ViT output is not guaranteed to be unit-norm)
        patch_norms = np.linalg.norm(all_patches, axis=1, keepdims=True)
        patch_norms = np.where(patch_norms < 1e-8, 1.0, patch_norms)
        all_patches = all_patches / patch_norms                     # (N*P, H)

        text_norms = np.linalg.norm(valid_text_tokens, axis=1, keepdims=True)
        text_norms = np.where(text_norms < 1e-8, 1.0, text_norms)
        valid_text_tokens = valid_text_tokens / text_norms          # (T, H)

        # FILIP score:
        # dot_matrix : (T, N*P)  — cosine similarities
        # for each text token: max over all patches
        # filip = mean over text tokens
        dot_matrix = valid_text_tokens @ all_patches.T              # (T, N*P)
        max_per_text = dot_matrix.max(axis=1)                       # (T,)
        filip_score  = float(max_per_text.mean())

        return filip_score

    def _sample_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
    ) -> List[np.ndarray]:
        """
        Sample n_frames uniformly from [start_time, end_time].

        Returns a list of BGR uint8 numpy arrays (any resolution — the engine
        preprocessor handles resizing).  Returns an empty list on failure.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("[FILIPReranker] Cannot open: %s", video_path)
            return []

        fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_total  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n_total / fps if n_total > 0 else (end_time - start_time)
        t1       = min(end_time, duration)

        n = self.n_frames
        if n == 1:
            timestamps = [(start_time + t1) / 2.0]
        else:
            step = (t1 - start_time) / max(n - 1, 1)
            timestamps = [start_time + i * step for i in range(n)]

        frames: List[np.ndarray] = []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()

        return frames
