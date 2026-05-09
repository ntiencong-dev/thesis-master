"""
engines/kria_engine.py
-----------------------
Edge backend: CLIP Vision + Text encoders compiled to .xmodel format,
executed on the DPU B4096 of AMD Kria KV260 via VART (Vitis AI Runtime).

Deployment workflow (NOT implemented here — requires Kria hardware):
1. Quantise CLIP with vai_q_pytorch (INT8, calibration dataset).
2. Compile with vai_c_xir → clip_vision.xmodel + clip_text.xmodel.
3. Deploy xmodel files to /usr/share/vitis_ai_library/models/clip_vit_b16/.
4. Set config.yaml: engine.type = "kria" and set xmodel_path values.

This module raises ImportError if vart is not installed so that the
factory can gracefully reject kria config on non-Kria hardware.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .base_engine import InferenceEngine


class KriaEngine(InferenceEngine):
    """
    Parameters (from config['engine'] dict)
    ----------------------------------------
    xmodel_path      : str   path to compiled vision encoder (.xmodel)
    xmodel_text_path : str   path to compiled text encoder (.xmodel)
    device_id        : int   DPU device index (default 0)
    batch_size       : int   DPU batch size (default 4)
    """

    # DPU output from ViT-B/16 INT8 projected back to 512-d float32
    _EMBED_DIM = 512

    def __init__(self, engine_cfg: dict) -> None:
        try:
            import vart          # Vitis AI Runtime
            import xir           # XIR graph loader
        except ImportError as exc:
            raise ImportError(
                "VART / XIR Python packages not found. "
                "Install Vitis AI 3.5 runtime on AMD Kria KV260. "
                "On PC prototype, set engine.type = 'pc' in config."
            ) from exc

        self._vart = vart
        self._xir  = xir

        self._vision_runner = self._load_runner(
            engine_cfg["xmodel_path"], device_id=engine_cfg.get("device_id", 0)
        )
        self._text_runner = self._load_runner(
            engine_cfg["xmodel_text_path"], device_id=engine_cfg.get("device_id", 0)
        )
        self._batch_size = engine_cfg.get("batch_size", 4)

    # ------------------------------------------------------------------
    # InferenceEngine contract
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._EMBED_DIM

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Run DPU inference on BGR frames using the compiled vision xmodel.

        Each frame is pre-processed (normalise, layout: NHWC INT8) before
        being fed to the DPU runner.  The output activation is L2-normalised
        to match the PC engine's contract.
        """
        results: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self._batch_size):
            batch = frames_bgr[i : i + self._batch_size]
            preprocessed = self._preprocess_frames(batch)

            # Allocate DPU input/output tensors
            input_tensors  = self._vision_runner.get_input_tensors()
            output_tensors = self._vision_runner.get_output_tensors()
            inputs  = [np.empty(t.dims, dtype=np.int8) for t in input_tensors]
            outputs = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]

            inputs[0][:len(batch)] = preprocessed
            job_id = self._vision_runner.execute_async(inputs, outputs)
            self._vision_runner.wait(job_id)

            # Dequantise INT8 → float32, then L2-normalise
            scale  = output_tensors[0].get_attr("fix_point")
            feats  = outputs[0][:len(batch)].astype(np.float32) * (2.0 ** (-scale))
            norms  = np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-8)
            results.append((feats / norms).astype(np.float32))

        return np.vstack(results)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Run DPU inference using the compiled text encoder xmodel.

        BPE tokenisation is performed in Python (same tokeniser as PC).
        Only the Transformer forward pass runs on DPU.
        """
        if isinstance(texts, str):
            texts = [texts]

        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        token_ids = tokenizer(texts).numpy()    # (N, 77) int32

        results: List[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch_tokens = token_ids[i : i + self._batch_size]

            input_tensors  = self._text_runner.get_input_tensors()
            output_tensors = self._text_runner.get_output_tensors()
            inputs  = [np.empty(t.dims, dtype=np.int8) for t in input_tensors]
            outputs = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]

            inputs[0][:len(batch_tokens)] = batch_tokens.astype(np.int8)
            job_id = self._text_runner.execute_async(inputs, outputs)
            self._text_runner.wait(job_id)

            scale  = output_tensors[0].get_attr("fix_point")
            feats  = outputs[0][:len(batch_tokens)].astype(np.float32) * (2.0 ** (-scale))
            norms  = np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-8)
            results.append((feats / norms).astype(np.float32))

        return np.vstack(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_runner(self, xmodel_path: str, device_id: int):
        graph    = self._xir.Graph.deserialize(xmodel_path)
        subgraph = self._get_dpu_subgraph(graph)
        return self._vart.Runner.create_runner(subgraph, "run")

    @staticmethod
    def _get_dpu_subgraph(graph):
        """Return the first DPU-type subgraph in the compiled xmodel."""
        for sg in graph.get_root_subgraph().toposort_child_subgraph():
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                return sg
        raise RuntimeError("No DPU subgraph found in xmodel.")

    @staticmethod
    def _preprocess_frames(frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Convert BGR float frames to INT8 NHWC format for DPU.
        Applies CLIP normalisation (mean/std) then quantises to INT8.
        """
        import cv2

        MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

        batch = []
        for frame in frames_bgr:
            rgb  = frame[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, /255
            norm = (rgb - MEAN) / STD
            batch.append(norm)

        arr = np.stack(batch)                           # (N, H, W, 3) float32
        return np.clip(arr * 128, -128, 127).astype(np.int8)
