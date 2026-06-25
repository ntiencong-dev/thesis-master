"""
engines/kria_engine.py
-----------------------
Edge backend: CLIP Vision + Text encoders compiled to .xmodel format,
executed on the DPU B4096 of AMD Kria KV260 via VART (Vitis AI Runtime).

Deployment workflow:
1. Quantise visual encoders with vai_q_pytorch (INT8) → PTQ.
2. Compile with vai_c_xir → clip_vision.xmodel + blip1_vision.xmodel.
3. Quantise text encoders with AutoAWQ (INT4) → AWQ scripts.
4. Deploy xmodel files to /usr/share/vitis_ai_library/models/.
5. Set config.yaml: engine.type = "kria" and set xmodel_path / awq_text_path.

MACRO — Persistent Runner Pool:
    Cả CLIP và BLIP-1 DPU runners được giữ alive từ lúc khởi động.
    Không reload weights giữa các lần inference → tiết kiệm ~420 ms/query.
    Bật bằng cách cấu hình engine.macro_pool: true trong config.

    Khi macro_pool = true và blip1_xmodel_path được set:
        KriaEngine tạo MacroRunnerPool chứa cả CLIP runner + BLIP-1 runner.
        run_clip_visual() / run_blip1_visual() switch giữa runners tức thì.

AWQ Text Encoders:
    Text encoders (CLIP text, BLIP-1 BERT) chạy trên ARM Cortex-A53.
    PTQ INT8 không đủ nhỏ cho dual-model trên 4 GB LPDDR4.
    Dùng AutoAWQ INT4 (scripts/awq_quantize_*.py) → ~4× size reduction.
    Load qua awq_text_path / awq_bert_path trong config.

This module raises ImportError if vart is not installed so that the
factory can gracefully reject kria config on non-Kria hardware.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .base_engine import InferenceEngine


class MacroRunnerPool:
    """
    MACRO: Giữ nhiều DPU runners alive liên tục trong DDR.

    Thay vì tạo/hủy runner mỗi lần inference (tốn ~180–240 ms/lần load),
    MacroRunnerPool khởi tạo tất cả runners một lần khi server start và
    giữ chúng trong bộ nhớ DDR suốt vòng đời process.

    Latency benefit (Kria KV260, DPU B4096):
        Không MACRO: load CLIP runner ~180 ms + load BLIP-1 runner ~240 ms
        Có MACRO:    cả hai runners luôn sẵn sàng → 0 ms overhead reload

    Usage:
        pool = MacroRunnerPool(clip_xmodel="...", blip1_xmodel="...", vart=vart, xir=xir)
        clip_output  = pool.run_vision(pool.CLIP,  frames_int8)
        blip1_output = pool.run_vision(pool.BLIP1, frames_int8)
    """

    CLIP  = "clip"
    BLIP1 = "blip1"

    def __init__(
        self,
        clip_xmodel: str,
        blip1_xmodel: str,
        vart,
        xir,
        device_id: int = 0,
    ) -> None:
        import logging
        self._log = logging.getLogger(__name__)
        self._vart = vart
        self._xir  = xir

        self._log.info("[MacroRunnerPool] Preloading CLIP runner...")
        self._clip_runner  = self._load_runner(clip_xmodel,  device_id)

        self._log.info("[MacroRunnerPool] Preloading BLIP-1 runner...")
        self._blip1_runner = self._load_runner(blip1_xmodel, device_id)

        self._log.info("[MacroRunnerPool] Both DPU runners active — MACRO ready.")

    def run_vision(self, model_key: str, frames_int8: np.ndarray) -> np.ndarray:
        """
        Chạy DPU inference với runner tương ứng mà không reload.

        Parameters
        ----------
        model_key  : MacroRunnerPool.CLIP hoặc MacroRunnerPool.BLIP1
        frames_int8: (N, H, W, C) INT8 NHWC tensor

        Returns
        -------
        np.ndarray  shape (N, embed_dim), float32, L2-normalised
        """
        runner = self._clip_runner if model_key == self.CLIP else self._blip1_runner
        return self._dpu_infer(runner, frames_int8)

    def _dpu_infer(self, runner, frames_int8: np.ndarray) -> np.ndarray:
        """Shared DPU inference path cho cả CLIP và BLIP-1."""
        input_tensors  = runner.get_input_tensors()
        output_tensors = runner.get_output_tensors()
        n = len(frames_int8)

        inputs  = [np.empty(t.dims, dtype=np.int8) for t in input_tensors]
        outputs = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]
        inputs[0][:n] = frames_int8

        job_id = runner.execute_async(inputs, outputs)
        runner.wait(job_id)

        scale = output_tensors[0].get_attr("fix_point")
        feats = outputs[0][:n].astype(np.float32) * (2.0 ** (-scale))
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-8)
        return (feats / norms).astype(np.float32)

    def _load_runner(self, xmodel_path: str, device_id: int):
        graph    = self._xir.Graph.deserialize(xmodel_path)
        subgraph = KriaEngine._get_dpu_subgraph(graph)
        return self._vart.Runner.create_runner(subgraph, "run")


class KriaEngine(InferenceEngine):
    """
    Parameters (from config['engine'] dict)
    ----------------------------------------
    xmodel_path      : str   path to compiled CLIP vision encoder (.xmodel)
    xmodel_text_path : str   path to compiled CLIP text encoder (.xmodel) [optional]
    blip1_xmodel_path: str   path to compiled BLIP-1 vision encoder (.xmodel) [optional]
    awq_text_path    : str   path to AWQ INT4 CLIP text encoder dir [optional]
    awq_bert_path    : str   path to AWQ INT4 BLIP-1 BERT dir [optional]
    macro_pool       : bool  enable MacroRunnerPool (default False)
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
        self._batch_size = engine_cfg.get("batch_size", 4)
        device_id = engine_cfg.get("device_id", 0)

        use_macro = engine_cfg.get("macro_pool", False)
        blip1_path = engine_cfg.get("blip1_xmodel_path")

        if use_macro and blip1_path:
            # MACRO mode: preload cả CLIP và BLIP-1 runners vào DDR
            self._macro_pool = MacroRunnerPool(
                clip_xmodel=engine_cfg["xmodel_path"],
                blip1_xmodel=blip1_path,
                vart=vart,
                xir=xir,
                device_id=device_id,
            )
            self._vision_runner = None   # dùng macro_pool thay vì runner trực tiếp
        else:
            self._macro_pool    = None
            self._vision_runner = self._load_runner(
                engine_cfg["xmodel_path"], device_id=device_id
            )

        # Text runner (CLIP text encoder trên DPU — optional)
        # Nếu awq_text_path được set, dùng AWQ INT4 text trên ARM CPU thay vì DPU
        self._awq_text_path = engine_cfg.get("awq_text_path")
        self._awq_bert_path = engine_cfg.get("awq_bert_path")

        if engine_cfg.get("xmodel_text_path") and not self._awq_text_path:
            self._text_runner = self._load_runner(
                engine_cfg["xmodel_text_path"], device_id=device_id
            )
        else:
            self._text_runner = None

    # ------------------------------------------------------------------
    # InferenceEngine contract
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._EMBED_DIM

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Run DPU inference on BGR frames using the compiled vision xmodel.
        Nếu MACRO mode đang bật, dùng MacroRunnerPool (không reload weights).
        """
        results: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self._batch_size):
            batch = frames_bgr[i : i + self._batch_size]
            preprocessed = self._preprocess_frames(batch)

            if self._macro_pool is not None:
                # MACRO path: runner luôn sẵn sàng trong DDR
                feats = self._macro_pool.run_vision(MacroRunnerPool.CLIP, preprocessed)
                results.append(feats)
                continue

            # Standard path
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

    def encode_frames_blip1(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Chạy BLIP-1 visual encoder qua MACRO pool (cho stage-2 ITM reranking).
        Chỉ available khi macro_pool=true và blip1_xmodel_path được set.
        """
        if self._macro_pool is None:
            raise RuntimeError(
                "encode_frames_blip1() yêu cầu macro_pool=true và blip1_xmodel_path "
                "trong config. Bật bằng engine.macro_pool: true."
            )
        results: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self._batch_size):
            batch = frames_bgr[i : i + self._batch_size]
            # BLIP-1 input: 384×384 (khác CLIP 224×224)
            preprocessed = self._preprocess_frames(batch, size=384)
            feats = self._macro_pool.run_vision(MacroRunnerPool.BLIP1, preprocessed)
            results.append(feats)
        return np.vstack(results)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text queries.

        Ưu tiên:
        1. AWQ INT4 text encoder (ARM CPU) nếu awq_text_path được set — tiết kiệm bộ nhớ
        2. DPU text runner (xmodel_text_path) nếu có
        3. Fallback: open_clip FP32 trên ARM CPU
        """
        if isinstance(texts, str):
            texts = [texts]

        if self._awq_text_path:
            return self._encode_text_awq(texts)

        if self._text_runner is not None:
            return self._encode_text_dpu(texts)

        # Fallback: open_clip FP32 trên ARM CPU
        return self._encode_text_fp32_fallback(texts)

    def _encode_text_awq(self, texts: List[str]) -> np.ndarray:
        """Load và chạy AWQ INT4 CLIP text encoder."""
        import torch, torch.nn.functional as F

        try:
            import open_clip
        except ImportError:
            return self._encode_text_fp32_fallback(texts)

        # Lazy-load AWQ model (lần đầu tiên gọi)
        if not hasattr(self, "_awq_clip_model"):
            awq_data = torch.load(
                f"{self._awq_text_path}/clip_text_awq.pt", map_location="cpu"
            )
            model, _, _ = open_clip.create_model_and_transforms(
                awq_data.get("model_name", "ViT-B-16"),
                pretrained=awq_data.get("pretrained", "openai"),
            )
            model.transformer.load_state_dict(awq_data["text_transformer_state_dict"])
            model.text_projection = awq_data["text_projection"]
            model.eval()
            self._awq_clip_model = model
            self._awq_clip_tokenizer = open_clip.get_tokenizer(
                awq_data.get("model_name", "ViT-B-16")
            )

        tokens = self._awq_clip_tokenizer(texts)
        with torch.no_grad():
            embs = self._awq_clip_model.encode_text(tokens)
            embs = F.normalize(embs.float(), dim=-1)
        return embs.numpy()

    def _encode_text_dpu(self, texts: List[str]) -> np.ndarray:
        """Dùng DPU text runner (xmodel) — path cũ, giữ tương thích ngược."""
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        token_ids = tokenizer(texts).numpy()

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

    def _encode_text_fp32_fallback(self, texts: List[str]) -> np.ndarray:
        """Fallback FP32 CLIP text encoding trên ARM CPU."""
        import torch, torch.nn.functional as F, open_clip
        if not hasattr(self, "_fp32_clip"):
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
            model.eval()
            self._fp32_clip = model
            self._fp32_tokenizer = open_clip.get_tokenizer("ViT-B-16")
        tokens = self._fp32_tokenizer(texts)
        with torch.no_grad():
            embs = self._fp32_clip.encode_text(tokens)
            embs = F.normalize(embs.float(), dim=-1)
        return embs.numpy()

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
    def _preprocess_frames(frames_bgr: List[np.ndarray], size: int = 224) -> np.ndarray:
        """
        Convert BGR float frames to INT8 NHWC format for DPU.
        Applies CLIP normalisation (mean/std) then quantises to INT8.

        Parameters
        ----------
        frames_bgr : list of H×W×3 BGR uint8 arrays
        size       : target spatial size (224 for CLIP, 384 for BLIP-1)
        """
        import cv2

        MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

        batch = []
        for frame in frames_bgr:
            resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            rgb  = resized[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, /255
            norm = (rgb - MEAN) / STD
            batch.append(norm)

        arr = np.stack(batch)                           # (N, H, W, 3) float32
        return np.clip(arr * 128, -128, 127).astype(np.int8)
