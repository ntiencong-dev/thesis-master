"""
engines/factory.py
-------------------
Creates the correct InferenceEngine instance from a config dict.

Usage
-----
    from src.engines.factory import create_engine
    import yaml

    with open("config/pc_blip1.yaml") as f:
        cfg = yaml.safe_load(f)

    engine = create_engine(cfg)
    # → BLIP1Engine (type="blip1")
    # → KriaEngine  (type="kria")
"""

from __future__ import annotations

from .base_engine import InferenceEngine


def create_engine(config: dict) -> InferenceEngine:
    """
    Parameters
    ----------
    config : dict loaded from pc_blip1.yaml or kria_blip1.yaml.

    Returns
    -------
    InferenceEngine subclass matching config['engine']['type'].

    Raises
    ------
    ValueError  if engine type is unknown.
    ImportError if Kria engine is requested but VART is not installed.
    """
    engine_cfg  = config.get("engine", {})
    engine_type = engine_cfg.get("type", "pc").lower()

    if engine_type == "pc":
        from .pc_engine import PCEngine
        return PCEngine(engine_cfg)

    if engine_type == "kria":
        from .kria_engine import KriaEngine
        return KriaEngine(engine_cfg)

    if engine_type == "xclip":
        from .xclip_engine import XCLIPEngine
        return XCLIPEngine(engine_cfg)

    if engine_type == "siglip":
        from .siglip_engine import SigLIPEngine
        return SigLIPEngine(engine_cfg)

    if engine_type == "languagebind":
        from .languagebind_engine import LanguageBindEngine
        return LanguageBindEngine(engine_cfg)

    if engine_type == "internvideo2":
        from .intern_video2_engine import InternVideo2Engine
        return InternVideo2Engine(engine_cfg)

    if engine_type == "blip1":
        from .blip1_engine import BLIP1Engine
        return BLIP1Engine(engine_cfg)

    if engine_type == "clip_blip1":
        # Dual-model: CLIP (PCEngine/KriaEngine) for stage-1 ITC retrieval
        # + BLIP-1 ITM for stage-2 re-ranking.
        # Returns the CLIP engine as primary; the BLIP-1 engine is stored
        # as _blip1_engine attribute for the searcher to attach as a reranker.
        from .blip1_engine import BLIP1Engine

        # Stage-1: CLIP engine (type overridden to 'pc' or 'kria')
        clip_cfg = {**engine_cfg, "type": engine_cfg.get("clip_engine_type", "pc")}
        if clip_cfg["type"] == "kria":
            from .kria_engine import KriaEngine
            clip_engine = KriaEngine(clip_cfg)
        else:
            from .pc_engine import PCEngine
            clip_engine = PCEngine(clip_cfg)

        # Stage-2: BLIP-1 engine using the itm_model sub-config
        blip1_cfg = {
            "model_name": config.get("search", {}).get(
                "itm_model_name", "Salesforce/blip-itm-base-coco"
            ),
            "device":     engine_cfg.get("device", None),
            "batch_size": engine_cfg.get("blip1_batch_size", 4),
            "awq_bert_path": engine_cfg.get("awq_bert_path", ""),
        }
        blip1_engine = BLIP1Engine(blip1_cfg)
        # Attach BLIP-1 engine as a sidecar on the CLIP engine
        clip_engine._blip1_engine = blip1_engine  # type: ignore[attr-defined]
        return clip_engine

    raise ValueError(
        f"Unknown engine type '{engine_type}'. "
        "Valid options: 'pc', 'kria', 'xclip', 'siglip', 'languagebind', "
        "'internvideo2', 'blip1', 'clip_blip1'."
    )

