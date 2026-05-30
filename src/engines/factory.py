"""
engines/factory.py
-------------------
Creates the correct InferenceEngine instance from a config dict.

Usage
-----
    from src.engines.factory import create_engine
    import yaml

    with open("config/pc.yaml") as f:
        cfg = yaml.safe_load(f)

    engine = create_engine(cfg)
    # → PCEngine  (type="pc")
    # → KriaEngine (type="kria")
"""

from __future__ import annotations

from .base_engine import InferenceEngine


def create_engine(config: dict) -> InferenceEngine:
    """
    Parameters
    ----------
    config : dict loaded from pc.yaml or kria.yaml.

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

    raise ValueError(
        f"Unknown engine type '{engine_type}'. "
        "Valid options: 'pc', 'kria', 'xclip', 'siglip', 'languagebind', 'internvideo2', 'blip1'."
    )
