"""
scripts/build_calibration_dataset.py
--------------------------------------
Bước 1.2 — Thu thập calibration frames từ video segments để dùng
cho PTQ (Post-Training Quantization) với vai_q_pytorch.

Yêu cầu: video MP4 files trong ./segments/ (hoặc --segments-dir chỉ định).

Output:
  exported_models/calibration_frames_blip1.npy
      shape: (N_FRAMES, 3, 384, 384), dtype=float32
      Đã qua BlipProcessor normalisation (RGB, ImageNet-style mean/std).
      Sẵn sàng feed trực tiếp vào TorchScript model.

Usage:
  source venv/bin/activate
  python scripts/build_calibration_dataset.py
  python scripts/build_calibration_dataset.py --segments-dir ./segments --n-frames 1000
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME    = "Salesforce/blip-itm-base-coco"
DEFAULT_DIR   = Path("segments")
OUTPUT_DIR    = Path("exported_models")
N_FRAMES      = 500          # Vitis AI recommends 100–1000, 500 is a good balance
INPUT_SIZE    = 384


# ---------------------------------------------------------------------------

def sample_frames_from_video(path: Path, n: int) -> list[np.ndarray]:
    """Sample n frames uniformly from a video file. Returns list of BGR uint8 arrays."""
    cap   = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def build_dataset(segments_dir: Path, n_frames: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(segments_dir.glob("**/*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {segments_dir}")

    log.info("Found %d video files in %s", len(video_files), segments_dir)
    log.info("Target: %d calibration frames total", n_frames)

    # -----------------------------------------------------------------------
    # Sample frames evenly across all videos
    # -----------------------------------------------------------------------
    per_video = max(1, n_frames // len(video_files))
    all_bgr: list[np.ndarray] = []

    for vpath in video_files:
        frames = sample_frames_from_video(vpath, per_video)
        log.info("  %s → %d frames sampled", vpath.name, len(frames))
        all_bgr.extend(frames)

    # Shuffle + trim to exactly n_frames
    random.shuffle(all_bgr)
    all_bgr = all_bgr[:n_frames]
    log.info("Total frames collected: %d", len(all_bgr))

    # -----------------------------------------------------------------------
    # Preprocess with BlipProcessor (same normalisation as inference)
    # -----------------------------------------------------------------------
    log.info("Preprocessing with BlipProcessor (384×384, RGB normalised) …")
    from transformers import BlipProcessor
    proc = BlipProcessor.from_pretrained(MODEL_NAME)

    # Convert BGR → RGB for BlipProcessor
    rgb_frames = [f[:, :, ::-1] for f in all_bgr]

    batch_size  = 32
    all_tensors = []

    for i in range(0, len(rgb_frames), batch_size):
        batch   = rgb_frames[i : i + batch_size]
        inputs  = proc(images=batch, return_tensors="pt")
        pv      = inputs["pixel_values"]        # (B, 3, 384, 384)
        all_tensors.append(pv.numpy())
        if (i // batch_size) % 5 == 0:
            log.info("  Processed %d / %d frames …", min(i + batch_size, len(rgb_frames)), len(rgb_frames))

    calibration_array = np.concatenate(all_tensors, axis=0).astype(np.float32)
    log.info("Calibration array shape: %s  dtype=%s", calibration_array.shape, calibration_array.dtype)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = output_dir / "calibration_frames_blip1.npy"
    np.save(str(out_path), calibration_array)
    size_mb = out_path.stat().st_size / 1e6
    log.info("Saved: %s  (%.1f MB)", out_path, size_mb)

    # Also save a small stats file for verification
    stats = {
        "n_frames":    calibration_array.shape[0],
        "shape":       list(calibration_array.shape),
        "mean":        float(calibration_array.mean()),
        "std":         float(calibration_array.std()),
        "min":         float(calibration_array.min()),
        "max":         float(calibration_array.max()),
        "source_dir":  str(segments_dir),
        "n_videos":    len(video_files),
    }
    import json
    stats_path = output_dir / "calibration_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("Stats: %s", stats)

    log.info("")
    log.info("=== Calibration dataset ready ===")
    log.info("  Frames:  %d", calibration_array.shape[0])
    log.info("  Shape:   %s", calibration_array.shape)
    log.info("  Saved:   %s", out_path)
    log.info("")
    log.info("Next step (inside Vitis AI Docker):")
    log.info("  python scripts/quantize_blip1.py")

    return out_path


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BLIP-1 PTQ calibration dataset")
    p.add_argument("--segments-dir", default=str(DEFAULT_DIR),
                   help=f"Directory containing .mp4 segment files (default: {DEFAULT_DIR})")
    p.add_argument("--n-frames", type=int, default=N_FRAMES,
                   help=f"Number of calibration frames to collect (default: {N_FRAMES})")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help=f"Output directory (default: {OUTPUT_DIR})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        segments_dir=Path(args.segments_dir),
        n_frames=args.n_frames,
        output_dir=Path(args.output_dir),
    )
