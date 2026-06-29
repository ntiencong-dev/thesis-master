"""
scripts/quantize_all.py
------------------------
Master quantization orchestrator — chuẩn bị TẤT CẢ models cần thiết
cho Option B (FILIP), Option C (Dual-model), và Hướng 2 (BLIP-1 ITC+ITM).

Chạy lần lượt từng bước hoặc chọn option/model cụ thể.

Usage
-----
  # Chuẩn bị đầy đủ cho mọi option (PC side, không cần Docker):
  source venv/bin/activate
  python scripts/quantize_all.py --step awq

  # Chỉ AWQ cho Option B (CLIP text encoder):
  python scripts/quantize_all.py --step awq --option b

  # Chỉ AWQ cho Option C (CLIP text + BLIP-1 BERT):
  python scripts/quantize_all.py --step awq --option c

  # Build calibration dataset (cần video files trong ./segments/):
  python scripts/quantize_all.py --step calib

  # PTQ INT8 (phải chạy trong Vitis AI Docker):
  #   → Xem hướng dẫn phần PTQ bên dưới

Deployment map
--------------
Option B (FILIP):
  Stage-1 (ITC):    CLIP ViT-B/16 PTQ INT8 → clip_vision.xmodel     [DPU]
  Stage-1 (text):   CLIP text AWQ INT4 → clip_text_awq_int4/         [ARM CPU]
  Stage-2 (FILIP):  Cùng CLIP engine — không cần model riêng

Option C (Dual-model):
  Stage-1 (visual): CLIP ViT-B/16 PTQ INT8 → clip_vision.xmodel     [DPU]
  Stage-1 (text):   CLIP text AWQ INT4 → clip_text_awq_int4/         [ARM CPU]
  Stage-2 (visual): BLIP-1 ViT-B/16 PTQ INT8 → blip1_vision.xmodel  [DPU]
  Stage-2 (text):   BLIP-1 BERT AWQ INT4 → blip1_bert_awq_int4/      [ARM CPU]

Hướng 2 (BLIP-1 ITC+ITM — current main pipeline):
  Stage-1 (visual): BLIP-1 ViT-B/16 PTQ INT8 → blip1_vision.xmodel  [DPU]
  Stage-1 (text):   BLIP-1 BERT AWQ INT4 → blip1_bert_awq_int4/      [ARM CPU]
  Stage-2 (ITM):    Cùng BLIP-1 engine — không cần model riêng
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected output paths
# ---------------------------------------------------------------------------

PATHS = {
    # AWQ INT4 — ARM CPU (runnable on PC + Kria)
    "clip_text_awq":   Path("exported_models/clip_text_awq_int4"),
    "blip1_bert_awq":  Path("exported_models/blip1_bert_awq_int4"),

    # PTQ INT8 calibration data — needed by Vitis AI quantizer in Docker
    "calib_blip1":     Path("exported_models/calibration_frames_blip1.npy"),
    "calib_clip":      Path("exported_models/calibration_frames_clip.npy"),

    # PTQ INT8 xmodel — produced in Vitis AI Docker, NOT by this script
    "blip1_xmodel":    Path("compiled/blip1_vision.xmodel"),
    "clip_xmodel":     Path("compiled/clip_vision.xmodel"),
}


def _check_requirements() -> None:
    """Verify key packages are installed."""
    missing = []
    for pkg, import_name in [
        ("open-clip-torch", "open_clip"),
        ("transformers",    "transformers"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s", ", ".join(missing))
        log.error("Install: pip install %s", " ".join(missing))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Build calibration dataset (PC side)
# ---------------------------------------------------------------------------

def step_calib(segments_dir: str, n_frames: int, model: str) -> None:
    """Build PTQ calibration datasets (.npy) from video segments."""
    log.info("=" * 60)
    log.info("STEP 1: Build calibration datasets (%s)", model)
    log.info("=" * 60)

    args = [
        sys.executable, "scripts/build_calibration_dataset.py",
        "--segments-dir", segments_dir,
        "--n-frames", str(n_frames),
        "--model", model,
    ]
    log.info("Running: %s", " ".join(args))
    result = subprocess.run(args)
    if result.returncode != 0:
        log.error("Calibration dataset build failed.")
        sys.exit(result.returncode)

    log.info("")
    log.info("✅ Calibration datasets ready:")
    if model in ("blip1", "all"):
        log.info("   %s", PATHS["calib_blip1"])
    if model in ("clip", "all"):
        log.info("   %s", PATHS["calib_clip"])


# ---------------------------------------------------------------------------
# Step 2: AWQ INT4 quantization (PC or Kria ARM CPU — no Docker needed)
# ---------------------------------------------------------------------------

def step_awq(option: str) -> None:
    """Run AWQ INT4 quantization for text encoders."""
    log.info("=" * 60)
    log.info("STEP 2: AWQ INT4 quantization (option=%s)", option)
    log.info("=" * 60)

    tasks = []
    if option in ("b", "c", "all"):
        tasks.append({
            "name": "CLIP text encoder",
            "script": "scripts/awq_quantize_clip_text.py",
            "output": PATHS["clip_text_awq"],
            "fp32_mb": 250,
        })
    if option in ("c", "2", "all"):
        tasks.append({
            "name": "BLIP-1 BERT text encoder",
            "script": "scripts/awq_quantize_bert.py",
            "output": PATHS["blip1_bert_awq"],
            "fp32_mb": 440,
        })

    for task in tasks:
        out = task["output"]
        log.info("")
        log.info("── %s ──", task["name"])
        expected_pt = out / "clip_text_awq.pt" if "clip" in task["name"].lower() else out / "blip1_text_awq.pt"
        if expected_pt.exists():
            log.info("   Already exists at %s — skipping (delete to re-quantize)", expected_pt)
            continue

        args = [sys.executable, task["script"], "--output", str(out)]
        log.info("Running: %s", " ".join(args))
        result = subprocess.run(args)
        if result.returncode != 0:
            log.error("AWQ quantization failed for %s", task["name"])
        else:
            size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
            log.info("✅ %s → %s  (%.1f MB vs FP32 %.0f MB)",
                     task["name"], out, size, task["fp32_mb"])

    log.info("")
    _print_awq_summary(option)


def _print_awq_summary(option: str) -> None:
    log.info("AWQ outputs:")
    if option in ("b", "c", "all") and PATHS["clip_text_awq"].exists():
        s = sum(f.stat().st_size for f in PATHS["clip_text_awq"].rglob("*") if f.is_file()) / 1e6
        log.info("  ✅ CLIP text AWQ:   %s  (%.1f MB)", PATHS["clip_text_awq"], s)
    else:
        log.info("  ❌ CLIP text AWQ:  NOT FOUND — run with --step awq --option b")

    if option in ("c", "2", "all") and PATHS["blip1_bert_awq"].exists():
        s = sum(f.stat().st_size for f in PATHS["blip1_bert_awq"].rglob("*") if f.is_file()) / 1e6
        log.info("  ✅ BLIP-1 BERT AWQ: %s  (%.1f MB)", PATHS["blip1_bert_awq"], s)
    else:
        log.info("  ❌ BLIP-1 BERT AWQ: NOT FOUND — run with --step awq --option c")


# ---------------------------------------------------------------------------
# Step 3: PTQ INT8 — Vitis AI Docker (instructions only, not auto-run)
# ---------------------------------------------------------------------------

def step_ptq_instructions(option: str) -> None:
    """Print PTQ INT8 instructions (must run in Vitis AI Docker)."""
    log.info("=" * 60)
    log.info("STEP 3: PTQ INT8 → .xmodel  (Vitis AI Docker required)")
    log.info("=" * 60)
    log.info("")
    log.info("PTQ KHÔNG thể chạy trực tiếp trên PC — cần Vitis AI Docker.")
    log.info("Chạy các lệnh sau:")
    log.info("")
    log.info("  # 1. Start Vitis AI Docker:")
    log.info("  docker run \\")
    log.info("    -v $(pwd):/workspace \\")
    log.info("    -v ~/.cache/huggingface:/home/vitis-ai-user/.cache/huggingface \\")
    log.info("    -it xilinx/vitis-ai-pytorch-cpu:3.5.0 bash")
    log.info("  conda activate vitis-ai-pytorch")
    log.info("  cd /workspace")
    log.info("  pip install transformers open-clip-torch -q")
    log.info("")

    if option in ("2", "c", "all"):
        log.info("  # 2a. BLIP-1 visual encoder PTQ (Hướng 2 + Option C):")
        log.info("  python scripts/quantize_blip1.py --step calib")
        log.info("  python scripts/quantize_blip1.py --step export")
        log.info("")

    if option in ("b", "c", "all"):
        log.info("  # 2b. CLIP visual encoder PTQ (Option B + Option C):")
        log.info("  python scripts/quantize_clip_visual.py --step calib")
        log.info("  python scripts/quantize_clip_visual.py --step export")
        log.info("")

    log.info("  # 3. Compile xmodel for KV260 DPU B4096:")
    if option in ("2", "c", "all"):
        log.info("  vai_c_xir -x quantized/BlipVisualITCWrapper_int.xmodel \\")
        log.info("            -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \\")
        log.info("            -o compiled/ -n blip1_vision")
    if option in ("b", "c", "all"):
        log.info("  vai_c_xir -x quantized/CLIPVisualWrapper_int.xmodel \\")
        log.info("            -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \\")
        log.info("            -o compiled/ -n clip_vision")
    log.info("")
    log.info("  # Output:")
    if option in ("2", "c", "all"):
        log.info("  #   compiled/blip1_vision.xmodel  (~161 MB)")
    if option in ("b", "c", "all"):
        log.info("  #   compiled/clip_vision.xmodel   (~160 MB)")


# ---------------------------------------------------------------------------
# Step 4: Validate AWQ token quality for FILIP (Option B)
# ---------------------------------------------------------------------------

def step_validate_filip() -> None:
    """Run awq_validate_filip_tokens.py to verify FILIP token quality after AWQ."""
    log.info("=" * 60)
    log.info("STEP 4: Validate FILIP token quality (Option B)")
    log.info("=" * 60)

    awq_path = PATHS["clip_text_awq"]
    if not awq_path.exists():
        log.warning("AWQ CLIP text not found at %s — run --step awq --option b first", awq_path)
        return

    args = [
        sys.executable, "scripts/awq_validate_filip_tokens.py",
        "--awq_path", str(awq_path / "clip_text_awq.pt"),
    ]
    log.info("Running: %s", " ".join(args))
    result = subprocess.run(args)
    if result.returncode == 0:
        log.info("✅ FILIP token quality: PASS")
    else:
        log.warning("⚠️  FILIP token quality: FAIL — consider --q_group_size 256")


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def step_status() -> None:
    """Print current quantization status for all options."""
    log.info("=" * 60)
    log.info("QUANTIZATION STATUS")
    log.info("=" * 60)

    def _check(path: Path, name: str) -> bool:
        exists = path.exists()
        size   = ""
        if exists and path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            size  = f"  ({total / 1e6:.1f} MB)"
        elif exists:
            size = f"  ({path.stat().st_size / 1e6:.1f} MB)"
        icon = "✅" if exists else "❌"
        log.info("  %s %-45s %s%s", icon, name, path, size)
        return exists

    log.info("")
    log.info("── AWQ INT4 (PC + Kria ARM CPU) ─────────────────────────")
    _check(PATHS["clip_text_awq"],  "CLIP text AWQ INT4")
    _check(PATHS["blip1_bert_awq"], "BLIP-1 BERT AWQ INT4")

    log.info("")
    log.info("── PTQ calibration data (.npy) ──────────────────────────")
    _check(PATHS["calib_blip1"],    "BLIP-1 calibration frames")
    _check(PATHS["calib_clip"],     "CLIP calibration frames")

    log.info("")
    log.info("── PTQ INT8 xmodel (Vitis AI Docker output) ─────────────")
    _check(PATHS["blip1_xmodel"],   "BLIP-1 vision xmodel")
    _check(PATHS["clip_xmodel"],    "CLIP vision xmodel")

    log.info("")
    log.info("── Required per option ───────────────────────────────────")
    log.info("  Hướng 2  (BLIP-1 ITC+ITM):  blip1_xmodel + blip1_bert_awq")
    log.info("  Option B (FILIP):            clip_xmodel  + clip_text_awq  + calib_clip")
    log.info("  Option C (Dual-model):       clip_xmodel  + clip_text_awq")
    log.info("                               blip1_xmodel + blip1_bert_awq")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Master quantization script for NLVS (all options)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quantize_all.py --step status          # check what's done
  python scripts/quantize_all.py --step awq             # AWQ for all options
  python scripts/quantize_all.py --step awq --option b  # AWQ for Option B only
  python scripts/quantize_all.py --step calib           # build calibration data
  python scripts/quantize_all.py --step ptq             # print PTQ instructions
  python scripts/quantize_all.py --step all             # everything except PTQ Docker
        """,
    )
    p.add_argument(
        "--step",
        choices=["status", "calib", "awq", "ptq", "validate", "all"],
        default="status",
        help="Which step to run (default: status)",
    )
    p.add_argument(
        "--option",
        choices=["b", "c", "2", "all"],
        default="all",
        help=(
            "b:   Option B (FILIP) — CLIP only\n"
            "c:   Option C (Dual-model) — CLIP + BLIP-1\n"
            "2:   Hướng 2 (BLIP-1 ITC+ITM) — BLIP-1 only\n"
            "all: All options (default)"
        ),
    )
    p.add_argument(
        "--segments-dir", default="segments",
        help="Video segments directory for calibration (default: segments/)",
    )
    p.add_argument(
        "--n-frames", type=int, default=500,
        help="Calibration frame count (default: 500)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    _check_requirements()

    # Map option → which models to calibrate
    calib_model = {
        "b":   "clip",
        "c":   "all",
        "2":   "blip1",
        "all": "all",
    }[args.option]

    if args.step == "status":
        step_status()

    elif args.step == "calib":
        step_calib(args.segments_dir, args.n_frames, calib_model)

    elif args.step == "awq":
        step_awq(args.option)

    elif args.step == "ptq":
        step_ptq_instructions(args.option)

    elif args.step == "validate":
        step_validate_filip()

    elif args.step == "all":
        log.info("Running ALL steps (calib → awq → validate → ptq instructions)")
        log.info("")
        step_calib(args.segments_dir, args.n_frames, calib_model)
        log.info("")
        step_awq(args.option)
        log.info("")
        if args.option in ("b", "all"):
            step_validate_filip()
        log.info("")
        step_ptq_instructions(args.option)
        log.info("")
        step_status()


if __name__ == "__main__":
    main()
