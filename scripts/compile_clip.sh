#!/usr/bin/env bash
# scripts/compile_clip.sh
# ------------------------
# Bước cuối — Compile CLIP ViT-B/16 INT8 xmodel sang DPU executable cho Kria KV260.
#
# PHẢI chạy TRONG Vitis AI Docker container:
#   docker run -v $(pwd):/workspace \
#              -v /home/tienc/.cache/huggingface:/home/vitis-ai-user/.cache/huggingface \
#              -it xilinx/vitis-ai-pytorch-cpu:latest bash
#   conda activate vitis-ai-pytorch
#   pip install open-clip-torch -q
#   bash scripts/compile_clip.sh
#
# Requires:
#   exported_models/calibration_frames_clip.npy   (từ build_calibration_dataset.py)
#
# Output:
#   compiled/clip_vision.xmodel    (~160 MB — deploy lên KV260)
#
# After compile:
#   scp compiled/clip_vision.xmodel ubuntu@<kria-ip>:/usr/share/vitis_ai_library/models/clip_vit_b16/

set -euo pipefail

WORKSPACE="${1:-/workspace}"
OUTPUT_DIR="${WORKSPACE}/compiled"
QUANT_DIR="${WORKSPACE}/quantized"
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"

echo "=================================================="
echo "  CLIP ViT-B/16 → DPU B4096 Pipeline"
echo "=================================================="

# ── Step 1: PTQ calibration + export → INT8 xmodel ──────────────────────────
echo ""
echo "[1/3] PTQ INT8: calibration + xmodel export ..."
cd "${WORKSPACE}"
python scripts/quantize_clip_visual.py --step all --output-dir "${QUANT_DIR}"

# ── Find the quantized xmodel ────────────────────────────────────────────────
QUANT_XMODEL=""
for candidate in \
    "${QUANT_DIR}/CLIPVisualWrapper_int.xmodel" \
    "${QUANT_DIR}/quantize_result/CLIPVisualWrapper.xmodel" \
    "${QUANT_DIR}/clip_visual_int.xmodel"; do
    if [[ -f "$candidate" ]]; then
        QUANT_XMODEL="$candidate"
        break
    fi
done

if [[ -z "$QUANT_XMODEL" ]]; then
    echo "ERROR: Could not find quantized CLIP xmodel in ${QUANT_DIR}/"
    echo "       Available files:"
    ls "${QUANT_DIR}/" 2>/dev/null || echo "  (empty)"
    exit 1
fi

echo ""
echo "  Found quantized xmodel: ${QUANT_XMODEL}"

# ── Step 2: Compile → DPU executable ─────────────────────────────────────────
echo ""
echo "[2/3] Compiling to DPU xmodel ..."

if [[ ! -f "$ARCH" ]]; then
    echo "ERROR: DPU arch file not found: ${ARCH}"
    echo "       Is this the Vitis AI Docker container?"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

vai_c_xir \
    --xmodel     "${QUANT_XMODEL}" \
    --arch       "${ARCH}" \
    --net_name   "clip_vision" \
    --output_dir "${OUTPUT_DIR}" \
    --options    '{"input_shape": "4,3,224,224"}' \
    2>&1 | tee "${OUTPUT_DIR}/compile_clip_log.txt"

# ── Step 3: Verify ───────────────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying output ..."
COMPILED="${OUTPUT_DIR}/clip_vision.xmodel"

if [[ ! -f "${COMPILED}" ]]; then
    echo "ERROR: Compiled xmodel not found at ${COMPILED}"
    echo "       Check ${OUTPUT_DIR}/compile_clip_log.txt"
    exit 1
fi

SIZE_MB=$(du -m "${COMPILED}" | cut -f1)
echo "  ✅ Compiled: ${COMPILED}  (${SIZE_MB} MB)"

# Subgraph analysis
if command -v python3 &>/dev/null && [[ -f "${WORKSPACE}/scripts/_analyze_xmodel.py" ]]; then
    python3 "${WORKSPACE}/scripts/_analyze_xmodel.py" "${COMPILED}"
fi

echo ""
echo "=================================================="
echo "  Compilation complete!"
echo "=================================================="
echo ""
echo "Deploy to KV260:"
echo "  ssh ubuntu@<kria-ip> 'mkdir -p /usr/share/vitis_ai_library/models/clip_vit_b16'"
echo "  scp ${COMPILED} ubuntu@<kria-ip>:/usr/share/vitis_ai_library/models/clip_vit_b16/"
echo ""
echo "Update config/kria_clip_blip1.yaml:"
echo "  engine:"
echo "    type: kria"
echo "    xmodel_path: /usr/share/vitis_ai_library/models/clip_vit_b16/clip_vision.xmodel"
