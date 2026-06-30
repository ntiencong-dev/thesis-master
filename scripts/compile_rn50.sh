#!/usr/bin/env bash
# scripts/compile_rn50.sh
# ------------------------
# Compile CLIP RN50 INT8 xmodel to DPU executable for Kria KV260.
#
# MUST run inside Vitis AI Docker container:
#   docker run -v $(pwd):/workspace \
#              -v /home/tienc/.cache/huggingface:/home/vitis-ai-user/.cache/huggingface \
#              -it xilinx/vitis-ai-pytorch-cpu:3.5.0 bash
#   conda activate vitis-ai-pytorch
#   pip install open-clip-torch -q
#   bash scripts/compile_rn50.sh

set -euo pipefail

WORKSPACE="${1:-/workspace}"
OUTPUT_DIR="${WORKSPACE}/compiled"
QUANT_DIR="${WORKSPACE}/quantized"
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"

echo "=================================================="
echo "  CLIP RN50 → DPU B4096 Pipeline"
echo "=================================================="

# ── Step 1: PTQ calibration + export → INT8 xmodel ──────────────────────────
echo ""
echo "[1/3] PTQ INT8: calibration + xmodel export ..."
cd "${WORKSPACE}"

# Clear previous artifacts to prevent cache conflicts
rm -rf "${QUANT_DIR}" "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python scripts/quantize_clip_rn50.py --step all --output-dir "${QUANT_DIR}"

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
    echo "ERROR: Could not find quantized RN50 xmodel in ${QUANT_DIR}/"
    exit 1
fi

echo "  Found quantized xmodel: ${QUANT_XMODEL}"

# ── Step 2: Compile → DPU executable ─────────────────────────────────────────
echo ""
echo "[2/3] Compiling to DPU xmodel ..."

if [[ ! -f "$ARCH" ]]; then
    echo "ERROR: DPU arch file not found: ${ARCH}"
    exit 1
fi

vai_c_xir \
    --xmodel     "${QUANT_XMODEL}" \
    --arch       "${ARCH}" \
    --net_name   "clip_vision_rn50" \
    --output_dir "${OUTPUT_DIR}" \
    --options    '{"input_shape": "1,3,224,224"}' \
    2>&1 | tee "${OUTPUT_DIR}/compile_rn50_log.txt"

# ── Step 3: Verify ───────────────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying output ..."
COMPILED="${OUTPUT_DIR}/clip_vision_rn50.xmodel"

if [[ ! -f "${COMPILED}" ]]; then
    echo "ERROR: Compiled xmodel not found at ${COMPILED}"
    echo "       Check ${OUTPUT_DIR}/compile_rn50_log.txt"
    exit 1
fi

SIZE_MB=$(du -m "${COMPILED}" | cut -f1)
echo "  ✅ Compiled: ${COMPILED}  (${SIZE_MB} MB)"

echo "=================================================="
echo "  Compilation complete!"
echo "=================================================="
