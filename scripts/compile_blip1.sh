#!/usr/bin/env bash
# scripts/compile_blip1.sh
# --------------------------
# Bước 1.4 — Compile INT8 xmodel sang DPU executable cho Kria KV260 (DPU B4096).
#
# PHẢI chạy TRONG Vitis AI Docker container (cùng session với quantize_blip1.py):
#   docker run -v $(pwd):/workspace -it xilinx/vitis-ai-pytorch-cpu:3.5.0 bash
#   conda activate vitis-ai-pytorch
#   bash scripts/compile_blip1.sh
#
# Requires:
#   quantized/blip1_visual_fp32_int.xmodel    (từ quantize_blip1.py)
#
# Output:
#   compiled/blip1_vision.xmodel              (~86 MB — deploy lên KV260)
#   compiled/blip1_vision_deploy_check.txt    (compatibility report)
#
# After compile: scp compiled/blip1_vision.xmodel ubuntu@<kria-ip>:/usr/share/vitis_ai_library/models/blip1_vit_b16/

set -euo pipefail

WORKSPACE="${1:-/workspace}"
QUANT_DIR="${WORKSPACE}/quantized"
OUTPUT_DIR="${WORKSPACE}/compiled"
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"

# ── Step 1: PTQ calibration + export → INT8 xmodel ──────────────────────────
echo ""
echo "[1/3] PTQ INT8: calibration + xmodel export ..."
cd "${WORKSPACE}"
python scripts/quantize_blip1.py --step all --output-dir "${QUANT_DIR}"

# Input xmodel name produced by vai_q_pytorch (may vary by version)
# Try common naming patterns
QUANT_XMODEL=""
for candidate in \
    "${QUANT_DIR}/BlipVisionModel_int.xmodel" \
    "${QUANT_DIR}/quantize_result/BlipVisionModel.xmodel" \
    "${QUANT_DIR}/blip1_visual_fp32_int.xmodel" \
    "${QUANT_DIR}/BlipVisualITCWrapper_int.xmodel" \
    "${QUANT_DIR}/quantize_result/BlipVisualITCWrapper.xmodel"; do
    if [[ -f "$candidate" ]]; then
        QUANT_XMODEL="$candidate"
        break
    fi
done

if [[ -z "$QUANT_XMODEL" ]]; then
    echo "ERROR: Could not find quantized xmodel in ${QUANT_DIR}/"
    echo "       Run quantize_blip1.py first."
    echo "       Available files:"
    ls "${QUANT_DIR}/" 2>/dev/null || echo "  (directory empty or missing)"
    exit 1
fi

echo "=================================================="
echo "  BLIP-1 ViT-B/16 → DPU B4096 Compilation"
echo "=================================================="
echo "  Input xmodel : ${QUANT_XMODEL}"
echo "  Target arch  : ${ARCH}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo ""

# Verify arch file exists
if [[ ! -f "$ARCH" ]]; then
    echo "ERROR: DPU arch file not found: ${ARCH}"
    echo "       Is Vitis AI installed? Is this the Vitis AI Docker container?"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------
# Compile: quantized xmodel → DPU executable xmodel
# ------------------------------------------------------------------
echo "[1/3] Running vai_c_xir …"
vai_c_xir \
    --xmodel     "${QUANT_XMODEL}" \
    --arch       "${ARCH}" \
    --net_name   "blip1_vision" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/compile_log.txt"

# ------------------------------------------------------------------
# Verify output
# ------------------------------------------------------------------
echo ""
echo "[2/3] Verifying output …"
COMPILED_XMODEL="${OUTPUT_DIR}/blip1_vision.xmodel"

if [[ ! -f "${COMPILED_XMODEL}" ]]; then
    echo "ERROR: Compiled xmodel not found at ${COMPILED_XMODEL}"
    echo "       Check compile_log.txt for errors."
    exit 1
fi

SIZE_MB=$(du -m "${COMPILED_XMODEL}" | cut -f1)
echo "  Compiled xmodel: ${COMPILED_XMODEL}  (${SIZE_MB} MB)"

# ------------------------------------------------------------------
# Subgraph analysis: check DPU coverage
# ------------------------------------------------------------------
echo ""
echo "[3/3] Subgraph analysis (DPU coverage) …"
python3 /workspace/scripts/_analyze_xmodel.py "${COMPILED_XMODEL}"

echo ""
echo "=================================================="
echo "  Compilation complete!"
echo "=================================================="
echo ""
echo "Deploy to KV260:"
echo "  ssh ubuntu@<kria-ip> 'mkdir -p /usr/share/vitis_ai_library/models/blip1_vit_b16'"
echo "  scp ${COMPILED_XMODEL} ubuntu@<kria-ip>:/usr/share/vitis_ai_library/models/blip1_vit_b16/"
echo ""
echo "Validate on KV260:"
echo "  xdputil query"
echo "  xdputil benchmark /usr/share/vitis_ai_library/models/blip1_vit_b16/blip1_vision.xmodel -i 1"
echo ""
echo "Then update config/kria_blip1.yaml:"
echo "  engine:"
echo "    type: kria"
echo "    xmodel_path: /usr/share/vitis_ai_library/models/blip1_vit_b16/blip1_vision.xmodel"
