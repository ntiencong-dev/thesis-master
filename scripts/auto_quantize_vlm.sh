#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào thất bại
set -e

echo "======================================================="
echo " BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG HÓA VLM CHO KRIA KV260"
echo "======================================================="

echo "[1/5] Xử lý xung đột phiên bản và cài đặt thư viện..."
pip install safetensors==0.3.1 -q
pip install open-clip-torch==2.20.0 -q
pip install transformers -q

echo "[2/5] Khởi tạo script và chuẩn bị dữ liệu Calibration từ ImageNet..."
python3 scripts/prepare_calib_data.py

# echo "[3/5] Lượng tử hóa mô hình CLIP RN50 (Tách luồng tránh tràn RAM)..."
python3 scripts/quantize_clip_rn50.py --step calib --output-dir quantized/rn50
python3 scripts/quantize_clip_rn50.py --step export --output-dir quantized/rn50

echo "[4/5] Lượng tử hóa mô hình BLIP-1 (Tách luồng tránh tràn RAM)..."
python3 scripts/quantize_blip1.py --step calib --output-dir quantized/blip1
python3 scripts/quantize_blip1.py --step export --output-dir quantized/blip1

echo "[5/5] Biên dịch đồ thị XIR sang mã máy DPU B4096..."
echo '{"target": "DPUCZDX8G_ISA1_B4096"}' > kv260_arch.json
mkdir -p compiled_vlm

# Trích xuất động tên file xmodel do trình biên dịch sinh ra để tránh sai sót tên class
CLIP_XMODEL=$(find quantized/rn50 -name "*.xmodel" | head -n 1)
vai_c_xir \
  --xmodel "$CLIP_XMODEL" \
  --arch kv260_arch.json \
  --output_dir compiled_vlm \
  --net_name clip_rn50_kv260

BLIP_XMODEL=$(find quantized/blip1 -name "*.xmodel" | head -n 1)
vai_c_xir \
  --xmodel "$BLIP_XMODEL" \
  --arch kv260_arch.json \
  --output_dir compiled_vlm \
  --net_name blip1_kv260

echo "======================================================="
echo " 🎉 HOÀN THÀNH TỰ ĐỘNG HÓA! CÁC TỆP ĐÃ BIÊN DỊCH:"
ls -lh compiled_vlm/
echo "======================================================="