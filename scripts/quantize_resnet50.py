"""
Script: quantize_resnet50.py
Mục đích: Lượng tử hóa PTQ INT8 cho ResNet-50 bằng dữ liệu Calibration thực tế.
"""

import os
import argparse
from pathlib import Path
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_nndct.apis import torch_quantizer

from datasets import load_dataset
from tqdm import tqdm

# def load_datasets(output_image_dir):
#     hf_token =''
    
#     # Tạo thư mục nếu chưa tồn tại
#     os.makedirs(output_image_dir, exist_ok=True)
    
#     # Tải dataset validation từ Hugging Face
#     dataset = load_dataset("imagenet-1k", split="validation", streaming=True, use_auth_token=hf_token)

#     print(f"Đang tải 500 ảnh từ ImageNet vào {output_image_dir}...")
#     for i, sample in enumerate(tqdm(dataset, total=500)):
#         if i >= 500:
#             break
        
#         # Lấy ảnh và nhãn (label)
#         image = sample["image"]
#         label = sample["label"]
        
#         # Đổi sang RGB nếu là ảnh đen trắng (L) để đồng bộ cho ResNet-50
#         if image.mode != "RGB":
#             image = image.convert("RGB")
            
#         # Lưu ảnh theo cấu trúc tên: idx_label.jpg
#         image.save(os.path.join(output_image_dir, f"img_{i}_{label}.jpg"))

#     print(f"Hoàn thành! Đã lưu 500 ảnh vào thư mục: {output_image_dir}")

# ---------------------------------------------------------------------------
# 1. Xây dựng Class Dataset để đọc ảnh từ thư mục phẳng
# ---------------------------------------------------------------------------
class CalibrationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Quét toàn bộ file ảnh (png, jpg, jpeg) trong thư mục được chỉ định.
        """
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Bắt buộc chuyển sang RGB để tránh lỗi với ảnh thang độ xám (Grayscale)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# ---------------------------------------------------------------------------
# 2. Quy trình Lượng tử hóa PTQ
# ---------------------------------------------------------------------------
def quantize_resnet50(quant_mode, output_dir, calib_dir, batch_size=1):
    print("[*] Đang tải mô hình ResNet-50 (FP32)...")
    model = models.resnet50(pretrained=True)
    model.eval().cpu()

    # --- TIỀN XỬ LÝ ẢNH CHUẨN IMAGENET ---
    # ResNet-50 yêu cầu: Resize(256) -> CenterCrop(224) -> Normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    print(f"[*] Đang tải dữ liệu hiệu chuẩn từ: {calib_dir}")
    calib_dataset = CalibrationDataset(calib_dir, transform=transform)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)

    if len(calib_dataset) == 0:
        raise ValueError(f"Không tìm thấy ảnh nào trong {calib_dir}!")

    # Tensor giả chỉ dùng để định hình cấu trúc đồ thị (Graph Tracing)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[*] Khởi tạo torch_quantizer ở chế độ: {quant_mode}")
    quantizer = torch_quantizer(
        quant_mode=quant_mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=output_dir,
    )
    
    quant_model = quantizer.quant_model

    # --- CHẠY SUY LUẬN ---
    with torch.no_grad():
        if quant_mode == "calib":
            print(f"[*] Bắt đầu Hiệu chuẩn với {len(calib_dataset)} ảnh...")
            
            # Đẩy ảnh thực tế vào mô hình thay vì torch.randn
            for idx, images in enumerate(calib_loader):
                quant_model(images)
                
                # In tiến độ cho dễ theo dõi
                if (idx + 1) % 50 == 0:
                    print(f"    Đã xử lý {idx + 1}/{len(calib_loader)} batch.")
                    
            print("[*] Hoàn thành quá trình chạy ảnh. Đang tính toán và xuất cấu hình lượng tử...")
            quantizer.export_quant_config()
            
        elif quant_mode == "test":
            print("[*] Bắt đầu quá trình Test và xuất Xmodel...")
            # Quá trình test/export chỉ cần dummy input để lấy graph architecture
            quant_model(dummy_input)
            quantizer.export_xmodel(deploy_check=True)
            print(f"[*] Hoàn thành! File Xmodel đã được lưu tại: {output_dir}")

# ---------------------------------------------------------------------------
# 3. Điểm nạp (Entry Point)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["calib", "export", "all"], default="all")
    parser.add_argument("--output_dir", type=str, default="quantized_resnet50")
    parser.add_argument("--calib_dir", type=str, default="data/imagenet")
    args = parser.parse_args()

    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)

    if args.step in ["calib", "all"]:
        # Truyền args.calib_dir vào hàm tải dữ liệu để giải quyết lỗi NameError
        quantize_resnet50(quant_mode="calib", output_dir=args.output_dir, calib_dir=args.calib_dir)
        
    if args.step in ["export", "all"]:
        quantize_resnet50(quant_mode="test", output_dir=args.output_dir, calib_dir=args.calib_dir)