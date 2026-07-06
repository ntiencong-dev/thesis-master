import os
import argparse
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def load_datasets(output_image_dir):
    from datasets import load_dataset
    from tqdm import tqdm
    
    # Token Hugging Face của bạn
    hf_token = ''
    
    os.makedirs(output_image_dir, exist_ok=True)
    
    print("Đang kết nối tới Hugging Face để tải dataset...")
    # Tải dataset dạng streaming
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True, use_auth_token=hf_token)
    
    # BƯỚC QUAN TRỌNG: Trộn ngẫu nhiên luồng dữ liệu bằng buffer_size lớn
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    print(f"Đang tải ngẫu nhiên 500 ảnh ĐA DẠNG từ ImageNet vào {output_image_dir}...")
    for i, sample in enumerate(tqdm(dataset, total=500)):
        if i >= 500:
            break
        
        image = sample["image"]
        label = sample["label"]
        
        # Đổi sang RGB nếu là ảnh đen trắng để đồng bộ cho các mô hình
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Lưu ảnh theo cấu trúc tên: img_idx_label_xxx.jpg
        image.save(os.path.join(output_image_dir, f"img_{i}_label_{label}.jpg"))

    print(f"Hoàn thành! Đã tải và lưu 500 ảnh vào thư mục: {output_image_dir}")

def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu Calibration cho VLM (CLIP & BLIP-1)")
    parser.add_argument("--refresh-data", action="store_true", help="Kích hoạt để tải lại 500 ảnh mới từ ImageNet")
    args = parser.parse_args()

    image_dir = "data/imagenet"
    out_dir = "exported_models"

    # Kiểm tra cờ --refresh-data
    if args.refresh_data:
        print("--- KÍCH HOẠT CHẾ ĐỘ TẢI LẠI DỮ LIỆU ---")
        load_datasets(image_dir)
    else:
        print("--- SỬ DỤNG DỮ LIỆU ĐÃ CÓ TRONG THƯ MỤC ---")

    os.makedirs(out_dir, exist_ok=True)

    print("Đang quét cấu trúc thư mục dữ liệu...")
    all_images = []

    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, f))

    if len(all_images) == 0:
        raise FileNotFoundError(f"Không tìm thấy ảnh hợp lệ trong {image_dir}. Hãy chạy lệnh với cờ --refresh-data")

    # Áp dụng chiến lược Lấy mẫu đa dạng (Diverse Sampling)
    random.seed(42)
    random.shuffle(all_images)

    # Lấy 200 ảnh cho Calibration (đủ tốt cho PTQ)
    num_calib = min(200, len(all_images))
    selected_images = all_images[:num_calib]
    print(f"Đã trộn ngẫu nhiên và chọn ra {num_calib} ảnh đa dạng cho Calibration.")

    # Transform cho CLIP (224x224)
    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # Transform cho BLIP-1 (384x384)
    blip_transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    clip_tensors = []
    blip_tensors = []

    print("Đang tiền xử lý ảnh sang ma trận Numpy...")
    for img_path in selected_images:
        try:
            img = Image.open(img_path).convert('RGB')
            clip_tensors.append(clip_transform(img).numpy())
            blip_tensors.append(blip_transform(img).numpy())
        except Exception as e:
            print(f"Bỏ qua ảnh lỗi: {img_path} - Lỗi: {e}")

    np.save(os.path.join(out_dir, "calibration_frames_clip.npy"), np.stack(clip_tensors))
    np.save(os.path.join(out_dir, "calibration_frames_blip1.npy"), np.stack(blip_tensors))
    print(f"Hoàn thành! Đã lưu ma trận đa dạng vào {out_dir}/")

if __name__ == "__main__":
    main()


# python3 scripts/prepare_calib_data.py --refresh-data