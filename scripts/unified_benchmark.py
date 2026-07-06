"""
Script: unified_benchmark.py
Benchmark tối đa hiệu năng ResNet-50 trên cả Laptop (GPU) và Kria KV260 (DPU).
"""

import os
import time
import argparse
import numpy as np
import cv2
import threading

# ---------------------------------------------------------------------------
# Tiền xử lý ảnh chung (Pure OpenCV/NumPy)
# ---------------------------------------------------------------------------
def preprocess_image(image_path, backend):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = 256, int(w * 256 / h)
    else:
        new_h, new_w = int(h * 256 / w), 256
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    start_y = (new_h - 224) // 2
    start_x = (new_w - 224) // 2
    img = img[start_y:start_y+224, start_x:start_x+224]
    
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    if backend == "pytorch":
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
    elif backend == "dpu":
        img = np.expand_dims(img, axis=0)
        
    return img

# ---------------------------------------------------------------------------
# Đánh giá trên Laptop (NVIDIA GPU / CPU)
# ---------------------------------------------------------------------------
def run_pytorch_benchmark(image_dir):
    # Import cục bộ để tránh lỗi trên board Kria
    import torch
    import torchvision.models as models

    print("[INFO] Khởi tạo môi trường PyTorch...")
    
    # 1. Tối ưu hóa cho phần cứng Laptop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Đã nhận diện GPU: {torch.cuda.get_device_name(0)}. Bật tối ưu hóa cuDNN...")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARNING] Không tìm thấy GPU, hệ thống sẽ chạy trên CPU.")

    model = models.resnet50(pretrained=True)
    model.eval().to(device)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    correct_top1 = 0
    total_time = 0.0
    
    # Khởi động GPU (Warm-up) để đo thời gian chính xác hơn
    dummy_tensor = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        _ = model(dummy_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"[INFO] Bắt đầu benchmark với {len(image_files)} ảnh...")
    with torch.no_grad():
        for filename in image_files:
            true_label = int(filename.split('.')[0].split('_')[-1])
            img_path = os.path.join(image_dir, filename)
            
            input_data = preprocess_image(img_path, backend="pytorch")
            input_tensor = torch.from_numpy(input_data).to(device)
            
            # Sử dụng mixed-precision (FP16) để tận dụng lõi Tensor Core trên GTX 1650 Ti
            with torch.amp.autocast(device_type=device.type):
                start_t = time.time()
                output = model(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_t = time.time()
            
            total_time += (end_t - start_t)
            
            pred_label = output.argmax(dim=1).item()
            if pred_label == true_label:
                correct_top1 += 1
                
    fps = len(image_files) / total_time
    accuracy = (correct_top1 / len(image_files)) * 100
    
    print("=" * 45)
    print(f"✅ KẾT QUẢ PYTORCH (LAPTOP - {device.type.upper()})")
    print(f"  - Tốc độ (FPS)   : {fps:.2f} frames/sec")
    print(f"  - Độ chính xác   : {accuracy:.2f}% (Top-1)")
    print("=" * 45)

# ---------------------------------------------------------------------------
# Luồng xử lý Worker cho DPU (Kria KV260)
# ---------------------------------------------------------------------------
def dpu_worker(runner, image_chunk, image_dir, input_scale, result_dict, thread_id):
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    out_shape = tuple(output_tensors[0].dims)
    
    correct = 0
    start_t = time.time()
    
    for filename in image_chunk:
        true_label = int(filename.split('.')[0].split('_')[-1])
        img_path = os.path.join(image_dir, filename)
        
        # Tiền xử lý
        input_data = preprocess_image(img_path, backend="dpu")
        input_data = (input_data * input_scale).astype(np.int8)
        output_data = np.empty(out_shape, dtype=np.int8, order="C")
        
        # Đẩy xuống DPU
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)
        
        pred_label = np.argmax(output_data[0])
        if pred_label == true_label:
            correct += 1
            
    end_t = time.time()
    
    # Ghi nhận kết quả của Thread
    result_dict[thread_id] = {
        "time": end_t - start_t,
        "correct": correct,
        "processed": len(image_chunk)
    }

# ---------------------------------------------------------------------------
# Đánh giá trên Board Kria KV260 (Đa luồng DPU)
# ---------------------------------------------------------------------------
def run_dpu_benchmark(image_dir, xmodel_path, num_threads):
    # Import cục bộ
    import xir
    import vart

    print(f"[INFO] Khởi tạo VART API cho DPU với {num_threads} luồng...")
    
    graph = xir.Graph.deserialize(xmodel_path)
    root_subgraph = graph.get_root_subgraph()

    subgraphs = root_subgraph.get_children()
    dpu_subgraphs = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    
    # Tạo pool các Runner cho từng luồng
    runners = [vart.Runner.create_runner(dpu_subgraphs[0], "run") for _ in range(num_threads)]
    
    input_fixpos = runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Chia nhỏ lượng ảnh cho các Thread
    chunk_size = len(image_files) // num_threads
    chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]
    
    threads = []
    result_dict = {}
    
    print(f"[INFO] Bắt đầu benchmark song song trên {num_threads} Thread(s)...")
    
    start_total = time.time()
    
    for i in range(num_threads):
        # Đảm bảo luồng cuối cùng xử lý các ảnh bị lẻ
        chunk = chunks[i] if i < num_threads - 1 else image_files[i * chunk_size:]
        
        t = threading.Thread(
            target=dpu_worker, 
            args=(runners[i], chunk, image_dir, input_scale, result_dict, i)
        )
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end_total = time.time()
    total_time_multi_thread = end_total - start_total
    
    # Tổng hợp dữ liệu
    total_correct = sum([res["correct"] for res in result_dict.values()])
    total_processed = sum([res["processed"] for res in result_dict.values()])
    
    fps = total_processed / total_time_multi_thread
    accuracy = (total_correct / total_processed) * 100
    
    print("=" * 45)
    print(f"✅ KẾT QUẢ DPU INT8 (KRIA KV260 - {num_threads} THREADS)")
    print(f"  - Tổng số ảnh đã xử lý : {total_processed}")
    print(f"  - Tốc độ đỉnh (FPS)    : {fps:.2f} frames/sec")
    print(f"  - Độ chính xác         : {accuracy:.2f}% (Top-1)")
    print("=" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Benchmark cho ResNet-50")
    parser.add_argument("--backend", choices=["pytorch", "dpu"], required=True, help="Môi trường chạy: 'pytorch' (Laptop) hoặc 'dpu' (KV260)")
    parser.add_argument("--image_dir", type=str, required=True, help="Thư mục chứa ảnh benchmark")
    parser.add_argument("--xmodel", type=str, default="", help="Đường dẫn Xmodel (Dùng cho backend DPU)")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng DPU Runner (Mặc định: 4)")
    
    args = parser.parse_args()

    if args.backend == "pytorch":
        run_pytorch_benchmark(args.image_dir)
    elif args.backend == "dpu":
        if not args.xmodel:
            raise ValueError("[ERROR] Cần cung cấp đường dẫn --xmodel khi chạy trên DPU!")
        run_dpu_benchmark(args.image_dir, args.xmodel, args.threads)