import argparse
import time
import numpy as np
import yaml
from pathlib import Path

# Provide a mock dataset for benchmarking
MOCK_QUERIES = [
    "a person walking near the entrance",
    "security camera footage of an empty corridor",
    "a man carrying a bag",
    "red car driving past the gate"
]

def generate_mock_frames(num_frames: int, h=224, w=224):
    """Generates random BGR frames to simulate video segments."""
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(num_frames)]

def main():
    parser = argparse.ArgumentParser(description="Kria KV260 Hardware Latency Benchmark")
    parser.add_argument("--config", type=str, default="config/kria_clip_blip1.yaml")
    parser.add_argument("--num-frames", type=int, default=100)
    args = parser.parse_args()

    print(f"==================================================")
    print(f"  Kria KV260 Pipeline Benchmark")
    print(f"==================================================")
    print(f"  Config: {args.config}")
    print(f"  Frames: {args.num_frames}")
    
    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 1. Initialize Engine
    print("\n[1/3] Initializing KriaEngine (Loading DPU & AWQ models)...")
    t0 = time.time()
    from src.engines.factory import create_engine
    engine = create_engine(config)
    t1 = time.time()
    print(f"      Engine loaded in {t1 - t0:.2f} seconds.")

    # 2. Benchmark Visual Encoder (DPU)
    print(f"\n[2/3] Benchmarking Stage-1 Visual Encoder (DPU B4096)...")
    
    # We must determine the correct input size
    is_blip = config["engine"]["type"] == "blip1" or (
        config["engine"]["type"] == "kria" and "blip1" in config["engine"].get("xmodel_path", "")
    )
    frame_size = 384 if is_blip else 224
    
    frames = generate_mock_frames(args.num_frames, h=frame_size, w=frame_size)
    
    # Warmup
    _ = engine.encode_frames(frames[:1])
    
    # Benchmark
    t0 = time.time()
    _ = engine.encode_frames(frames)
    t1 = time.time()
    
    total_time = t1 - t0
    fps = args.num_frames / total_time
    ms_per_frame = (total_time / args.num_frames) * 1000
    
    print(f"      Total Time : {total_time:.2f} s")
    print(f"      Throughput : {fps:.2f} FPS")
    print(f"      Latency    : {ms_per_frame:.2f} ms / frame")

    # 3. Benchmark Text Encoder (CPU AWQ)
    print(f"\n[3/3] Benchmarking Stage-1 Text Encoder (ARM Cortex-A53 AWQ)...")
    
    # Warmup
    _ = engine.encode_text([MOCK_QUERIES[0]])
    
    # Benchmark
    t0 = time.time()
    _ = engine.encode_text(MOCK_QUERIES)
    t1 = time.time()
    
    total_time = t1 - t0
    ms_per_query = (total_time / len(MOCK_QUERIES)) * 1000
    
    print(f"      Total Time : {total_time:.2f} s")
    print(f"      Latency    : {ms_per_query:.2f} ms / query")

    # 4. Benchmark Stage-2 Reranker (ITM Cross-Attention on ARM CPU)
    if hasattr(engine, "score_itm"):
        print(f"\n[4/4] Benchmarking Stage-2 ITM Reranker (ARM Cortex-A53 AWQ)...")
        # We need to simulate the pipeline: extracting 5 uniform frames from a segment
        itm_frames = generate_mock_frames(5, h=384, w=384)
        
        # Warmup
        _ = engine.score_itm(itm_frames, MOCK_QUERIES[0])
        
        # Benchmark
        t0 = time.time()
        for query in MOCK_QUERIES:
            _ = engine.score_itm(itm_frames, query)
        t1 = time.time()
        
        total_time = t1 - t0
        ms_per_itm = (total_time / len(MOCK_QUERIES)) * 1000
        
        print(f"      Total Time : {total_time:.2f} s")
        print(f"      Latency    : {ms_per_itm:.2f} ms / segment-query pair")
    else:
        print("\n[!] Stage-2 ITM Benchmarking skipped (Not a BLIP-1 or Hybrid engine).")
    
    print(f"\n==================================================")
    print(f"  Benchmark Complete.")
    print(f"==================================================")

if __name__ == "__main__":
    main()
