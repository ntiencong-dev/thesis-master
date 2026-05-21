"""One-time migration: copy all Faiss vectors → Qdrant nlvs_segments collection."""
import faiss
import pickle
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

INDEX_DIR = "index_store"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "nlvs_segments"
BATCH = 100

idx = faiss.read_index(f"{INDEX_DIR}/faiss.index")
with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
    meta = pickle.load(f)

n = idx.ntotal
print(f"Loaded: {n} vectors, {len(meta)} metadata entries")

raw = np.zeros((n, idx.d), dtype="float32")
for i in range(n):
    idx.reconstruct(i, raw[i])
print(f"Vectors shape: {raw.shape}, norm[0]={np.linalg.norm(raw[0]):.4f}")

client = QdrantClient(url=QDRANT_URL)

total_inserted = 0
for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    points = []
    for i, m in enumerate(meta[start:end]):
        points.append(PointStruct(
            id=start + i,
            vector=raw[start + i].tolist(),
            payload={
                "cam_id":     m.cam_id,
                "video_path": m.video_path,
                "start_time": float(getattr(m, "relative_start", getattr(m, "start_time", 0.0))),
                "end_time":   float(getattr(m, "relative_end",   getattr(m, "end_time",   0.0))),
                "capture_ts": float(getattr(m, "segment_wall_start", 0.0)),
            }
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    total_inserted += len(points)
    print(f"  Inserted {total_inserted}/{n}...")

info = client.get_collection(COLLECTION)
print(f"Qdrant vectors_count after migration: {info.vectors_count}")
print("Migration DONE")
