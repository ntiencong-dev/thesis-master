# NLVS Project: Master Blueprint for Kria KV260 Deployment
**Target Device:** AMD Xilinx Kria KV260 (4GB LPDDR4, DPUCZDX8G_ISA1_B4096)
**Objective:** Deploy a Real-time Text-to-Video Retrieval System using Vision-Language Models (VLMs).
**Deadline Constraint:** July 15, 2026 (Prepare system metrics for academic paper submission).

## 1. System Architecture: Dual-Model Hybrid Quantization (Option C)
The system will implement a two-stage retrieval pipeline using both CLIP and BLIP-1, heavily optimized for the memory and computing constraints of the KV260 board.

*   **Stage-1 (Broad Retrieval):** CLIP Visual Encoder (512d) indexes frames into a local Vector DB. Retrieves top-K candidates.
*   **Stage-2 (Fine-grained Reranking):** BLIP-1 ITM (Image-Text Matching) uses BERT cross-attention to rerank the top-K candidates.

### 1.1. Hardware-Aware Split-Quantization Strategy
To fit within the 4GB RAM limit and optimize latency:
*   **Visual Encoders (CLIP & BLIP-1):** Deployed on DPU B4096 using **PTQ INT8** (via `vai_q_pytorch`).
*   **Text Encoders (CLIP Text & BLIP-1 BERT):** Deployed on ARM Cortex-A53 CPU using **AWQ INT4** (via `AutoAWQ`). This reduces the text model memory footprint significantly (e.g., CLIP text from ~250MB to ~63MB, BLIP-1 BERT to ~110MB). The total system memory is reduced from ~1.37 GB to ~494 MB[cite: 2].

### 1.2. Memory Management (MACRO)
*   **MACRO (Persistent DPU Runners):** Maintain both CLIP and BLIP-1 DPU runners actively in DDR memory to eliminate the ~150-300ms overhead of loading/unloading models during the pipeline execution[cite: 2].

### 1.3. Vector Database
*   **Qdrant (Local/Embedded Mode):** Use `qdrant-client` with `path="./local_qdrant_db"` to run directly within the Python process. **Do not** use Qdrant Docker Server to prevent Out-Of-Memory (OOM) crashes.

---

## 2. Agent Execution Tasks

### Task 1: Fix DPU Compilation Errors (`[XIR_VALUE_UNMATCH]`)
When exporting the PyTorch model for `vai_q_pytorch` and compiling with `vai_c_xir`, the graph must be **absolutely static**. Dynamic shapes (`.size()`, `.expand()`, `.repeat()`) and memory discontinuity cause XIR compiler crashes.

**Agent Actions for Visual Encoders (Transformer/Attention blocks):**
1.  **Hardcode Batch Size:** Force `BATCH_SIZE = 1` for calibration and export.
2.  **Enforce Memory Contiguity:** Always apply `.contiguous()` after `.transpose()` or `.permute()` before reshaping.
3.  **Static Shape Mapping:** Use `.view(1, seq_len, dim)` instead of dynamic `.reshape()`. Use `.unsqueeze(0)` for positional embeddings instead of relying on implicit broadcasting.

*Reference Code Pattern for Agent:*
```python
# DO NOT USE: x = x.reshape(x.size(0), -1)
# DO NOT USE: cls_token.expand_as(...)

# USE STATIC SHAPES & CONTIGUOUS MEMORY:
B, T, C = 1, 197, 768
x = x.flatten(2).transpose(1, 2).contiguous() 
cls_token = self.class_embedding.view(1, 1, 768).to(x.dtype)
x = torch.cat([cls_token, x], dim=1) 
pos_emb = self.positional_embedding.unsqueeze(0).to(x.dtype)
x = x + pos_emb