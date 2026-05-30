# ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH
## TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN

---

# CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
### *Độc lập – Tự do – Hạnh phúc*

---

# PHIẾU ĐĂNG KÝ ĐỀ TÀI LUẬN VĂN THẠC SĨ

---

**Tên đề tài tiếng Việt:**

> NGHIÊN CỨU TỐI ƯU HÓA VÀ TRIỂN KHAI MÔ HÌNH ĐA PHƯƠNG THỨC TRÊN NỀN TẢNG NHÚNG AMD KRIA KV260 CHO ỨNG DỤNG TÌM KIẾM VIDEO BẰNG NGÔN NGỮ TỰ NHIÊN

**Tên đề tài tiếng Anh:**

> RESEARCH ON OPTIMIZATION AND DEPLOYMENT OF VISION-LANGUAGE MODELS ON AMD KRIA KV260 EMBEDDED PLATFORM FOR NATURAL LANGUAGE VIDEO SEARCH APPLICATIONS

---

**Hướng đề tài:** ☑ Hướng ứng dụng (12 TC)

**Ngành học:** ☑ Khoa học máy tính — Mã ngành: **8480101**

---

| | **Cán bộ hướng dẫn 1** |
|---|---|
| **Họ tên** | ĐỖ TRÍ NHỰT |
| **Email** | trinhutdo@uit.edu.vn |
| **Điện thoại** | 0938 113 898 |
| **Đơn vị công tác** | Trường Đại học Công nghệ Thông tin, ĐHQG-HCM |

**Thời gian thực hiện:** 6 tháng — từ tháng 03/2026 đến tháng 09/2026.

| | **Học viên thực hiện** |
|---|---|
| **Họ tên** | *(Học viên điền)* |
| **Mã số** | *(Học viên điền)* |
| **Khóa** | *(Học viên điền)* |
| **Email** | *(Học viên điền)* |

---

# ĐỀ CƯƠNG ĐỀ TÀI LUẬN VĂN THẠC SĨ

---

## 1. NỘI DUNG ĐỀ CƯƠNG

---

### 1.1 Tổng quan đề tài

#### 1.1.1 Bối cảnh và tính cấp thiết

Sự bùng nổ của hệ thống camera giám sát an ninh (CCTV) toàn cầu — với hơn 1 tỷ thiết bị đang hoạt động theo IHS Markit (2023) và dự kiến vượt 1.4 tỷ vào năm 2028 — đặt ra nhu cầu cấp thiết về khả năng truy xuất thông minh dữ liệu video. Một hệ thống giám sát quy mô trung bình (100 camera, hoạt động 24/7) tạo ra xấp xỉ 2.4 TB dữ liệu video mỗi ngày. Quy trình xem lại thủ công hiện tại không chỉ tốn kém thời gian và nhân lực, mà còn tiềm ẩn nguy cơ bỏ sót do mệt mỏi của người vận hành.

Lĩnh vực **Tìm kiếm Video bằng Ngôn ngữ Tự nhiên** (Natural Language Video Search — NLVS) giải quyết vấn đề này bằng cách cho phép truy vấn dạng văn bản tự do như *"người mặc áo khoác đen đi lạng lách qua cổng"* và nhận về các đoạn video phù hợp nhất được xếp hạng theo độ tương đồng ngữ nghĩa. Nền tảng kỹ thuật của hướng tiếp cận này là các mô hình **Vision-Language Model (VLM)** — đặc biệt là CLIP (Radford et al., 2021) và các biến thể EVA-CLIP (Sun et al., 2023), X-CLIP (Ni et al., 2022) — với khả năng ánh xạ ảnh và văn bản vào cùng một không gian biểu diễn ngữ nghĩa thông qua huấn luyện đối nghịch trên hàng tỷ cặp dữ liệu.

Tuy nhiên, các mô hình VLM hiện tại đặt ra yêu cầu tính toán cao: EVA-CLIP ViT-L/14 cần ~1.4 GB VRAM; một máy chủ GPU A100 SXM tiêu thụ ~400W và có giá ~10.000 USD. Mô hình xử lý tập trung (server-centric) này dẫn đến: (i) chi phí hạ tầng cao, không phù hợp cho triển khai quy mô lớn; (ii) độ trễ mạng khi truyền dữ liệu video từ camera về máy chủ; (iii) rủi ro về quyền riêng tư — toàn bộ dữ liệu video nhạy cảm phải rời khỏi vị trí camera.

Xu hướng tất yếu là dịch chuyển năng lực suy luận AI về phía **biên mạng** (*edge computing*), đặt trực tiếp tại điểm thu thập dữ liệu. Nền tảng nhúng **AMD Kria KV260** — tích hợp ARM Cortex-A53, FPGA Zynq UltraScale+ MPSoC và đơn vị xử lý học sâu DPU B4096 (~1.4 TOPS INT8, 300 MHz) — cung cấp ~56 GOPS/W hiệu suất năng lượng, gấp ~3.7 lần so với GPU rời thông thường, với mức tiêu thụ chỉ 15–25W và giá thành Starter Kit ~249 USD. Đây là nền tảng hứa hẹn để triển khai VLM cho hệ thống NLVS tại biên mà không cần kết nối máy chủ GPU.

Tuy nhiên, việc triển khai VLM trên DPU FPGA đặt ra các thách thức kỹ thuật đặc thù chưa được giải quyết đầy đủ trong tài liệu khoa học hiện có. Kiến trúc **Vision Transformer (ViT)** — backbone của các VLM hiện đại — chứa nhiều tầng (*LayerNormalization*, *Softmax trong MultiHeadAttention*, *GELU activation*) không được DPU B4096 hỗ trợ phần cứng, buộc phải fallback sang ARM CPU. Kết quả là **đồ thị thực thi hỗn hợp** (mixed-execution graph) với overhead DMA (Direct Memory Access) khi chuyển tensor giữa CPU (PS) và DPU (PL) sau mỗi Transformer block. Với 12 blocks của CLIP ViT-B/16, ước tính 60 lần DMA transfer mỗi forward pass — đây là bottleneck tính toán chính chưa có giải pháp tối ưu được công bố cho KV260.

**Đề tài luận văn** tập trung nghiên cứu và giải quyết vấn đề này: xây dựng quy trình **lượng tử hóa sau huấn luyện** (Post-Training Quantization — PTQ) INT8 cho mô hình CLIP ViT-B/16 bằng Vitis AI 3.5, biên dịch sang định dạng `.xmodel` cho DPU B4096, tích hợp với pipeline xử lý video phần cứng VVAS trên Kria KV260, và đánh giá định lượng sự đánh đổi giữa chất lượng truy xuất — hiệu năng tính toán — tiêu thụ năng lượng. Tính khả thi kỹ thuật của toàn bộ pipeline NLVS đã được kiểm chứng qua một hệ thống prototype chạy trên PC (154 test cases, 100% pass rate), tạo nền tảng vững chắc để chuyển sang giai đoạn nghiên cứu triển khai biên.

#### 1.1.2 Tổng quan tài liệu

**a) Vision-Language Models và bài toán triển khai trên thiết bị biên**

CLIP (Radford et al., 2021) thiết lập kiến trúc *dual-encoder* học biểu diễn chung cho ảnh và văn bản theo mục tiêu InfoNCE:

$$\mathcal{L}_{CLIP} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{\langle v_i,t_i\rangle/\tau}}{\sum_{j=1}^{N}e^{\langle v_i,t_j\rangle/\tau}} + \log\frac{e^{\langle v_i,t_i\rangle/\tau}}{\sum_{j=1}^{N}e^{\langle v_j,t_i\rangle/\tau}}\right]$$

Sau L2-normalization, tìm kiếm cosine similarity tương đương inner product trên unit hypersphere, cho phép lập chỉ mục và tìm kiếm hiệu quả bằng ANN index (HNSW). EVA-CLIP ViT-L/14 (Sun et al., 2023) đạt ImageNet zero-shot accuracy 79.8% và là baseline chính trong hệ thống prototype PC. Tuy nhiên, trọng tâm của nghiên cứu này không nằm ở việc cải tiến chất lượng VLM, mà ở câu hỏi: **làm thế nào để VLM vận hành được trong giới hạn phần cứng của thiết bị biên?**

Các nghiên cứu VLM nhỏ gọn như MobileVLM (Chu et al., 2023) và TinyVLM chủ yếu nhắm vào mobile CPU (ARM Cortex-A78) chứ không phải FPGA DPU. Nghiên cứu triển khai ViT trên FPGA (Lu et al., 2021; Li et al., 2023) chủ yếu nhắm vào Xilinx Alveo data-center với tài nguyên LUT/DSP lớn hơn nhiều KV260, và không giải quyết bài toán mixed-execution graph đặc thù cho DPU B4096.

**b) Lượng tử hóa mạng nơ-ron: PTQ cho Vision Transformer**

Lượng tử hóa tuyến tính uniform INT8 ánh xạ giá trị thực $x$ sang số nguyên 8-bit:

$$x_q = \text{clip}\!\left(\left\lfloor\frac{x}{s}\right\rceil + z,\; -128,\; 127\right), \quad s = \frac{x_{max} - x_{min}}{255}$$

**PTQ** (Post-Training Quantization) sử dụng calibration set nhỏ (~1000 mẫu) để ước lượng phân phối kích hoạt mà không cần training lại. So với QAT, PTQ nhanh hơn nhưng có thể suy giảm chất lượng nhiều hơn với các kiến trúc phi tuyến như ViT.

Thách thức đặc thù khi lượng tử hóa ViT: (i) *LayerNormalization* — phép tính $\frac{x-\mu}{\sigma+\epsilon}$ yêu cầu độ chính xác cao cho $\mu$, $\sigma$; (ii) *Softmax* trong MHA — phân phối activations biến thiên lớn theo độ dài sequence; (iii) *GELU* — hàm phi tuyến phức tạp, khó xấp xỉ bằng INT8. Bondarenko et al. (2021) và SmoothQuant (Xiao et al., 2023) chỉ ra rằng *per-channel* quantization kết hợp với smooth activation migration cải thiện đáng kể chất lượng PTQ cho Transformer.

Với DPU B4096, các tầng không được hỗ trợ (LayerNorm, Softmax, GELU) tự động phân vào ARM CPU. Mỗi lần chuyển tiếp CPU↔DPU tốn overhead DMA và synchronization. Gschwend (2020) ước tính overhead này chiếm 15–40% tổng thời gian suy luận tùy theo kích thước tensor.

**c) Kiến trúc DPU B4096 và FPGA-based AI acceleration**

DPU B4096 (AMD Xilinx, PG338) là hardened IP core tối ưu cho CNN workload: Conv2D, DepthwiseConv, BatchNorm, ReLU/ReLU6, MaxPool, AvgPool, Concat, Add. Kiến trúc tính toán: systolic array MAC INT8, output buffer FP32 (accumulation) → INT8 (output quantization). Băng thông bộ nhớ: AXI HP ports kết nối LPDDR4 ~19.2 GB/s.

Công cụ `vaitrace` (Vitis AI 3.5) cho phép profiling từng subgraph DPU/CPU, đo chính xác thời gian thực thi và overhead DMA — đây là công cụ quan trọng để định lượng bottleneck và đánh giá hiệu quả các chiến lược tối ưu trong phạm vi nghiên cứu này.

**d) Tìm kiếm vector xấp xỉ với HNSW và Qdrant**

Sau khi encode segment video thành embedding vector, bài toán tìm kiếm trở thành *Approximate Nearest Neighbor (ANN) search*. HNSW (Malkov & Yashunin, 2020) xây dựng đồ thị phân cấp với độ phức tạp tìm kiếm $O(\log N)$ và recall@10 đạt ~99% trong benchmark. Qdrant (Rust, v1.9+) triển khai HNSW kết hợp payload filtering và persistence, phù hợp cho triển khai nhúng với bộ nhớ eMMC giới hạn.

**e) Khoảng trống nghiên cứu**

Tổng hợp tài liệu xác nhận khoảng trống rõ ràng: chưa có công trình nào công bố quy trình đầy đủ triển khai CLIP ViT-B/16 trên DPU B4096 của Kria KV260, phân tích chi tiết mixed-execution graph, và đánh giá định lượng trade-off chất lượng–hiệu năng–năng lượng cho ứng dụng NLVS end-to-end. Luận văn này nhắm trực tiếp vào khoảng trống đó.

#### 1.1.3 Phạm vi và giới hạn nghiên cứu

**Phạm vi bao gồm:**

- Quy trình PTQ INT8 và biên dịch `.xmodel` cho CLIP ViT-B/16 trên Vitis AI 3.5.
- Tích hợp inference DPU B4096 với VART runtime và pipeline VVAS trên Kria KV260.
- Đánh giá định lượng: embedding quality (cosine similarity FP32 vs. INT8), retrieval quality (Recall@K trên DiDeMo/Charades-STA), encode throughput (FPS), query latency (ms), tiêu thụ điện năng (W).
- Hệ thống prototype PC (EVA-CLIP ViT-L/14 + Qdrant) đóng vai trò nền tảng kỹ thuật đã kiểm chứng và baseline so sánh.

**Phạm vi không bao gồm:**

- Training hoặc fine-tuning mô hình từ đầu; QAT (Quantization-Aware Training).
- Triển khai đa camera phân tán với quản lý tập trung.
- Mô hình video-language phức tạp (InternVideo2, X-CLIP) trên Kria — chỉ CLIP ViT-B/16 phù hợp với giới hạn tài nguyên DPU.

---
### 1.2 Mục tiêu nghiên cứu

Luận văn đặt ra **mục tiêu tổng quát**: Xây dựng và đánh giá một hệ thống NLVS có khả năng vận hành hoàn toàn trên nền tảng nhúng AMD Kria KV260, sử dụng mô hình CLIP ViT-B/16 được lượng tử hóa INT8 và biên dịch cho DPU B4096, đáp ứng yêu cầu triển khai thực tế về hiệu năng, chất lượng và tiêu thụ năng lượng.

Các **mục tiêu cụ thể** bao gồm:

| Mã | Mục tiêu | Chỉ tiêu đánh giá |
|---|---|---|
| **MT-01** | Xây dựng quy trình PTQ INT8 cho CLIP ViT-B/16 với Vitis AI 3.5 | Quy trình được tài liệu hóa, `.xmodel` biên dịch thành công cho DPU B4096 |
| **MT-02** | Phân tích và tối ưu hóa mixed-execution graph CPU↔DPU | Giảm số lần DMA transfer; profiling vaitrace định lượng overhead |
| **MT-03** | Triển khai inference DPU B4096 tích hợp pipeline VVAS trên Kria KV260 | Hệ thống end-to-end hoạt động, encode throughput ≥ 2 FPS |
| **MT-04** | Đánh giá chất lượng embedding sau lượng tử hóa | Cosine similarity INT8 vs. FP32 ≥ 0.95; Recall@10 suy giảm ≤ 5% trên tập benchmark |
| **MT-05** | Đánh giá hiệu năng và năng lượng Kria so với baseline PC | Đo throughput (FPS), latency (ms/query), power (W), GOPS/W; kết quả định lượng có thể tái tạo |
| **MT-06** | Xác định giới hạn và hướng mở rộng | Danh sách bottleneck kỹ thuật còn tồn tại và đề xuất cải tiến tiếp theo |

---

### 1.3 Nội dung nghiên cứu

Luận văn được tổ chức thành **5 nội dung nghiên cứu** (NC) có quan hệ kế thừa:

---

#### NC1: Nghiên cứu cơ sở lý thuyết

**Mục đích:** Xây dựng nền tảng lý luận phục vụ cho NC3, NC4, NC5.

**Nội dung:**

*a) Vision Transformer và CLIP Architecture*

ViT (Dosovitskiy et al., 2021) phân chia ảnh đầu vào $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ thành $N = \frac{HW}{P^2}$ patches, mỗi patch được project tuyến tính thành token embedding, thêm positional encoding và xử lý qua $L$ lớp Transformer. CLIP ViT-B/16 (H=W=224, P=16, L=12, D=512) tạo ra visual embedding 512 chiều; CLIP Text Encoder (Transformer 12 lớp) tạo text embedding 512 chiều cùng chiều.

Hàm tương đồng tìm kiếm: $\text{sim}(q, v) = \frac{\mathbf{q}^T \mathbf{v}}{|\mathbf{q}||\mathbf{v}|}$, trong đó $\mathbf{q}$ là text query embedding và $\mathbf{v}$ là video segment embedding. Kết quả tìm kiếm được xếp hạng giảm dần theo $\text{sim}$.

*b) Quantization lý thuyết và thực tiễn trên DPU*

Phân tích sâu ba loại phép tính không được DPU B4096 hỗ trợ:

- **LayerNorm**: $y = \frac{x - E[x]}{\sqrt{\text{Var}[x] + \epsilon}} \cdot \gamma + \beta$ — yêu cầu tính mean và variance động theo sequence, không thể biểu diễn bằng fixed-point arithmetic của DPU.
- **Softmax**: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ — phép tính $e^x$ không có hardware support trong DPU.
- **GELU**: $\text{GELU}(x) = x \cdot \Phi(x)$ — hàm CDF Gaussian phức tạp.

Khi `vai_q_pytorch` phân tích đồ thị PyTorch, các tầng này được tách thành subgraph riêng chạy trên CPU. Kết quả: mỗi Transformer block tạo ra ít nhất 2 lần context switch CPU↔DPU (trước và sau LayerNorm). Với L=12 blocks, tổng: ~24–60 DMA transfers/forward pass tùy cấu hình batch.

*c) HNSW và ANN Search*

HNSW (Malkov & Yashunin, 2020) xây dựng đồ thị phân cấp $G = \{G_0, G_1, \ldots, G_{l_{max}}\}$ trong đó $G_0$ chứa tất cả nodes với cạnh kết nối láng giềng gần nhất. Tìm kiếm: greedy traversal từ entry point ở tầng cao nhất xuống $G_0$, độ phức tạp $O(\log N)$. Tham số quan trọng: $M$ (số cạnh mỗi node), $ef_{construction}$, $ef_{search}$ — trade-off recall vs. latency.

*d) VVAS (Vitis Video Analytics SDK)*

VVAS cung cấp GStreamer plugin pipeline tích hợp VCU (Video Codec Unit) hardware H.264/H.265 decoder với DPU inference: `filesrc → h264parse → vvas_xvcudec → vvas_xdpuinfer → appsink`. Ưu điểm: zero-copy buffer transfer giữa VCU output và DPU input thông qua shared CMA memory, giảm overhead copy CPU.

---

#### NC2: Hệ thống NLVS Prototype trên PC — Cơ sở kiểm chứng khả thi

**Mục đích:** Xác nhận tính khả thi kỹ thuật của toàn bộ pipeline NLVS trước khi chuyển sang nghiên cứu tối ưu hóa cho nền tảng Kria.

Hệ thống prototype đã được xây dựng sẵn trên nền tảng PC với kiến trúc sau:

- **Visual encoder:** EVA-CLIP ViT-L/14 (FP32, CUDA) — trích xuất embedding 768 chiều cho mỗi video segment.
- **Text encoder:** EVA-CLIP Text Transformer (FP32) — trích xuất embedding 768 chiều cho query văn bản.
- **Vector store:** Qdrant v1.9 với HNSW index, payload filtering, gRPC/REST API.
- **Segmentation:** Scene detection (PySceneDetect) + fixed-window sliding để tạo segments từ 2–30 giây.
- **API:** FastAPI + async job queue phục vụ indexing và search requests.

Hệ thống đã được kiểm thử với **154 test cases** (pytest, 100% pass rate) bao gồm unit tests cho tất cả components và integration tests cho API. Bộ test này xác nhận pipeline hoạt động đúng về chức năng và là nền tảng để đặt baseline so sánh hiệu năng cho NC5.

**Vai trò trong luận văn:** NC2 không phải là đóng góp khoa học mới của luận văn này. Hệ thống prototype PC đóng vai trò (a) minh chứng tính khả thi của toàn bộ pipeline NLVS, và (b) baseline tham chiếu (FP32 on GPU) để đánh giá sự đánh đổi trong NC5.

---

#### NC3: Lượng tử hóa INT8 và Biên dịch .xmodel cho DPU B4096

**Mục đích:** Đây là nội dung nghiên cứu trung tâm, trực tiếp tạo ra artifact kỹ thuật chính của luận văn.

**3.1 Chuẩn bị và phân tích mô hình**

Sử dụng CLIP ViT-B/16 pre-trained từ `openai/clip-vit-base-patch16` (HuggingFace). Trước khi lượng tử hóa, phân tích đồ thị PyTorch để xác định tất cả tầng không tương thích DPU:

```python
from pytorch_nndct.apis import Inspector
inspector = Inspector("DPUCZDX8G_ISA1_B4096")
inspector.inspect(visual_encoder, dummy_input, device="cpu")
```

Output: danh sách tầng phân vào DPU subgraph vs. CPU subgraph, số lượng context switches dự kiến.

**3.2 Calibration Dataset**

Xây dựng calibration set từ ~1000 frames trích xuất từ video giám sát đại diện (đa dạng điều kiện ánh sáng, mật độ người, góc nhìn). Calibration set này quyết định chất lượng scale factors $s$ và zero points $z$ cho từng tầng. Nghiên cứu ảnh hưởng của kích thước calibration set (100, 500, 1000, 2000 mẫu) đến chất lượng embedding sau PTQ.

**3.3 PTQ với vai_q_pytorch**

```python
from pytorch_nndct.apis import torch_quantizer
quantizer = torch_quantizer(
    quant_mode="calib",
    module=visual_encoder,
    input_args=dummy_input,
    device=torch.device("cpu"),
    quant_config_file="quant_config.json"
)
# Calibration pass
for batch in calibration_loader:
    quantizer.quant_model(batch)
quantizer.export_quant_config()
```

Tham số quan trọng cần khảo sát:
- `weight_bit`: 8 (fixed) — quantize weights xuống INT8
- `act_bit`: 8 (fixed) — quantize activations
- `calib_algo`: `percentile` vs. `minmax` vs. `entropy` — so sánh ảnh hưởng đến chất lượng embedding

**3.4 Biên dịch sang .xmodel với vai_c_xir**

Sau calibration, biên dịch đồ thị lượng tử hóa sang `.xmodel`:

```bash
vai_c_xir \
  --xmodel clip_vitb16_quantized.xmodel \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  --output_dir ./compiled_model \
  --net_name clip_vitb16
```

Phân tích output: số lượng DPU subgraphs, số lượng CPU subgraphs, phân bố tham số trên DPU vs. CPU, tỷ lệ MAC operations trên DPU.

**3.5 Phân tích Mixed-Execution Graph**

Đây là phần nghiên cứu đặc thù nhất. Sau biên dịch, sử dụng `xdputil` để phân tích:
- Tổng số subgraphs và thứ tự thực thi
- Kích thước tensor tại mỗi điểm CPU↔DPU transfer
- Ước tính thời gian DMA transfer dựa trên kích thước tensor và băng thông AXI HP

Đề xuất và đánh giá các chiến lược giảm overhead:
- **Subgraph merging:** Điều chỉnh threshold lượng tử hóa để merge các DPU subgraph nhỏ liền kề
- **Operator substitution:** Thay thế GELU bằng ReLU6 (DPU-native) trong fine-tune — đánh giá ảnh hưởng đến embedding quality
- **Batch size optimization:** Khảo sát batch={1,2,4,8} — trade-off between throughput và latency

**3.6 Kiểm chứng chất lượng embedding**

So sánh định lượng embedding chất lượng:

| Metric | Phương pháp đo |
|---|---|
| Cosine similarity drift | $\Delta_{\cos} = 1 - \frac{\mathbf{e}_{FP32}^T \mathbf{e}_{INT8}}{|\mathbf{e}_{FP32}||\mathbf{e}_{INT8}|}$ trên 1000 cặp ảnh-text |
| Ranking correlation | Spearman $\rho$ giữa ranking FP32 và INT8 trên 100 queries |
| Recall@K | Trên tập DiDeMo test split (K = 1, 5, 10) |

---

#### NC4: Triển khai và Tối ưu hóa Trên Kria KV260

**Mục đích:** Xây dựng hệ thống NLVS đầy đủ chức năng trên Kria KV260, tối ưu hóa để đạt hiệu năng thực tế chấp nhận được.

**4.1 Cài đặt môi trường Kria**

- OS: Ubuntu 22.04 for KV260 (Xilinx Board Support Package)
- Vitis AI Runtime (VART) 3.5 với Python bindings
- Qdrant ARM64 binary (pre-compiled từ source)
- Firmware: DPUCZDX8G xclbin cho KV260

**4.2 VART Inference Engine**

Triển khai inference pipeline sử dụng VART Python API:

```python
import vart, xir

# Load compiled model
graph = xir.Graph.deserialize("clip_vitb16.xmodel")
runner = vart.Runner.create_runner(subgraph, "run")

# Inference với buffer pre-allocation
input_tensor_buffers = runner.get_inputs()
output_tensor_buffers = runner.get_outputs()
job_id = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
runner.wait(job_id)
```

Tối ưu hóa: pre-allocate tensor buffers, sử dụng numpy zero-copy interface với CMA buffers khi khả thi.

**4.3 Tích hợp VVAS Pipeline**

Xây dựng GStreamer pipeline tích hợp VCU decoder và DPU inference:

```
filesrc location=video.mp4 ! \
  qtdemux ! h264parse ! \
  vvas_xvcudec dev-idx=0 ! \
  video/x-raw,format=NV12 ! \
  vvas_xdpuinfer \
    model-path=/path/to/clip_vitb16.xmodel \
    batch-size=4 ! \
  appsink
```

Lợi ích: VCU hardware decoder tiêu thụ ít năng lượng hơn software decoder ~5×; NV12 → RGB conversion trên PL giảm tải ARM CPU.

**4.4 Chiến lược tối ưu hóa hiệu năng**

*Double-buffering:* Trong khi DPU xử lý batch $B_t$, CPU chuẩn bị batch $B_{t+1}$ (decode frames, preprocess). Giảm DPU idle time.

*NEON SIMD:* Sử dụng ARM NEON intrinsics cho các CPU subgraph (LayerNorm, Softmax, GELU) thay vì plain NumPy. Thư viện `nnpack` cung cấp NEON-optimized softmax và normalization.

*Memory bandwidth:* Phân tích cache locality của CPU subgraph tensors; cố gắng giữ intermediate tensors trong L2 cache Cortex-A53 (256 KB) để giảm DRAM access.

*Thermal management:* Monitor nhiệt độ DPU qua sysfs (`/sys/class/thermal/thermal_zone*/temp`). Triển khai Dynamic Voltage and Frequency Scaling (DVFS) để duy trì hiệu năng bền vững trong điều kiện ambient nhiệt cao.

**4.5 Tích hợp Qdrant và API**

Triển khai Qdrant ARM64 trực tiếp trên KV260, cấu hình lưu trữ trên eMMC:
- Collection `nlvs_kria`: HNSW M=16, ef=100, vector_size=512 (CLIP ViT-B/16)
- Index segments: encode frame → DPU → embedding 512D → upsert Qdrant
- Search: text query → DPU → text embedding 512D → Qdrant search → top-K results

Xây dựng REST API tối giản (FastAPI + uvicorn) trên ARM CPU phục vụ search requests từ client.

---

#### NC5: Đánh giá Hiệu năng và So sánh

**Mục đích:** Cung cấp đánh giá định lượng toàn diện về trade-off giữa chất lượng truy xuất — hiệu năng tính toán — tiêu thụ năng lượng giữa Kria KV260 và baseline PC.

**5.1 Thiết kế thực nghiệm**

*Tập dữ liệu đánh giá:*
- **DiDeMo** (Hendricks et al., 2017): 10,464 video clips, 40,543 text descriptions — benchmark chuẩn cho NLVS
- **Charades-STA** (Gao et al., 2017): 16,128 activity clips — temporal grounding
- **Custom security footage set**: ~200 clips từ camera giám sát thực tế (đa điều kiện ánh sáng, ngày/đêm)

*Cấu hình so sánh:*

| Cấu hình | Hardware | Model | Precision |
|---|---|---|---|
| Baseline-PC | i7-12700K + GTX 1650 Ti | EVA-CLIP ViT-L/14 | FP32 |
| Kria-FP32 | Kria KV260 (CPU only) | CLIP ViT-B/16 | FP32 |
| Kria-INT8 | Kria KV260 (DPU B4096) | CLIP ViT-B/16 | INT8 |

**5.2 Các chỉ số đánh giá**

*Chất lượng truy xuất:*
- **Recall@K** (K=1,5,10): Tỷ lệ truy vấn có video liên quan trong top-K kết quả
- **mAP** (Mean Average Precision): Đánh giá tổng hợp ranking quality
- **Embedding similarity drift**: $\Delta_{\cos}$ giữa FP32 và INT8 embeddings

*Hiệu năng tính toán:*
- **Index throughput**: số frames/giây có thể encode (video indexing speed)
- **Query latency**: thời gian từ khi nhận text query đến khi trả về top-K (ms)
- **Memory footprint**: peak RAM sử dụng trong quá trình indexing và search

*Năng lượng tiêu thụ:*
- **Idle power**: mức tiêu thụ khi không có tải (W)
- **Peak power**: mức tiêu thụ khi encode video liên tục (W)
- **Energy per query**: năng lượng tiêu thụ cho một search query (mJ)
- **GOPS/W**: hiệu quả tính toán (Giga Operations Per Second / Watt)

*Đo lường:*
- PC: `nvidia-smi` + `powerstat` cho GPU power; Python `time.perf_counter()` cho latency
- Kria: INA219 current sensor trên power rail (đo trực tiếp); `vaitrace` cho DPU profiling

**5.3 Phân tích kết quả**

Kỳ vọng kết quả dự kiến (dựa trên tài liệu tham khảo và phân tích lý thuyết):

| Metric | Baseline-PC | Kria-FP32 | Kria-INT8 (dự kiến) |
|---|---|---|---|
| Recall@10 (DiDeMo) | ~65% | ~55% | ~60-63% |
| Index throughput | ~8 FPS | ~0.5 FPS | ~2-5 FPS |
| Query latency | ~80ms | ~500ms | ~150-300ms |
| Power | ~120W | ~8W | ~15-20W |
| GOPS/W | ~15 | ~5 | ~50-60 |

Phân tích sẽ đi sâu vào: (i) nguồn gốc suy giảm Recall@10 từ PC sang Kria (model size reduction ViT-L→ViT-B vs. quantization error); (ii) bottleneck throughput (DPU latency vs. DMA overhead vs. video decode); (iii) điểm hòa vốn năng lượng (energy break-even point) khi scale số camera.

---
---

### 1.4 Phương pháp nghiên cứu

Luận văn áp dụng phương pháp nghiên cứu thực nghiệm định lượng (*quantitative experimental research*) kết hợp với nghiên cứu thiết kế hệ thống (*design science research*). Quá trình triển khai được chia thành **4 pha** liên tiếp:

**Pha 1 — Nghiên cứu lý thuyết và thu thập dữ liệu** *(Tháng 1–2)*

- Khảo sát tài liệu có hệ thống (systematic literature review) về: VLM deployment trên edge, INT8 PTQ cho ViT, DPU B4096 architecture, VVAS programming model.
- Phân tích tài liệu kỹ thuật AMD: PG338 (DPU TRM), UG1414 (Vitis AI User Guide), UG1085 (Zynq MPSoC TRM).
- Thu thập và chuẩn hóa tập dữ liệu DiDeMo và Charades-STA; tạo calibration dataset từ video giám sát đại diện.
- Thiết lập môi trường phát triển: Docker Vitis AI 3.5 trên máy host (x86), cross-compilation toolchain.

**Pha 2 — Lượng tử hóa và biên dịch mô hình** *(Tháng 2–3)*

- Thực hiện PTQ với `vai_q_pytorch`: thử nghiệm nhiều cấu hình calibration (kích thước dataset, thuật toán calib_algo).
- Phân tích đồ thị thực thi: đếm subgraphs, đo kích thước tensors tại các điểm CPU↔DPU transition.
- Biên dịch `.xmodel` với `vai_c_xir` và validate trên Kria KV260.
- Đo lường chất lượng embedding: so sánh cosine similarity FP32 vs. INT8 trên 1000 cặp test.

**Pha 3 — Triển khai và tối ưu hóa trên Kria KV260** *(Tháng 3–4)*

- Lập trình VART runtime wrapper cho CLIP visual encoder và text encoder.
- Tích hợp VVAS pipeline; kiểm thử end-to-end với video test clips.
- Triển khai double-buffering và NEON optimization cho CPU subgraphs.
- Tích hợp Qdrant ARM64; kiểm thử indexing và search functionality.
- Phát triển REST API và giao diện demo.

**Pha 4 — Đánh giá và viết luận văn** *(Tháng 4–6)*

- Chạy benchmark hệ thống đầy đủ: Recall@K, throughput, latency, power consumption.
- Phân tích kết quả, xác định bottleneck, rút ra kết luận về khả năng ứng dụng thực tế.
- Viết luận văn, chuẩn bị slides và demo cho buổi bảo vệ.

---

### 1.5 Kết quả dự kiến

Kết thúc luận văn, các sản phẩm nghiên cứu dự kiến bao gồm:

**Sản phẩm kỹ thuật:**

| Sản phẩm | Mô tả |
|---|---|
| `clip_vitb16_dpu.xmodel` | CLIP ViT-B/16 visual encoder lượng tử hóa INT8, biên dịch cho DPU B4096 |
| `clip_text_dpu.xmodel` | CLIP Text encoder lượng tử hóa INT8, biên dịch cho DPU B4096 |
| VART inference module | Python/C++ wrapper cho DPU inference, tích hợp double-buffering |
| VVAS pipeline config | GStreamer pipeline cấu hình đầy đủ cho video decode + DPU inference |
| Hệ thống NLVS-Kria | End-to-end working system: REST API + Qdrant + DPU inference trên KV260 |
| Quantization pipeline script | Reproducible PTQ pipeline với Vitis AI 3.5, được tài liệu hóa đầy đủ |

**Đóng góp khoa học:**

1. **Phân tích thực nghiệm mixed-execution graph** của CLIP ViT-B/16 trên DPU B4096: số subgraphs, overhead DMA, tỷ lệ MAC operations trên DPU vs. CPU — dữ liệu chưa được công bố trong tài liệu hiện có.

2. **Đánh giá định lượng trade-off** chất lượng embedding (Recall@K) — throughput (FPS) — năng lượng (GOPS/W) giữa PC GPU và Kria KV260 trong ứng dụng NLVS thực tế.

3. **Quy trình tái tạo** (reproducible pipeline) từ pre-trained CLIP → PTQ → `.xmodel` → Kria deployment, có thể áp dụng cho các VLM ViT-based khác.

**Tiêu chí thành công:**

- Recall@10 trên DiDeMo test set ≥ 55% (Kria-INT8) — chấp nhận được cho ứng dụng giám sát thực tế.
- Index throughput ≥ 2 FPS (Kria-INT8) — đủ để xử lý real-time stream 1 camera 2FPS.
- Query latency ≤ 500ms (Kria-INT8) — đáp ứng yêu cầu interactive search.
- Power consumption ≤ 25W — phù hợp với power budget của thiết bị edge.

---

### 1.6 Tài liệu tham khảo

**VLM và CLIP:**

[1] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning transferable visual models from natural language supervision*. ICML 2021.

[2] Sun, Q., Fang, Y., Wu, L., et al. (2023). *EVA-CLIP: Improved training techniques for CLIP at scale*. arXiv:2303.15389.

[3] Ni, B., Peng, H., Chen, M., et al. (2022). *Expanding language-image pretrained models for general video recognition*. ECCV 2022.

[4] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). *An image is worth 16x16 words: Transformers for image recognition at scale*. ICLR 2021.

[5] Zhai, X., Kolesnikov, A., Houlsby, N., & Beyer, L. (2022). *Scaling vision transformers*. CVPR 2022.

**Quantization của Transformer:**

[6] Bondarenko, Y., Nagel, M., & Blankevoort, T. (2021). *Understanding and overcoming the challenges of efficient transformer quantization*. EMNLP 2021.

[7] Xiao, G., Lin, J., Seznec, M., et al. (2023). *SmoothQuant: Accurate and efficient post-training quantization for large language models*. ICML 2023.

[8] Liu, Z., Wang, Y., Han, K., et al. (2021). *Post-training quantization for vision transformer*. NeurIPS 2021.

[9] Nagel, M., Amjad, R. A., Van Baalen, M., et al. (2020). *Up or down? Adaptive rounding for post-training quantization*. ICML 2020.

[10] Kim, S., Agrawal, A., & Choudhury, R. (2021). *I-BERT: Integer-only BERT quantization*. ICML 2021.

**FPGA và DPU Acceleration:**

[11] AMD Xilinx. (2023). *Deep Learning Processing Unit v3.3 Product Guide* (PG338). AMD Developer Resources.

[12] AMD Xilinx. (2023). *Vitis AI User Guide v3.5* (UG1414). AMD Developer Resources.

[13] Gschwend, D. (2020). *ZynqNet: An FPGA-accelerated embedded convolutional neural network*. arXiv:2005.03921.

[14] Lu, L., Liang, Y., Xiao, Q., & Yan, S. (2017). *Evaluating fast algorithms for convolutional neural networks on FPGAs*. FCCM 2017.

[15] Li, Z., Gu, Q., Wang, J., et al. (2023). *HALO: Hardware-aware learning to optimize for efficient vision transformers on FPGAs*. DAC 2023.

[16] Nguyen, T., Yoon, K., & Park, S. (2022). *Efficient deployment of vision transformers on FPGA with INT8 quantization*. FPL 2022.

**ANN Search và Vector Databases:**

[17] Malkov, Y. A., & Yashunin, D. A. (2020). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE TPAMI.

[18] Johnson, J., Douze, M., & Jégou, H. (2021). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.

[19] Qdrant Team. (2024). *Qdrant Vector Database Documentation v1.9*. https://qdrant.tech/documentation/.

[20] Simhadri, H. V., Williams, G., Aujoux, M., et al. (2022). *Results of the NeurIPS 2021 challenge on billion-scale approximate nearest neighbor search*. NeurIPS 2021 Competition Track.

**NLVS và Video Retrieval:**

[21] Hendricks, L. A., Wang, O., Shechtman, E., et al. (2017). *Localizing moments in video with natural language*. ICCV 2017. *(DiDeMo dataset)*

[22] Gao, J., Sun, C., Yang, Z., & Nevatia, R. (2017). *TALL: Temporal activity localization via language query*. ICCV 2017. *(Charades-STA)*

[23] Luo, H., Ji, L., Zhong, M., et al. (2022). *CLIP4Clip: An empirical study of CLIP for end to end video clip retrieval*. Neurocomputing.

[24] Portillo-Quintero, J. A., Ortiz-Bayliss, J. C., & Terashima-Marín, H. (2021). *A straightforward framework for video retrieval using CLIP*. MCPR 2021.

**Edge AI và VVAS:**

[25] AMD Xilinx. (2023). *Vitis Video Analytics SDK (VVAS) User Guide* (UG1362). AMD Developer Resources.

[26] Banbury, C., Reddi, V. J., Torelli, P., et al. (2021). *Benchmarking TinyML systems: Challenges and direction*. arXiv:2003.04821.

---

## 2. KẾ HOẠCH THỰC HIỆN

---

### 2.1 Phân chia giai đoạn

| Giai đoạn | Nội dung | Thời gian |
|---|---|---|
| **Giai đoạn 1** | Nghiên cứu lý thuyết: ViT, quantization, DPU B4096, VVAS. Thu thập dữ liệu benchmark. Phân tích cấu trúc CLIP ViT-B/16 với DPU Inspector. | Tháng 1–2 |
| **Giai đoạn 2** | PTQ với vai_q_pytorch. Khảo sát calibration configs. Biên dịch .xmodel. Phân tích mixed-execution graph. Đánh giá embedding quality FP32 vs. INT8. | Tháng 2–3 |
| **Giai đoạn 3** | Triển khai VART runtime trên Kria KV260. Tích hợp VVAS pipeline. Tích hợp Qdrant ARM64. Xây dựng REST API. Double-buffering + NEON optimization. | Tháng 3–4 |
| **Giai đoạn 4** | Benchmark đầy đủ: Recall@K, throughput, latency, power. Phân tích kết quả và bottleneck. | Tháng 4–5 |
| **Giai đoạn 5** | Viết luận văn. Chuẩn bị báo cáo và slides. Bảo vệ luận văn. | Tháng 5–6 |

---

### 2.2 Biểu đồ Gantt

```
Nội dung                               | T1 | T2 | T3 | T4 | T5 | T6 |
---------------------------------------|----|----|----|----|----|----|
NC1: Lý thuyết & tài liệu             | ██ | █  |    |    |    |    |
NC1: Thu thập & chuẩn bị dữ liệu      | █  | █  |    |    |    |    |
NC2: Review hệ thống prototype PC     |    | █  |    |    |    |    |
NC3: PTQ & phân tích mixed-exec graph |    | ██ | █  |    |    |    |
NC3: Biên dịch .xmodel & kiểm tra     |    |    | █  |    |    |    |
NC4: VART runtime trên Kria KV260     |    |    | ██ | █  |    |    |
NC4: VVAS integration & tối ưu hóa    |    |    | █  | ██ |    |    |
NC4: Qdrant ARM64 & REST API          |    |    |    | █  |    |    |
NC5: Benchmark & đánh giá kết quả     |    |    |    | █  | ██ |    |
Viết luận văn                         |    |    | █  | █  | ██ | ██ |
Bảo vệ luận văn                       |    |    |    |    |    | █  |
```

**Chú thích:** ██ = Hoạt động chính / █ = Hoạt động phụ hoặc tổng hợp

---

### 2.3 Yêu cầu tài nguyên

| Tài nguyên | Mô tả | Ghi chú |
|---|---|---|
| **AMD Kria KV260 Starter Kit** | Phần cứng nghiên cứu chính | Giá ~249 USD, cần mua trước Giai đoạn 3 |
| **Máy host x86** | Chạy Vitis AI Docker, cross-compile | CPU ≥ 8 cores, RAM ≥ 32 GB, GPU tùy chọn (cho PTQ) |
| **INA219 Power Monitor** | Đo công suất tiêu thụ Kria | I2C interface, độ chính xác ±1% |
| **Dataset DiDeMo** | ~50 GB video + annotations | Tải từ official repo; cần xin access |
| **Vitis AI 3.5 Docker** | Môi trường PTQ và biên dịch | AMD developer account, miễn phí |
| **MicroSD card 32 GB** | OS cho Kria KV260 | Class 10 / UHS-I |

---

### 2.4 Rủi ro và biện pháp ứng phó

| Rủi ro | Xác suất | Ảnh hưởng | Biện pháp ứng phó |
|---|---|---|---|
| CLIP ViT-B/16 PTQ giảm Recall@10 > 10% | Trung bình | Cao | Thử QAT cục bộ (fine-tune 2 epochs); hoặc thay bằng ViT-S/16 nhỏ hơn |
| DPU B4096 không đủ tài nguyên cho full ViT-B/16 | Thấp | Cao | Dùng CLIP ViT-S/16 (6 blocks thay vì 12) |
| Throughput Kria < 1 FPS sau tất cả tối ưu | Thấp | Trung bình | Chấp nhận offline indexing (không real-time); điều chỉnh tiêu chí thành công |
| Qdrant ARM64 không ổn định trên eMMC | Thấp | Thấp | Dùng SQLite + tự cài đặt HNSW từ `hnswlib` |
| Kria KV260 hết hàng / giao hàng chậm | Thấp | Cao | Đặt hàng sớm từ Tháng 1; phương án dự phòng: Xilinx ZCU104 |

---

*Hà Nội / Thành phố Hồ Chí Minh, ngày     tháng     năm 2026*

**Học viên thực hiện** *(ký và ghi rõ họ tên)*

&nbsp;

**Cán bộ hướng dẫn** *(ký và ghi rõ họ tên)*

ĐỖ TRÍ NHỰT

---

*Đề cương luận văn thạc sĩ — Chương trình Khoa học Máy tính (8480101) — Hướng ứng dụng*
*Trường Đại học Công nghệ Thông tin, ĐHQG-HCM*
