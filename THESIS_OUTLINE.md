# ĐỀ CƯƠNG LUẬN VĂN THẠC SĨ

---

**Tên đề tài:**
**NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG TÌM KIẾM VIDEO BẰNG NGÔN NGỮ TỰ NHIÊN DỰA TRÊN MÔ HÌNH HỌC SÂU ĐA PHƯƠNG THỨC VÀ TRIỂN KHAI TRÊN NỀN TẢNG NHÚNG AMD KRIA KV260**

---

**Học viên thực hiện:** [Họ và tên học viên]
**Chuyên ngành:** Kỹ thuật Điện tử
**Mã số:** [Mã số chuyên ngành]
**Người hướng dẫn khoa học:** [Học hàm, học vị, Họ tên]
**Cơ sở đào tạo:** [Tên trường]
**Năm thực hiện:** 2026

---

## 1. NỘI DUNG ĐỀ CƯƠNG

---

### 1.1 Tổng quan đề tài

#### 1.1.1 Bối cảnh và tính cấp thiết

Trong bối cảnh hệ thống giám sát an ninh và quản lý video ngày càng được triển khai rộng rãi, lượng dữ liệu video tích lũy mỗi ngày đã vượt xa khả năng xem xét thủ công của con người. Theo ước tính, một hệ thống giám sát đô thị trung bình tạo ra hàng trăm giờ video mỗi ngày, trong đó chỉ một tỷ lệ nhỏ chứa các sự kiện có ý nghĩa. Việc truy xuất thủ công các đoạn video liên quan đến một sự kiện cụ thể — chẳng hạn "người mặc áo đỏ chạy qua cổng" hay "xe máy vượt đèn đỏ" — không chỉ tốn kém thời gian mà còn tiềm ẩn sai sót do mệt mỏi của người vận hành.

Trước yêu cầu đó, lĩnh vực **Tìm kiếm Video bằng Ngôn ngữ Tự nhiên** (Natural Language Video Search — NLVS) đã nổi lên như một hướng nghiên cứu có giá trị ứng dụng cao, cho phép hệ thống tự động ánh xạ truy vấn văn bản của người dùng sang không gian biểu diễn đa phương thức (multimodal) và truy xuất các đoạn video phù hợp nhất. Nền tảng kỹ thuật cho hướng tiếp cận này được đặt nền móng bởi mô hình **CLIP** (Contrastive Language-Image Pretraining) do Radford và cộng sự (2021) đề xuất, mở ra khả năng học biểu diễn chung cho ngôn ngữ và hình ảnh trong một không gian nhúng thống nhất thông qua huấn luyện đối nghịch trên quy mô lớn (400 triệu cặp ảnh-văn bản).

Tuy nhiên, phần lớn các hệ thống NLVS hiện tại được thiết kế để vận hành trên máy chủ GPU hiệu năng cao, đặt ra rào cản về chi phí hạ tầng và khả năng triển khai thực địa. Trong khi đó, các nền tảng nhúng thế hệ mới như **AMD Kria KV260** — tích hợp bộ xử lý ARM Cortex-A53 với cổng logic khả trình FPGA Zynq UltraScale+ MPSoC và đơn vị xử lý học sâu DPU B4096 (1.4 TOPS INT8) — đang mở ra khả năng triển khai các mô hình học sâu có quy mô vừa ngay tại điểm thu thập dữ liệu (edge deployment) với mức tiêu thụ điện năng thấp (≈ 15–25W), phù hợp cho các hệ thống giám sát nhúng hoạt động liên tục.

Đề tài nghiên cứu này nhằm giải quyết đồng thời hai thách thức: (i) thiết kế và tối ưu một pipeline NLVS đạt chất lượng truy xuất cạnh tranh với trạng thái nghệ thuật hiện tại (state-of-the-art), và (ii) nghiên cứu phương pháp lượng tử hóa và triển khai pipeline đó trên nền tảng nhúng AMD Kria KV260 thông qua bộ công cụ Vitis AI 3.5, đảm bảo cân bằng giữa độ chính xác và hiệu năng tính toán trong điều kiện tài nguyên hạn chế.

#### 1.1.2 Tổng quan về hướng tiếp cận kỹ thuật

Hệ thống NLVS được đề xuất trong luận văn này xây dựng dựa trên kiến trúc **truy xuất hai giai đoạn** (two-stage retrieval): giai đoạn thứ nhất thực hiện tìm kiếm thô (coarse retrieval) bằng cách ánh xạ truy vấn văn bản và các đoạn video vào cùng một không gian biểu diễn nhúng thông qua các mô hình đa phương thức; giai đoạn thứ hai tinh chỉnh xếp hạng kết quả (reranking) bằng mô hình hiểu ngôn ngữ-thị giác (Vision-Language Model) để nâng cao độ chính xác.

Về phía trích xuất đặc trưng video, luận văn nghiên cứu và đánh giá một tập hợp các backbone từ nhẹ đến nặng, bao gồm: EVA-CLIP ViT-L/14 (Sun et al., 2023), X-CLIP (Ni et al., 2022), SigLIP (Zhai et al., 2023), LanguageBind (Zhu et al., 2023) và InternVideo2 (Wang et al., 2024). Về phía lập chỉ mục và tìm kiếm vector, hệ thống sử dụng thư viện Faiss (Johnson et al., 2021) với cấu trúc `IndexFlatIP` cho tìm kiếm chính xác cosine similarity, kết hợp kỹ thuật loại trừ kết quả trùng lặp theo thời gian (Temporal Non-Maximum Suppression — T-NMS) và ngưỡng điểm thích ứng (adaptive score threshold).

Về triển khai nhúng, luận văn nghiên cứu quy trình lượng tử hóa sau huấn luyện (Post-Training Quantization — PTQ) INT8 bằng `vai_q_pytorch`, biên dịch sang định dạng `.xmodel` bằng `vai_c_xir`, và tích hợp với pipeline xử lý video phần cứng thông qua VVAS (Vitis Video Analytics SDK).

#### 1.1.3 Phạm vi và giới hạn

Luận văn tập trung vào bài toán truy xuất đoạn video (video segment retrieval) từ tập dữ liệu video được lập chỉ mục trước (offline indexing — online search), không giải quyết bài toán nhận dạng hành động theo thời gian thực (real-time action recognition streaming). Phần cứng mục tiêu là AMD Kria KV260; phần cứng phát triển là PC với GPU NVIDIA GTX 1650 Ti (4 GB VRAM). Ngôn ngữ truy vấn chính là tiếng Anh và tiếng Việt.

---

### 1.2 Mục tiêu đề tài

Luận văn đặt ra hai nhóm mục tiêu chính:

#### 1.2.1 Mục tiêu khoa học

1. **Phân tích và đánh giá** các phương pháp biểu diễn đặc trưng đa phương thức (multimodal feature representation) hiện đại, bao gồm các kiến trúc CLIP, X-CLIP, SigLIP, LanguageBind và InternVideo2, trên bài toán tìm kiếm video bằng ngôn ngữ tự nhiên, từ đó xác định sự đánh đổi giữa chất lượng truy xuất, tài nguyên bộ nhớ và tốc độ suy luận trong điều kiện thiết bị hạn chế.

2. **Nghiên cứu và phân tích** lý thuyết lượng tử hóa mạng nơ-ron sâu (Neural Network Quantization), đặc biệt là phương pháp PTQ INT8, và đánh giá tác động của quá trình lượng tử hóa đến chất lượng embedding của mô hình Vision Transformer (ViT) trong bối cảnh triển khai trên FPGA DPU.

3. **Đề xuất và xây dựng** kiến trúc pipeline NLVS hoàn chỉnh gồm: phân đoạn video thích ứng (adaptive scene segmentation), trích xuất đặc trưng đa khung hình (multi-frame feature extraction), lập chỉ mục vector và tìm kiếm ngữ nghĩa, cùng cơ chế tái xếp hạng hai giai đoạn.

#### 1.2.2 Mục tiêu kỹ thuật

1. **Xây dựng thành công** hệ thống NLVS chạy được trên PC (GPU NVIDIA) với chất lượng truy xuất được kiểm chứng trên tập dữ liệu thực tế.

2. **Triển khai và kiểm chứng** pipeline NLVS trên nền tảng nhúng AMD Kria KV260 thông qua quy trình Vitis AI 3.5 (PTQ → biên dịch xmodel → VART runtime), đạt độ suy giảm chất lượng embedding sau lượng tử hóa dưới 5% (cosine similarity INT8 vs. FP32 ≥ 0.95).

3. **Đạt được** tốc độ tìm kiếm dưới 200ms/truy vấn trên Kria KV260, phù hợp cho tương tác người dùng trong thời gian thực.

4. **So sánh định lượng** hiệu năng tính toán, mức tiêu thụ điện năng và chất lượng truy xuất giữa hai môi trường triển khai: PC GPU và Kria KV260, cung cấp cơ sở dữ liệu thực nghiệm cho việc lựa chọn nền tảng triển khai trong các ứng dụng giám sát thực tế.

---

### 1.3 Nội dung nghiên cứu của đề tài

Đề tài được tổ chức thành năm nội dung nghiên cứu chính, được thực hiện theo tiến độ tuần tự và có sự liên kết chặt chẽ:

#### Nội dung 1: Nghiên cứu cơ sở lý thuyết và rà soát tài liệu

Nội dung này tập trung rà soát có hệ thống (systematic review) các công trình nghiên cứu nền tảng và liên quan, bao gồm:

- **Lý thuyết CLIP và biểu diễn đa phương thức:** Phân tích cơ chế huấn luyện đối nghịch (contrastive learning) trong mô hình CLIP (Radford et al., 2021), trong đó hàm mục tiêu InfoNCE cực đại hóa tương đồng cosine giữa các cặp ảnh-văn bản tương ứng và cực tiểu hóa với các cặp không tương ứng trong cùng một mini-batch. Nghiên cứu sự mở rộng của CLIP sang không gian video qua các mô hình X-CLIP (Ni et al., 2022) với cơ chế cross-frame attention, LanguageBind (Zhu et al., 2023) với multi-modal binding, và InternVideo2 (Wang et al., 2024) với mô hình 1B tham số đạt 57.2% R@1 trên MSR-VTT.

- **Kiến trúc Vision Transformer (ViT):** Phân tích kiến trúc ViT-L/14 được sử dụng làm backbone cho EVA-CLIP (Sun et al., 2023), bao gồm patch embedding (patch size 14×14, image size 224×224, sequence length 257), multi-head self-attention với 16 heads, và projection head tạo embedding 768 chiều.

- **Lý thuyết lập chỉ mục vector và tìm kiếm tương đồng:** Nghiên cứu các phương pháp tìm kiếm cận đúng láng giềng (Approximate Nearest Neighbor — ANN), đặc biệt `IndexFlatIP` (tìm kiếm chính xác, O(N·D)) và `IndexIVFFlat` (tìm kiếm gần đúng, O(N/nlist·D)) trong thư viện Faiss (Johnson et al., 2021), cùng các kỹ thuật Product Quantization để nén bộ nhớ.

- **Lý thuyết lượng tử hóa mạng nơ-ron:** Nghiên cứu toán học của quá trình lượng tử hóa tuyến tính INT8 (Nagel et al., 2021), trong đó giá trị thực $x$ được ánh xạ sang số nguyên 8-bit theo công thức $x_q = \text{clip}\left(\left\lfloor\frac{x}{s}\right\rceil + z,\ -128,\ 127\right)$ với $s$ là scale factor và $z$ là zero point. Phân tích tác động của lượng tử hóa đến từng loại tầng trong ViT, đặc biệt là LayerNormalization, MultiHeadAttention (Softmax) và GELU — các tầng yêu cầu fallback về CPU khi triển khai trên DPU B4096.

- **Phân đoạn cảnh video:** Nghiên cứu thuật toán phát hiện ranh giới cảnh dựa trên sự khác biệt histogram màu sắc (HSV color histogram difference) được triển khai trong PySceneDetect (ContentDetector), kết hợp với cơ chế cửa sổ trượt (sliding window) để phân đoạn video thành các đơn vị ngữ nghĩa nhất quán.

#### Nội dung 2: Thiết kế và xây dựng kiến trúc hệ thống NLVS

Trên cơ sở lý thuyết được rà soát, nội dung này tiến hành thiết kế và hiện thực hóa kiến trúc hệ thống hoàn chỉnh với hai luồng xử lý chính:

- **Luồng lập chỉ mục (Indexing Pipeline):** Đầu vào là tệp video (.mp4/.avi/.mov). Video được đưa qua lớp thu nạp video (`VideoPipeline`) hỗ trợ ba backend: OpenCV, GStreamer và VVAS (dành cho Kria). Tiếp theo, module `SceneSegmenter` sử dụng PySceneDetect với `ContentDetector` (ngưỡng 27.0, phát hiện thay đổi HSV) để phân đoạn video thích ứng theo nội dung, tránh tạo ra các "embedding chimera" khi một cửa sổ cố định bao gồm nhiều cảnh khác nhau. Mỗi đoạn video được trích xuất $n=5$ khung hình đều nhau, sau đó mã hóa bằng engine đa phương thức được chọn. Embedding của đoạn video được tính bằng phép trung bình cộng (mean pooling) của $n$ embedding khung hình rời rạc, sau đó chuẩn hóa L2:
$$\mathbf{v}_{seg} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{v}_{f_i}, \quad \hat{\mathbf{v}}_{seg} = \frac{\mathbf{v}_{seg}}{\|\mathbf{v}_{seg}\|_2}$$
Module `CaptionAugmenter` (tùy chọn) tổng hợp hybrid embedding từ biểu diễn thị giác và biểu diễn văn bản của caption được tạo tự động: $\mathbf{h} = \alpha \cdot \hat{\mathbf{v}}_{seg} + (1-\alpha) \cdot \hat{\mathbf{c}}$, với $\alpha = 0.7$. Vector cuối cùng được lưu vào chỉ mục Faiss `IndexFlatIP` cùng siêu dữ liệu (video_id, t_start, t_end).

- **Luồng tìm kiếm (Search Pipeline):** Truy vấn văn bản được chuẩn hóa (loại bỏ tiền tố mệnh lệnh, dịch Việt–Anh), sau đó mã hóa qua tập hợp 12 prompt template ("a video of {}", "footage showing {}", v.v.) và lấy trung bình cộng embedding, tương tự kỹ thuật ensemble template được đề xuất trong Radford et al. (2021). Tìm kiếm cosine similarity bằng Faiss, kết quả được lọc bằng ngưỡng điểm thích ứng (adaptive threshold) và loại trừ trùng lặp bằng thuật toán T-NMS với IoU ngưỡng 0.30. Cuối cùng, giai đoạn tái xếp hạng sử dụng BLIP-2 (Li et al., 2023) với cơ chế Visual Question Answering ("Does this scene contain X?") để tính điểm tổng hợp: $s_{final} = 0.6 \cdot s_{cosine} + 0.4 \cdot s_{BLIP2}$.

- **Thiết kế lớp engine đa mô hình (Multi-Engine Abstraction):** Xây dựng giao diện trừu tượng `InferenceEngine` với phương thức `encode_frames()` và `encode_text()`, cho phép hoán đổi backend (EVA-CLIP, X-CLIP, SigLIP, LanguageBind, InternVideo2, KriaEngine) thông qua cấu hình YAML mà không cần thay đổi mã nguồn logic tìm kiếm.

#### Nội dung 3: Thực nghiệm và đánh giá chất lượng trên PC GPU

Nội dung này tiến hành lập chỉ mục và đánh giá hệ thống trên môi trường PC với GPU NVIDIA GTX 1650 Ti (4 GB VRAM):

- **Xây dựng và chuẩn bị tập dữ liệu thực nghiệm:** Lập chỉ mục tập dữ liệu video giám sát thực tế, đánh giá chất lượng truy xuất thông qua các chỉ số Recall@K (R@1, R@5, R@10) trên tập câu truy vấn được xây dựng thủ công.

- **So sánh các backbone model:** Đánh giá định lượng chất lượng truy xuất và tốc độ suy luận giữa EVA-CLIP ViT-L/14 (embed_dim=768), X-CLIP ViT-B/16 (512-d, temporal attention), SigLIP ViT-L-16 (1024-d, multilingual), LanguageBind (768-d) và InternVideo2-1B (768-d) trên cùng một tập câu hỏi và tập dữ liệu video.

- **Đánh giá các thành phần kỹ thuật:** Thực nghiệm ablation study đánh giá đóng góp riêng lẻ của từng thành phần: phân đoạn cảnh thích ứng (vs. cửa sổ trượt cố định), template ensemble (vs. single template), T-NMS (vs. không lọc), tái xếp hạng BLIP-2 (vs. single-stage), và caption augmentation.

- **Đánh giá hỗ trợ tiếng Việt:** So sánh chất lượng truy vấn tiếng Việt thông qua dịch máy Việt–Anh (deep_translator) và truy vấn trực tiếp qua SigLIP multilingual (WebLI).

#### Nội dung 4: Nghiên cứu triển khai trên nền tảng nhúng AMD Kria KV260

Đây là nội dung trọng tâm phân biệt luận văn với các nghiên cứu NLVS trên PC thông thường:

- **Phân tích kiến trúc phần cứng AMD Kria KV260:** Nghiên cứu chi tiết kiến trúc Zynq UltraScale+ MPSoC, DPU B4096 (4096 MAC operations/cycle, 300 MHz, ~1.4 TOPS INT8), hệ thống bộ nhớ LPDDR4 4GB dùng chung (PS/PL shared), và băng thông bus AXI HP (~19.2 GB/s) ảnh hưởng đến thông lượng DMA khi truyền tensor giữa ARM CPU và DPU.

- **Quy trình lượng tử hóa và biên dịch model (Vitis AI 3.5):**
  - *Giai đoạn 1 — Export TorchScript:* Xuất CLIP ViT-B/16 visual encoder và text encoder sang định dạng TorchScript với input shape cố định để tương thích với compiler Vitis AI.
  - *Giai đoạn 2 — Post-Training Quantization (PTQ):* Sử dụng `vai_q_pytorch` trong môi trường Docker Vitis AI để thực hiện hiệu chỉnh (calibration) trên 1000 khung hình đại diện từ tập dữ liệu giám sát thực tế, xuất mô hình lượng tử INT8.
  - *Giai đoạn 3 — Biên dịch sang .xmodel:* Sử dụng `vai_c_xir` với file mô tả kiến trúc DPU `arch.json` của KV260 để biên dịch sang định dạng `.xmodel` (~86 MB cho visual encoder, ~63 MB cho text encoder).
  - *Giai đoạn 4 — Kiểm chứng chất lượng:* Đánh giá sự suy giảm chất lượng embedding giữa FP32 (PC) và INT8 (Kria DPU) thông qua cosine similarity trên tập kiểm chứng.

- **Tích hợp pipeline xử lý video phần cứng (VVAS):** Nghiên cứu và triển khai pipeline GStreamer với các plugin VVAS (`vvas_xmultisrc`, `vvas_xdec`, `vvas_xfilter`) để tận dụng bộ giải mã video phần cứng H.264/H.265 (VCU IP Core) của Zynq, đạt tốc độ giải mã lên đến 60fps cho video 1080p.

- **Phân tích thách thức kỹ thuật đặc thù của Transformer trên DPU:** Phân tích biểu đồ thực thi hỗn hợp DPU/CPU (mixed-execution graph) phát sinh do các tầng không được hỗ trợ bởi DPU (LayerNormalization, MultiHeadAttention Softmax, GELU), định lượng overhead của các lần chuyển tiếp CPU↔DPU, và so sánh với CNN thuần túy.

- **Tối ưu hóa hiệu năng trên Kria:** Nghiên cứu các kỹ thuật tối ưu bao gồm: double-buffering DPU runner, batch_size tuning (1→4), NEON-accelerated preprocessing trên ARM, cấu hình CPU frequency scaling (performance governor), và quản lý nhiệt độ DPU trong điều kiện tải liên tục.

#### Nội dung 5: Đánh giá và so sánh tổng thể

- **So sánh hiệu năng PC vs. Kria KV260:** Đo lường và so sánh định lượng thời gian lập chỉ mục 1 giờ video (dự kiến ~2 phút trên PC vs. ~17 phút trên Kria), độ trễ tìm kiếm/truy vấn (dự kiến ~20ms trên PC vs. ~110ms trên Kria), và mức tiêu thụ điện năng (dự kiến ~80W trên PC vs. ~20W trên Kria).

- **Đánh giá tính di động của chỉ mục (index portability):** Kiểm chứng khả năng chuyển đổi tệp chỉ mục Faiss giữa hai môi trường và mức độ suy giảm chất lượng truy xuất khi sử dụng index được lập từ embedding FP32 (PC) để tìm kiếm bằng embedding INT8 (Kria) và ngược lại.

- **Đánh giá tính khả thi triển khai sản xuất:** Kiểm tra hệ thống trong điều kiện vận hành kéo dài (soak test), bao gồm giám sát rò rỉ bộ nhớ, nhiệt độ DPU, và độ ổn định độ trễ.

---

### 1.4 Phương pháp thực hiện

#### 1.4.1 Phương pháp nghiên cứu lý thuyết

Phương pháp nghiên cứu chính được sử dụng là **rà soát tài liệu có hệ thống** (systematic literature review) kết hợp với **phân tích lý thuyết và suy luận toán học**. Các bài báo được thu thập từ các nguồn khoa học uy tín (NeurIPS, ICCV, ECCV, CVPR, arXiv) và đánh giá theo tiêu chí mức độ liên quan đến bài toán tìm kiếm video, khả năng tái hiện (reproducibility), và tính khả thi triển khai trong điều kiện tài nguyên hạn chế.

Mô hình toán học của từng thành phần được phân tích từ góc độ:
- **Biểu diễn:** Không gian nhúng (embedding space), chiều, tính chất chuẩn hóa.
- **Tối ưu hóa:** Hàm mục tiêu huấn luyện (InfoNCE loss, sigmoid binary CE loss).
- **Tìm kiếm:** Độ phức tạp tính toán và không gian của các cấu trúc chỉ mục.
- **Lượng tử hóa:** Sai số lượng tử hóa và ảnh hưởng đến khoảng cách cosine trong không gian nhúng.

#### 1.4.2 Phương pháp kỹ thuật và thực nghiệm

Hệ thống được phát triển theo phương pháp **thiết kế lặp tăng dần** (iterative incremental design) qua ba pha:

- **Pha 1 — Xây dựng hệ thống nền tảng (Baseline):** Hiện thực pipeline CLIP ViT-B/16 cơ bản với cửa sổ trượt cố định, Faiss IndexFlatIP, và tìm kiếm cosine similarity đơn giai đoạn. Pha này thiết lập đường cơ sở (baseline) để so sánh.

- **Pha 2 — Tích hợp các thành phần nâng cao:** Tích hợp EVA-CLIP ViT-L/14 làm backbone chính (thay thế ViT-B/16), module phân đoạn cảnh PySceneDetect, engine X-CLIP với temporal attention, engine SigLIP multilingual, kỹ thuật T-NMS, và ngưỡng điểm thích ứng.

- **Pha 3 — Tích hợp reranker và caption augmentation:** Tích hợp BLIP-2 reranker (Li et al., 2023), module `CaptionAugmenter` tạo hybrid embedding, và các engine LanguageBind, InternVideo2-1B. Sau Pha 3, thực hiện đánh giá toàn diện trên PC.

- **Pha 4 — Nghiên cứu triển khai Kria KV260:** Thực hiện quy trình Vitis AI (export → PTQ → biên dịch xmodel → deploy), tích hợp VVAS pipeline, đánh giá hiệu năng và chất lượng trên phần cứng nhúng.

Mỗi pha được kiểm chứng bằng bộ kiểm thử đơn vị (unit tests) và kiểm thử tích hợp (integration tests) được xây dựng với framework `pytest`, đảm bảo tính đúng đắn của từng thành phần trước khi tích hợp vào hệ thống tổng thể.

Giao diện người dùng thử nghiệm được xây dựng bằng Streamlit, cho phép tìm kiếm văn bản, xem trước thumbnail kết quả và tải xuống đoạn video tương ứng. REST API được cung cấp qua FastAPI để tích hợp với các hệ thống ngoài.

#### 1.4.3 Môi trường và công cụ phát triển

| Môi trường | Cấu hình |
|---|---|
| **PC phát triển** | CPU Intel/AMD, GPU NVIDIA GTX 1650 Ti (4 GB VRAM), RAM 16 GB, Ubuntu 20.04/22.04 |
| **Nền tảng nhúng** | AMD Kria KV260 (ARM Cortex-A53 ×4, DPU B4096, 4 GB LPDDR4), Ubuntu 22.04 |
| **Framework ML** | PyTorch 2.x, open_clip_torch, Transformers (Hugging Face) |
| **Công cụ Vitis AI** | vai_q_pytorch (PTQ), vai_c_xir (compiler), VART Python runtime |
| **Thư viện tìm kiếm** | Faiss (IndexFlatIP, IndexIVFFlat) |
| **Xử lý video** | OpenCV, GStreamer, VVAS (Kria) |
| **Phát hiện cảnh** | PySceneDetect (ContentDetector) |
| **Kiểm thử** | pytest, với bộ unit tests và integration tests cho từng module |
| **Container** | Docker (xilinx/vitis-ai-pytorch-cpu:3.5.0) cho môi trường quantization |

---

### 1.5 Kết quả, sản phẩm dự kiến

#### 1.5.1 Sản phẩm phần mềm

1. **Hệ thống NLVS hoàn chỉnh** dưới dạng mã nguồn Python có cấu trúc module hóa, bao gồm:
   - Pipeline lập chỉ mục video (indexing pipeline) hỗ trợ các định dạng .mp4, .avi, .mov, .mkv.
   - Pipeline tìm kiếm (search pipeline) hỗ trợ truy vấn tiếng Anh và tiếng Việt.
   - Sáu engine suy luận có thể hoán đổi: EVA-CLIP ViT-L/14, X-CLIP, SigLIP, LanguageBind, InternVideo2-1B, và KriaEngine (VART/DPU).
   - Giao diện người dùng web (Streamlit) và REST API (FastAPI).
   - Bộ kiểm thử tự động (pytest) với coverage cho tất cả các module chính.

2. **Model đã lượng tử hóa INT8 và biên dịch cho Kria KV260:**
   - `clip_vision.xmodel` (~86 MB): CLIP ViT-B/16 visual encoder, INT8, DPU-ready.
   - `clip_text.xmodel` (~63 MB): CLIP ViT-B/16 text encoder, INT8, DPU-ready.
   - Các script hỗ trợ: `export_clip_for_vitis.py`, `quantize_clip.py`, `compile_xmodel.sh`, `validate_quantization.py`.

3. **Tệp cấu hình hệ thống (YAML)** cho từng môi trường triển khai (PC, Kria), bao gồm các tham số đã được tối ưu thực nghiệm: cửa sổ phân đoạn 10s, độ chồng lấp 30%, 5 khung hình/đoạn, ngưỡng điểm 0.20.

#### 1.5.2 Kết quả khoa học kỳ vọng

1. **Bộ số liệu đánh giá so sánh** các backbone model (EVA-CLIP, X-CLIP, SigLIP, LanguageBind, InternVideo2) trên bài toán NLVS với tập dữ liệu video giám sát thực tế, được trình bày theo các chỉ số Recall@1, Recall@5, Recall@10 và độ trễ suy luận.

2. **Kết quả đánh giá định lượng** tác động của lượng tử hóa INT8 PTQ lên chất lượng embedding của CLIP ViT-B/16, bao gồm: cosine similarity trung bình giữa embedding FP32 và INT8 (kỳ vọng ≥ 0.96), và suy giảm R@1 truy xuất (kỳ vọng < 4%).

3. **Bảng so sánh hiệu năng** PC GPU vs. AMD Kria KV260 theo các chỉ tiêu: thông lượng lập chỉ mục (segments/giây), độ trễ tìm kiếm (ms/truy vấn), mức tiêu thụ điện năng (W), hiệu suất tính toán trên watt (TOPS/W), và chi phí phần cứng.

4. **Kết quả ablation study** định lượng đóng góp của từng thành phần kỹ thuật (phân đoạn cảnh, template ensemble, T-NMS, BLIP-2 reranker, caption augmentation) đến chất lượng truy xuất tổng thể.

5. **Báo cáo lộ trình triển khai Kria KV260** đầy đủ (Phase 0–4) có thể tái sử dụng cho các hệ thống NLVS tương tự trên nền tảng nhúng AMD Zynq/Kria.

#### 1.5.3 Đóng góp khoa học

Luận văn dự kiến đóng góp vào lĩnh vực nghiên cứu các điểm sau:

- Đề xuất và kiểm chứng kiến trúc pipeline NLVS hoàn chỉnh phù hợp với điều kiện tài nguyên phần cứng hạn chế (consumer GPU 4 GB VRAM và FPGA DPU edge), có thể được tái hiện và mở rộng bởi cộng đồng nghiên cứu.
- Cung cấp phân tích chi tiết về thách thức kỹ thuật của việc triển khai Vision Transformer (ViT) trên DPU FPGA, đặc biệt là vấn đề đồ thị thực thi hỗn hợp (mixed-execution graph) do các tầng LayerNorm, Softmax và GELU không được DPU hỗ trợ, và các chiến lược giảm thiểu overhead DMA.
- Cung cấp cơ sở dữ liệu thực nghiệm về sự đánh đổi giữa chất lượng truy xuất — hiệu năng — tiêu thụ điện năng cho bài toán NLVS trên hai nền tảng phần cứng có đặc điểm tương phản (high-performance GPU vs. low-power FPGA edge SOM).

---

### 1.6 Tài liệu tham khảo

#### Các công trình nền tảng về biểu diễn đa phương thức

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, và I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. of the 38th International Conference on Machine Learning (ICML)*, PMLR, vol. 139, pp. 8748–8763, 2021.

[2] Q. Sun, Y. Fang, L. Wu, X. Wang, và Y. Cao, "EVA-CLIP: Improved Training Techniques for CLIP at Scale," *arXiv preprint arXiv:2303.15389*, 2023.

[3] X. Zhai, B. Kolesnikov, N. Houlsby, và L. Beyer, "Scaling Vision Transformers," in *Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 12104–12113, 2022.

[4] X. Zhai, B. Mustafa, A. Kolesnikov, và L. Beyer, "Sigmoid Loss for Language Image Pre-Training (SigLIP)," in *Proc. of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 11975–11986, 2023.

#### Các công trình về tìm kiếm video và mô hình video-language

[5] B. Ni, H. Peng, M. Chen, S. Zhang, G. Meng, J. Fu, S. Xiang, và H. Ling, "Expanding Language-Image Pretrained Models for General Video Recognition (X-CLIP)," in *Proc. of the European Conference on Computer Vision (ECCV)*, pp. 1–18, 2022.

[6] B. Zhu, B. Lin, B. Ning, Y. Yang, J. Yan, C. Zhang, B. Li, X. Zhang, Y. Wei, và B. Li, "LanguageBind: Extending Video-Language Pretraining to N-Modality by Language-based Semantic Alignment," in *Proc. of the 12th International Conference on Learning Representations (ICLR)*, 2024.

[7] Y. Wang, K. Li, X. Li, J. Yu, Y. He, J. Wang, Y. Wang, Z. Chen, B. Zhu, B. Lin, Z. Yao, T. Zheng, X. Yin, S. Li, và Y. Li, "InternVideo2: Scaling Foundation Models for Multimodal Video Understanding," in *Proc. of the European Conference on Computer Vision (ECCV)*, 2024.

[8] H. Luo, L. Ji, M. Zhong, Y. Chen, W. Lei, N. Duan, và T. Li, "CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval," *Neurocomputing*, vol. 508, pp. 293–304, 2022.

#### Các công trình về Vision Transformer

[9] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, và N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)," in *Proc. of the 9th International Conference on Learning Representations (ICLR)*, 2021.

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, và I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

#### Các công trình về reranking và hiểu ngôn ngữ-thị giác

[11] J. Li, D. Li, S. Savarese, và S. Hoi, "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," in *Proc. of the 40th International Conference on Machine Learning (ICML)*, PMLR, vol. 202, pp. 19730–19742, 2023.

[12] J. Li, D. Li, C. Xiong, và S. Hoi, "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation," in *Proc. of the 39th International Conference on Machine Learning (ICML)*, PMLR, vol. 162, pp. 12888–12900, 2022.

#### Các công trình về lập chỉ mục vector

[13] J. Johnson, M. Douze, và H. Jégou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021.

[14] H. Jégou, M. Douze, và C. Schmid, "Product Quantization for Nearest Neighbor Search," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 33, no. 1, pp. 117–128, 2011.

[15] Y. Malkov và D. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (HNSW)," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 4, pp. 824–836, 2020.

#### Các công trình về lượng tử hóa mạng nơ-ron

[16] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, và T. Blankevoort, "A White Paper on Neural Network Quantization," *arXiv preprint arXiv:2106.08295*, 2021.

[17] R. Krishnamoorthi, "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper," *arXiv preprint arXiv:1806.08342*, 2018.

[18] Y. Bengio, N. Léonard, và A. Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation," *arXiv preprint arXiv:1308.3432*, 2013.

#### Tài liệu kỹ thuật nền tảng nhúng và FPGA

[19] AMD Xilinx, *Vitis AI User Guide (UG1414)*, v3.5, AMD Xilinx, San Jose, CA, USA, 2023. [Online]. Available: https://docs.xilinx.com/r/en-US/ug1414-vitis-ai

[20] AMD Xilinx, *Kria KV260 Vision AI Starter Kit — Product Brief*, AMD Xilinx, San Jose, CA, USA, 2022.

[21] AMD Xilinx, *Deep Learning Processing Unit (DPU) IP Product Guide (PG338)*, v4.1, AMD Xilinx, San Jose, CA, USA, 2023. [Online]. Available: https://docs.xilinx.com/r/en-US/pg338-dpu

[22] AMD Xilinx, *Vitis Video Analytics SDK (VVAS) Documentation*, v3.0, AMD Xilinx, San Jose, CA, USA, 2023. [Online]. Available: https://xilinx.github.io/VVAS/

#### Các công trình về phân đoạn cảnh video

[23] B. Castellano, *PySceneDetect: Python and OpenCV-Based Scene Cut Detection and Video Splitting Tool*, Version 0.6, 2023. [Online]. Available: https://www.scenedetect.com

[24] P. Sidiropoulos, V. Mezaris, I. Kompatsiaris, H. Meinedo, M. Bugalho, và I. Trancoso, "Temporal Video Segmentation to Scenes Using High-Level Audiovisual Features," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 21, no. 8, pp. 1163–1177, 2011.

#### Các công trình về học biểu diễn đối nghịch

[25] T. Chen, S. Kornblith, M. Norouzi, và G. Hinton, "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)," in *Proc. of the 37th International Conference on Machine Learning (ICML)*, PMLR, vol. 119, pp. 1597–1607, 2020.

[26] A. van den Oord, Y. Li, và O. Vinyals, "Representation Learning with Contrastive Predictive Coding," *arXiv preprint arXiv:1807.03748*, 2018. *(InfoNCE loss)*

---

*Đề cương này được xây dựng dựa trên các kết quả nghiên cứu và kỹ thuật được ghi nhận trong hệ thống prototype NLVS v3.0, bao gồm toàn bộ kiến trúc phần mềm đã hiện thực hóa qua ba pha phát triển và nghiên cứu triển khai trên AMD Kria KV260.*
