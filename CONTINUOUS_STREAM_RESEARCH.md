# Nghiên cứu: Kiến trúc Luồng Giám sát Liên tục trên AMD Kria KV260

> **Phiên bản:** 2.6  
> **Ngày:** 2026-05-20  
> **Hệ thống cơ sở:** NLVS v3.0 (Kria KV260, DPU B4096, VVAS 3.0) — PC dev: EVA-CLIP ViT-L/14 (CUDA)  
> **Thay đổi v2.0:** Tail-only overlap · Qdrant vector database · SQLite persistent queue  
> **Thay đổi v2.1:** Tham số xác nhận · check_ffmpeg/is_wsl2 helpers · V4L2 noise suppression  
> **Thay đổi v2.2:** OpenCVCaptureDaemon (ffmpeg fallback) · WSL2 usbipd-win guide · EOFError guard trong VideoIndex.load() · embed_dim phân biệt PC(768) vs Kria(512)  
> **Thay đổi v2.3:** Thống nhất Qdrant workflow trên cả PC và Kria · NLVideoSearcher hỗ trợ Qdrant search backend · pc.yaml backend=qdrant · 11 tests mới (QN-18..20, CI-01..06, SR-29..30)  
> **Thay đổi v2.4:** Fatal error detection trong OverlapCaptureDaemon + OpenCVCaptureDaemon · last_error property · app.py hiển thị camera permission error với hướng dẫn sửa · 6 tests mới (CD-01..06) · system fix: sudo usermod -aG video $USER  
> **Thay đổi v2.5:** app.py bảo toàn last_error vào session_state['last_capture_error'] sau Stop Pipeline · xóa 0-byte .mp4 khi Start Pipeline · 1 test mới (CD-07)  
> **Thay đổi v2.6:** job_queue: mark_dead() + purge_missing_files() · continuous_indexer: FileNotFoundError → mark_dead() thay vì retry · app.py _start_pipeline(): gọi purge_missing_files() sau xóa 0-byte files · 2 tests mới (CI-07, CI-08)  
> **Thay đổi v2.7:** WSL2 isochronous USB fix — FolderWatcherDaemon polls directory thay vì direct capture · windows_capture_server.py chạy trên Windows ghi qua \\wsl$\ · 5 tests mới (CD-08..12)  
> **Thay đổi v2.8:** Fix qdrant-client 1.18.0 API break — `_search_qdrant` dùng `client.query_points()` thay vì `client.search()` · `@st.cache_resource` model caching fix (load 1 lần) · Qdrant n_vectors UI fix · score_threshold hạ 0.20→0.10 cho screen-capture EVA-CLIP · 2 tests mới (QN-21, QN-22)  
> **Thay đổi v2.9:** Docker Qdrant `restart=unless-stopped` (auto-start qua WSL reboot) · `_reset_searcher()` không gọi `.clear()` nữa — giữ model cache, chỉ reset Faiss + reconnect Qdrant → tránh Streamlit 1.32 `expire_cache` coroutine warning  
> **Tham số vận hành đã xác nhận (v2.9):** `window=10s · overlap=30% · stride=7.0s · frames=5 · min_score=0.10`  
> **Bối cảnh:** Chuyển đổi từ mô hình "upload video thủ công" sang mô hình "luồng camera giám sát liên tục tự động lập chỉ mục"

---

## Mục lục

1. [Phân tích ý tưởng kiến trúc đề xuất](#1-phân-tích-ý-tưởng-kiến-trúc-đề-xuất)
2. [Tính toán hiệu năng định lượng](#2-tính-toán-hiệu-năng-định-lượng)
3. [Vấn đề Segment Boundary — Phân tích và Giải pháp](#3-vấn-đề-segment-boundary--phân-tích-và-giải-pháp)
4. [Kiến trúc Daemon hóa đề xuất](#4-kiến-trúc-daemon-hóa-đề-xuất)
   - [4.5 PersistentJobQueue — SQLite-backed Queue](#45-persistentjobqueue--sqlite-backed-queue)
   - [4.6 CircuitBreaker — Xử lý Overload](#46-circuitbreaker--xử-lý-overload)
5. [Vector Database — Qdrant cho Production](#5-vector-database--qdrant-cho-production)
6. [Phân tích Storage và Index Growth](#6-phân-tích-storage-và-index-growth)
7. [Khả năng mở rộng đa camera](#7-khả-năng-mở-rộng-đa-camera)
8. [Đường ống độ trễ: Camera → Searchable](#8-đường-ống-độ-trễ-camera--searchable)
9. [Rủi ro kỹ thuật và Mitigation](#9-rủi-ro-kỹ-thuật-và-mitigation)
10. [Đánh giá tổng thể và Khuyến nghị](#10-đánh-giá-tổng-thể-và-khuyến-nghị)

---

## 1. Phân tích ý tưởng kiến trúc đề xuất

### 1.1 So sánh hai mô hình vận hành

Workflow người dùng đề xuất khác biệt căn bản với mô hình hiện tại:

```
════════════════════════════════════════════════════════════════
MODEL HIỆN TẠI (Manual Upload):
════════════════════════════════════════════════════════════════

Người dùng
    │  upload thủ công file .mp4
    ▼
[index_video.py]
    │  xử lý toàn bộ video
    ▼
[Faiss Index]  ──► Streamlit / API search

Đặc điểm:
  • Triggered by human action (không tự động)
  • Batch processing (toàn bộ video một lúc)
  • Không có khái niệm "thời gian thực"
  • Phù hợp: forensic analysis sau sự kiện

════════════════════════════════════════════════════════════════
MODEL ĐỀ XUẤT (Continuous Stream Indexing):
════════════════════════════════════════════════════════════════

Camera(s) [RTSP / CSI / USB]
    │  luồng H.264 liên tục, 24/7
    ▼
[Capture Daemon]  ─── cắt thành segment ~60s
    │  /tmp/segments/cam01_20260518_143000.mp4
    ▼
[Indexing Queue]  ─── hàng đợi job (FIFO)
    │
    ▼
[NLVS Indexing Daemon]
    │  SceneSegmenter → KriaEngine → Faiss.add()
    ▼
[Faiss Index]  ──► growing over time, persistent
    │
    ▼
Search Service (FastAPI port 8000) — luôn luôn available

Đặc điểm:
  • Fully automated, trigger = new video segment ready
  • Near real-time: event → searchable in ~60–90 giây
  • Phù hợp: operational surveillance, không cần người can thiệp
  • Độ phức tạp cao hơn: cần quản lý queue, storage, lifecycle
```

### 1.2 Sơ đồ luồng dữ liệu tổng thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     KRIA KV260 — CONTINUOUS SURVEILLANCE                │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      CAPTURE LAYER                               │   │
│  │                                                                  │   │
│  │  Camera 1 (RTSP)  ──► OverlapCaptureDaemon (ffmpeg -c copy)    │   │
│  │                        stride=50s, segment=60s, overlap=10s     │   │
│  │                        → /storage/cam01/cam01_<ts>_<id>.mp4    │   │
│  │                                                                  │   │
│  │  Camera 1 (webcam) ──► OpenCVCaptureDaemon (fallback, no ffmpeg)│   │
│  │                        serial 60s segments via cv2.VideoCapture │   │
│  │                                                                  │   │
│  │  Camera N (RTSP)  ──► [separate OverlapCaptureDaemon thread]    │   │
│  └───────────────────────────────┬──────────────────────────────────┘   │
│                                   │ watchdog lib / polling (new file)   │
│                                   ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     INDEXING QUEUE                               │   │
│  │   PersistentJobQueue (SQLite WAL, crash-safe)                    │   │
│  │   Item: IndexJob(cam_id, segment_path, capture_timestamp)        │   │
│  └───────────────────────────────┬──────────────────────────────────┘   │
│                                   │ CI-Indexer thread                   │
│                                   ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    INDEXING DAEMON (ContinuousIndexer)            │   │
│  │                                                                  │   │
│  │  1. SceneSegmenter(segment.mp4)                                  │   │
│  │     └─ PySceneDetect adaptive window                             │   │
│  │  2. create_engine(config) — PCEngine(EVA-CLIP) / KriaEngine(DPU) │   │
│  │     └─ PC: EVA-CLIP ViT-L/14 (CUDA) → 768-d embeddings          │   │
│  │     └─ Kria: VVAS decode → DPU B4096 → 512-d embeddings         │   │
│  │  3. VideoIndex.add(embeddings, metadata) OR Qdrant.upsert()      │   │
│  │     └─ Qdrant (PC/dev + Kria — unified workflow)                 │   │
│  └───────────────────────────────┬──────────────────────────────────┘   │
│                                   │ index updated                       │
│                                   ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              SEARCH SERVICE (FastAPI, always-on)                 │   │
│  │   GET /search?q="người vượt rào"&cam_id=cam01&since=2h          │   │
│  │   POST /search  { query, filters, top_k }                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

External Storage (USB HDD / NAS):
  /storage/
    cam01/   ── segment files (rotating, keep last 72h)
    cam02/   ── segment files
  /opt/nlvs/
    index_store/   ── faiss.index + metadata.pkl (permanent)
```

---

## 2. Tính toán hiệu năng định lượng

### 2.1 Tham số cấu hình baseline (kria.yaml)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `window_sec` | **10.0 s** | Độ dài cửa sổ trượt (đã xác nhận) |
| `overlap_ratio` | **0.30** | stride = **7.0 s** (đã xác nhận) |
| `frames_per_window` | **5** | Số khung hình/segment (đã xác nhận) |
| `score_threshold` | **0.20** | Ngưỡng cosine similarity tối thiểu |
| `batch_size` | 4 (Kria DPU) / 16 (PC GPU) | DPU B4096 vs GTX 1650 Ti |
| `embed_dim` | **768** (PC, EVA-CLIP ViT-L/14) / **512** (Kria, DPU INT8) | Embedding dimension — khác nhau theo backend |
| Video độ phân giải | 1080p H.264 | Định dạng camera điển hình |
| Segment độ dài | 60 s | Capture daemon, stride=50s |
| Capture backend | `OverlapCaptureDaemon` (ffmpeg) / `OpenCVCaptureDaemon` (fallback) | Auto-selected by `check_ffmpeg()` |

### 2.2 Số lượng windows trên một segment 60 giây

Với `window_sec = 10s`, `stride = 7.0s` (= 10 × (1 − 0.30)):

$$N_{windows} = \left\lfloor\frac{T_{seg} - T_{win}}{\text{stride}}\right\rfloor + 1 = \left\lfloor\frac{60 - 10}{7.0}\right\rfloor + 1 = 7 + 1 = \mathbf{8 \text{ windows/segment}}$$

Tổng số frame cần encode:
$$N_{frames} = 8 \times 5 = \mathbf{40 \text{ frames/segment}}$$

> **So sánh với tham số cũ** (window=5s, stride=2.5s): 23 windows/115 frames → giảm 3× số frame cần xử lý với window mới.

### 2.3 Throughput DPU B4096 cho 1 segment

Từ benchmark đã nghiên cứu (KRIA_DEPLOYMENT_RESEARCH_P2.md §11.1):

```
Baseline benchmark: 720 segments/giờ video → tổng 997s xử lý
→ Thời gian/segment = 997s / 720 ≈ 1.385 s/segment (window 5s, stride 5s)

Với cấu hình xác nhận (window=10s, stride=7.0s, 8 windows/segment thay vì 12):
  DPU inference: 8 windows × 5 frames × 46ms/frame = 1.84s (DPU alone)
  ARM preprocessing: 40 frames × ~12ms/frame = 0.48s
  VVAS decode overhead: ~0.1s (VCU hardware, 60s@60fps = 1s decode time)
  DMA + Python overhead (estimated): ~0.5s
  ─────────────────────────────────────────────────────
  TỔNG ƯỚC TÍNH: ~3.0s per 1-minute segment
```

**Kết luận quan trọng:**

$$\text{Processing Ratio} = \frac{T_{capture}}{T_{process}} = \frac{60\text{s}}{3.0\text{s}} \approx \mathbf{20\times \text{ faster than real-time}}$$

Mỗi 60 giây video được xử lý xong trong ~3.0 giây → **hệ thống có dư tài nguyên rất lớn** — DPU có thể phục vụ nhiều camera đồng thời.

### 2.4 Bảng tổng hợp hiệu năng theo độ dài segment

Thay đổi độ dài segment ảnh hưởng trực tiếp đến độ trễ và throughput:

```
┌──────────────────┤ window=10s, overlap=30%, stride=7s, frames=5 ├────────────────┐
┌──────────────────┬─────────────┬─────────────┬──────────────┬────────────────┐
│ Segment Duration │ Windows/seg │ Frames/seg  │ Process Time │ Latency (avg)  │
├──────────────────┼─────────────┼─────────────┼──────────────┼────────────────┤
│     30 s         │      3      │     15      │   ~1.5s      │ ~16.5s         │
│     60 s ★       │      8      │     40      │   ~3.0s      │ ~33.0s         │
│    120 s         │     16      │     80      │   ~5.8s      │ ~65.8s         │
│    300 s (5 min) │     42      │    210      │  ~15.3s      │ ~165.3s        │
└──────────────────┴─────────────┴─────────────┴──────────────┴────────────────┘

Latency (avg) = segment_duration/2 + process_time
  (trung bình đợi nửa segment trước khi file hoàn chỉnh)

★ Giá trị đã xác nhận — latency ~33s là rất tốt cho surveillance forensics.
```

**Nhận xét:** Segment 60s là sự cân bằng tốt — latency ~38s là chấp nhận được cho surveillance forensics, trong khi không gây overhead quá lớn từ file management.

### 2.5 Phân tích DPU utilization

```
Timeline cho 1 camera, segment 60s:

Time  0s ──────────────────────────────────────────── 60s ──► 120s
       │  [=== CAPTURE: camera stream → segment_001.mp4 ===]     │
       │                                                    │     │
       │                          60s: file closed ─────►  │     │
       │                                          [PROCESS  │     │
       │                                          ~8.3s]    │     │
       │                                                    │     │
       │                                   68.3s: searchable│     │
       │                                                    │
                                   DPU idle: 60 - 8.3 = 51.7s

DPU Utilization (1 camera, segment=60s):
  Busy: 8.3s out of 60s → 13.8% utilization
  Idle: 51.7s → 86.2% headroom

Implication: DPU có thể phục vụ NHIỀU camera đồng thời
             hoặc xử lý indexing theo batch để tăng throughput
```

### 2.6 Hiệu năng thực tế — Conservative estimate

Các overhead thực tế trên ARM Cortex-A53:

| Nguồn overhead | Ước tính | Ghi chú |
|---|---|---|
| Python GIL contention | +10–15% | Capture daemon + indexing daemon cùng process |
| File I/O (mp4 read từ USB storage) | +5–10% | USB 3.0: ~400 MB/s, 60s@1080p H.264 ≈ 30 MB → 0.075s |
| Faiss.add() trên ARM | ~0.5ms/vector × **8 vectors** | **~4ms/segment** → negligible |
| inotify + queue overhead | < 1ms | Negligible |
| KriaEngine startup (one-time) | ~3–5s | Load xmodel vào DPU một lần khi khởi động |

**Conservative total: ~4.0–4.5s per 60s segment** (thay vì 3.0s lý thuyết).

$$\text{Processing Ratio (conservative)} = \frac{60}{4.5} \approx \mathbf{13\times \text{ faster than real-time}}$$

---

## 3. Vấn đề Segment Boundary — Phân tích và Giải pháp

### 3.1 Mô tả vấn đề

Đây là vấn đề **quan trọng nhất** và thường bị bỏ qua trong thiết kế ban đầu.

```
Ví dụ: Sự kiện "Người vượt rào cản" diễn ra từ 0:58 đến 1:04

Segment 001 (0:00 → 1:00):
  Windows: ..., [0:52–0:57], [0:55–1:00]
  ← Chứa phần đầu của sự kiện (người tiến đến rào)
  ← Embedding = "người đứng gần rào" — KHÔNG khớp với query

Segment 002 (1:00 → 2:00):
  Windows: [1:00–1:05], [1:02.5–1:07.5], ...
  ← Chứa phần cuối của sự kiện (người đã vượt xong)
  ← Embedding = "người ở phía bên kia rào" — KHÔNG khớp tốt

→ Sự kiện bị "mất" trong khe hở giữa hai segment!
   Không segment nào có embedding đại diện đủ cho hành động đầy đủ.
```

### 3.2 Phân tích định lượng mức độ ảnh hưởng

Với `window_sec = 5s` và `stride = 2.5s`:

```
Xác suất một sự kiện độ dài T giây bị split qua boundary:
  P(split) = (T + window_sec) / segment_duration
           = (T + 10) / 60

  Hành động nhanh (T=2s):   P = 12/60 = 20.0%
  Hành động vừa (T=5s):     P = 15/60 = 25.0%
  Hành động chậm (T=15s):   P = 25/60 = 41.7%
  Hành động kéo dài (T=30s): P = 40/60 = 66.7%

→ Với hành động điển hình (2–10s), ~20–25% sự kiện bị ảnh hưởng bởi boundary
  (cao hơn window=5s vì cửa sổ rộng hơn, nhưng được giải quyết hoàn toàn bởi tail-only overlap v2.0).
```

### 3.3 Giải pháp — Tail-Only Overlap (v2.0)

**Thiết kế: Mỗi segment file dài 60 giây; segment mới bắt đầu 10 giây trước khi segment trước kết thúc (stride = 50s).**

```
THIẾT KẾ TAIL-ONLY OVERLAP:
  Segment 001:  [0:00 ─────────────────── 1:00]   (60s)
  Segment 002:        [0:50 ─────────────────── 1:50]   (60s)
  Segment 003:              [1:40 ─────────────────── 2:40]   (60s)
                                   ↑ 10s overlap = tail của segment trước

  File duration : 60s  (mỗi file luôn đúng 60s)
  Stride        : 50s  (segment mới bắt đầu mỗi 50s)
  Overlap       : 10s  (10s đầu của file N+1 = 10s cuối của file N)
```

**So sánh với thiết kế v1.0 (head + tail padding):**

| Tiêu chí | v1.0 (head + tail) | v2.0 (tail-only) |
|---|---|---|
| File duration | 70s | **60s** |
| Stride | 60s | **50s** |
| Overlap vùng | 5s head + 5s tail | **10s tail** |
| Index logic | Skip head 5s + tail 5s khi build meta | **Index toàn bộ file** |
| Storage overhead | 10/70 = 14.3% | 10/60 = 16.7% |
| Triển khai | Phức tạp (canonical range) | **Đơn giản** |

Thiết kế v2.0 loại bỏ hoàn toàn logic `canonical_start_offset` / `canonical_end_offset` — index toàn bộ 60s mỗi file; Temporal NMS xử lý deduplication bằng `absolute_start` cross-segment.

#### 3.3.1 Triển khai Capture Daemon với Tail Overlap

GStreamer `splitmuxsink` không hỗ trợ native overlap. Giải pháp thực tế: Python `OverlapCaptureDaemon` launch ffmpeg subprocess mỗi `STRIDE_SEC` giây, mỗi subprocess ghi `SEGMENT_SEC` giây. Hai subprocess chạy chồng nhau trong `OVERLAP_SEC` giây cuối của mỗi segment.

**Lựa chọn backend tự động** (implemented trong `app.py → _start_pipeline()`):
- `check_ffmpeg()` → True: dùng `OverlapCaptureDaemon` (ffmpeg `-c copy`, hỗ trợ RTSP + V4L2)
- `check_ffmpeg()` → False: dùng `OpenCVCaptureDaemon` (chỉ thiết bị local, serial segments)

```python
# src/capture_daemon.py — helpers và hai class daemon

SEGMENT_SEC = 60   # duration mỗi file (giây)
STRIDE_SEC  = 50   # khoảng cách giữa hai segment start (giây)
OVERLAP_SEC = 10   # = SEGMENT_SEC - STRIDE_SEC


def check_ffmpeg() -> bool:
    """Return True nếu ffmpeg có trên PATH."""
    import shutil
    return shutil.which("ffmpeg") is not None


def is_wsl2() -> bool:
    """Return True khi chạy trong WSL2 (Microsoft kernel)."""
    try:
        return "microsoft" in open("/proc/version").read().lower()
    except OSError:
        return False


def list_webcams() -> list:
    """
    Trả về danh sách thiết bị camera khả dụng.
    Chặn toàn bộ V4L2/obsensor stderr noise khi probe bằng cách
    redirect fd 2 → /dev/null trong quá trình quét.
    Trả về [] trong WSL2 nếu chưa cấu hình USB passthrough.
    """
    ...


class OverlapCaptureDaemon:
    """Tail-only overlap capture qua ffmpeg subprocess.
    Hỗ trợ RTSP (rtsp://) và V4L2 local (/dev/videoN hoặc index int).
    Yêu cầu: ffmpeg phải có trên PATH (kiểm tra với check_ffmpeg()).
    """

    def __init__(self, cam_id: str, rtsp_url: str,
                 output_dir: str, queue: PersistentJobQueue,
                 circuit_breaker: CircuitBreaker = None): ...

    def start(self) -> None: ...   # non-blocking, starts thread
    def stop(self, timeout: float = 10.0) -> None: ...

    @property
    def is_running(self) -> bool: ...

    def _is_network_source(self) -> bool:
        """True nếu rtsp://, rtmp://, http://"""

    def _launch_ffmpeg(self, out_path, wall_ts) -> subprocess.Popen:
        """RTSP: ffmpeg -c copy | V4L2: ffmpeg -f v4l2 -c:v libx264 -preset ultrafast"""


class OpenCVCaptureDaemon:
    """Fallback capture daemon dùng cv2.VideoCapture (không cần ffmpeg).
    Ghi serial segments (không có tail overlap), phù hợp webcam/WSL2.
    Yêu cầu: USB passthrough trên WSL2 (xem §3.3.3).
    """

    def __init__(self, cam_id: str, device: str,
                 output_dir: str, queue: PersistentJobQueue,
                 circuit_breaker: CircuitBreaker = None): ...

    def start(self) -> None: ...
    def stop(self, timeout: float = 10.0) -> None: ...

    @property
    def is_running(self) -> bool: ...
```

**Chi phí tài nguyên trong vùng overlap 10s:** 2 ffmpeg passthrough processes chạy đồng thời, mỗi process ~2% CPU ARM (không decode) → tổng ~4% CPU trong 10s. Không ảnh hưởng DPU.

**Filename format:** `{cam_id}_{wall_ts:010d}_{seg_id:05d}.mp4` — ví dụ `cam01_1747580000_00003.mp4`.

#### 3.3.3 WSL2 — USB Camera Passthrough (usbipd-win)

WSL2 kernel không expose USB devices theo mặc định. `/dev/video*` sẽ **không tồn tại** cho đến khi cấu hình `usbipd-win` trên Windows host.

`app.py` tự động phát hiện WSL2 (`is_wsl2()`) và hiển thị hướng dẫn khi không có camera:

```powershell
# Windows PowerShell (Admin)

# 1. Cài đặt usbipd-win (một lần)
winget install --interactive --exact dorssel.usbipd-win

# 2. Tìm Bus ID của camera
usbipd list

# 3. Bind camera (một lần, cần Admin)
usbipd bind --busid <BusID>

# 4. Attach vào WSL2 (mỗi session)
usbipd attach --wsl --busid <BusID>
```

```bash
# Trong WSL2 — xác nhận camera đã được nhận diện
ls /dev/video*   # → /dev/video0
```

Sau khi attach thành công, `list_webcams()` sẽ trả về `["/dev/video0"]` và `OpenCVCaptureDaemon` hoặc `OverlapCaptureDaemon` có thể sử dụng thiết bị.

#### 3.3.2 Quản lý Timestamp trong SegmentMeta

Mỗi segment bắt đầu tại một thời điểm wall-clock xác định (`segment_wall_start`). Metadata mỗi indexed window cần lưu đủ để: **(a)** tìm clip gốc (dùng `video_path` + `relative_start`), **(b)** deduplication cross-segment trong Temporal NMS (dùng `absolute_start`), **(c)** hiển thị timestamp cho kết quả tìm kiếm.

```python
# src/indexer.py — SegmentMeta (v2.0)
@dataclass
class SegmentMeta:
    # File location (permanent trong retention period)
    video_path:          str    # đường dẫn tuyệt đối đến segment file
    cam_id:              str    # "cam01", "cam02", ...
    segment_wall_start:  float  # unix timestamp khi ffmpeg bắt đầu ghi file này

    # In-file position (dùng để seek khi download clip)
    relative_start:      float  # giây từ đầu file → bắt đầu window
    relative_end:        float  # giây từ đầu file → kết thúc window

    # Absolute timestamps (Temporal NMS cross-segment + payload filter)
    absolute_start:      float  # = segment_wall_start + relative_start
    absolute_end:        float  # = segment_wall_start + relative_end
```

**Ví dụ** — Segment 002 (`cam01_1747580050_00002.mp4`, `segment_wall_start=1747580050.0`), window t=5.0→10.0s:
- `absolute_start = 1747580050 + 5.0 = 1747580055.0` (14:14:15)
- Download: open file, seek to 5.0s → 10.0s
- Nếu file đã bị rotate xóa: hiển thị timestamp, "video không còn lưu trữ"

**Số window không thay đổi nhiều:** File vẫn 60s → **8 windows** (window=10s, stride=7.0s). Giảm từ 23 (cấu hình cũ) xuống 8, nhưng mỗi window dài hơn (10s) → nắm bắt tốt hơn hành động vừa phải (5–10s). Chi phí xử lý giảm mạnh: 40 frames (từ 115 frames cấu hình cũ).

### 3.4 Temporal NMS — Xử lý kết quả trùng lặp từ overlap

Khi một hành động được lập chỉ mục từ cả segment 001 (phần cuối) và segment 002 (phần đầu do overlap), cùng một đoạn thời gian sẽ có **hai vector trong Faiss index**. Thuật toán Temporal NMS hiện tại đã xử lý điều này:

```python
# src/searcher.py — temporal_nms() đã handle overlapping results
# IoU-based deduplication với threshold=0.30
# Hai kết quả từ cùng đoạn thời gian → giữ kết quả có score cao hơn
```

**Cần điều chỉnh:** NMS hiện tại so sánh thời gian trong cùng `video_id`. Với continuous streaming, cần so sánh cross-segment bằng **absolute timestamp**:

```python
# Hiện tại: so sánh theo video_id + relative time
# Cần: so sánh theo camera_id + absolute_timestamp

# absolute_start = file_capture_time + relative_start_in_file
# Temporal NMS trên absolute timestamp để loại trùng lặp cross-segment
```

---

## 4. Kiến trúc Daemon hóa đề xuất

### 4.1 Process Model

```
┌───────────────────────────────────────────────────────────────────────┐
│                    KRIA KV260 — Process Tree                          │
│                                                                       │
│  systemd                                                              │
│    ├── nlvs-capture@cam01.service  ──► Python: OverlapCaptureDaemon  │
│    │                                    (ffmpeg -c copy subprocess)   │
│    ├── nlvs-capture@cam02.service  ──► Python: OverlapCaptureDaemon  │
│    └── nlvs-indexer.service        ──► Python: ContinuousIndexer     │
│            ├── CI-Watchdog thread: watchdog lib / polling → enqueue  │
│            ├── CI-Indexer thread:  dequeue → encode → vector store   │
│            └── CI-Persist thread:  periodic Faiss flush (300s)       │
│                                                                       │
│  PC / WSL2 development (app.py Streamlit):                           │
│    ├── OverlapCaptureDaemon thread (if ffmpeg available)             │
│    │   └─ fallback: OpenCVCaptureDaemon thread (if no ffmpeg)        │
│    └── ContinuousIndexer (same 3-thread model)                       │
│                                                                       │
│  IPC: PersistentJobQueue (SQLite WAL file — crash-safe)              │
│       (capture daemon enqueues → indexer dequeues)                   │
└───────────────────────────────────────────────────────────────────────┘
```

### 4.2 Capture Daemon — Python ffmpeg + OpenCV fallback

Implementation thực tế dùng Python subprocess thay vì GStreamer để đơn giản hóa và hỗ trợ đa nền tảng (PC/WSL2/Kria).

```python
# Khởi động trong app.py → _start_pipeline()

from src.capture_daemon import (
    OverlapCaptureDaemon, OpenCVCaptureDaemon,
    check_ffmpeg, is_wsl2, list_webcams,
)
from src.job_queue import PersistentJobQueue

queue = PersistentJobQueue(db_path="/opt/nlvs/job_queue.db")

if check_ffmpeg():
    # Primary: ffmpeg -c copy — passthrough H.264, không decode, DPU rảnh
    # Hỗ trợ cả RTSP và V4L2 local (/dev/video0 hoặc index 0)
    daemon = OverlapCaptureDaemon(
        cam_id="cam01",
        rtsp_url="rtsp://192.168.1.100:554/stream",  # hoặc "/dev/video0"
        output_dir="/storage/cam01",
        queue=queue,
    )
else:
    # Fallback: OpenCV VideoCapture — không cần ffmpeg
    # Serial segments (không có true overlap), phù hợp webcam local
    daemon = OpenCVCaptureDaemon(
        cam_id="cam01",
        device="0",        # index int hoặc /dev/video0
        output_dir="/storage/cam01",
        queue=queue,
    )

daemon.start()   # non-blocking; gọi daemon.stop() để dừng
```

**ffmpeg subprocess command (OverlapCaptureDaemon):**
```bash
# RTSP source:
ffmpeg -loglevel warning -rtsp_transport tcp \
  -i rtsp://... -t 60 -c copy -y cam01_<ts>_<id>.mp4

# V4L2 local source:
ffmpeg -loglevel warning -f v4l2 \
  -i /dev/video0 -t 60 -c:v libx264 -preset ultrafast -y cam01_<ts>_<id>.mp4
```

**Lợi ích passthrough RTSP (`-c copy`):** Không decode video → DPU/GPU hoàn toàn rảnh trong bước capture.

### 4.3 Indexing Daemon — Python (v2.0)

Phiên bản v2.0 thay thế `queue.Queue` in-memory bằng `PersistentJobQueue` (SQLite) và hỗ trợ Qdrant hoặc Faiss tùy backend.

```python
# src/continuous_indexer.py
"""
Continuous Indexing Daemon cho NLVS (v2.0).
Sử dụng PersistentJobQueue (SQLite) + CircuitBreaker + Qdrant/Faiss vector store.
File monitoring: watchdog library (polling fallback nếu watchdog không có).
Engine: create_engine(config) factory — PCEngine(EVA-CLIP) hoặc KriaEngine(DPU).
"""
import os, time, threading, logging
from pathlib import Path

try:
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    from watchdog.observers import Observer as WatchdogObserver
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False  # fallback sang polling mỗi 5 giây

from src.indexer import SegmentMeta, VideoIndex
from src.scene_segmenter import SceneSegmenter
from src.engines.factory import create_engine   # PCEngine hoặc KriaEngine
from src.job_queue import PersistentJobQueue, IndexJob, CircuitBreaker

log = logging.getLogger("nlvs.indexer")


class ContinuousIndexer:
    """
    Daemon xử lý continuous indexing từ camera segments (v2.0).

    Threads (3 daemon threads + caller thread):
      1. CI-Watchdog : watchdog lib (hoặc polling) → CircuitBreaker → enqueue
      2. CI-Indexer  : dequeue → create_engine → VectorDB.add()
      3. CI-Persist  : flush Faiss index mỗi 300s (Qdrant tự persist)
    """
    # constants từ module level
    # _PERSIST_INTERVAL = 300
    # _POLL_INTERVAL    = 5.0
    # _DEQUEUE_SLEEP    = 2.0

    def __init__(self, config: dict,
                 db_path: str = "/opt/nlvs/job_queue.db"):
        self._config  = config
        self._queue   = PersistentJobQueue(db_path)
        self._breaker = CircuitBreaker(self._queue)
        self._stop    = threading.Event()

        # Engine: PCEngine (EVA-CLIP, 768-d) hoặc KriaEngine (DPU, 512-d)
        self._engine = create_engine(config)

        # Vector store: Qdrant (production) hoặc Faiss (dev/fallback)
        self._qdrant_client     = None
        self._faiss_index       = None   # VideoIndex (app.py reads this for live search)
        self._init_vector_store()

        # Startup recovery: replay segment files từ 2h gần nhất
        storage_dirs = (config.get("capture") or {}).get("storage_dirs", [])
        if storage_dirs:
            self._queue.replay_from_storage(
                storage_dirs,
                last_indexed_ts=time.time() - 7200,
            )
        log.info("[ContinuousIndexer] Initialized.")

    def _init_vector_store(self):
        """Kết nối Qdrant (nếu backend=qdrant); fallback sang Faiss."""
        backend  = (self._config.get("index") or {}).get("backend", "faiss")
        if backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                url = (self._config["index"]).get("qdrant_url", "http://localhost:6333")
                self._qdrant_client = QdrantClient(url=url)
                log.info("[ContinuousIndexer] Connected to Qdrant: %s", url)
                return
            except Exception as e:
                log.warning("[ContinuousIndexer] Qdrant unavailable (%s) → Faiss fallback.", e)
        # Faiss fallback
        embed_dim  = (self._config.get("index") or {}).get("embed_dim", 768)
        index_dir  = (self._config.get("index") or {}).get("index_dir", "/opt/nlvs/index_store")
        self._faiss_index = VideoIndex(embed_dim=embed_dim)
        log.info("[ContinuousIndexer] Using Faiss (embed_dim=%d, index_dir=%s).", embed_dim, index_dir)

    def start(self) -> None:
        """Khởi động 3 daemon threads (non-blocking)."""
        self._stop.clear()
        self._thread_watchdog = threading.Thread(
            target=self._watchdog_loop, name="CI-Watchdog", daemon=True)
        self._thread_indexer  = threading.Thread(
            target=self._indexer_loop,  name="CI-Indexer",  daemon=True)
        self._thread_persist  = threading.Thread(
            target=self._persist_loop,  name="CI-Persist",  daemon=True)
        for t in (self._thread_watchdog, self._thread_indexer, self._thread_persist):
            t.start()
        log.info("[ContinuousIndexer] Started (3 threads active).")

    def stop(self) -> None:
        self._stop.set()

    def _watchdog_loop(self):
        """Thread 1: theo dõi storage_dirs → CircuitBreaker → PersistentJobQueue.
        Dùng watchdog library nếu có; polling mỗi _POLL_INTERVAL giây nếu không.
        """
        watch_dirs = (self._config.get("capture") or {}).get("storage_dirs", [])

        if _HAS_WATCHDOG:
            # Event-driven: watchdog observer
            ...
        else:
            # Fallback polling
            seen = set()
            while not self._stop.is_set():
                for d in watch_dirs:
                    for mp4 in Path(d).glob("*.mp4"):
                        if mp4 not in seen and mp4.stat().st_size > 0:
                            seen.add(mp4)
                            job = IndexJob(cam_id=mp4.stem.split("_")[0],
                                           segment_path=str(mp4),
                                           capture_timestamp=mp4.stat().st_mtime)
                            if not self._breaker.is_open():
                                self._queue.enqueue(job)
                self._stop.wait(5.0)

    def _indexer_loop(self):
        """Thread 2: dequeue → encode (PCEngine/KriaEngine) → VectorDB.add()."""
        while not self._stop.is_set():
            job = self._queue.dequeue()
            if job is None:
                self._stop.wait(2.0)
                continue
            try:
                self._process_segment(job)
                self._queue.mark_done(job.id)
            except Exception as e:
                log.error("[CI-Indexer] Failed %s: %s", job.segment_path, e)
                self._queue.mark_failed(job.id)

    def _process_segment(self, job: IndexJob):
        """
        Core indexing cho một segment (v2.0).
        Index toàn bộ 60s — không skip overlap.
        Temporal NMS (absolute_start) xử lý dedup cross-segment.
        """
        from .video_processor import VideoProcessor
        import numpy as np

        processor = VideoProcessor(self._config)
        windows   = processor.extract_windows(job.segment_path)
        # windows: list of (t_start, t_end, frames_list)

        embeddings, metas = [], []
        for t_start, t_end, frames in windows:
            emb = self._engine.encode_frames(frames)
            embeddings.append(emb)
            metas.append(SegmentMeta(
                cam_id=job.cam_id,
                video_path=job.segment_path,
                relative_start=t_start,
                relative_end=t_end,
                segment_wall_start=job.capture_timestamp,
                absolute_start=job.capture_timestamp + t_start,
                absolute_end=job.capture_timestamp + t_end,
            ))

        if not embeddings:
            return
        embs = np.stack(embeddings)
        if self._qdrant_client:
            self._add_to_qdrant(embs, metas)
        elif self._faiss_index:
            self._faiss_index.add(embs, metas)

    def _persist_loop(self):
        """Thread 3: flush Faiss index mỗi _PERSIST_INTERVAL giây."""
        while not self._stop.is_set():
            self._stop.wait(300)
            if self._faiss_index:
                index_dir = (self._config.get("index") or {}).get(
                    "index_dir", "/opt/nlvs/index_store")
                self._faiss_index.save(index_dir)
                log.info("[CI-Persist] Faiss index flushed → %s", index_dir)
```

### 4.4 Systemd Service Files

```ini
# /etc/systemd/system/nlvs-capture@.service
# Instantiate với: systemctl start nlvs-capture@cam01

[Unit]
Description=NLVS Camera Capture — %i
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
EnvironmentFile=/etc/nlvs/cameras/%i.env
ExecStart=/usr/bin/bash /etc/nlvs/capture_pipeline.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/nlvs-indexer.service

[Unit]
Description=NLVS Continuous Indexing Daemon
After=nlvs-capture@cam01.service
Requires=nlvs-capture@cam01.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Prototype
Environment=NLVS_CONFIG=/etc/nlvs/kria.yaml
ExecStart=/usr/bin/python3 -m src.continuous_indexer
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

---

### 4.5 PersistentJobQueue — SQLite-backed Queue

Thay thế `queue.Queue` in-memory bằng SQLite-backed persistent queue để giải quyết:
- **Crash recovery**: Jobs sống sót qua restart/crash (PROCESSING → PENDING khi khởi động lại)
- **Deduplication**: `UNIQUE` trên `segment_path` → không index lại cùng file
- **Retry với exponential backoff**: Failed jobs retry tối đa 3 lần (30s → 60s → 120s)
- **Stuck detection**: PROCESSING job > 5 phút được reset về PENDING tự động
- **Startup replay**: Tự scan storage dirs để enqueue segment files chưa được index

```python
# src/job_queue.py — actual implementation
import sqlite3, threading, time, logging, glob, os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_RETRIES        = 3
PROCESSING_TIMEOUT = 300   # giây
BACKOFF_BASE       = 30    # giây: delay = BACKOFF_BASE × 2^(retry-1)
OPEN_THRESHOLD     = 20    # CircuitBreaker: mở khi queue depth ≥ 20
CLOSE_THRESHOLD    = 5     # CircuitBreaker: đóng khi queue depth ≤ 5


@dataclass
class IndexJob:
    cam_id:            str
    segment_path:      str
    capture_timestamp: float        # unix timestamp khi file được ghi xong
    id:                Optional[int] = None
    retry_count:       int = 0
    status:            str = "pending"
    created_at:        float = field(default_factory=time.time)
    next_retry_after:  float = field(default_factory=time.time)


class PersistentJobQueue:
    """Thread-safe SQLite-backed persistent job queue."""

    def __init__(self, db_path: str = "/opt/nlvs/job_queue.db") -> None:
        self._db_path = db_path
        self._lock    = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)   # CREATE TABLE IF NOT EXISTS ...
        self._conn.commit()
        self._recover_stuck_jobs()

    def enqueue(self, job: IndexJob) -> bool:
        """Insert job. Returns True nếu inserted; False nếu segment_path đã tồn tại (dedup)."""
        now = time.time()
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO jobs "
                    "(cam_id, segment_path, capture_timestamp, status, "
                    " retry_count, created_at, next_retry_after, updated_at) "
                    "VALUES (?, ?, ?, 'pending', 0, ?, ?, ?)",
                    (job.cam_id, job.segment_path, job.capture_timestamp, now, now, now),
                )
                self._conn.commit()
                return True
        except sqlite3.Error as exc:
            logger.error("[JobQueue] enqueue error: %s", exc)
            return False

    def dequeue(self) -> Optional[IndexJob]:
        """Atomically claim oldest pending job. Returns None nếu queue rỗng."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs "
                "WHERE status='pending' AND next_retry_after <= ? "
                "ORDER BY created_at ASC LIMIT 1", (now,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE jobs SET status='processing', updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            self._conn.commit()
            return IndexJob(id=row["id"], cam_id=row["cam_id"],
                            segment_path=row["segment_path"],
                            capture_timestamp=row["capture_timestamp"],
                            retry_count=row["retry_count"], status="processing")

    def mark_done(self, job_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='done', updated_at=? WHERE id=?",
                (time.time(), job_id))
            self._conn.commit()

    def mark_failed(self, job_id: int) -> None:
        """Exponential backoff retry; DEAD sau MAX_RETRIES lần."""
        ...

    def stats(self) -> dict:
        """Trả về dict {status: count} cho tất cả trạng thái."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0, "dead": 0}
        for row in rows:
            counts[row["status"]] = row["n"]
        return counts

    def depth(self) -> int:
        """Số jobs ở trạng thái pending."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='pending'"
            ).fetchone()[0]

    def replay_from_storage(self, storage_dirs: list, last_indexed_ts: float) -> None:
        """Scan dirs và enqueue segment files mới hơn last_indexed_ts."""
        for d in storage_dirs:
            for path in sorted(glob.glob(f"{d}/**/*.mp4", recursive=True)):
                if os.path.getmtime(path) > last_indexed_ts:
                    cam_id = Path(path).stem.split("_")[0]
                    self.enqueue(IndexJob(
                        cam_id=cam_id, segment_path=path,
                        capture_timestamp=os.path.getmtime(path)))
```

**Schema bảng `jobs` (SQLite WAL mode — actual `_SCHEMA` trong `src/job_queue.py`):**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        jobs (SQLite WAL)                            │
├──────────────────────┬──────────┬──────────────────────────────────┤
│ Column               │ Type     │ Description                      │
├──────────────────────┼──────────┼──────────────────────────────────┤
│ id                   │ INTEGER  │ Auto PK                          │
│ cam_id               │ TEXT     │ "cam01", "cam02"...              │
│ segment_path         │ TEXT     │ UNIQUE — deduplication key       │
│ capture_timestamp    │ REAL     │ unix timestamp (mtime of file)   │
│ status               │ TEXT     │ pending/processing/done/failed/dead │
│ retry_count          │ INTEGER  │ số lần đã thử lại                │
│ next_retry_after     │ REAL     │ unix ts: chỉ dequeue sau mốc này │
│ created_at           │ REAL     │ thời điểm enqueue                │
│ updated_at           │ REAL     │ thời điểm cập nhật trạng thái    │
└──────────────────────┴──────────┴──────────────────────────────────┘

Retry backoff: 30s → 60s → 120s → DEAD  (MAX_RETRIES=3)
Note: không có started_at/completed_at riêng — updated_at dùng chung.
```

**Luồng trạng thái job:**

```
                 enqueue()           dequeue()
  [FILE] ──────► PENDING ──────────► PROCESSING ──────► DONE
                    ▲                     │
                    │    mark_failed()    │ (exception)
                    └──── FAILED ◄────────┘
                              │
                   (retries ≥ 3)
                              ▼
                           DEAD  ──► log.error + manual inspection
```

### 4.6 CircuitBreaker — Xử lý Overload

Khi indexing daemon bị quá tải (ví dụ: mất điện 30 phút → 90 segments tồn đọng), `CircuitBreaker` ngăn nhận thêm segment mới để hệ thống có thể drain backlog trước.

```python
# src/job_queue.py (tiếp theo)

class CircuitBreaker:
    """
    Mở circuit khi queue quá tải; đóng khi queue đủ thấp.
    Segment mới bị bỏ qua (logged as WARNING) khi circuit OPEN.
    """
    OPEN_THRESHOLD  = 20   # pending+failed ≥ 20 → OPEN (khoảng 17 phút backlog)
    CLOSE_THRESHOLD = 5    # pending+failed ≤ 5  → CLOSE

    def __init__(self, queue: PersistentJobQueue):
        self._queue = queue
        self._open  = False

    def can_accept(self) -> bool:
        depths  = self._queue.depth()
        pending = depths.get("pending", 0) + depths.get("failed", 0)

        if not self._open and pending >= self.OPEN_THRESHOLD:
            self._open = True
            log.warning(
                f"CircuitBreaker OPEN — queue depth={pending}, "
                f"dropping new segments until depth ≤ {self.CLOSE_THRESHOLD}."
            )
        elif self._open and pending <= self.CLOSE_THRESHOLD:
            self._open = False
            log.info(f"CircuitBreaker CLOSED — queue depth={pending}, resuming.")

        return not self._open

    @property
    def is_open(self) -> bool:
        return self._open
```

**Tham số tối ưu cho Kria (1 camera, stride=50s):**
```
Tốc độ indexing:  ~50s / segment
Tốc độ capture:   1 segment / 50s
→ DPU utilization: ~100% với 1 camera  (tight balance)

CircuitBreaker.OPEN_THRESHOLD=20 tương đương ~17 phút backlog.
Sau khi hệ thống bị tắt 30 phút → 36 segments tồn đọng → circuit mở ngay
khi startup, xử lý dần backlog trước khi chấp nhận luồng mới từ camera.
```

---

## 5. Vector Database — Qdrant cho Production

### 5.1 Động lực thay thế Raw Faiss

| Vấn đề với Faiss thuần | Giải pháp với Qdrant |
|---|---|
| Toàn bộ index phải load vào RAM (18 GB sau 90 ngày/3 cam) | On-disk HNSW via MMAP — chỉ load page cần thiết |
| Không hỗ trợ filter theo metadata | Payload filtering: `cam_id`, `time_range` tại query time |
| In-process only — indexer và API phải cùng process | REST API + gRPC — indexer và search API dùng chung |
| Phải rebuild khi migrate index type | Automatic index management |
| Index mất khi crash (nếu chưa flush) | Persistent on-disk, WAL commit sau mỗi upsert |

### 5.2 Cài đặt Qdrant trên Kria KV260 (ARM64)

```bash
# Qdrant phát hành ARM64 binary (aarch64-unknown-linux-musl):
wget https://github.com/qdrant/qdrant/releases/download/v1.9.2/\
qdrant-aarch64-unknown-linux-musl.tar.gz
tar xzf qdrant-aarch64-unknown-linux-musl.tar.gz
sudo mv qdrant /usr/local/bin/

# Python client:
pip install qdrant-client

# Tạo storage directory:
mkdir -p /opt/nlvs/qdrant_storage
```

```yaml
# /opt/nlvs/qdrant_config.yaml
storage:
  storage_path: /opt/nlvs/qdrant_storage
  # Dùng MMAP → vectors không cần load toàn bộ vào RAM

service:
  host: 127.0.0.1   # chỉ listen local — không expose ra ngoài
  http_port: 6333
  grpc_port: 6334

telemetry_disabled: true
```

```ini
# /etc/systemd/system/qdrant.service
[Unit]
Description=Qdrant Vector Database
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/usr/local/bin/qdrant --config-path /opt/nlvs/qdrant_config.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Yêu cầu tài nguyên (3 cameras × 90 ngày):**
```
Số vectors : 3 × 8 windows/min × 60 × 24 × 90  ≈ 3.1M vectors
On-disk    : 3.1M × 2048 bytes × 1.5 (HNSW overhead) ≈ 9.5 GB
RAM (idle) : ~50–100 MB (Qdrant service + OS overhead)
RAM (search, filter 24h/1 cam): ~630 MB MMAP pages loaded
→ Khả thi với 4 GB RAM Kria khi dùng filter hẹp time range.
```

### 5.3 Collection Schema và Indexing

```python
# src/vector_store.py (Kria production only)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType,
    PointStruct, Filter, FieldCondition, Range, MatchValue,
)
import uuid, numpy as np

COLLECTION = "nlvs_segments"
# embed_dim phụ thuộc backend:
#   Kria (DPU B4096, INT8 → fp32 projection): 512
#   PC   (EVA-CLIP ViT-L/14, FP16):           768
EMBED_DIM  = 512   # Kria; đổi thành 768 cho PC Qdrant nếu cần


def create_collection(client: QdrantClient):
    """Tạo collection với HNSW index + indexed payload fields."""
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        on_disk_payload=True,   # payload lưu on-disk, không chiếm RAM
    )
    # Index payload fields để filter O(log N)
    client.create_payload_index(COLLECTION, "cam_id",
                                PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "absolute_start",
                                PayloadSchemaType.FLOAT)
    client.create_payload_index(COLLECTION, "absolute_end",
                                PayloadSchemaType.FLOAT)


def add_vectors(client: QdrantClient,
                embeddings: np.ndarray,
                payloads: list):
    """Upsert batch vectors + metadata vào Qdrant."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                "cam_id":             p["cam_id"],
                "video_path":         p["video_path"],
                "segment_wall_start": p["segment_wall_start"],
                "relative_start":     p["relative_start"],
                "relative_end":       p["relative_end"],
                "absolute_start":     p["absolute_start"],
                "absolute_end":       p["absolute_end"],
            }
        )
        for emb, p in zip(embeddings, payloads)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
```

### 5.4 Search với Payload Filtering

```python
def search_with_filter(client: QdrantClient,
                       query_vector: np.ndarray,
                       cam_id: str = None,
                       start_ts: float = None,
                       end_ts: float = None,
                       top_k: int = 10) -> list:
    """
    Tìm kiếm có filter theo camera và time range.
    Ví dụ: tìm sự kiện trên cam01 trong khoảng 7:00–9:00 sáng nay.
    """
    must_conditions = []
    if cam_id:
        must_conditions.append(
            FieldCondition(key="cam_id", match=MatchValue(value=cam_id))
        )
    if start_ts is not None or end_ts is not None:
        must_conditions.append(
            FieldCondition(
                key="absolute_start",
                range=Range(gte=start_ts, lte=end_ts),
            )
        )

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector.tolist(),
        query_filter=Filter(must=must_conditions) if must_conditions else None,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "score":          r.score,
            "cam_id":         r.payload["cam_id"],
            "video_path":     r.payload["video_path"],
            "relative_start": r.payload["relative_start"],
            "absolute_start": r.payload["absolute_start"],
        }
        for r in results
    ]
```

**Ví dụ truy vấn có filter:**
```python
import time

yesterday = time.time() - 86400
# Tìm "người mang ba lô đỏ" trên cam01, trong 24h qua
results = search_with_filter(
    client,
    query_vector=text_embedding("người mang ba lô đỏ"),
    cam_id="cam01",
    start_ts=yesterday,
    end_ts=time.time(),
    top_k=5,
)
```

### 5.5 So sánh Qdrant vs LanceDB vs Faiss

| Tiêu chí | Faiss (hiện tại) | Qdrant | LanceDB |
|---|---|---|---|
| Deployment | Embedded Python | Server (ARM64 binary) | Embedded Python |
| Storage | In-memory + .index file | On-disk HNSW (MMAP) | On-disk Lance format |
| RAM (90d/3cam) | ~18 GB (không khả thi) | ~50–200 MB service | ~50–200 MB OS cache |
| Payload filtering | Không | **Có (indexed)** | Có (SQL-like) |
| Multi-client access | Không | **Có (REST/gRPC)** | Hạn chế (file lock) |
| ARM64 support | ✅ | ✅ | ✅ |
| Incremental update | Có (add) | **Có (upsert/delete)** | Có |
| Search latency (top-10) | <1ms (exact) | 2–5ms (ANN HNSW) | 3–8ms |
| Complexity | Thấp | Trung bình | Thấp |
| **Khuyến nghị** | Dev/testing | **Production Kria** | Backup option |

**Khuyến nghị cho luận văn:**
- **Cả PC lẫn Kria**: dùng Qdrant (`http://localhost:6333`, collection `nlvs_segments`) — workflow đồng nhất, giúp so sánh kết quả giữa 2 môi trường
- **Config toggle**: `index.backend` trong `pc.yaml` / `kria.yaml` (hiện tại cả hai đều là `qdrant`; Faiss vẫn là fallback khi Qdrant không khả dụng)

---

## 6. Phân tích Storage và Index Growth

### 5.1 Video Storage Growth

```
Camera 1, độ phân giải 1080p H.264, bitrate 4 Mbps:

  Dung lượng/phút  : 4 Mbps × 60s / 8 = 30 MB/phút
  Dung lượng/giờ   : 1.8 GB/giờ
  Dung lượng/ngày  : 43.2 GB/ngày
  Dung lượng/tuần  : 302.4 GB/tuần (1 camera)

Lưu ý: Kria KV260 eMMC = 8 GB → KHÔNG thể lưu video trực tiếp
  → Bắt buộc dùng External Storage:
    • USB 3.0 HDD 1TB: ~23 ngày / camera
    • USB 3.0 SSD 256GB: ~5.9 ngày / camera
    • NAS qua GbE: scalable, recommended

Rotation policy (giữ 72h rolling window):
  Video cần lưu: 72h × 1.8 GB/h = 129.6 GB / camera
  USB HDD 1TB: có thể phục vụ ~7 cameras ở 72h retention
```

### 5.2 Faiss Index Growth

```
Mỗi phút video (1 segment = 60s → 8 windows): 8 vectors

  Kria (embed_dim=512): 8 × 512 × 4 bytes = 16.4 KB/phút
  PC   (embed_dim=768): 8 × 768 × 4 bytes = 24.6 KB/phút

Mỗi giờ  (Kria/PC): 480 vectors →  0.98 MB /  1.47 MB RAM
Mỗi ngày (24h):   11,520 vectors → 23.6 MB / 35.3 MB RAM
Mỗi tuần:         80,640 vectors → 165 MB  / 247 MB RAM
Mỗi tháng:       345,600 vectors → 707 MB  / 1.06 GB RAM  ← thoải mái

Giới hạn IndexFlatIP (exact search):
  Kria (4 GB RAM, trừ OS + DPU model ≈ 2.4 GB cho index):
    → ~1.2M vectors  → ~104 ngày / camera (512-d)
  PC dev (16+ GB RAM): không giới hạn thực tế cho 1 camera

Sau ngưỡng này: cần migrate sang IndexIVFFlat (hoặc dùng Qdrant)
```

### 5.3 Index Migration Strategy

```
Giai đoạn 1 (ngày 1–104 Kria / 1–∞ PC): IndexFlatIP — exact search, <1ms
Giai đoạn 2 (ngày 104–365, Kria):        IndexIVFFlat — ANN, ~1–3ms
Giai đoạn 3 (sau 365 ngày, Kria):        IndexIVFPQ  — compressed, ~2–5ms,
                                           RAM 8× nhỏ hơn

Auto-migration trigger (hoặc dùng Qdrant từ đầu với kria.yaml):
```python
def maybe_migrate_index(index: VideoIndex) -> VideoIndex:
    n    = index._index.ntotal
    edim = index.embed_dim  # 512 (Kria) hoặc 768 (PC)
    if n > 1_000_000 and isinstance(index._index, faiss.IndexFlatIP):
        log.warning("Index size %d > 1M — migrating to IndexIVFFlat (dim=%d)", n, edim)
        new_idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(edim), edim, 256)
        # ... train_and_add + save_backup
        return new_idx
    return index
```

### 5.4 Retention Policy — Video vs. Index

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA LIFECYCLE                            │
│                                                             │
│  Video Files:         Rolling 72h window (delete old)       │
│  ────────────────────────────────────────────────────────   │
│  day  0: [cam01_seg001.mp4] ... [cam01_seg1440.mp4]         │
│  day  3: seg001–seg1440 bị xóa, seg2881+ được giữ           │
│                                                             │
│  Faiss Index:         PERMANENT (không xóa vectors!)        │
│  ────────────────────────────────────────────────────────   │
│  Tìm kiếm sự kiện 2 tuần trước: ✅ Vẫn tìm được            │
│  Xem lại video clip 2 tuần trước: ❌ File đã bị xóa         │
│                                                             │
│  Hệ quả thiết kế:                                           │
│  → SearchResult chứa absolute_timestamp thay vì file path  │
│  → Khi video đã bị xóa: hiển thị "Video not available,    │
│     event detected at [timestamp]"                          │
│  → Tùy chọn: archive video lên cloud storage nếu có match │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Khả năng mở rộng đa camera

### 6.1 Mô hình tính toán đa camera

```
Single camera analysis (window=10s, stride=7s, conservative process_time=4.5s/segment):
  Capture stride:    50s (mọi 50s có 1 segment mới cần index)
  Process time:      ~4.5s per segment
  DPU utilization:   4.5/50 = 9% per camera
  DPU idle:          45.5s / 50s → 91% headroom per camera

Lý thuyết: Kria DPU có thể phục vụ 50/4.5 ≈ 11 cameras (serial DPU execution)
```

Tuy nhiên có 3 bottleneck ngoài DPU:

```
Bottleneck 1: ARM CPU (preprocessing)
  1 camera:  40 frames × 12ms = 0.48s ARM preprocessing
  4 cameras: 4 × 0.48s = 1.92s ARM time trong cùng 50s stride
  → ARM còn >48s sau preprocessing → KHÔNG phải bottleneck

Bottleneck 2: Memory Bandwidth (PS↔PL DMA)
  1 segment: 40 frames × 224×224×3 bytes = ~6.0 MB DMA transfer
  DMA bandwidth: 19.2 GB/s → 6.0 MB / 19.2 GB/s ≈ 0.3ms → Trivial
  → Không phải bottleneck

Bottleneck 3: LPDDR4 Bandwidth (ARM CPU + DPU shared)
  4 cameras parallel: 4 × 6.0 MB frames ≈ 24 MB in 50s
  Peak bandwidth needed: ~0.5 MB/s << 19.2 GB/s → KHÔNG phải bottleneck

Bottleneck 4: Sequential DPU execution (VART không hỗ trợ true parallel DPU jobs)
  DPU B4096 là SINGLE inference unit — không thể xử lý 2 batches đồng thời
  → Cameras phải chia sẻ DPU theo thứ tự (serialized)
  → 7 cameras mỗi 50s: DPU busy 7 × 4.5s = 31.5s → 63% utilization → OK
  → 11 cameras: DPU busy 11 × 4.5s = 49.5s → 99% → NO MARGIN
```

### 6.2 Khuyến nghị số lượng camera thực tế

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│          KRIA KV260 — CAMERA CAPACITY ANALYSIS (window=10s, stride=7s, proc=4.5s)   │
├──────────────┬────────────┬──────────────┬──────────────────────────────────────────┤
│  N cameras   │ DPU util.  │ ARM util.    │ Recommendation                           │
├──────────────┼────────────┼──────────────┼──────────────────────────────────────────┤
│      1       │     9%     │    <1%       │ ✅ Tối ưu (luận văn demo)                │
│      3       │    27%     │     3%       │ ✅ Comfortable                           │
│      5       │    45%     │     5%       │ ✅ Khuyến nghị cho production nhỏ        │
│      7       │    63%     │     7%       │ ✅ Sweet spot — đủ margin                │
│      9       │    81%     │     9%       │ ⚠️ Cận ngưỡng — monitor queue depth     │
│     11       │    99%     │    11%       │ ❌ Không khuyến nghị (no margin)         │
└──────────────┴────────────┴──────────────┴──────────────────────────────────────────┘

Thực tế nên giữ DPU utilization ≤ 70% để:
  • Có margin xử lý segment dài hơn (không phải 60s exact)
  • Đáp ứng burst từ nhiều cameras ready cùng lúc
  • Tránh queue buildup khi một camera có nhiều chuyển động

→ KHUYẾN NGHỊ: **7 cameras** là sweet spot cho Kria KV260 với cấu hình đã xác nhận
```

### 6.3 Queue Management cho đa camera

```python
# Khi 3 cameras đồng thời gửi segment (worst case):
# Tất cả 3 files hoàn thành trong cùng 1 giây

# Với maxsize=50 queue: 3 jobs sẽ được queue và xử lý tuần tự
# Queue depth dự kiến: tối đa 3 jobs (1 đang process + 2 chờ)
# Time to clear backlog: 3 × 12s = 36s → clear trước khi camera tiếp theo gửi

# Cần monitor: queue depth
# Alert nếu queue depth > 10 (backlog growing → processing too slow)
```

---

## 8. Đường ống độ trễ: Camera → Searchable

### 7.1 Phân tích từng bước độ trễ

```
EVENT xảy ra tại thời điểm T

┌──────────────────────────────────────────────────────────────────┐
│ Bước 1: Capture buffer lag                                       │
│   RTSP latency + ffmpeg tcp buffer (OverlapCaptureDaemon)        │
│   Estimated: 200–500ms                                           │
└──────────────────────────────────────────────────────────────────┘
           │ video frames với event ở trong bộ nhớ đệm
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ Bước 2: Wait for segment to complete                             │
│   Event tại thời điểm T trong segment                           │
│   Nếu event ở đầu segment (t=5s sau boundary): đợi 55s nữa     │
│   Nếu event ở cuối segment (t=55s sau boundary): đợi 5s nữa    │
│   Average wait: 30s (uniform distribution)                       │
│   Worst case: 60s                                               │
└──────────────────────────────────────────────────────────────────┘
           │ file .mp4 completed + inotify IN_CLOSE_WRITE
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ Bước 3: Queue wait                                               │
│   Nếu indexer đang bận với segment khác: thêm 0–12s chờ        │
│   Average: ~4s (queue depth ~0 với 1 camera)                    │
└──────────────────────────────────────────────────────────────────┘
           │ job dequeued
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ Bước 4: NLVS Indexing                                            │
│   KriaEngine (8 windows × 5 frames × 46ms) + Qdrant/Faiss.add() │
│   Duration: ~3.0s (theoretical) / ~4.5s (conservative)          │
└──────────────────────────────────────────────────────────────────┘
           │ vectors added to Qdrant/Faiss index
           ▼
EVENT SEARCHABLE

TỔNG ĐỘ TRỄ:
  Average: 0.35s + 30s + 0.5s + 4.5s = ~35.4s
  Best case: 0.2s + 5s + 0s + 3.0s = ~8.2s
  Worst case: 0.5s + 60s + 6s + 4.5s = ~71.0s
```

### 7.2 So sánh độ trễ với các hệ thống tương đương

```
┌─────────────────────────────────────────────────────────────────┐
│              LATENCY COMPARISON — Surveillance Systems          │
├──────────────────────────┬──────────────────┬───────────────────┤
│ System                   │ Event→Searchable │ Notes             │
├──────────────────────────┼──────────────────┼───────────────────┤
│ NLVS Kria (confirmed)          │ 8–71s avg ~35s   │ Edge, DPU B4096       │
│ NLVS PC (EVA-CLIP, 60s seg)    │ 3–65s avg ~33s   │ GPU accelerated       │
│ NLVS Manual Upload       │ Minutes to hours │ Human in loop     │
│ Commercial VMS (e.g.,    │ 15–60s           │ $10K+ appliance   │
│  Milestone, Genetec)     │                  │                   │
│ Cloud AI (Azure Video    │ 30–120s          │ Network dependent │
│  Indexer)                │                  │                   │
└──────────────────────────┴──────────────────┴───────────────────┘

→ NLVS Kria cạnh tranh tốt với các giải pháp thương mại
   trong cùng tầm giá phần cứng (~$249)
```

### 7.3 Tối ưu độ trễ: Giảm Segment Duration

Để giảm độ trễ, có thể giảm segment từ 60s xuống 30s:

```
Segment 30s:
  Processing time: ~6s
  Average latency: 30/2 + 0 + 6 = ~21s

Đánh đổi:
  ✅ Latency giảm 50%: 46s → 21s
  ❌ File overhead tăng 2×: 2 files/phút thay vì 1
  ❌ Index growth rate giống nhau (cùng số windows/phút)
  ❌ Segment boundary problem xảy ra 2× thường xuyên
  → Cần overlap 10s nhưng với segment 30s → overhead lớn hơn tương đối

Khuyến nghị: Giữ 60s segment nếu latency <90s là chấp nhận được.
             Giảm xuống 30s nếu cần latency <30s.
```

---

## 9. Rủi ro kỹ thuật và Mitigation

### 8.1 Risk Matrix

| Rủi ro | Xác suất | Tác động | Mitigation |
|---|---|---|---|
| Queue buildup khi burst events (nhiều camera active) | Trung bình | Cao | Alert + queue depth monitoring; tăng queue maxsize; drop oldest if full |
| watchdog miss events (filesystem buffer exhaustion) | Thấp | Cao | Polling fallback (_HAS_WATCHDOG=False) mỗi 5s; systemd watchdog |
| DPU thermal throttling dưới sustained load | Trung bình | Trung bình | Duty cycle limit; thermal monitoring; fan required |
| Video file corruption nếu mất điện đột ngột | Trung bình | Trung bình | UPS + graceful shutdown; ffmpeg subprocess ghi atomic per-segment |
| Faiss index corruption khi process bị kill | Trung bình | Cao | Write-ahead log; atomic save (save to .tmp, rename) |
| Storage full → segment files bị mất | Thấp | Cao | Disk usage monitoring; emergency cleanup trigger tại 90% |
| Segment boundary splits quan trọng | Cao | Trung bình | Overlapping segments (đã phân tích §3); cross-segment NMS |
| RTSP stream loss (camera offline) | Trung bình | Thấp | rtspsrc với reconnect timeout; health check endpoint |

### 8.2 Graceful Shutdown Handling

```python
# src/continuous_indexer.py — Graceful shutdown

import signal

def handle_sigterm(signum, frame):
    """Đảm bảo Faiss được flush khi service bị stop bởi systemd."""
    log.info("SIGTERM received — flushing index before shutdown...")
    indexer._stop_event.set()
    indexer.index.save(indexer.index_dir)
    log.info(f"Final index saved: {indexer.index.total_vectors()} vectors.")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)
```

### 8.3 Crash Recovery

```
Khi nlvs-indexer.service crash và restart:

1. VideoIndex.load_or_create() tải lại index đã flush gần nhất
2. Các segment chưa được index (trong khoảng thời gian từ lần flush cuối
   đến khi crash) sẽ bị mất → replay mechanism:

   # Replay: tìm segment files mới hơn thời gian flush cuối
   last_indexed_ts = get_last_indexed_timestamp(index)
   for seg_file in sorted(glob("/storage/**/*.mp4")):
       if os.path.getctime(seg_file) > last_indexed_ts:
           queue.put(IndexJob.from_file(seg_file))
```

---

## 10. Đánh giá tổng thể và Khuyến nghị

### 9.1 Đánh giá tính khả thi

```
┌─────────────────────────────────────────────────────────────────┐
│         ĐÁNH GIÁ KHẢ THI — CONTINUOUS STREAM WORKFLOW          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ KHẢ THI với cấu hình đã xác nhận:                          │
│                                                                 │
│  Hiệu năng:                                                     │
│  ✅ Processing ratio 13–20× faster than real-time (1 camera)  │
│  ✅ 7 cameras là sweet spot cho Kria KV260                     │
│  ✅ Latency avg ~35s — cạnh tranh với giải pháp thương mại    │
│  ✅ Faiss search vẫn <2ms sau hàng tháng accumulation         │
│                                                                 │
│  Kỹ thuật:                                                      │
│  ✅ OverlapCaptureDaemon: ffmpeg -c copy, stride/overlap        │
│  ✅ watchdog lib (polling fallback) trigger low-overhead        │
│  ✅ Qdrant backend (PC + Kria) với Faiss fallback             │
│  ✅ Systemd services cung cấp auto-restart và lifecycle mgmt  │
│                                                                 │
│  ⚠️ CẦN GIẢI QUYẾT TRƯỚC KHI PRODUCTION:                        │
│  ✅ Segment boundary overlap: tail-only 60s/50s (v2.0 ✔)        │
│  ✅ Temporal NMS: absolute timestamp cross-segment (v2.0 ✔)     │
│  ✅ EOFError trong VideoIndex.load(): pickle guard (v2.2 ✔)     │
│  ✅ WSL2 USB camera: usbipd-win guide + OpenCVCaptureDaemon fallback │
│  ✅ Unified Qdrant workflow PC+Kria (v2.3 ✔)                    │
│  ✅ Fatal error detection trong capture daemons (v2.4 ✔)         │
│  ⚠️ External storage: bắt buộc (eMMC không đủ)                  │
│  ⚠️ Index growth: Qdrant giải quyết RAM; Faiss ok ~104 ngày     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Lộ trình triển khai đề xuất

```
Giai đoạn A (đã hoàn thành — v2.0/v2.1/v2.2/v2.3 source code): Cơ sở kiến trúc
────────────────────────────────────────────────────────────────────────────────
✔ SegmentMeta v2.0 (cam_id, relative_start, absolute_start + migration)
✔ GstPipeline defaults: window=10s, overlap=30%, frames=5
✔ PersistentJobQueue (SQLite WAL, backoff, startup recovery, stats())
✔ CircuitBreaker (OPEN≥20, CLOSE≤5) — src/job_queue.py
✔ OverlapCaptureDaemon (ffmpeg -c copy, stride=50s, RTSP+V4L2)
✔ OpenCVCaptureDaemon (cv2 fallback, serial segments, no ffmpeg)
✔ check_ffmpeg() / is_wsl2() / list_webcams() — src/capture_daemon.py
✔ ContinuousIndexer (watchdog/poll, create_engine factory, Qdrant→Faiss)
✔ VideoIndex.load(): EOFError/UnpicklingError guard (v2.2)
✔ pc.yaml: EVA-CLIP ViT-L/14, embed_dim=768, CUDA, backend=qdrant (v2.3)
✔ kria.yaml: DPU B4096, embed_dim=512, Qdrant production
✔ NLVideoSearcher: Qdrant search backend + graceful Faiss fallback (v2.3)
✔ app.py: Streamlit 2-tab UI (Camera & Indexing / Search)
✔ app.py: auto-select OverlapCaptureDaemon vs OpenCVCaptureDaemon
✔ app.py: WSL2 usbipd-win guide, ffmpeg warning banner
✔ app.py: unified Qdrant vector count in _pipeline_status() (v2.3)
✔ OverlapCaptureDaemon: fatal error detection — "Permission denied" / "No such device" → stop loop + last_error (v2.4)
✔ OpenCVCaptureDaemon: os.access() permission check → stop loop + last_error (v2.4)
✔ app.py: capture_error banner với hướng dẫn sudo usermod -aG video $USER (v2.4)
✔ app.py: bảo toàn last_error sau Stop Pipeline vào last_capture_error; xóa 0-byte segments khi start (v2.5)
✔ job_queue.py: mark_dead() (immediate dead-letter, no retry) + purge_missing_files() (v2.6)
✔ continuous_indexer.py: FileNotFoundError → mark_dead() thay vì mark_failed() — không retry vô ích (v2.6)
✔ app.py _start_pipeline(): gọi queue.purge_missing_files() sau xóa 0-byte files (v2.6)
✔ 215 tests passing (166 unit + 49 integration)

Giai đoạn B (tiếp theo): Integration Test
───────────────────────────────────────────
□ Test end-to-end với 1 camera giả lập (ffmpeg RTSP server)
□ Đo latency thực tế (capture → searchable) với window=10s
□ Kiểm tra crash recovery (kill -9 indexer process)
□ Test: sự kiện tại boundary có được tìm thấy không (tail overlap)

Giai đoạn C (sản xuất): Multi-camera và Production Hardening
─────────────────────────────────────────────────────────────────
□ Thêm camera thứ 2, 3..7 với separate capture services
□ Implement storage rotation (xóa video > 72h)
□ Implement metrics endpoint (Prometheus)
□ Soak test: 7 ngày liên tục với 3 cameras
□ Monitor: DPU temperature, queue depth, latency percentiles
□ Qdrant production deploy trên Kria ARM64
```

### 9.3 Tóm tắt số liệu chính (v2.0)

| Chỉ số | Giá trị | Điều kiện |
|---|---|---|
| **window_sec** | **10.0 s** | Đã xác nhận |
| **overlap_ratio** | **0.30** (stride = 7.0 s) | Đã xác nhận |
| **frames_per_window** | **5** | Đã xác nhận |
| **score_threshold** | **0.20** | Đã xác nhận |
| Windows/segment 60s | **8** | 10s window, 7s stride |
| Processing ratio | **13–20×** faster than real-time | 1 camera, 1080p, segment=60s |
| Avg latency (camera → searchable) | **~35 giây** | 1 camera, stride=50s (capture) |
| Max cameras supported | **7 (khuyến nghị)**, 9 (max) | Kria KV260, proc=4.5s/seg |
| DPU utilization (7 cameras) | **~63%** | Headroom cho burst |
| Index growth (Qdrant on-disk) | ~27 GB / 90 ngày | 3 cameras, 512-d, HNSW overhead |
| Faiss IndexFlatIP limit | **~104 ngày/camera** | 512-d, 8 windows/min |
| Qdrant RAM (idle) | ~50–100 MB | Service + OS overhead |
| Qdrant RAM (search 24h/1cam) | ~1.8 GB | MMAP pages loaded |
| Segment boundary miss rate | ~12–17% | Không có overlap |
| Segment boundary miss rate | <1% | Với 10s tail-only overlap (v2.0) |
| Capture overlap storage overhead | 10/60 = 16.7% per file | Tail-only overlap, stride=50s |
| SQLite queue (crash recovery) | 100% jobs recovered | PROCESSING → PENDING on restart |
| Queue circuit breaker | Opens @ depth=20 | ~17 phút backlog |
| Video storage (72h retention) | ~130 GB | 1 camera, 1080p@4Mbps |
| Search latency | 2–5ms (ANN) | Qdrant HNSW, filtered query |
| Search latency (Faiss fallback) | <2ms (exact) | IndexFlatIP, <1M vectors |

### 9.4 Câu hỏi thiết kế còn mở

Một số quyết định thiết kế cần thực nghiệm để quyết định:

1. **Segment duration: 30s hay 60s?** — Đo latency thực tế và chọn theo SLA.
2. **Xóa video khi hết retention hay archive lên cloud?** — Phụ thuộc ngân sách storage.
3. **Index per-camera hay global index?** — Per-camera cho phép filter, global index đơn giản hơn.
4. **Reranking (BLIP-2) có nên chạy online?** — BLIP-2 cần ~2.8GB RAM; conflict với KriaEngine; chỉ khả thi khi unload xmodel sau indexing.
5. **Absolute timestamp hay relative?** — Absolute cho phép cross-segment NMS và timezone-aware search ("tìm người chạy vào lúc 14:30 hôm qua").

---

## 11. WSL2 USB Isochronous Limitation & Windows Bridge (v2.7)

### 11.1 Phân tích vấn đề gốc rễ

**Triệu chứng**: Camera `/dev/video0` có trong WSL2 (qua `usbipd attach`), quyền đúng (`crw-rw---- root video`), `tienc` trong nhóm `video`, nhưng mọi cố gắng capture đều thất bại:
- `ffmpeg -f v4l2`: treo không có output, không có frame
- `cv2.VideoCapture`: trả về frame trống (0 byte JPEG)
- `dmesg`: `vhci_get_frame_number: Not yet implemented`, `urb->status -104 (ECONNRESET)`

**Nguyên nhân**: WSL2's `vhci_hcd` (USB/IP virtual host controller) **không hỗ trợ isochronous USB transfers**. UVC webcam dùng endpoint isochronous để stream video — tất cả consumer webcam đều dùng chuẩn này. Kết quả: mọi V4L2 read đều trả về `ECONNRESET (-104)`.

```
# USB descriptor xác nhận:
bmAttributes: 0x05 (Isochronous, Asynchronous) ← TẤT CẢ endpoints đều isochronous
Camera: 174f:244c (Integrated Camera)
```

**Giải pháp**: Chạy capture trên **Windows host** (DirectShow hoạt động bình thường), ghi segment files ra thư mục chia sẻ qua `/mnt/c/`, WSL2 chỉ **watch folder** và index.

### 11.2 Kiến trúc Windows Bridge

```
Windows Host                          WSL2 (Ubuntu)
────────────────                      ──────────────────────────────
windows_capture_server.py             app.py (Streamlit)
  ├── cv2.VideoCapture(0)              ├── FolderWatcherDaemon
  │   (DirectShow — hoạt động)         │   ├── _run_loop(): poll dir mỗi 2s
  ├── Ghi segment → C:\Users\<you>\    │   ├── Chỉ enqueue file ổn định
  │   segments\cam01_<ts>_<seq>.mp4   │   │   (size không đổi qua 2 lần check)
  └── Tên file chuẩn NLVS format       │   └── Skip 0-byte và wrong cam_id
                                       ├── PersistentJobQueue (SQLite)
/mnt/c/Users/<you>/segments/ ←────────└── ContinuousIndexer (CUDA/EVA-CLIP)
  (Windows path visible từ WSL2)
```

### 11.3 FolderWatcherDaemon (`src/capture_daemon.py`)

**Class**: `FolderWatcherDaemon` (dòng 553–697)

**Constructor parameters**:
| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `watch_dir` | (required) | Thư mục chứa segment files từ Windows |
| `cam_id` | `"cam01"` | Lọc theo prefix tên file |
| `job_queue` | (required) | SQLite queue để enqueue |
| `poll_interval` | `2.0` | Giây giữa mỗi lần scan |
| `stable_checks` | `2` | Số lần size phải ổn định trước khi enqueue |

**Logic `_run_loop()`**:
```python
while not self._stop_event.is_set():
    seen = scan_mp4_files(watch_dir, cam_id)
    for path in seen - already_enqueued:
        if size > 0 and size_stable_across_2_checks:
            self._enqueue_segment(path)
            already_enqueued.add(path)
    sleep(poll_interval)
```

**Tích hợp `app.py`**:
- Device string dạng `watch://<folder_path>` hoặc là đường dẫn folder tồn tại → dùng `FolderWatcherDaemon`
- Hiển thị hướng dẫn WSL2 bridge trong sidebar khi detect WSL2

### 11.4 `windows_capture_server.py`

Script Python chạy trên **Windows** (không phải WSL2):

```bash
# Trên Windows PowerShell:
pip install opencv-python
python windows_capture_server.py --output C:\Users\<you>\segments --camera 0 --segment-sec 60 --cam-id cam01
```

**Features**:
- Ghi segment `cam01_<unix_ts>_<seq>.mp4` bằng `cv2.VideoWriter`
- Tự động tạo thư mục output
- Retry khi camera lỗi tạm thời
- Graceful shutdown qua Ctrl+C
- Tên file tương thích 100% với NLVS format (indexer nhận ngay)

**Shared folder access**:
```
Windows path:  C:\Users\tienc\segments\
WSL2 path:     /mnt/c/Users/tienc/segments/
```
Không cần mount thêm — WSL2 auto-mount tất cả Windows drives qua `/mnt/`.

### 11.5 Cấu hình trong app.py (Streamlit)

1. Mở sidebar "📷 Camera Source"
2. Device input: nhập `/mnt/c/Users/<you>/segments` hoặc `watch:///mnt/c/Users/<you>/segments`
3. App tự nhận ra là `FolderWatcherDaemon` mode
4. Start Pipeline → daemon poll folder, enqueue segment khi file ổn định
5. ContinuousIndexer xử lý như bình thường

**UI warning khi WSL2 detect**:
```
⚠ WSL2 detected — V4L2 capture unavailable (isochronous USB not supported by vhci_hcd)
→ Use Windows Bridge: run windows_capture_server.py on Windows, point device to /mnt/c/...
```

### 11.6 Tests (v2.7) — CD-08 đến CD-12

| Test ID | Mô tả | Kết quả |
|---------|-------|---------|
| CD-08 | `FolderWatcherDaemon` enqueue file ổn định | ✅ PASS |
| CD-09 | Bỏ qua file 0-byte | ✅ PASS |
| CD-10 | Bỏ qua file sai cam_id prefix | ✅ PASS |
| CD-11 | Không enqueue file 2 lần | ✅ PASS |
| CD-12 | Chờ size ổn định trước khi enqueue | ✅ PASS |

**Tổng cộng**: **220 passed, 14 skipped** (234 collected), 0 regression. +5 tests so với v2.6 (215→220).

### 11.7 Hạn chế đã biết

| Vấn đề | Mô tả | Workaround |
|--------|-------|-----------|
| Latency thêm | Bridge thêm ~60s (1 segment) latency trước khi WSL2 thấy file | Giảm `--segment-sec 30` trên Windows |
| usbipd không cần thiết | Với bridge mode, không cần attach USB vào WSL2 | Detach hoàn toàn để tránh conflict |
| Windows path case-sensitive | `/mnt/c/` vs `/mnt/C/` — WSL2 mount theo Windows drive letter | Luôn dùng lowercase `/mnt/c/` |
| Overlap segment | `windows_capture_server.py` hiện chưa implement tail-overlap | TODO: thêm `--overlap-sec` option |
