# ============================================================
# NLVS — Multi-target Dockerfile
#
# Build for PC  (CUDA, GTX 1650 Ti):
#   docker build --build-arg TARGET=pc -t nlvs:pc .
#
# Build for Kria KV260 (Vitis AI 3.5):
#   docker build --build-arg TARGET=kria -t nlvs:kria .
#
# Run (PC):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/videos:/data \
#     -e CONFIG=config/pc.yaml \
#     nlvs:pc
# ============================================================

ARG TARGET=pc

# ---- PC base (NVIDIA CUDA 11.8) ----------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-pc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        libglib2.0-0 libgl1-mesa-glx libgstreamer1.0-0 \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ---- Kria base (Vitis AI 3.5 runtime) ------------------------------------
FROM xilinx/vitis-ai-cpu:3.5.0 AS base-kria

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Vitis AI runtime already includes Python 3.8 + VART + XIR
# Install only the missing packages
RUN pip install --no-cache-dir \
        open-clip-torch transformers opencv-python-headless \
        faiss-cpu numpy Pillow tqdm fastapi "uvicorn[standard]" pydantic pyyaml streamlit

# ---- Common final layer (selected by TARGET) -----------------------------
FROM base-${TARGET} AS final

WORKDIR /app

# Copy only application code (not venv — already in base)
COPY src/       ./src/
COPY api/       ./api/
COPY config/    ./config/
COPY app.py     ./app.py
COPY index_video.py ./index_video.py

# Install remaining Python dependencies (PC only; Kria base pre-installs them)
COPY requirements.txt .
RUN if [ "$TARGET" = "pc" ]; then \
        pip3 install --no-cache-dir -r requirements.txt; \
    fi

# Default: start the FastAPI server
# Override with: docker run ... streamlit run app.py
ENV CONFIG=config/pc.yaml
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
