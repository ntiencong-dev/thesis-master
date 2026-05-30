"""
api/main.py
-----------
FastAPI search & indexing service for NLVS.

Endpoints
---------
GET  /health          — liveness check + index stats.
POST /index           — add a video file to the index.
POST /index/directory — index all videos in a directory.
POST /search          — natural language query → ranked clip list.
GET  /thumbnail       — extract a JPEG frame from any video at a given timestamp.
GET  /debug/query     — vector diagnostics for a query (scores, templates, NMS).
DELETE /index         — reset the in-memory index.

Run
---
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Or via config:
    CONFIG=config/pc.yaml uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import time
from typing import List, Optional

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from src.searcher import NLVideoSearcher, SearchResult

# ---------------------------------------------------------------------------
# Application init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NLVS — Natural Language Video Search",
    description="Open-vocabulary video search powered by CLIP + Qdrant.",
    version="1.0.0",
)

# Load config from environment variable; fall back to pc.yaml
_CONFIG_PATH = os.environ.get("CONFIG", "config/pc.yaml")

# Lazy-initialise the searcher (model loading is expensive)
_searcher: Optional[NLVideoSearcher] = None


def _get_searcher() -> NLVideoSearcher:
    global _searcher
    if _searcher is None:
        if os.path.isfile(_CONFIG_PATH):
            _searcher = NLVideoSearcher.from_config(_CONFIG_PATH)
        else:
            _searcher = NLVideoSearcher.from_params()
    return _searcher


def _qdrant_total(searcher: NLVideoSearcher) -> int:
    """Return number of vectors in the Qdrant collection (0 if unavailable)."""
    try:
        if searcher._qdrant_client is None:
            return 0
        info = searcher._qdrant_client.get_collection(searcher._qdrant_collection)
        return info.points_count or 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the video file.")


class IndexDirRequest(BaseModel):
    video_dir: str  = Field(..., description="Absolute path to the directory.")
    extensions: List[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv"],
        description="Video file extensions to include.",
    )


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Free-text query (any language). E.g. 'người leo rào'.",
        min_length=1,
        max_length=512,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )
    score_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score to include in results.",
    )
    nms_iou: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Temporal IoU threshold for NMS deduplication.",
    )


class ClipResult(BaseModel):
    rank:       int
    score:      float
    video_id:   str
    video_path: str
    start_time: float
    end_time:   float
    duration:   float


class SearchResponse(BaseModel):
    query:       str
    num_results: int
    query_ms:    float
    results:     List[ClipResult]


class IndexResponse(BaseModel):
    video_path:      str
    segments_added:  int
    total_in_index:  int


class HealthResponse(BaseModel):
    status:       str
    backend:      str
    total_vectors: int
    config_path:  str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — returns backend type and index size."""
    searcher = _get_searcher()
    backend  = searcher._config.get("backend", "unknown")
    return HealthResponse(
        status="ok",
        backend=backend,
        total_vectors=_qdrant_total(searcher),
        config_path=_CONFIG_PATH,
    )


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
def index_video(req: IndexRequest):
    """
    Add a single video file to the Faiss index.

    The video is processed through the configured pipeline
    (OpenCV or GStreamer), embeddings are computed, and the index
    is saved to disk if index_dir is set.
    """
    if not os.path.isfile(req.video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: {req.video_path}",
        )

    searcher = _get_searcher()
    try:
        n = searcher.index_video(req.video_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IndexResponse(
        video_path=req.video_path,
        segments_added=n,
        total_in_index=_qdrant_total(searcher),
    )


@app.post("/index/directory", response_model=IndexResponse, tags=["Indexing"])
def index_directory(req: IndexDirRequest):
    """Index all compatible video files in a directory."""
    if not os.path.isdir(req.video_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {req.video_dir}",
        )

    searcher = _get_searcher()
    try:
        n = searcher.index_directory(req.video_dir, extensions=tuple(req.extensions))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IndexResponse(
        video_path=req.video_dir,
        segments_added=n,
        total_in_index=_qdrant_total(searcher),
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search(req: SearchRequest):
    """
    Natural language video search.

    Pipeline:
    1. Encode query with CLIP Text Encoder.
    2. Faiss nearest-neighbour search (cosine similarity).
    3. Score threshold filtering.
    4. Temporal NMS to remove duplicate time windows.
    5. Return top-K ranked results with timestamps.

    Example query: "người leo rào" / "red truck passing the gate"
    """
    searcher = _get_searcher()
    if _qdrant_total(searcher) == 0:
        raise HTTPException(
            status_code=409,
            detail="Index is empty. POST /index first.",
        )

    t0 = time.perf_counter()
    try:
        results: List[SearchResult] = searcher.search(
            query_text=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            nms_iou=req.nms_iou,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    clips = [
        ClipResult(
            rank=r.rank,
            score=round(r.score, 6),
            video_id=r.video_id,
            video_path=r.video_path,
            start_time=round(r.start_time, 3),
            end_time=round(r.end_time, 3),
            duration=round(r.end_time - r.start_time, 3),
        )
        for r in results
    ]

    return SearchResponse(
        query=req.query,
        num_results=len(clips),
        query_ms=round(elapsed_ms, 2),
        results=clips,
    )


@app.delete("/index", tags=["Indexing"])
def reset_index():
    """
    Reset the in-memory index (does NOT delete saved files on disk).
    Useful for re-indexing from scratch.
    """
    global _searcher
    _searcher = None
    return {"status": "index reset", "message": "Call /index to rebuild."}


# ---------------------------------------------------------------------------
# Thumbnail endpoint
# ---------------------------------------------------------------------------

@app.get("/thumbnail", tags=["Search"],
         responses={200: {"content": {"image/jpeg": {}}},
                    404: {"description": "Video not found or seek failed"}})
def get_thumbnail(
    video_path: str = Query(..., description="Absolute path to the video file."),
    t: float        = Query(..., ge=0.0, description="Timestamp in seconds."),
    width:  int     = Query(320, ge=32, le=1920, description="Output width in px."),
    height: int     = Query(180, ge=18, le=1080, description="Output height in px."),
):
    """
    Extract a single JPEG frame from *video_path* at time *t* seconds.

    Useful for displaying result thumbnails when calling the API from an
    external client (curl, JS, Python) without the Streamlit UI.

    Example
    -------
    .. code-block:: bash

        curl "http://localhost:8000/thumbnail?video_path=/data/fi003.mp4&t=12.5" \
             -o thumb.jpg
    """
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404,
                            detail=f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=404,
                            detail=f"OpenCV cannot open: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=404,
                            detail=f"Seek to {t}s failed — beyond video duration?")

    resized = cv2.resize(frame, (width, height))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encoding failed.")

    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Debug endpoint — vector diagnostics
# ---------------------------------------------------------------------------

@app.get("/debug/query", tags=["System"])
def debug_query(
    query: str  = Query(..., min_length=1, max_length=512,
                        description="Search query to diagnose."),
    top_k: int  = Query(10, ge=1, le=50,
                        description="Number of raw results to show before NMS."),
):
    """
    Return detailed vector diagnostics for a search query.

    Shows:
    - cleaned (normalised) query text
    - which CLIP prompt templates are used
    - cosine score distribution across the entire index
    - top-N raw Faiss hits before NMS, compared between raw query and
      template-ensemble query

    Useful for debugging why a query returns unexpected results.
    """
    searcher = _get_searcher()
    if _qdrant_total(searcher) == 0:
        raise HTTPException(status_code=409, detail="Index is empty.")
    try:
        return searcher.search_debug(query, top_k=top_k, score_threshold=0.0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
