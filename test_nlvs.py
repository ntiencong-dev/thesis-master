"""
test_nlvs.py
------------
End-to-end test suite for the NLVS prototype.

Kịch bản kiểm thử
-----------------
T1  VideoProcessor  – 1-FPS keyframe mode
T2  VideoProcessor  – Sliding-window mode (50 % overlap)
T3  CLIPFeatureExtractor – encode_frames (L2-norm, shape, dtype)
T4  CLIPFeatureExtractor – encode_text  (L2-norm, shape, dtype)
T5  CLIPFeatureExtractor – cosine gap: matching text > unrelated text
T6  VideoIndex          – add + search correctness (self-query score ≈ 1.0)
T7  VideoIndex          – save / load persistence round-trip
T8  NLVideoSearcher     – full index_video pipeline on real file
T9  NLVideoSearcher     – text search returns correct number of results
T10 NLVideoSearcher     – search result timestamps are within video duration
T11 NLVideoSearcher     – save_index / load_index persistence
T12 CLI index_video.py  – smoke-test via subprocess
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import List

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_PATH = "/home/tienc/Prototype/data/fi003.mp4"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results: list[tuple[str, bool, str]] = []


def test(name: str):
    """Decorator factory that catches exceptions and records pass/fail."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, True, ""))
                print(f"{PASS} {name}")
            except Exception as exc:
                tb = traceback.format_exc()
                results.append((name, False, str(exc)))
                print(f"{FAIL} {name}\n       {exc}\n{tb}")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Import modules under test (lazy, so import errors show per-test)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.video_processor import VideoProcessor, VideoSegment
from src.feature_extractor import CLIPFeatureExtractor
from src.indexer import VideoIndex, SegmentMeta
from src.searcher import NLVideoSearcher


# ---------------------------------------------------------------------------
# Shared fixtures (loaded once)
# ---------------------------------------------------------------------------

def _get_video_info(path: str):
    cap = cv2.VideoCapture(path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    cap.release()
    return fps, n_frames, duration


print(f"\n{INFO} Loading CLIP model (first run downloads weights ~600 MB)…")
_extractor = CLIPFeatureExtractor()
print(f"{INFO} CLIP model ready on {_extractor.device}\n")


# ---------------------------------------------------------------------------
# T1 – VideoProcessor 1-FPS keyframe mode
# ---------------------------------------------------------------------------

@test("T1: VideoProcessor – 1-FPS keyframe extraction")
def t1():
    proc = VideoProcessor(fps_mode=1.0)
    segments: List[VideoSegment] = proc.extract_segments(VIDEO_PATH)
    _, _, duration = _get_video_info(VIDEO_PATH)
    expected_min = max(1, int(duration) - 2)
    expected_max = int(duration) + 2
    assert len(segments) >= expected_min, \
        f"Too few segments: {len(segments)} (duration={duration:.1f}s)"
    assert len(segments) <= expected_max, \
        f"Too many segments: {len(segments)}"
    for seg in segments:
        assert seg.frame.shape == (224, 224, 3), \
            f"Wrong frame shape: {seg.frame.shape}"
        assert seg.frame.dtype == np.uint8
        assert seg.start_time >= 0
        assert seg.end_time > seg.start_time
    print(f"       {len(segments)} segments, duration={duration:.1f}s")

t1()

# ---------------------------------------------------------------------------
# T2 – VideoProcessor sliding-window mode
# ---------------------------------------------------------------------------

@test("T2: VideoProcessor – sliding-window 5s/50% overlap")
def t2():
    proc = VideoProcessor(fps_mode=None, window_sec=5.0, stride_sec=2.5)
    segments = proc.extract_segments(VIDEO_PATH)
    _, _, duration = _get_video_info(VIDEO_PATH)
    # At 2.5s stride: roughly duration / 2.5 windows
    expected_min = max(1, int(duration / 2.5) - 3)
    assert len(segments) >= expected_min, \
        f"Too few sliding-window segments: {len(segments)}"
    # Overlap: consecutive windows should overlap by ~50%
    for a, b in zip(segments[:-1], segments[1:]):
        overlap = a.end_time - b.start_time
        assert overlap >= 0, "Windows should not have a gap"
    print(f"       {len(segments)} windows, stride=2.5s, duration={duration:.1f}s")

t2()

# ---------------------------------------------------------------------------
# T3 – CLIPFeatureExtractor encode_frames
# ---------------------------------------------------------------------------

@test("T3: CLIPFeatureExtractor – encode_frames shape/norm/dtype")
def t3():
    proc = VideoProcessor(fps_mode=1.0)
    segments = proc.extract_segments(VIDEO_PATH)[:8]
    frames = [s.frame for s in segments]
    embs = _extractor.encode_frames(frames)
    assert embs.shape == (len(frames), 512), f"Shape mismatch: {embs.shape}"
    assert embs.dtype == np.float32
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Embeddings not unit-norm: min={norms.min():.4f} max={norms.max():.4f}"
    print(f"       {embs.shape}, all norms ≈ 1.0")

t3()

# ---------------------------------------------------------------------------
# T4 – CLIPFeatureExtractor encode_text
# ---------------------------------------------------------------------------

@test("T4: CLIPFeatureExtractor – encode_text shape/norm/dtype")
def t4():
    queries = ["person climbing a fence", "red truck", "empty road"]
    embs = _extractor.encode_text(queries)
    assert embs.shape == (3, 512), f"Shape mismatch: {embs.shape}"
    assert embs.dtype == np.float32
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    print(f"       {embs.shape}, all norms ≈ 1.0")

t4()

# ---------------------------------------------------------------------------
# T5 – CLIP cosine ranking: matching text > unrelated text
# ---------------------------------------------------------------------------

@test("T5: CLIP – cosine score: 'person' > 'spaceship' for human activity frame")
def t5():
    proc = VideoProcessor(fps_mode=1.0)
    frames = [s.frame for s in proc.extract_segments(VIDEO_PATH)[:5]]
    frame_embs = _extractor.encode_frames(frames)   # (5, 512)

    texts = ["person", "spaceship launching into orbit"]
    text_embs = _extractor.encode_text(texts)        # (2, 512)

    # cosine = dot product (already L2-normalised)
    scores_person    = float((frame_embs @ text_embs[0:1].T).mean())
    scores_spaceship = float((frame_embs @ text_embs[1:2].T).mean())

    print(f"       avg cosine — 'person':{scores_person:.4f}  "
          f"'spaceship':{scores_spaceship:.4f}")
    assert scores_person > scores_spaceship, \
        "Expected 'person' to score higher than 'spaceship'"

t5()

# ---------------------------------------------------------------------------
# T6 – VideoIndex self-query score ≈ 1.0
# ---------------------------------------------------------------------------

@test("T6: VideoIndex – add/search, self-query cosine score ≈ 1.0")
def t6():
    idx = VideoIndex(embed_dim=512)
    rng = np.random.default_rng(42)
    N = 20
    embs = rng.random((N, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    metas = [SegmentMeta(f"v{i}", f"/tmp/v{i}.mp4", i*5.0, (i+1)*5.0) for i in range(N)]
    idx.add(embs, metas)

    for i in range(N):
        res = idx.search(embs[i:i+1], top_k=1)
        assert len(res) == 1
        score, meta = res[0]
        assert abs(score - 1.0) < 1e-4, f"Self-query score={score} ≠ 1.0"
        assert meta.video_id == f"v{i}"
    print(f"       {N} self-queries all passed (score ≈ 1.0)")

t6()

# ---------------------------------------------------------------------------
# T7 – VideoIndex save / load persistence
# ---------------------------------------------------------------------------

@test("T7: VideoIndex – save/load round-trip")
def t7():
    tmp_dir = tempfile.mkdtemp(prefix="nlvs_test_")
    try:
        idx = VideoIndex(embed_dim=512)
        rng = np.random.default_rng(7)
        embs = rng.random((5, 512)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        metas = [SegmentMeta(f"v{i}", f"/tmp/v{i}.mp4", i*3.0, (i+1)*3.0) for i in range(5)]
        idx.add(embs, metas)
        idx.save(tmp_dir)

        idx2 = VideoIndex.load(tmp_dir)
        assert idx2.total_vectors() == 5

        for i in range(5):
            res = idx2.search(embs[i:i+1], top_k=1)
            score, meta = res[0]
            assert abs(score - 1.0) < 1e-4
            assert meta.video_id == f"v{i}"
        print(f"       saved+loaded 5 vectors, all scores ≈ 1.0")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

t7()

# ---------------------------------------------------------------------------
# T8 – NLVideoSearcher full index_video on real file
# ---------------------------------------------------------------------------

@test("T8: NLVideoSearcher – index_video on fi003.mp4")
def t8():
    tmp_dir = tempfile.mkdtemp(prefix="nlvs_test_")
    try:
        searcher = NLVideoSearcher(
            index_dir=tmp_dir,
            use_sliding_window=True,
            window_sec=5.0,
            overlap_ratio=0.5,
        )
        n = searcher.index_video(VIDEO_PATH)
        assert n > 0, "No segments indexed"
        assert searcher._index.total_vectors() == n
        print(f"       {n} segments indexed from fi003.mp4")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

t8()

# ---------------------------------------------------------------------------
# T9 – NLVideoSearcher search returns correct top_k
# ---------------------------------------------------------------------------

@test("T9: NLVideoSearcher – search returns exactly top_k results (or total if fewer)")
def t9():
    tmp_dir = tempfile.mkdtemp(prefix="nlvs_test_")
    try:
        searcher = NLVideoSearcher(index_dir=tmp_dir, use_sliding_window=True,
                                   window_sec=5.0, overlap_ratio=0.5)
        n = searcher.index_video(VIDEO_PATH)

        for top_k in [1, 3, 5]:
            results_list = searcher.search("người đi bộ", top_k=top_k)
            expected = min(top_k, n)
            assert len(results_list) == expected, \
                f"top_k={top_k}: got {len(results_list)}, expected {expected}"
            # ranks must be 1-based and ascending
            ranks = [r.rank for r in results_list]
            assert ranks == list(range(1, len(results_list)+1)), \
                f"Ranks out of order: {ranks}"
            # scores must be in non-increasing order
            for a, b in zip(results_list[:-1], results_list[1:]):
                assert a.score >= b.score - 1e-6, \
                    f"Scores not sorted: {a.score} < {b.score}"
        print(f"       top_k=[1,3,5] all returned correct counts & ordering")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

t9()

# ---------------------------------------------------------------------------
# T10 – result timestamps within video duration
# ---------------------------------------------------------------------------

@test("T10: NLVideoSearcher – result timestamps within video duration")
def t10():
    _, _, duration = _get_video_info(VIDEO_PATH)
    tmp_dir = tempfile.mkdtemp(prefix="nlvs_test_")
    try:
        searcher = NLVideoSearcher(index_dir=tmp_dir, use_sliding_window=True,
                                   window_sec=5.0, overlap_ratio=0.5)
        searcher.index_video(VIDEO_PATH)
        for query in ["người đi bộ", "xe máy", "cửa", "trên đường"]:
            results_list = searcher.search(query, top_k=5)
            for r in results_list:
                assert r.start_time >= 0, f"Negative start_time: {r.start_time}"
                assert r.end_time <= duration + 0.5, \
                    f"end_time {r.end_time:.1f} > duration {duration:.1f}"
                assert r.start_time < r.end_time, \
                    f"start >= end: {r.start_time} >= {r.end_time}"
        print(f"       all timestamps valid for duration={duration:.1f}s")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

t10()

# ---------------------------------------------------------------------------
# T11 – NLVideoSearcher save_index / load_index round-trip
# ---------------------------------------------------------------------------

@test("T11: NLVideoSearcher – save_index/load_index round-trip")
def t11():
    tmp_dir = tempfile.mkdtemp(prefix="nlvs_test_")
    try:
        s1 = NLVideoSearcher(index_dir=tmp_dir, use_sliding_window=True,
                             window_sec=5.0, overlap_ratio=0.5)
        n = s1.index_video(VIDEO_PATH)
        r1 = s1.search("người đi bộ", top_k=3)

        # Load into a fresh searcher
        s2 = NLVideoSearcher(index_dir=tmp_dir)
        s2.load_index()
        assert s2._index.total_vectors() == n
        r2 = s2.search("người đi bộ", top_k=3)

        assert len(r1) == len(r2), "Result count mismatch after reload"
        for a, b in zip(r1, r2):
            assert abs(a.score - b.score) < 1e-5, \
                f"Score mismatch after reload: {a.score} vs {b.score}"
        print(f"       {n} vectors persisted & reloaded, scores identical")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

t11()

# ---------------------------------------------------------------------------
# T12 – CLI smoke-test
# ---------------------------------------------------------------------------

@test("T12: CLI index_video.py – index + search via subprocess")
def t12():
    tmp_dir = tempfile.mkdtemp(prefix="nlvs_test_")
    venv_python = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python"
    )
    python_bin = venv_python if os.path.isfile(venv_python) else sys.executable
    try:
        cmd = [
            python_bin, "index_video.py", "index",
            "--video",     VIDEO_PATH,
            "--index-dir", tmp_dir,
            "--window-sec", "5",
            "--overlap",   "0.5",
            "--query",     "người đi bộ",
            "--top-k",     "3",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if proc.returncode != 0:
            raise RuntimeError(f"CLI exited {proc.returncode}\n{proc.stderr[-1500:]}")
        out = proc.stdout + proc.stderr
        assert "Total segments indexed" in out, \
            f"Expected indexing summary in output.\n{out[:800]}"
        assert "Score" in out or "Rank" in out, \
            f"Expected search table in output.\n{out[:800]}"
        print(f"       CLI exited 0, found indexing summary + search table")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

t12()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  Results: {passed}/{len(results)} tests passed"
      + (f"  ❌ {failed} FAILED" if failed else "  ✅ ALL PASSED"))
print("="*60)

if failed:
    print("\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
