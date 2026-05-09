#!/usr/bin/env python3
"""
scripts/debug_vectors.py
------------------------
CLI diagnostic tool for understanding why a query returns poor results.

Usage
-----
    # Load existing index and debug a query
    python scripts/debug_vectors.py --query "man waving hand"

    # Specify a custom index directory
    python scripts/debug_vectors.py \\
        --index-dir ./index_store \\
        --query "find the man is waving hand" \\
        --top-k 10 \\
        --save-frames  # save top-3 result frames as JPEG to /tmp/

    # Re-index a video and debug
    python scripts/debug_vectors.py \\
        --video data/fi003.mp4 \\
        --query "man waving hand"

"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Colour helpers ────────────────────────────────────────────────────────

def _clr(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def bold(t):  return _clr(t, "1")
def green(t): return _clr(t, "92")
def yellow(t): return _clr(t, "93")
def red(t):   return _clr(t, "91")
def cyan(t):  return _clr(t, "96")


def _bar(value: float, max_val: float = 0.5, width: int = 30) -> str:
    filled = int(min(value / max(max_val, 1e-8), 1.0) * width)
    return "[" + "█" * filled + "·" * (width - filled) + f"] {value:.4f}"


# ── Main diagnostic function ──────────────────────────────────────────────

def run_diagnostics(
    index_dir: str,
    query: str,
    top_k: int = 10,
    save_frames: bool = False,
    score_threshold: float = 0.0,
) -> None:
    import warnings
    warnings.filterwarnings("ignore")

    from src.searcher import NLVideoSearcher, _normalize_query, _CLIP_TEMPLATES
    from src.engines.factory import create_engine

    print()
    print(bold("=" * 70))
    print(bold("  NLVS Vector Diagnostics"))
    print(bold("=" * 70))

    # ── 1. Load index ──────────────────────────────────────────────────────
    print(f"\n{cyan('▸ Index directory:')} {index_dir}")
    try:
        searcher = NLVideoSearcher.from_params(index_dir=index_dir)
        searcher.load_index(index_dir)
        n = searcher._index.total_vectors()
        print(f"  {green('✓')} Loaded {n} segment vectors  (dim={searcher._engine.embed_dim})")
    except FileNotFoundError as e:
        print(red(f"  ✗ Index not found: {e}"))
        print("  Run: python index_video.py <video_path>  to build the index first.")
        return

    # ── 2. Query normalisation ─────────────────────────────────────────────
    print(f"\n{cyan('▸ Query analysis')}")
    print(f"  Original : {bold(repr(query))}")
    cleaned = _normalize_query(query)
    if cleaned != query:
        print(f"  Cleaned  : {bold(repr(cleaned))}  {yellow('(prefix stripped)')}")
    else:
        print(f"  Cleaned  : {bold(repr(cleaned))}  (no prefix stripped)")

    templates = [t.format(cleaned) for t in _CLIP_TEMPLATES]
    print(f"\n  {cyan('Templates used for embedding ensemble:')}")
    for i, t in enumerate(templates, 1):
        print(f"    {i}. {t}")

    # ── 3. Encode both variants ────────────────────────────────────────────
    print(f"\n{cyan('▸ Encoding queries …')}")
    engine = searcher._engine

    vec_original  = engine.encode_text(query)           # (1, D)
    vec_cleaned   = engine.encode_text(cleaned)          # (1, D)
    vec_templates = searcher._encode_with_templates(cleaned)  # (D,)

    # Cosine similarity between original and template-ensemble
    sim_orig_tmpl = float(
        np.dot(vec_original[0], vec_templates)
        / (np.linalg.norm(vec_original[0]) * np.linalg.norm(vec_templates) + 1e-8)
    )
    print(f"  cos(original, template-ensemble) = {sim_orig_tmpl:.4f}")
    if sim_orig_tmpl < 0.9:
        print(f"  {yellow('⚠ Low similarity')} — templates shift the embedding significantly")
    else:
        print(f"  {green('✓')} Embeddings are similar — templates are consistent")

    # ── 4. Score distribution ─────────────────────────────────────────────
    print(f"\n{cyan('▸ Score distribution (all {n} segments, template-ensemble)')}")
    raw = searcher._index.search(vec_templates, top_k=n)
    scores = [s for s, _ in raw]
    p = lambda q: float(np.percentile(scores, q))

    max_s, p90, p75, med, mean_s, min_s = (
        max(scores), p(90), p(75), p(50), np.mean(scores), min(scores)
    )
    print(f"  max={max_s:.4f}  p90={p90:.4f}  p75={p75:.4f}  "
          f"median={med:.4f}  mean={mean_s:.4f}  min={min_s:.4f}")

    if max_s < 0.25:
        print(f"  {red('⚠ Very low max score')} — the video may not contain this action,")
        print(f"    or try rephrasing as a visual description (not an action command).")
    elif max_s < 0.35:
        print(f"  {yellow('⚠ Moderate scores')} — results exist but may not be precise.")
        print(f"    Tip: lower score_threshold to 0.10–0.15 and review visually.")
    else:
        print(f"  {green('✓')} Good score range — query is well-matched.")

    # Histogram (ASCII)
    print()
    buckets = np.linspace(min_s, max_s, 10)
    hist, _ = np.histogram(scores, bins=buckets)
    for i, (lo, hi, cnt) in enumerate(zip(buckets, buckets[1:], hist)):
        bar = "█" * min(cnt, 40)
        mark = green(" ←") if lo >= 0.20 else ""
        print(f"  [{lo:.3f}–{hi:.3f}] {bar:40s} {cnt:3d}{mark}")
    print(f"  {'':46s} ^ default threshold (0.20)")

    # ── 5. Top-K comparison ───────────────────────────────────────────────
    for label, vec in [
        ("Original query", vec_original),
        ("Cleaned query",  vec_cleaned),
        ("Template ensemble", vec_templates),
    ]:
        print(f"\n{cyan(f'▸ Top-{top_k} — {label}')}")
        hits = searcher._index.search(vec, top_k=top_k)
        for rank, (score, meta) in enumerate(hits, 1):
            marker = green("●") if score >= 0.25 else (yellow("◑") if score >= 0.15 else red("○"))
            print(f"  {rank:2d}. {marker} {_bar(score, 0.5)}  "
                  f"[{meta.start_time:6.1f}s – {meta.end_time:6.1f}s]  {meta.video_id}")

    # ── 6. Save frames ────────────────────────────────────────────────────
    if save_frames:
        print(f"\n{cyan('▸ Saving top-3 frames …')}")
        hits = searcher._index.search(vec_templates, top_k=3)
        for rank, (score, meta) in enumerate(hits, 1):
            mid = (meta.start_time + meta.end_time) / 2.0
            cap = cv2.VideoCapture(meta.video_path)
            if not cap.isOpened():
                print(f"  {red('✗')} Cannot open: {meta.video_path}")
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000.0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"  {red('✗')} Seek failed at {mid:.1f}s in {meta.video_path}")
                continue
            out_path = f"/tmp/nlvs_debug_rank{rank}_{meta.video_id}_{mid:.1f}s.jpg"
            cv2.imwrite(out_path, frame)
            print(f"  {green('✓')} Rank {rank} (score={score:.4f}) → {out_path}")

    print()
    print(bold("=" * 70))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug NLVS vector similarity for a text query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", "-q", required=True,
        help='Text query, e.g. "man waving hand"',
    )
    parser.add_argument(
        "--index-dir", "-i", default="./index_store",
        help="Path to saved Faiss index directory (default: ./index_store)",
    )
    parser.add_argument(
        "--video", "-v", default=None,
        help="Re-index this video before running diagnostics (optional).",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=10,
        help="Number of top results to display (default: 10)",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save top-3 result frames as JPEG to /tmp/",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.0,
        help="Score threshold for summary (default: 0.0 = show all)",
    )
    args = parser.parse_args()

    # Optionally re-index a video
    if args.video:
        import warnings; warnings.filterwarnings("ignore")
        from src.searcher import NLVideoSearcher
        print(f"Re-indexing {args.video} …")
        s = NLVideoSearcher.from_params(
            index_dir=args.index_dir,
            use_sliding_window=True,
            window_sec=5.0,
            overlap_ratio=0.5,
            frames_per_window=5,
        )
        n = s.index_video(args.video)
        s.save_index()
        print(f"Done — {n} segments indexed, saved to {args.index_dir}\n")

    run_diagnostics(
        index_dir=args.index_dir,
        query=args.query,
        top_k=args.top_k,
        save_frames=args.save_frames,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
