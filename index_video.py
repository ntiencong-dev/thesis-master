"""
index_video.py
--------------
Command-line tool for building or querying a NLVS Faiss index.

Examples
--------
# Index a single video
python index_video.py index --video ./data/sample.mp4 --index-dir ./index_store

# Index all videos in a directory
python index_video.py index --video-dir ./data/videos --index-dir ./index_store

# Search using the saved index
python index_video.py search \
    --query "person climbing a fence" \
    --index-dir ./index_store \
    --top-k 5

# Index + immediately search (pipeline mode)
python index_video.py index \
    --video ./data/sample.mp4 \
    --index-dir ./index_store \
    --query "red truck passing the gate"
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Natural Language Video Search — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- index sub-command ---
    idx = sub.add_parser("index", help="Build the Faiss index from video(s).")
    grp = idx.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video",     metavar="PATH",  help="Single video file.")
    grp.add_argument("--video-dir", metavar="DIR",   help="Directory of videos.")
    idx.add_argument(
        "--index-dir", metavar="DIR", default="./index_store",
        help="Directory to save the Faiss index.",
    )
    idx.add_argument(
        "--window-sec", type=float, default=10.0,
        help="Sliding-window duration (seconds).",
    )
    idx.add_argument(
        "--overlap", type=float, default=0.30,
        help="Overlap ratio [0, 1). 0.30 = 30%%.",
    )
    idx.add_argument(
        "--keyframe", action="store_true",
        help="Use 1-FPS keyframe mode instead of sliding window.",
    )
    idx.add_argument("--query", metavar="TEXT", help="Optional: search immediately after indexing.")
    idx.add_argument("--top-k", type=int, default=5)

    # --- search sub-command ---
    srch = sub.add_parser("search", help="Query an existing index.")
    srch.add_argument("--query",     required=True, metavar="TEXT")
    srch.add_argument("--index-dir", required=True, metavar="DIR")
    srch.add_argument("--top-k",     type=int, default=5)

    return parser


def _print_results(results) -> None:
    if not results:
        print("No results found.")
        return
    print(f"\n{'Rank':<6} {'Score':<8} {'Video':<30} {'Start':>8} {'End':>8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r.rank:<6} {r.score:<8.4f} {r.video_id[:30]:<30} "
            f"{r.start_time:>8.2f} {r.end_time:>8.2f}"
        )
    print()


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # Lazy import — avoids loading PyTorch when running --help
    from src.searcher import NLVideoSearcher

    if args.command == "index":
        searcher = NLVideoSearcher(
            index_dir=args.index_dir,
            use_sliding_window=not args.keyframe,
            window_sec=args.window_sec,
            overlap_ratio=args.overlap,
        )

        if args.video:
            n = searcher.index_video(args.video)
        else:
            n = searcher.index_directory(args.video_dir)

        print(f"\nTotal segments indexed: {n}")

        if args.query:
            print(f"\nSearching for: \"{args.query}\"")
            results = searcher.search(args.query, top_k=args.top_k)
            _print_results(results)

    elif args.command == "search":
        searcher = NLVideoSearcher(index_dir=args.index_dir)
        try:
            searcher.load_index()
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Searching for: \"{args.query}\"")
        results = searcher.search(args.query, top_k=args.top_k)
        _print_results(results)


if __name__ == "__main__":
    main()
