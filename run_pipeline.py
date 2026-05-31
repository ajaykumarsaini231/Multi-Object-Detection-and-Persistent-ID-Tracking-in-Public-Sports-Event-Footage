#!/usr/bin/env python3
"""
run_pipeline.py – One-shot entry point
=======================================
Usage examples
--------------
# Basic run (person tracking):
python run_pipeline.py --input path/to/video.mp4

# With all optional enhancements:
python run_pipeline.py --input path/to/video.mp4 \
    --model yolov8m.pt \
    --conf 0.35 \
    --frame-skip 2 \
    --heatmap \
    --count-chart \
    --speed \
    --device cuda

# Quick smoke-test (first 120 frames only):
python run_pipeline.py --input path/to/video.mp4 --max-frames 120
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracker import SportTracker
from analytics import (
    generate_heatmap,
    plot_object_count,
    estimate_speeds,
    build_metrics_text,
)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    # --- Core ---------------------------------------------------------------
    parser.add_argument("--input",      required=True, help="Input video path or URL")
    parser.add_argument("--output",     default="outputs/tracked.mp4")
    parser.add_argument("--model",      default="yolov8m.pt",
                        help="YOLOv8 weights: yolov8n/s/m/l/x.pt")
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--frame-skip", type=int,   default=1)
    parser.add_argument("--classes",    type=int,   nargs="+", default=[0],
                        help="COCO class IDs (0=person, 32=sports ball, …)")
    parser.add_argument("--no-traj",    action="store_true",
                        help="Disable trajectory lines")
    parser.add_argument("--device",     default="cpu",
                        help="cpu | cuda | mps")
    parser.add_argument("--max-frames", type=int, default=None)

    # --- Enhancements -------------------------------------------------------
    parser.add_argument("--heatmap",     action="store_true", help="Generate movement heatmap")
    parser.add_argument("--count-chart", action="store_true", help="Save object-count-over-time chart")
    parser.add_argument("--speed",       action="store_true", help="Print speed estimates")
    parser.add_argument("--px-per-m",   type=float, default=30.0,
                        help="Pixels-per-metre calibration for speed (default 30)")
    parser.add_argument("--source-url",  default="N/A", help="Original video URL for report")

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Run tracker
    # -----------------------------------------------------------------------
    tracker = SportTracker(
        model_path  = args.model,
        conf_thresh = args.conf,
        iou_thresh  = args.iou,
        classes     = args.classes,
        frame_skip  = args.frame_skip,
        draw_traj   = not args.no_traj,
        device      = args.device,
    )

    import cv2
    cap = cv2.VideoCapture(args.input)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stats = tracker.process_video(
        input_path  = args.input,
        output_path = args.output,
        max_frames  = args.max_frames,
    )

    # -----------------------------------------------------------------------
    # 2. Optional enhancements
    # -----------------------------------------------------------------------
    if args.heatmap:
        generate_heatmap(
            tracker.tracks,
            frame_size=(src_w, src_h),
            output_path="outputs/heatmap.png",
        )

    active_counts = [
        len([tid for tid, ts in tracker.tracks.items()
             if ts.frame_last_seen == fi])
        for fi in range(stats["frames_processed"])
    ]

    if args.count_chart:
        plot_object_count(
            active_counts,
            output_path="outputs/object_count.png",
            fps=src_fps,
            frame_skip=args.frame_skip,
        )

    speeds = {}
    if args.speed:
        speeds = estimate_speeds(
            tracker.tracks,
            fps=src_fps,
            frame_skip=args.frame_skip,
            pixels_per_meter=args.px_per_m,
        )

    # -----------------------------------------------------------------------
    # 3. Metrics report
    # -----------------------------------------------------------------------
    report_text = build_metrics_text(stats, speeds, video_source=args.source_url)
    print("\n" + report_text)

    report_path = "outputs/metrics_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"[INFO] Metrics report saved → {report_path}")
    print(f"[INFO] Annotated video saved → {args.output}")


if __name__ == "__main__":
    main()
