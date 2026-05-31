"""
Multi-Object Detection and Persistent ID Tracking Pipeline
===========================================================
Uses YOLOv8 (detection) + ByteTrack (multi-object tracking)
"""

import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Optional imports – fail gracefully so the file can be imported without GPU
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import supervision as sv
    SV_AVAILABLE = True
except ImportError:
    SV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Stores per-track history for trajectory drawing and statistics."""
    track_id: int
    centroid_history: List[Tuple[int, int]] = field(default_factory=list)
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    frame_first_seen: int = 0
    frame_last_seen: int = 0
    total_distance: float = 0.0
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 255, 0))

    def update(self, cx: int, cy: int, bbox: Tuple, frame_idx: int):
        if self.centroid_history:
            px, py = self.centroid_history[-1]
            self.total_distance += np.hypot(cx - px, cy - py)
        self.centroid_history.append((cx, cy))
        self.bbox_history.append(bbox)
        self.frame_last_seen = frame_idx
        # Keep only last 60 points for trajectory
        if len(self.centroid_history) > 60:
            self.centroid_history.pop(0)


# ---------------------------------------------------------------------------
# Colour palette (visually distinct, BGR)
# ---------------------------------------------------------------------------
PALETTE = [
    (0, 165, 255),   # orange
    (255, 0,   0),   # blue
    (0, 255,   0),   # green
    (0,   0, 255),   # red
    (255, 255,  0),  # cyan
    (180,   0, 255), # magenta-ish
    (0, 255, 200),   # mint
    (200, 255,  0),  # lime
    (255, 128,  0),  # deep-sky blue
    (128,   0, 255), # purple
    (0, 128, 255),   # gold-orange
    (255,   0, 128), # pink-blue
]


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_bbox_with_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    confidence: float,
    color: Tuple[int, int, int],
    label_prefix: str = "ID",
) -> None:
    """Draw a rounded-corner bbox and ID label on the frame."""
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"{label_prefix} {track_id}  {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Label background
    pad = 3
    cv2.rectangle(
        frame,
        (x1, y1 - th - 2 * pad - baseline),
        (x1 + tw + 2 * pad, y1),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (x1 + pad, y1 - pad - baseline),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_trajectory(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    """Draw a fading polyline for the track trajectory."""
    n = len(points)
    for i in range(1, n):
        alpha = i / n  # newer = brighter
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, points[i - 1], points[i], c, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    active_count: int,
    total_ids: int,
) -> None:
    """Top-left HUD: frame counter, FPS, active/total counts."""
    lines = [
        f"Frame : {frame_idx}",
        f"FPS   : {fps:.1f}",
        f"Active: {active_count}",
        f"Total IDs: {total_ids}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y0 = 10, 30
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(frame, line, (x, y), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class SportTracker:
    """
    End-to-end detection + tracking pipeline.

    Parameters
    ----------
    model_path : str
        YOLOv8 model weights (e.g. 'yolov8n.pt', 'yolov8m.pt').
    conf_thresh : float
        Minimum detection confidence.
    iou_thresh : float
        NMS IoU threshold.
    classes : list[int] | None
        COCO class IDs to keep (None = keep all). Use [0] for 'person'.
    frame_skip : int
        Process every N-th frame (1 = every frame).
    draw_trajectory : bool
        Overlay track trajectory polylines.
    device : str
        'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        classes: Optional[List[int]] = None,
        frame_skip: int = 1,
        draw_traj: bool = True,
        device: str = "cpu",
    ):
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes if classes else [0]  # default: person
        self.frame_skip = max(1, frame_skip)
        self.draw_traj = draw_traj
        self.device = device

        print(f"[INFO] Loading model: {model_path}  device={device}")
        self.model = YOLO(model_path)

        # Track state registry
        self.tracks: Dict[int, TrackState] = {}
        self.total_ids_seen: set = set()

    # ------------------------------------------------------------------
    def _run_detection(self, frame: np.ndarray):
        """Run YOLOv8 + ByteTrack on a single frame."""
        results = self.model.track(
            frame,
            persist=True,               # ByteTrack memory across calls
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            device=self.device,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        return results[0]

    # ------------------------------------------------------------------
    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        Run the full pipeline on a video file.

        Returns
        -------
        dict with summary statistics.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_fps = src_fps / self.frame_skip

        print(f"[INFO] Input : {input_path}")
        print(f"[INFO] Resolution: {src_w}x{src_h}  FPS: {src_fps:.1f}  Frames: {total_src}")
        print(f"[INFO] Output: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (src_w, src_h))

        frame_idx = 0
        written   = 0
        t_start   = time.time()

        # Per-frame active count for statistics
        active_counts: List[int] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and written >= max_frames:
                break

            # Skip frames
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            # ---- Detection + Tracking ----
            result = self._run_detection(frame)

            active_ids_this_frame = set()

            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                track_ids  = result.boxes.id.cpu().numpy().astype(int)
                confs      = result.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), tid, conf in zip(boxes_xyxy, track_ids, confs):
                    tid = int(tid)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Register new track
                    if tid not in self.tracks:
                        self.tracks[tid] = TrackState(
                            track_id=tid,
                            frame_first_seen=frame_idx,
                            color=id_to_color(tid),
                        )

                    ts = self.tracks[tid]
                    ts.update(cx, cy, (x1, y1, x2, y2), frame_idx)
                    self.total_ids_seen.add(tid)
                    active_ids_this_frame.add(tid)

                    # Draw
                    draw_bbox_with_label(
                        frame, x1, y1, x2, y2,
                        tid, conf, ts.color,
                    )
                    if self.draw_traj and len(ts.centroid_history) > 1:
                        draw_trajectory(frame, ts.centroid_history, ts.color)

            # HUD
            elapsed = time.time() - t_start
            proc_fps = (written + 1) / elapsed if elapsed > 0 else 0
            draw_overlay(
                frame, frame_idx, proc_fps,
                len(active_ids_this_frame), len(self.total_ids_seen)
            )

            writer.write(frame)
            active_counts.append(len(active_ids_this_frame))
            written  += 1
            frame_idx += 1

            if written % 50 == 0:
                print(
                    f"  frame {frame_idx:>5} | written {written:>5} "
                    f"| active {len(active_ids_this_frame):>3} "
                    f"| total IDs {len(self.total_ids_seen):>3}"
                )

        cap.release()
        writer.release()

        elapsed = time.time() - t_start
        stats = {
            "frames_processed": written,
            "elapsed_seconds":  round(elapsed, 2),
            "avg_proc_fps":     round(written / elapsed, 2) if elapsed > 0 else 0,
            "total_unique_ids": len(self.total_ids_seen),
            "avg_active_per_frame": round(np.mean(active_counts), 2) if active_counts else 0,
            "max_active_per_frame": int(np.max(active_counts)) if active_counts else 0,
        }

        print("\n[DONE] Summary:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection and Persistent ID Tracking"
    )
    parser.add_argument("--input",       required=True, help="Path to input video")
    parser.add_argument("--output",      default="outputs/tracked.mp4", help="Output video path")
    parser.add_argument("--model",       default="yolov8m.pt", help="YOLOv8 weights")
    parser.add_argument("--conf",        type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou",         type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--frame-skip",  type=int,   default=1,    help="Process every N frames")
    parser.add_argument("--classes",     type=int,   nargs="+", default=[0], help="COCO class IDs (0=person)")
    parser.add_argument("--no-traj",     action="store_true", help="Disable trajectory drawing")
    parser.add_argument("--device",      default="cpu", help="Device: cpu / cuda / mps")
    parser.add_argument("--max-frames",  type=int, default=None, help="Limit output frames (for testing)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    tracker = SportTracker(
        model_path  = args.model,
        conf_thresh = args.conf,
        iou_thresh  = args.iou,
        classes     = args.classes,
        frame_skip  = args.frame_skip,
        draw_traj   = not args.no_traj,
        device      = args.device,
    )

    tracker.process_video(
        input_path  = args.input,
        output_path = args.output,
        max_frames  = args.max_frames,
    )


if __name__ == "__main__":
    main()
