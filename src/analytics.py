"""
analytics.py – Optional post-processing enhancements
======================================================
• Movement heatmap (per-pixel density)
• Object-count-over-time chart
• Approximate speed estimation (pixels / second → scaled to km/h equivalent)
• Team / role cluster suggestion (colour-k-means on crops)

All functions accept the `SportTracker` instance after video processing.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# 1. Movement heatmap
# ---------------------------------------------------------------------------

def generate_heatmap(
    tracks: dict,          # SportTracker.tracks
    frame_size: tuple,     # (width, height)
    output_path: str = "outputs/heatmap.png",
    sigma: int = 25,
) -> np.ndarray:
    """
    Accumulate centroid positions into a density map and save as an image.
    Returns the colourised heatmap (BGR numpy array).
    """
    w, h = frame_size
    density = np.zeros((h, w), dtype=np.float32)

    for ts in tracks.values():
        for cx, cy in ts.centroid_history:
            if 0 <= cx < w and 0 <= cy < h:
                density[cy, cx] += 1.0

    # Gaussian blur to spread point masses
    density = cv2.GaussianBlur(density, (0, 0), sigma)

    # Normalise → 0–255
    if density.max() > 0:
        density = (density / density.max() * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(density, cv2.COLORMAP_JET)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, heatmap_bgr)
    print(f"[analytics] Heatmap saved → {output_path}")
    return heatmap_bgr


# ---------------------------------------------------------------------------
# 2. Object count over time
# ---------------------------------------------------------------------------

def plot_object_count(
    active_counts: list,       # list of int, one per processed frame
    output_path: str = "outputs/object_count.png",
    fps: float = 30.0,
    frame_skip: int = 1,
):
    """
    Save a simple line-chart of active object count per second using OpenCV
    (avoids matplotlib dependency).
    """
    if not active_counts:
        return

    n = len(active_counts)
    W, H = 960, 400
    margin = 60
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 30  # dark background

    max_val = max(active_counts) or 1
    effective_fps = fps / frame_skip

    # Grid lines
    for yv in range(0, max_val + 1, max(1, max_val // 5)):
        yp = H - margin - int(yv / max_val * (H - 2 * margin))
        cv2.line(canvas, (margin, yp), (W - margin, yp), (60, 60, 60), 1)
        cv2.putText(canvas, str(yv), (5, yp + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Plot line
    pts = []
    for i, cnt in enumerate(active_counts):
        xp = margin + int(i / (n - 1) * (W - 2 * margin)) if n > 1 else W // 2
        yp = H - margin - int(cnt / max_val * (H - 2 * margin))
        pts.append((xp, yp))

    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (0, 200, 255), 2, cv2.LINE_AA)

    # Axis labels
    cv2.putText(canvas, "Seconds", (W // 2 - 30, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "Active Objects", (5, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, "Object Count Over Time", (W // 2 - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # X-axis ticks (every 5 s)
    tick_interval = int(5 * effective_fps)
    for fi in range(0, n, max(1, tick_interval)):
        xp = margin + int(fi / (n - 1) * (W - 2 * margin)) if n > 1 else W // 2
        secs = fi / effective_fps
        cv2.putText(canvas, f"{secs:.0f}s", (xp - 10, H - margin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"[analytics] Object count chart saved → {output_path}")
    return canvas


# ---------------------------------------------------------------------------
# 3. Speed estimation
# ---------------------------------------------------------------------------

def estimate_speeds(
    tracks: dict,
    fps: float,
    frame_skip: int = 1,
    pixels_per_meter: float = 30.0,
) -> Dict[int, float]:
    """
    Estimate average speed (m/s) for each track.

    Parameters
    ----------
    pixels_per_meter : float
        Calibration constant. Adjust for your video's real-world scale.
        A rough default of 30 px/m covers many broadcast sports shots.
    """
    effective_fps = fps / frame_skip
    speeds = {}
    for tid, ts in tracks.items():
        n = len(ts.centroid_history)
        if n < 2:
            speeds[tid] = 0.0
            continue
        duration_sec = n / effective_fps
        dist_m = ts.total_distance / pixels_per_meter
        speeds[tid] = round(dist_m / duration_sec, 2)
    return speeds


# ---------------------------------------------------------------------------
# 4. Simple metrics report text
# ---------------------------------------------------------------------------

def build_metrics_text(
    stats: dict,
    speeds: Dict[int, float],
    video_source: str = "N/A",
) -> str:
    lines = [
        "=" * 60,
        "  TRACKING METRICS REPORT",
        "=" * 60,
        f"  Video source     : {video_source}",
        f"  Frames processed : {stats.get('frames_processed', '?')}",
        f"  Processing time  : {stats.get('elapsed_seconds', '?')} s",
        f"  Avg proc FPS     : {stats.get('avg_proc_fps', '?')}",
        f"  Total unique IDs : {stats.get('total_unique_ids', '?')}",
        f"  Avg active/frame : {stats.get('avg_active_per_frame', '?')}",
        f"  Max active/frame : {stats.get('max_active_per_frame', '?')}",
        "",
        "  Per-track speed estimates (m/s approx.):",
    ]
    for tid, spd in sorted(speeds.items()):
        lines.append(f"    ID {tid:>3}: {spd:.2f} m/s  (~{spd*3.6:.1f} km/h)")
    lines.append("=" * 60)
    return "\n".join(lines)
