# Multi-Object Detection & Persistent ID Tracking
### YOLOv8 + ByteTrack | Sports / Event Footage

---

## Quick start

```bash
git clone <repo-url>
cd sports_tracker
pip install -r requirements.txt

# Basic run
python run_pipeline.py --input path/to/your_video.mp4

# Full pipeline with all enhancements
python run_pipeline.py \
  --input    path/to/your_video.mp4 \
  --model    yolov8m.pt \
  --conf     0.35 \
  --frame-skip 2 \
  --heatmap --count-chart --speed \
  --source-url "https://youtube.com/watch?v=..."
```

Or open `notebook.ipynb` in JupyterLab for an interactive walkthrough.

---

## Installation

### Requirements
- Python 3.9+
- pip

### Steps

```bash
pip install -r requirements.txt
```

On first run, `ultralytics` will automatically download the chosen YOLOv8 weights
(e.g. `yolov8m.pt`, ~50 MB).

### GPU acceleration (optional but recommended for long videos)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# then pass --device cuda to run_pipeline.py
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.2 | YOLOv8 detection + ByteTrack tracker |
| `opencv-python` | ≥ 4.9 | Video I/O, drawing, heatmap |
| `numpy` | ≥ 1.24 | Numerical ops |
| `yt-dlp` | ≥ 2024.1 | Download public video (optional) |
| `pandas` | ≥ 2.0 | Track inspection table (notebook) |

---

## How to run

### CLI

```
python run_pipeline.py --help

Options:
  --input        Input video path (required)
  --output       Output annotated video path [outputs/tracked.mp4]
  --model        YOLOv8 weights file [yolov8m.pt]
                 Choices: yolov8n/s/m/l/x.pt (n=fastest, x=most accurate)
  --conf         Detection confidence threshold [0.35]
  --iou          NMS IoU threshold [0.45]
  --frame-skip   Process every N frames [1]
  --classes      COCO class IDs to detect [0] (0=person, 32=ball)
  --no-traj      Disable trajectory polylines
  --device       cpu | cuda | mps [cpu]
  --max-frames   Limit output frames (useful for quick tests)
  --heatmap      Save movement density heatmap
  --count-chart  Save object-count-over-time chart
  --speed        Print per-track speed estimates
  --px-per-m     Pixels per metre calibration [30.0]
  --source-url   Original video URL (for the metrics report)
```

### Download a public video first (optional)

```bash
# Low-res download for speed
yt-dlp -f "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]" \
       --merge-output-format mp4 \
       -o input_video.mp4 \
       "https://www.youtube.com/watch?v=REPLACE_ME"
```

### Notebook

```bash
jupyter lab notebook.ipynb
```

---

## Project structure

```
sports_tracker/
├── run_pipeline.py         # Main CLI entry point
├── notebook.ipynb          # Interactive Jupyter walkthrough
├── requirements.txt
├── README.md
├── src/
│   ├── tracker.py          # YOLOv8 + ByteTrack core pipeline
│   └── analytics.py        # Heatmap, count chart, speed estimation
├── outputs/                # Generated artifacts (gitignored)
│   ├── tracked.mp4
│   ├── heatmap.png
│   ├── object_count.png
│   └── metrics_report.txt
└── report/
    └── technical_report.md
```

---

## Model & tracker choices

### Detector — YOLOv8

YOLOv8 (Ultralytics, 2023) is selected because:

- **Speed / accuracy Pareto frontier**: `yolov8m` runs at ~25–40 FPS on a modern CPU
  for 640-pixel inputs, while achieving ~50 mAP on COCO.
- **COCO pre-training**: The `person` class (ID 0) is well-trained on diverse
  crowd and sports footage, minimising fine-tuning need.
- **Integrated ByteTrack**: Ultralytics ships ByteTrack as a first-class tracker
  accessible via `model.track(persist=True, tracker="bytetrack.yaml")`, keeping
  the codebase simple.

Model size selection guide:

| Weight | Speed (CPU) | mAP | Recommended for |
|--------|------------|-----|-----------------|
| `yolov8n` | Fastest | ~37 | Prototype / demo |
| `yolov8m` | Medium  | ~50 | **Default (balanced)** |
| `yolov8x` | Slowest | ~54 | High-accuracy offline runs |

### Tracker — ByteTrack

ByteTrack (Zhang et al., 2022) is selected because:

- **Low-confidence recovery**: Keeps track associations using *both* high- and
  low-confidence detections, dramatically reducing ID switches during momentary
  occlusion.
- **Kalman filter motion model**: Predicts where each object will be next frame,
  enabling robust re-identification after short disappearances.
- **No Re-ID network needed**: Operates on bounding-box IoU alone, making it
  model-agnostic and fast.

---

## How ID consistency is maintained

1. **Kalman Filter prediction** — Each active track predicts its next position;
   this prediction is matched against new detections via IoU even when the detector
   misses a frame.
2. **Two-stage matching** — ByteTrack first tries to match detections to active
   tracks (high confidence), then attempts to match remaining detections to
   `lost` tracks (low confidence) before finally spawning new IDs.
3. **`persist=True`** — Ultralytics' session state is retained across `model.track()`
   calls, so ByteTrack's internal memory (Kalman state, tracklet buffer) is not
   reset between frames.
4. **Trajectory history** — Up to 60 previous centroids are stored per track for
   visual overlay and analytics.

---

## Assumptions

- Input video is a standard mp4/avi with a steady frame rate.
- `pixels_per_meter = 30` is a rough default for broadcast-range sports shots.
  Recalibrate using a known field dimension for accurate speed numbers.
- The `person` class (COCO ID 0) covers all human participants. For vehicle
  tracking, add class IDs 2 (car), 5 (bus), 7 (truck), etc.
- Frame timestamps are derived from video FPS metadata; VFR videos may show
  slight drift.

---

## Known limitations

| Challenge | Current handling | Potential improvement |
|-----------|------------------|-----------------------|
| Prolonged occlusion (>2 s) | Track may be lost and reassigned a new ID | Appearance Re-ID (OSNet / BoT-SORT) |
| Very dense crowds (20+ people) | ID switches increase | Smaller grid / ROI crop |
| Fast camera pan | Kalman prediction diverges | Optical flow–guided Kalman |
| Similar jerseys | No colour-based Re-ID | Team clustering (k-means on crops) |
| Fisheye / wide-angle | Bounding-box distortion | Undistort before detect |

---

## Possible improvements

- **Appearance Re-ID**: Replace IoU-only matching with a lightweight embedding
  (e.g. OSNet) for person re-identification across longer occlusions.
- **Bird's-eye projection**: Apply homography to project players onto a top-down
  field view for tactical analysis.
- **Team clustering**: K-means on dominant jersey colours to label tracks as
  Team A / B / referee.
- **Evaluation metrics**: HOTA, MOTA, IDF1 on a ground-truth annotated clip.
- **ONNX / TensorRT export**: 3–10× speedup via `model.export(format="onnx")`.
- **Streamlit demo**: Wrap pipeline in a simple web UI for non-technical users.

---

## Video source

> **Replace this line with your actual public video URL before submission.**
>
> Example: `https://www.youtube.com/watch?v=XXXXXXX`

---

## References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- ByteTrack: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
- COCO Dataset class list: https://cocodataset.org/#explore
