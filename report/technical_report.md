# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Public Sports Footage

---

### 1. Overview

This report describes a computer vision pipeline that detects all human subjects in a publicly available sports video and assigns each a persistent unique ID across the full duration of the clip. The pipeline handles occlusion, motion blur, scale changes, and similar-looking subjects.

---

### 2. Model / Detector

**YOLOv8-Medium (`yolov8m.pt`)**

YOLOv8 is an anchor-free, single-stage object detector trained on the COCO dataset. The "medium" variant (25 M parameters) was chosen as the default because it achieves a strong speed–accuracy balance (~50 mAP@0.5:0.95 on COCO val, ~40 FPS on a modern CPU at 640 px input). It requires no fine-tuning for the `person` class (COCO class 0), which covers players, athletes, and participants in all tested footage categories.

Detection settings used:

| Hyperparameter | Value | Rationale |
|---|---|---|
| Confidence threshold | 0.35 | Reduces missed detections while limiting false positives |
| IoU NMS threshold | 0.45 | Balanced suppression in crowded scenes |
| Input class | 0 (person) | Sports footage focus; extend to ball (32) as needed |

---

### 3. Tracking Algorithm

**ByteTrack** (Zhang et al., ECCV 2022)

ByteTrack is integrated directly into Ultralytics via `model.track(persist=True, tracker="bytetrack.yaml")`. It operates by:

1. Running YOLO detection each frame.
2. Splitting detections into *high*-confidence (above a threshold, typically 0.5) and *low*-confidence groups.
3. Performing **two-stage Hungarian assignment**:
   - Stage 1: Match high-confidence detections to existing tracks using IoU distance.
   - Stage 2: Match remaining low-confidence detections to unmatched tracks (recovery of occluded subjects).
4. Propagating each track's bounding box using a **Kalman filter** when no detection is matched (track enters "lost" state).
5. Deleting tracks not matched within a configurable buffer window.

---

### 4. Why This Combination

| Consideration | YOLOv8 + ByteTrack |
|---|---|
| No fine-tuning needed | YOLOv8 COCO pre-training covers sports persons |
| Simple deployment | Single pip install, auto weight download |
| Re-ID not required | ByteTrack's two-stage approach recovers IDs via motion model alone |
| Real-time capable | 25–40 FPS on CPU; >100 FPS with CUDA |
| Active maintenance | Ultralytics + ByteTrack are both actively maintained (2024) |

Alternatives considered:

- **SORT** – simpler, but no low-confidence recovery; more ID switches.
- **DeepSORT** – adds appearance Re-ID; slower and needs separate Re-ID model.
- **BoT-SORT** – marginally better HOTA on MOT benchmarks; same inference cost but more complex.

ByteTrack's two-stage matching is the key differentiator for sports footage, where players momentarily overlap and then separate.

---

### 5. How ID Consistency Is Maintained

1. **Session-level `persist=True`**: Ultralytics stores the ByteTrack state (Kalman filter estimates, active tracklet buffer) across successive `model.track()` calls in the same Python session. IDs are never reused within a session.

2. **Kalman Filter motion prediction**: Even if the detector fails on frame *k*, the Kalman filter predicts where the track should be on frame *k+1*, keeping it alive during short occlusions.

3. **Two-stage matching**: Low-confidence detections that would be discarded by SORT are used to re-link tracks that have just re-emerged from occlusion, preventing premature ID termination.

4. **Frame-skip handling**: When `--frame-skip N` is set, ByteTrack's internal buffer and Kalman covariance are updated every N frames. We keep skip ≤ 3 to avoid Kalman divergence.

---

### 6. Challenges Faced

| Challenge | Observed impact | Mitigation applied |
|---|---|---|
| **Player overlap / scrum** | ID switches when players cross paths | ByteTrack two-stage; trajectory history gives visual continuity |
| **Camera pan** | Kalman prediction points to wrong region | Reduce frame-skip; consider optical-flow compensation |
| **Similar appearance** | Occasional Re-ID confusion at long range | Confidence threshold tuning; future: appearance embedding |
| **Partial visibility** | Detections near frame edge are unstable | Lower `conf` threshold helps; boundary filtering as post-process |
| **Motion blur on fast play** | Lower detection confidence → lost tracks | YOLOv8's data augmentation reduces sensitivity; ByteTrack's low-conf stage recovers |

---

### 7. Failure Cases Observed

- **ID proliferation in dense crowd**: In marathon footage with 50+ runners in a narrow band, ID switching increases significantly. Localised NMS and smaller model grid patches would help.
- **Re-entry after 3+ seconds off-screen**: ByteTrack's buffer (default 30 frames at 30 FPS) expires; player re-enters with a new ID. Appearance Re-ID models address this.
- **Goalpost / referee confusion**: In soccer footage, the YOLOv8 model occasionally detects part of a referee's arm as a separate "person". Post-processing minimum bounding-box area filter reduces this.

---

### 8. Possible Improvements

1. **Appearance Re-ID (BoT-SORT / StrongSORT)**: Add an OSNet or ResNet-50 Re-ID head to compare crops across a gallery; enables re-association after long occlusion.

2. **Bird's-eye projection**: Apply homography (manual or automatic via field-line detection) to project player positions to a top-down tactical map.

3. **Team clustering**: K-means on dominant HSV colour in the jersey region of each crop; auto-labels tracks as Team A / B / referee.

4. **Evaluation (HOTA / MOTA / IDF1)**: Annotate a short clip with ground-truth tracks using CVAT and run the py-motmetrics library for objective benchmarking.

5. **TensorRT / ONNX export**: `model.export(format="onnx")` reduces inference time by 3–5× on CPU and 8–10× with TensorRT on GPU.

6. **Streamlit or Gradio demo**: Wrap the pipeline in a lightweight web app for non-technical stakeholders.

---

### 9. References

- Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics
- Zhang, Y. et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- COCO Dataset — https://cocodataset.org
- Aharon, N. et al. *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*. arXiv 2206.14651.
