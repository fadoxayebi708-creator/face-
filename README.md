# face-pipeline

Modular, real-time facial-analysis pipeline. Each processing stage is an
independent, strictly-typed class that reads and writes a shared `FaceState`
data contract.

## Status

| Stage | Module | Status |
|-------|--------|--------|
| Face detection / 468-point landmarks | `face_pipeline.face_detector` | Implemented |
| Head pose (solvePnP) | _planned_ | Pending |
| Gaze tracking (iris) | _planned_ | Pending |
| Expression / Action Units | _planned_ | Pending |

## Install

```bash
pip install -r requirements.txt
# or, as a package (editable):
pip install -e .
```

Requires Python 3.10+.

## Quick start

```python
import cv2
from face_pipeline import FaceDetector

with FaceDetector(max_num_faces=1) as detector:
    frame = cv2.imread("face.jpg")          # (H, W, 3) uint8 BGR
    face = detector.detect_primary(frame)
    if face is not None:
        print(face.face_id, face.num_landmarks, face.bbox.xyxy)
        pose_pts = face.get_pose_landmarks_2d()   # (6, 2) for solvePnP
```

## Webcam smoke test

```bash
python -m face_pipeline.face_detector
```

Draws bounding boxes, landmark counts, in-frame confidence, and live FPS.
Press `q` to quit.
