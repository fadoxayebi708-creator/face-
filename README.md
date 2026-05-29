# face-pipeline

Modular, real-time facial-analysis pipeline. Each processing stage is an
independent, strictly-typed class that reads and writes a shared `FaceState`
data contract.

## Status

| Stage | Module | Status |
|-------|--------|--------|
| Face detection / 468-point landmarks | `face_pipeline.face_detector` | Implemented |
| Head pose (solvePnP) | `face_pipeline.pose_estimator` | Implemented |
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

Add head pose on top of detection:

```python
from face_pipeline import FaceDetector, PoseEstimator

detector = FaceDetector(max_num_faces=1)
estimator = PoseEstimator(temporal_guess=True)

face = detector.detect_primary(frame)
if face is not None:
    pose = estimator.estimate(face)           # also sets face.head_pose
    if pose is not None and pose.is_reliable():
        print(pose.yaw, pose.pitch, pose.roll)
```

## Webcam smoke test

```bash
python -m face_pipeline.face_detector     # landmarks + bbox + FPS
python -m face_pipeline.pose_estimator    # + projected pose axes & angles
```

Press `q` to quit.
