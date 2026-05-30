"""face_pipeline

Modular real-time facial-analysis pipeline.
"""

from __future__ import annotations

from face_pipeline.face_detector import (
    BoundingBox,
    FaceDetector,
    FaceMeshIndices,
    FaceState,
)
from face_pipeline.pose_estimator import HeadPose, PoseEstimator
from face_pipeline.gaze_tracker import EyeGaze, GazeDirection, GazeResult, GazeTracker

__all__ = [
    "BoundingBox",
    "FaceDetector",
    "FaceMeshIndices",
    "FaceState",
    "HeadPose",
    "PoseEstimator",
    "EyeGaze",
    "GazeDirection",
    "GazeResult",
    "GazeTracker",
]

__version__ = "0.3.0"
