"""face_pipeline

Modular real-time facial-analysis pipeline.

Currently implemented stages:
    * Face detection / landmarks (:mod:`face_pipeline.face_detector`)
    * Head pose / solvePnP (:mod:`face_pipeline.pose_estimator`)
"""

from __future__ import annotations

from face_pipeline.face_detector import (
    BoundingBox,
    FaceDetector,
    FaceMeshIndices,
    FaceState,
)
from face_pipeline.pose_estimator import HeadPose, PoseEstimator

__all__ = [
    "BoundingBox",
    "FaceDetector",
    "FaceMeshIndices",
    "FaceState",
    "HeadPose",
    "PoseEstimator",
]

__version__ = "0.2.0"
