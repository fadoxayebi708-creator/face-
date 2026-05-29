"""face_pipeline

Modular real-time facial-analysis pipeline.

Currently implemented stages:
    * Face detection / landmarks (:mod:`face_pipeline.face_detector`)
"""

from __future__ import annotations

from face_pipeline.face_detector import (
    BoundingBox,
    FaceDetector,
    FaceMeshIndices,
    FaceState,
)

__all__ = [
    "BoundingBox",
    "FaceDetector",
    "FaceMeshIndices",
    "FaceState",
]

__version__ = "0.1.0"
