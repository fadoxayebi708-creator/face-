"""gaze_tracker.py

Gaze-estimation stage: iris-position gaze direction and aversion cues.

Consumes a FaceState carrying refined iris landmarks (478-point topology) and
produces a GazeResult. The (gaze_x, gaze_y) vector is written back into
FaceState.gaze to honour the shared pipeline contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from face_pipeline.face_detector import FaceState

logger = logging.getLogger(__name__)

class GazeDirection(str, Enum):
    """Categorical gaze direction along a single axis."""

    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

@dataclass(frozen=True, slots=True)
class _EyeConfig:
    """Landmark indices defining one eye (image-space)."""

    outer: int
    inner: int
    top: int
    bottom: int
    iris_ring: Tuple[int, ...]
    ear_points: Tuple[int, int, int, int, int, int]

# Image-left eye (subject's right). Iris ring 468-472 belongs here.
_LEFT_EYE = _EyeConfig(
    outer=33, inner=133, top=159, bottom=145,
    iris_ring=(468, 469, 470, 471, 472),
    ear_points=(33, 160, 158, 133, 153, 144),
)

# Image-right eye (subject's left). Iris ring 473-477 belongs here.
_RIGHT_EYE = _EyeConfig(
    outer=263, inner=362, top=386, bottom=374,
    iris_ring=(473, 474, 475, 476, 477),
    ear_points=(263, 387, 385, 362, 380, 373),
)

_EPS: float = 1e-6

@dataclass(slots=True)
class EyeGaze:
    """Per-eye gaze descriptor."""

    center: Tuple[float, float]
    horizontal_ratio: float
    vertical_ratio: float
    aspect_ratio: float
    is_open: bool

@dataclass(slots=True)
class GazeResult:
    """Fused gaze estimate for a single face."""

    left_eye: EyeGaze = field(repr=False)
    right_eye: EyeGaze = field(repr=False)
    gaze_x: float
    gaze_y: float
    horizontal: GazeDirection
    vertical: GazeDirection
    is_averted: bool
    confidence: float

    @property
    def vector(self) -> Tuple[float, float]:
        """(gaze_x, gaze_y)."""
        return self.gaze_x, self.gaze_y

    @property
    def label(self) -> str:
        """Human-readable direction, e.g. 'up-left' or 'center'."""
        if self.horizontal is GazeDirection.CENTER and self.vertical is GazeDirection.CENTER:
            return GazeDirection.CENTER.value
        parts = [d.value for d in (self.vertical, self.horizontal) if d is not GazeDirection.CENTER]
        return "-".join(parts)

    def is_reliable(self, min_confidence: float = 0.5) -> bool:
        """Whether the estimate clears a confidence threshold."""
        return self.confidence >= min_confidence

class GazeTracker:
    """Estimates iris-position gaze direction from a FaceState."""

    def __init__(
        self,
        *,
        horizontal_threshold: float = 0.30,
        vertical_threshold: float = 0.30,
        aversion_threshold: float = 0.35,
        min_eye_aspect_ratio: float = 0.15,
        max_yaw: float = 30.0,
        max_pitch: float = 25.0,
    ) -> None:
        for name, value in (
            ("horizontal_threshold", horizontal_threshold),
            ("vertical_threshold", vertical_threshold),
            ("aversion_threshold", aversion_threshold),
            ("min_eye_aspect_ratio", min_eye_aspect_ratio),
            ("max_yaw", max_yaw),
            ("max_pitch", max_pitch),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold
        self.aversion_threshold = aversion_threshold
        self.min_eye_aspect_ratio = min_eye_aspect_ratio
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        logger.info("GazeTracker ready (ear_floor=%.2f)", min_eye_aspect_ratio)

    def estimate(self, face: FaceState) -> Optional[GazeResult]:
        """Estimate gaze for a face and populate face.gaze."""
        if not face.has_iris:
            logger.warning(
                "GazeTracker requires refined iris landmarks; face has %d points",
                face.num_landmarks,
            )
            return None

        landmarks = face.landmarks
        left = self._eye_gaze(landmarks, _LEFT_EYE)
        right = self._eye_gaze(landmarks, _RIGHT_EYE)

        open_eyes: List[EyeGaze] = [e for e in (left, right) if e.is_open]
        fused = open_eyes if open_eyes else [left, right]
        mean_h = float(np.mean([e.horizontal_ratio for e in fused]))
        mean_v = float(np.mean([e.vertical_ratio for e in fused]))

        gaze_x = float(np.clip((mean_h - 0.5) * 2.0, -1.0, 1.0))
        gaze_y = float(np.clip((mean_v - 0.5) * 2.0, -1.0, 1.0))

        horizontal = self._classify(
            gaze_x, self.horizontal_threshold, GazeDirection.LEFT, GazeDirection.RIGHT
        )
        vertical = self._classify(
            gaze_y, self.vertical_threshold, GazeDirection.UP, GazeDirection.DOWN
        )
        is_averted = float(np.hypot(gaze_x, gaze_y)) > self.aversion_threshold
        confidence = self._confidence(open_eyes, face.head_pose)

        face.gaze = (gaze_x, gaze_y)
        return GazeResult(
            left_eye=left, right_eye=right,
            gaze_x=gaze_x, gaze_y=gaze_y,
            horizontal=horizontal, vertical=vertical,
            is_averted=is_averted, confidence=confidence,
        )

    def draw_gaze(self, frame: np.ndarray, gaze: GazeResult, arrow_length: float = 40.0) -> np.ndarray:
        """Draw iris centres and a fused gaze arrow on a frame."""
        colour = (0, 255, 0) if gaze.is_reliable() else (0, 165, 255)
        for eye in (gaze.left_eye, gaze.right_eye):
            cx, cy = int(round(eye.center[0])), int(round(eye.center[1]))
            cv2.circle(frame, (cx, cy), 2, colour, -1, cv2.LINE_AA)
            tip = (
                int(round(cx + gaze.gaze_x * arrow_length)),
                int(round(cy + gaze.gaze_y * arrow_length)),
            )
            cv2.arrowedLine(frame, (cx, cy), tip, colour, 2, cv2.LINE_AA, tipLength=0.3)
        return frame

    def _eye_gaze(self, landmarks: np.ndarray, cfg: _EyeConfig) -> EyeGaze:
        """Compute the gaze descriptor for one eye."""
        iris = landmarks[np.asarray(cfg.iris_ring, dtype=np.intp), :2]
        center = iris.mean(axis=0)

        outer_x = float(landmarks[cfg.outer, 0])
        inner_x = float(landmarks[cfg.inner, 0])
        x_left, x_right = min(outer_x, inner_x), max(outer_x, inner_x)
        horizontal = (float(center[0]) - x_left) / max(x_right - x_left, _EPS)

        top_y = float(landmarks[cfg.top, 1])
        bottom_y = float(landmarks[cfg.bottom, 1])
        y_top, y_bottom = min(top_y, bottom_y), max(top_y, bottom_y)
        vertical = (float(center[1]) - y_top) / max(y_bottom - y_top, _EPS)

        ear = self._eye_aspect_ratio(landmarks, cfg.ear_points)
        return EyeGaze(
            center=(float(center[0]), float(center[1])),
            horizontal_ratio=float(np.clip(horizontal, 0.0, 1.0)),
            vertical_ratio=float(np.clip(vertical, 0.0, 1.0)),
            aspect_ratio=ear,
            is_open=ear >= self.min_eye_aspect_ratio,
        )

    @staticmethod
    def _eye_aspect_ratio(landmarks: np.ndarray, points: Tuple[int, ...]) -> float:
        """Eye-Aspect-Ratio from six ordered eye landmarks."""
        p1, p2, p3, p4, p5, p6 = (landmarks[i, :2] for i in points)
        vertical = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
        horizontal = float(np.linalg.norm(p1 - p4))
        return vertical / (2.0 * max(horizontal, _EPS))

    @staticmethod
    def _classify(
        value: float, threshold: float, negative: GazeDirection, positive: GazeDirection
    ) -> GazeDirection:
        """Map a signed gaze component to a categorical direction."""
        if value < -threshold:
            return negative
        if value > threshold:
            return positive
        return GazeDirection.CENTER

    def _confidence(
        self, open_eyes: List[EyeGaze], head_pose: Optional[Tuple[float, float, float]]
    ) -> float:
        """Combine eye-openness and head-frontality into a confidence score."""
        openness = len(open_eyes) / 2.0
        frontal = 1.0
        if head_pose is not None:
            yaw, pitch, _roll = head_pose
            frontal = 1.0 - 0.5 * (abs(yaw) / self.max_yaw + abs(pitch) / self.max_pitch)
            frontal = float(np.clip(frontal, 0.0, 1.0))
        return float(np.clip(openness * frontal, 0.0, 1.0))
