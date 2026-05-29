"""face_detector.py

MediaPipe Face Mesh detection stage for the real-time facial-analysis pipeline.

This module provides two things:

* :class:`FaceState` - the typed data contract that flows through every
  downstream pipeline stage (pose estimation, gaze tracking, action-unit
  analysis, behavioural scoring). Detection stages populate the core fields;
  later stages fill the optional slots.
* :class:`FaceDetector` - a stateful, context-managed wrapper around
  ``mediapipe`` Face Mesh that returns one :class:`FaceState` per detected face.

Design notes
------------
* ``mp.solutions.face_mesh`` exposes neither a detection bounding box nor a
  per-face confidence score, so the box is derived from landmark extents and
  ``detection_confidence`` carries an in-frame-ratio occlusion proxy instead.
* A MediaPipe graph is **not** thread-safe. Instantiate one detector per
  worker thread when this stage is parallelised in the pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Landmark index reference                                                    #
# --------------------------------------------------------------------------- #
class FaceMeshIndices:
    """Named MediaPipe Face Mesh landmark indices used across the pipeline.

    All indices follow MediaPipe's canonical 468-point topology. Iris indices
    (468-477) are only valid when the detector is created with
    ``refine_landmarks=True``. Left/right labels are in **image space** (the
    camera's perspective); the gaze stage canonicalises to subject-relative.
    """

    NOSE_TIP: Final[int] = 1
    CHIN: Final[int] = 152
    LEFT_EYE_OUTER: Final[int] = 33      # image-left eye, outer canthus
    RIGHT_EYE_OUTER: Final[int] = 263    # image-right eye, outer canthus
    LEFT_EYE_INNER: Final[int] = 133
    RIGHT_EYE_INNER: Final[int] = 362
    MOUTH_LEFT: Final[int] = 61
    MOUTH_RIGHT: Final[int] = 291
    FOREHEAD: Final[int] = 10

    #: Canonical 6-point subset consumed by the solvePnP head-pose stage.
    #: Order: nose tip, chin, left-eye outer, right-eye outer, mouth left, mouth right.
    POSE_LANDMARKS: Final[Tuple[int, ...]] = (1, 152, 33, 263, 61, 291)

    #: Iris point groups (require refine_landmarks=True). Index 0 is the centre.
    RIGHT_IRIS: Final[Tuple[int, ...]] = (468, 469, 470, 471, 472)
    LEFT_IRIS: Final[Tuple[int, ...]] = (473, 474, 475, 476, 477)


_NUM_LANDMARKS_BASE: Final[int] = 468
_NUM_LANDMARKS_REFINED: Final[int] = 478


# --------------------------------------------------------------------------- #
# Structured outputs                                                          #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned face bounding box in pixel coordinates.

    Attributes:
        x: Left edge (pixels).
        y: Top edge (pixels).
        width: Box width (pixels).
        height: Box height (pixels).
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """``(cx, cy)`` box centre in pixels."""
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """Corner format ``(x1, y1, x2, y2)``."""
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass(slots=True)
class FaceState:
    """Per-face state object: the shared contract for the whole pipeline.

    Detection stages populate the core fields. Downstream stages
    (pose/gaze/AU/emotion) populate the ``Optional`` slots in place, so a single
    typed object accumulates the full analysis for one face in one frame.

    Attributes:
        face_id: Area-ranked index within the frame (0 = largest/primary face).
        timestamp: Monotonic capture time in seconds.
        frame_shape: Source frame shape as ``(height, width)``.
        landmarks: ``(N, 3) float32`` landmarks in **pixel** space ``(x, y, z)``.
        landmarks_norm: ``(N, 3) float32`` landmarks normalized to ``[0, 1]``.
        bbox: Landmark-derived bounding box, clamped to the frame.
        detection_confidence: In-frame landmark ratio in ``[0, 1]`` (occlusion /
            out-of-frame proxy; MediaPipe Face Mesh exposes no true score).
        head_pose: Optional ``(yaw, pitch, roll)`` in degrees (pose stage).
        gaze: Optional gaze descriptor ``(x, y)`` or richer struct (gaze stage).
        action_units: Optional FACS AU intensities by name (expression stage).
        emotion: Optional emotion-probability map by label (emotion stage).
    """

    face_id: int
    timestamp: float
    frame_shape: Tuple[int, int]
    landmarks: np.ndarray = field(repr=False)
    landmarks_norm: np.ndarray = field(repr=False)
    bbox: BoundingBox
    detection_confidence: float

    # --- Optional slots populated by downstream stages -------------------- #
    head_pose: Optional[Tuple[float, float, float]] = None
    gaze: Optional[Tuple[float, float]] = None
    action_units: Optional[Dict[str, float]] = None
    emotion: Optional[Dict[str, float]] = None

    @property
    def num_landmarks(self) -> int:
        """Number of landmarks carried by this state."""
        return int(self.landmarks.shape[0])

    @property
    def has_iris(self) -> bool:
        """Whether refined iris landmarks (478-point topology) are present."""
        return self.num_landmarks >= _NUM_LANDMARKS_REFINED

    @property
    def area_ratio(self) -> float:
        """Face bbox area as a fraction of the full frame area."""
        h, w = self.frame_shape
        return self.bbox.area / float(max(h * w, 1))

    def is_reliable(self, min_confidence: float = 0.9) -> bool:
        """Return ``True`` when the in-frame landmark ratio clears a threshold.

        Args:
            min_confidence: Minimum acceptable in-frame ratio.

        Returns:
            Whether the detection is considered reliable for analysis.
        """
        return self.detection_confidence >= min_confidence

    def get_landmarks(
        self, indices: Sequence[int], *, normalized: bool = False
    ) -> np.ndarray:
        """Select a subset of landmarks by index.

        Args:
            indices: Landmark indices to extract (e.g. ``FaceMeshIndices.POSE_LANDMARKS``).
            normalized: If ``True`` return normalized coords, else pixel coords.

        Returns:
            ``(len(indices), 3) float32`` array.
        """
        source = self.landmarks_norm if normalized else self.landmarks
        return source[np.asarray(indices, dtype=np.intp)]

    def get_pose_landmarks_2d(self) -> np.ndarray:
        """Return the 6-point 2D pixel set used by the solvePnP head-pose stage.

        Returns:
            ``(6, 2) float64`` image points in :attr:`FaceMeshIndices.POSE_LANDMARKS`
            order, ready to hand to ``cv2.solvePnP``.
        """
        pts = self.landmarks[np.asarray(FaceMeshIndices.POSE_LANDMARKS, dtype=np.intp), :2]
        return pts.astype(np.float64)


# --------------------------------------------------------------------------- #
# Detector                                                                    #
# --------------------------------------------------------------------------- #
class FaceDetector:
    """Stateful MediaPipe Face Mesh wrapper producing :class:`FaceState` objects.

    The underlying graph instance is created once and reused across frames for
    throughput. Use as a context manager, or call :meth:`close` explicitly.

    Example:
        >>> with FaceDetector(max_num_faces=1) as detector:
        ...     faces = detector.detect(frame_bgr)
        ...     primary = faces[0] if faces else None

    Note:
        A MediaPipe graph is not thread-safe; create one detector per thread.
    """

    def __init__(
        self,
        *,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_face_area_ratio: float = 0.0,
    ) -> None:
        """Initialise the detector.

        Args:
            max_num_faces: Maximum number of faces to track per frame.
            refine_landmarks: If ``True`` use the 478-point topology with irises
                (required for the gaze stage).
            static_image_mode: If ``True`` treat every frame independently
                (no inter-frame tracking); use ``False`` for video streams.
            min_detection_confidence: MediaPipe detector gate in ``[0, 1]``.
            min_tracking_confidence: MediaPipe tracker gate in ``[0, 1]``.
            min_face_area_ratio: Reject faces whose bbox area is below this
                fraction of the frame (filters tiny/spurious detections).

        Raises:
            ValueError: If ``max_num_faces`` < 1 or ratios are out of range.
        """
        if max_num_faces < 1:
            raise ValueError("max_num_faces must be >= 1")
        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
            ("min_face_area_ratio", min_face_area_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        self.max_num_faces: int = max_num_faces
        self.refine_landmarks: bool = refine_landmarks
        self.min_face_area_ratio: float = min_face_area_ratio
        self._closed: bool = False

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "FaceDetector ready (max_faces=%d, refine=%s, static=%s)",
            max_num_faces,
            refine_landmarks,
            static_image_mode,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def detect(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> List[FaceState]:
        """Detect faces in a single BGR frame.

        Args:
            frame: ``(H, W, 3) uint8`` BGR image (OpenCV convention).
            timestamp: Optional capture time (seconds). Defaults to
                ``time.monotonic()``.

        Returns:
            A list of :class:`FaceState`, sorted by bbox area descending
            (``face_id == 0`` is the primary subject). Empty if no face passes
            the detector and area-ratio filters.

        Raises:
            RuntimeError: If the detector has been closed.
            TypeError: If ``frame`` is not a uint8 ndarray.
            ValueError: If ``frame`` is not a 3-channel image.
        """
        if self._closed:
            raise RuntimeError("detect() called on a closed FaceDetector")

        self._validate_frame(frame)
        ts: float = time.monotonic() if timestamp is None else timestamp
        height, width = frame.shape[:2]

        # Zero-copy handoff: convert once, mark read-only for MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)

        multi = getattr(results, "multi_face_landmarks", None)
        if not multi:
            logger.debug("No face detected (ts=%.3f)", ts)
            return []

        scale = np.array([width, height, width], dtype=np.float32)
        states: List[FaceState] = []

        for raw in multi:
            norm = self._landmarks_to_array(raw)
            pixel = norm * scale
            bbox = self._compute_bbox(pixel, width, height)

            area_ratio = bbox.area / float(width * height)
            if area_ratio < self.min_face_area_ratio:
                logger.debug("Discarded face: area_ratio %.4f below floor", area_ratio)
                continue

            states.append(
                FaceState(
                    face_id=-1,  # assigned after area ranking
                    timestamp=ts,
                    frame_shape=(height, width),
                    landmarks=pixel,
                    landmarks_norm=norm,
                    bbox=bbox,
                    detection_confidence=self._in_frame_ratio(pixel, width, height),
                )
            )

        states.sort(key=lambda s: s.bbox.area, reverse=True)
        for rank, state in enumerate(states):
            state.face_id = rank

        logger.debug("Detected %d face(s) at ts=%.3f", len(states), ts)
        return states

    def detect_primary(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> Optional[FaceState]:
        """Convenience wrapper returning only the largest face (or ``None``).

        Args:
            frame: ``(H, W, 3) uint8`` BGR image.
            timestamp: Optional capture time (seconds).

        Returns:
            The primary :class:`FaceState`, or ``None`` if no face was found.
        """
        faces = self.detect(frame, timestamp)
        return faces[0] if faces else None

    def close(self) -> None:
        """Release the underlying MediaPipe graph. Idempotent."""
        if not self._closed:
            self._mesh.close()
            self._closed = True
            logger.info("FaceDetector closed")

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate the input frame's type, dimensionality and channel count."""
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be np.ndarray, got {type(frame)!r}")
        if frame.dtype != np.uint8:
            raise TypeError(f"frame must be uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be (H, W, 3) BGR; got shape {frame.shape}"
            )

    @staticmethod
    def _landmarks_to_array(raw_landmarks) -> np.ndarray:  # noqa: ANN001 (protobuf type)
        """Marshal a MediaPipe landmark protobuf into an ``(N, 3) float32`` array."""
        lms = raw_landmarks.landmark
        out = np.empty((len(lms), 3), dtype=np.float32)
        for i, lm in enumerate(lms):
            out[i, 0] = lm.x
            out[i, 1] = lm.y
            out[i, 2] = lm.z
        return out

    @staticmethod
    def _compute_bbox(pixel_landmarks: np.ndarray, width: int, height: int) -> BoundingBox:
        """Derive a frame-clamped bbox from landmark pixel extents."""
        xs = pixel_landmarks[:, 0]
        ys = pixel_landmarks[:, 1]
        x1 = int(np.clip(np.floor(xs.min()), 0, width - 1))
        y1 = int(np.clip(np.floor(ys.min()), 0, height - 1))
        x2 = int(np.clip(np.ceil(xs.max()), 0, width - 1))
        y2 = int(np.clip(np.ceil(ys.max()), 0, height - 1))
        return BoundingBox(x=x1, y=y1, width=max(x2 - x1, 0), height=max(y2 - y1, 0))

    @staticmethod
    def _in_frame_ratio(pixel_landmarks: np.ndarray, width: int, height: int) -> float:
        """Fraction of landmarks inside the frame rectangle (occlusion proxy)."""
        xs = pixel_landmarks[:, 0]
        ys = pixel_landmarks[:, 1]
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        return float(np.count_nonzero(inside) / inside.size)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Manual smoke test (webcam). Run: python -m face_pipeline.face_detector       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera index 0")

    prev = time.monotonic()
    with FaceDetector(max_num_faces=2, refine_landmarks=True) as detector:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Frame grab failed; stopping")
                    break

                faces = detector.detect(frame)

                now = time.monotonic()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now

                for face in faces:
                    x1, y1, x2, y2 = face.bbox.xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"id={face.face_id} pts={face.num_landmarks} "
                        f"conf={face.detection_confidence:.2f}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame, f"FPS: {fps:5.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow("FaceDetector smoke test (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
