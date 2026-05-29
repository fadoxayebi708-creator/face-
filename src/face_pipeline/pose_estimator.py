"""pose_estimator.py

Head-pose estimation stage (yaw / pitch / roll) via ``cv2.solvePnP``.

Consumes a :class:`~face_pipeline.face_detector.FaceState`, solves the
perspective-n-point problem against a canonical 3D face model, and returns a
:class:`HeadPose`. The ``(yaw, pitch, roll)`` tuple is also written back into
``FaceState.head_pose`` to honour the shared pipeline contract.

Conventions
-----------
Angles are right-handed rotations about the OpenCV camera axes (X right,
Y down, Z into the scene), in **degrees**, and are ~0 when the subject faces
the camera:

* ``pitch`` - rotation about X (nodding; looking down is positive by default).
* ``yaw``   - rotation about Y (turning; toward image-left is positive by default).
* ``roll``  - rotation about Z (in-plane head tilt; clockwise-in-image positive).

Signs depend on camera mounting; use ``invert_yaw`` / ``invert_pitch`` /
``invert_roll`` to match your setup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from face_pipeline.face_detector import FaceMeshIndices, FaceState

logger = logging.getLogger(__name__)


#: Canonical 3D face model (millimetres) in a **Y-down** frame matching OpenCV
#: image axes, so a frontal face yields a near-identity rotation. Row order
#: matches :attr:`FaceMeshIndices.POSE_LANDMARKS`:
#: nose tip, chin, left-eye outer, right-eye outer, mouth-left, mouth-right.
_MODEL_POINTS_3D: np.ndarray = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, 330.0, -65.0),    # chin            (below nose -> +Y)
        (-225.0, -170.0, -135.0),  # left eye outer  (image-left, above -> -Y)
        (225.0, -170.0, -135.0),   # right eye outer
        (-150.0, 150.0, -125.0),   # mouth left
        (150.0, 150.0, -125.0),    # mouth right
    ],
    dtype=np.float64,
)

#: Minimum landmark count required (highest POSE_LANDMARKS index + 1).
_MIN_REQUIRED_LANDMARKS: int = max(FaceMeshIndices.POSE_LANDMARKS) + 1


# --------------------------------------------------------------------------- #
# Structured output                                                           #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class HeadPose:
    """Head-pose solution for a single face.

    Attributes:
        yaw: Rotation about the camera Y axis, degrees.
        pitch: Rotation about the camera X axis, degrees.
        roll: Rotation about the camera Z axis, degrees.
        rotation_vector: ``(3, 1) float64`` Rodrigues rotation vector.
        translation_vector: ``(3, 1) float64`` translation vector.
        rotation_matrix: ``(3, 3) float64`` rotation matrix.
        reprojection_error: RMS reprojection error of the model points (pixels).
    """

    yaw: float
    pitch: float
    roll: float
    rotation_vector: np.ndarray = field(repr=False)
    translation_vector: np.ndarray = field(repr=False)
    rotation_matrix: np.ndarray = field(repr=False)
    reprojection_error: float

    @property
    def angles(self) -> Tuple[float, float, float]:
        """``(yaw, pitch, roll)`` in degrees."""
        return self.yaw, self.pitch, self.roll

    def is_frontal(self, yaw_thresh: float = 15.0, pitch_thresh: float = 15.0) -> bool:
        """Whether the head is roughly facing the camera.

        Args:
            yaw_thresh: Max absolute yaw (degrees) considered frontal.
            pitch_thresh: Max absolute pitch (degrees) considered frontal.

        Returns:
            ``True`` if both yaw and pitch are within thresholds.
        """
        return abs(self.yaw) <= yaw_thresh and abs(self.pitch) <= pitch_thresh

    def is_reliable(self, max_reprojection_error: float = 10.0) -> bool:
        """Whether the solve is trustworthy based on reprojection error.

        Args:
            max_reprojection_error: Maximum acceptable RMS error (pixels).

        Returns:
            ``True`` if the reprojection error is within tolerance.
        """
        return self.reprojection_error <= max_reprojection_error


# --------------------------------------------------------------------------- #
# Estimator                                                                   #
# --------------------------------------------------------------------------- #
class PoseEstimator:
    """Estimates head pose from a :class:`FaceState` using ``cv2.solvePnP``.

    Example:
        >>> detector = FaceDetector(max_num_faces=1)
        >>> pose_estimator = PoseEstimator()
        >>> face = detector.detect_primary(frame_bgr)
        >>> if face is not None:
        ...     pose = pose_estimator.estimate(face)
        ...     if pose and pose.is_reliable():
        ...         print(pose.angles)
    """

    def __init__(
        self,
        *,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        model_points: Optional[np.ndarray] = None,
        solve_flags: int = cv2.SOLVEPNP_ITERATIVE,
        max_reprojection_error: float = 10.0,
        temporal_guess: bool = False,
        invert_yaw: bool = False,
        invert_pitch: bool = False,
        invert_roll: bool = False,
    ) -> None:
        """Initialise the estimator.

        Args:
            camera_matrix: Optional calibrated ``(3, 3)`` intrinsics. If ``None``,
                a per-frame-size matrix is synthesised (focal = width, centre =
                image centre).
            dist_coeffs: Optional distortion coefficients. Defaults to zeros.
            model_points: Optional ``(6, 3)`` 3D model overriding the default.
            solve_flags: ``cv2.SOLVEPNP_*`` flag for the solver.
            max_reprojection_error: Default RMS error tolerance for reliability.
            temporal_guess: If ``True`` warm-start each solve with the previous
                frame's pose (single-subject assumption; reduces jitter on video).
            invert_yaw: Negate the yaw output.
            invert_pitch: Negate the pitch output.
            invert_roll: Negate the roll output.

        Raises:
            ValueError: If ``model_points`` is not shaped ``(6, 3)`` or
                ``camera_matrix`` is not ``(3, 3)``.
        """
        model = _MODEL_POINTS_3D if model_points is None else np.asarray(model_points, np.float64)
        if model.shape != (6, 3):
            raise ValueError(f"model_points must be (6, 3); got {model.shape}")
        if camera_matrix is not None and np.asarray(camera_matrix).shape != (3, 3):
            raise ValueError("camera_matrix must be (3, 3)")

        self._model: np.ndarray = model
        self._fixed_k: Optional[np.ndarray] = (
            None if camera_matrix is None else np.asarray(camera_matrix, np.float64)
        )
        self._dist: np.ndarray = (
            np.zeros((4, 1), np.float64) if dist_coeffs is None
            else np.asarray(dist_coeffs, np.float64)
        )
        self._flags: int = solve_flags
        self.max_reprojection_error: float = max_reprojection_error
        self.temporal_guess: bool = temporal_guess
        self._signs: Tuple[float, float, float] = (
            -1.0 if invert_yaw else 1.0,
            -1.0 if invert_pitch else 1.0,
            -1.0 if invert_roll else 1.0,
        )

        self._k_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._prev_rvec: Optional[np.ndarray] = None
        self._prev_tvec: Optional[np.ndarray] = None
        logger.info("PoseEstimator ready (temporal_guess=%s, flags=%d)", temporal_guess, solve_flags)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def estimate(self, face: FaceState) -> Optional[HeadPose]:
        """Estimate head pose for a face and populate ``face.head_pose``.

        Args:
            face: A populated :class:`FaceState`.

        Returns:
            A :class:`HeadPose`, or ``None`` if landmarks are insufficient or
            the PnP solve fails.
        """
        if face.num_landmarks < _MIN_REQUIRED_LANDMARKS:
            logger.warning(
                "Insufficient landmarks for pose (%d < %d)",
                face.num_landmarks,
                _MIN_REQUIRED_LANDMARKS,
            )
            return None

        image_points = face.get_pose_landmarks_2d()
        height, width = face.frame_shape
        k = self._camera_matrix_for(width, height)

        use_guess = self.temporal_guess and self._prev_rvec is not None
        if use_guess:
            ok, rvec, tvec = cv2.solvePnP(
                self._model, image_points, k, self._dist,
                self._prev_rvec.copy(), self._prev_tvec.copy(),
                useExtrinsicGuess=True, flags=self._flags,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                self._model, image_points, k, self._dist, flags=self._flags,
            )

        if not ok:
            logger.warning("solvePnP failed for face_id=%d", face.face_id)
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
        sign_yaw, sign_pitch, sign_roll = self._signs
        yaw, pitch, roll = yaw * sign_yaw, pitch * sign_pitch, roll * sign_roll

        error = self._reprojection_error(image_points, rvec, tvec, k)

        if self.temporal_guess:
            self._prev_rvec, self._prev_tvec = rvec, tvec

        pose = HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rotation_vector=rvec,
            translation_vector=tvec,
            rotation_matrix=rotation_matrix,
            reprojection_error=error,
        )
        face.head_pose = pose.angles

        if error > self.max_reprojection_error:
            logger.debug(
                "High reprojection error %.2fpx for face_id=%d", error, face.face_id
            )
        return pose

    def draw_pose_axes(
        self, frame: np.ndarray, head_pose: HeadPose, length: float = 80.0
    ) -> np.ndarray:
        """Draw the projected 3D pose axes (X red, Y green, Z blue) on a frame.

        Args:
            frame: ``(H, W, 3) uint8`` BGR image, modified in place.
            head_pose: A :class:`HeadPose` produced by :meth:`estimate`.
            length: Axis length in model units (millimetres).

        Returns:
            The same frame, with axes drawn from the nose origin.
        """
        height, width = frame.shape[:2]
        k = self._camera_matrix_for(width, height)
        axes = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
        projected, _ = cv2.projectPoints(
            axes, head_pose.rotation_vector, head_pose.translation_vector, k, self._dist
        )
        origin, x_end, y_end, z_end = (self._as_point(p) for p in projected.reshape(-1, 2))
        cv2.line(frame, origin, x_end, (0, 0, 255), 2, cv2.LINE_AA)   # X
        cv2.line(frame, origin, y_end, (0, 255, 0), 2, cv2.LINE_AA)   # Y
        cv2.line(frame, origin, z_end, (255, 0, 0), 2, cv2.LINE_AA)   # Z
        return frame

    def reset(self) -> None:
        """Clear the temporal warm-start cache (call on scene/subject change)."""
        self._prev_rvec = None
        self._prev_tvec = None

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _camera_matrix_for(self, width: int, height: int) -> np.ndarray:
        """Return calibrated intrinsics, or a cached synthetic matrix by size."""
        if self._fixed_k is not None:
            return self._fixed_k
        key = (width, height)
        cached = self._k_cache.get(key)
        if cached is None:
            focal = float(width)
            cached = np.array(
                [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            self._k_cache[key] = cached
        return cached

    def _reprojection_error(
        self, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray
    ) -> float:
        """RMS reprojection error (pixels) of the model under the solved pose."""
        projected, _ = cv2.projectPoints(self._model, rvec, tvec, k, self._dist)
        projected = projected.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))

    @staticmethod
    def _rotation_matrix_to_euler(r: np.ndarray) -> Tuple[float, float, float]:
        """Decompose a rotation matrix into ``(pitch, yaw, roll)`` degrees.

        Uses a Tait-Bryan (X-Y-Z) decomposition with gimbal-singularity
        handling. Returned order is ``(pitch_about_x, yaw_about_y, roll_about_z)``.
        """
        sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
        if sy >= 1e-6:
            pitch = np.arctan2(r[2, 1], r[2, 2])
            yaw = np.arctan2(-r[2, 0], sy)
            roll = np.arctan2(r[1, 0], r[0, 0])
        else:  # gimbal lock
            pitch = np.arctan2(-r[1, 2], r[1, 1])
            yaw = np.arctan2(-r[2, 0], sy)
            roll = 0.0
        return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))

    @staticmethod
    def _as_point(p: np.ndarray) -> Tuple[int, int]:
        """Convert a projected ``(2,)`` point to an integer pixel tuple."""
        return int(round(float(p[0]))), int(round(float(p[1])))


# --------------------------------------------------------------------------- #
# Manual smoke test (webcam). Run: python -m face_pipeline.pose_estimator      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    from face_pipeline.face_detector import FaceDetector

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera index 0")

    prev = time.monotonic()
    with FaceDetector(max_num_faces=1) as detector:
        estimator = PoseEstimator(temporal_guess=True)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Frame grab failed; stopping")
                    break

                face = detector.detect_primary(frame)
                if face is not None:
                    pose = estimator.estimate(face)
                    if pose is not None:
                        estimator.draw_pose_axes(frame, pose)
                        label = (
                            f"yaw={pose.yaw:+6.1f} pitch={pose.pitch:+6.1f} "
                            f"roll={pose.roll:+6.1f} err={pose.reprojection_error:4.1f}px"
                        )
                        colour = (0, 255, 0) if pose.is_reliable() else (0, 165, 255)
                        cv2.putText(
                            frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2, cv2.LINE_AA,
                        )
                else:
                    estimator.reset()

                now = time.monotonic()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                cv2.putText(
                    frame, f"FPS: {fps:5.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow("PoseEstimator smoke test (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
