import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"

LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

NOSE_TIP_INDEX = 1
LEFT_EAR_INDEX = 234
RIGHT_EAR_INDEX = 454


@dataclass
class FaceAnalysisResult:
    face_detected: bool = False
    num_faces: int = 0
    eye_contact: float | None = None
    gaze_horizontal: float | None = None

    def to_dict(self) -> dict:
        return {
            "face_detected": self.face_detected,
            "num_faces": self.num_faces,
            "eye_contact": self.eye_contact,
            "gaze_horizontal": self.gaze_horizontal,
        }


class FaceAnalyzer:
    def __init__(self, model_path: str | None = None):
        self._detector: FaceLandmarker | None = None
        self._initialized = False
        self._init_error: str | None = None

        if not HAS_MEDIAPIPE:
            self._init_error = "mediapipe not installed"
            logger.warning("MediaPipe not available, face analysis disabled")
            return

        path = model_path or str(MODEL_PATH)
        if not Path(path).exists():
            self._init_error = f"Model file not found: {path}"
            logger.warning("FaceAnalyzer model not found at %s, face analysis disabled", path)
            return

        try:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=RunningMode.IMAGE,
                num_faces=2,
            )
            self._detector = FaceLandmarker.create_from_options(options)
            self._initialized = True
            logger.info("FaceAnalyzer initialized with MediaPipe Tasks FaceLandmarker")
        except Exception as e:
            self._init_error = str(e)
            logger.warning("FaceAnalyzer init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._initialized

    def analyze_frame(self, image_bytes: bytes) -> FaceAnalysisResult:
        if not self._initialized or self._detector is None:
            return FaceAnalysisResult()

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return FaceAnalysisResult()

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(data=rgb, image_format=mp.ImageFormat.SRGB)
            result = self._detector.detect(mp_image)

            if not result.face_landmarks:
                return FaceAnalysisResult(face_detected=False, num_faces=0)

            num_faces = len(result.face_landmarks)
            primary = result.face_landmarks[0]

            eye_contact = self._compute_eye_contact(primary, img.shape)
            gaze_h = self._compute_gaze_horizontal(primary)

            return FaceAnalysisResult(
                face_detected=True,
                num_faces=num_faces,
                eye_contact=eye_contact,
                gaze_horizontal=gaze_h,
            )
        except Exception as e:
            logger.warning("FaceAnalyzer frame error: %s", e)
            return FaceAnalysisResult()

    def analyze_base64(self, b64_image: str) -> FaceAnalysisResult:
        try:
            image_bytes = base64.b64decode(b64_image)
            return self.analyze_frame(image_bytes)
        except Exception as e:
            logger.warning("FaceAnalyzer base64 decode error: %s", e)
            return FaceAnalysisResult()

    @staticmethod
    def _compute_eye_contact(landmarks, img_shape) -> float | None:
        h, w = img_shape[:2]

        def _center(indices):
            xs = [landmarks[i].x * w for i in indices if i < len(landmarks)]
            ys = [landmarks[i].y * h for i in indices if i < len(landmarks)]
            if not xs:
                return None, None
            return sum(xs) / len(xs), sum(ys) / len(ys)

        left_eye_cx, left_eye_cy = _center(LEFT_EYE_INDICES)
        right_eye_cx, right_eye_cy = _center(RIGHT_EYE_INDICES)
        left_iris_cx, left_iris_cy = _center(LEFT_IRIS_INDICES)
        right_iris_cx, right_iris_cy = _center(RIGHT_IRIS_INDICES)

        if any(v is None for v in [left_eye_cx, right_eye_cx, left_iris_cx, right_iris_cx]):
            return None

        def _eye_width(indices):
            xs = [landmarks[i].x * w for i in indices if i < len(landmarks)]
            if len(xs) < 2:
                return 1.0
            return max(xs) - min(xs)

        left_w = _eye_width(LEFT_EYE_INDICES)
        right_w = _eye_width(RIGHT_EYE_INDICES)

        left_ratio = abs(left_iris_cx - left_eye_cx) / left_w if left_w > 0 else 1.0
        right_ratio = abs(right_iris_cx - right_eye_cx) / right_w if right_w > 0 else 1.0

        avg_offset = (left_ratio + right_ratio) / 2.0

        contact_score = max(0.0, min(100.0, (1.0 - avg_offset * 2.0) * 100.0))
        return round(contact_score, 1)

    @staticmethod
    def _compute_gaze_horizontal(landmarks) -> float | None:
        try:
            nose = landmarks[NOSE_TIP_INDEX]
            left_ear = landmarks[LEFT_EAR_INDEX]
            right_ear = landmarks[RIGHT_EAR_INDEX]

            face_width = right_ear.x - left_ear.x
            if face_width <= 0:
                return None

            nose_relative = (nose.x - left_ear.x) / face_width
            gaze = (nose_relative - 0.5) * 2.0
            return round(gaze, 3)
        except (IndexError, AttributeError):
            return None

    def close(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            self._initialized = False