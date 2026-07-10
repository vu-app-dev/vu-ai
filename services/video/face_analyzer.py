import base64
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None
    np = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None

logger = logging.getLogger(__name__)

MEDIAPIPE_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"
YUNET_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

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
        self._detector = None
        self._yunet = None
        self._backend: str | None = None
        self._initialized = False
        self._init_error: str | None = None

        if HAS_MEDIAPIPE and (model_path or MEDIAPIPE_MODEL_PATH).exists():
            path = model_path or str(MEDIAPIPE_MODEL_PATH)
            try:
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=path),
                    running_mode=RunningMode.IMAGE,
                    num_faces=2,
                )
                self._detector = FaceLandmarker.create_from_options(options)
                self._backend = "mediapipe"
                self._initialized = True
                logger.info("FaceAnalyzer initialized with MediaPipe FaceLandmarker")
                return
            except Exception as e:
                self._init_error = f"MediaPipe init failed: {e}"
                logger.warning("FaceAnalyzer MediaPipe init failed: %s, trying YuNet fallback", e)

        if HAS_CV2 and YUNET_MODEL_PATH.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(YUNET_MODEL_PATH), "", (320, 320),
                    score_threshold=0.6, nms_threshold=0.3, top_k=2,
                )
                self._backend = "yunet"
                self._initialized = True
                logger.info("FaceAnalyzer initialized with OpenCV YuNet fallback")
                return
            except Exception as e:
                self._init_error = f"YuNet init failed: {e}"
                logger.warning("FaceAnalyzer YuNet init failed: %s", e)

        if not self._initialized:
            if not HAS_CV2:
                self._init_error = "Neither MediaPipe nor OpenCV available"
            elif not YUNET_MODEL_PATH.exists() and not (model_path or MEDIAPIPE_MODEL_PATH).exists():
                self._init_error = "No face detection model files found"
            logger.warning("FaceAnalyzer disabled: %s", self._init_error)

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def backend(self) -> str | None:
        return self._backend

    def analyze_frame(self, image_bytes: bytes) -> FaceAnalysisResult:
        if not self._initialized:
            return FaceAnalysisResult()

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return FaceAnalysisResult()

            if self._backend == "mediapipe":
                return self._analyze_mediapipe(img)
            elif self._backend == "yunet":
                return self._analyze_yunet(img)
            return FaceAnalysisResult()
        except Exception as e:
            logger.warning("FaceAnalyzer frame error: %s", e)
            return FaceAnalysisResult()

    def _analyze_mediapipe(self, img) -> FaceAnalysisResult:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(data=rgb, image_format=mp.ImageFormat.SRGB)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return FaceAnalysisResult(face_detected=False, num_faces=0)

        num_faces = len(result.face_landmarks)
        primary = result.face_landmarks[0]

        eye_contact = self._compute_eye_contact_mp(primary, img.shape)
        gaze_h = self._compute_gaze_horizontal_mp(primary)

        return FaceAnalysisResult(
            face_detected=True,
            num_faces=num_faces,
            eye_contact=eye_contact,
            gaze_horizontal=gaze_h,
        )

    def _analyze_yunet(self, img) -> FaceAnalysisResult:
        h, w = img.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(img)

        if faces is None or len(faces) == 0:
            return FaceAnalysisResult(face_detected=False, num_faces=0)

        num_faces = len(faces)
        primary = faces[0]

        right_eye_x, right_eye_y = primary[4], primary[5]
        left_eye_x, left_eye_y = primary[6], primary[7]
        nose_x, nose_y = primary[8], primary[9]

        bbox_w = primary[2]
        bbox_h = primary[3]
        bbox_x = primary[0]

        eye_dx = abs(right_eye_x - left_eye_x)
        eye_dy = abs(right_eye_y - left_eye_y)
        eye_mid_x = (right_eye_x + left_eye_x) / 2.0
        eye_mid_y = (right_eye_y + left_eye_y) / 2.0

        nose_offset_x = (nose_x - eye_mid_x) / eye_dx if eye_dx > 0 else 0.0
        nose_offset_y = (nose_y - eye_mid_y) / eye_dy if eye_dy > 0 else 0.0

        horizontal_gaze = abs(nose_offset_x)
        vertical_gaze = abs(nose_offset_y)

        contact_score = max(0.0, min(100.0, (1.0 - horizontal_gaze * 3.0) * 100.0))
        contact_score = contact_score * max(0.0, min(1.0, 1.0 - vertical_gaze * 0.5))

        face_width = eye_dx if eye_dx > 0 else 1.0
        gaze_h = (nose_x - bbox_x - bbox_w / 2.0) / (face_width / 2.0) if face_width > 0 else 0.0
        gaze_h = max(-1.0, min(1.0, gaze_h))

        return FaceAnalysisResult(
            face_detected=True,
            num_faces=num_faces,
            eye_contact=round(contact_score, 1),
            gaze_horizontal=round(gaze_h, 3),
        )

    def analyze_base64(self, b64_image: str) -> FaceAnalysisResult:
        try:
            image_bytes = base64.b64decode(b64_image)
            return self.analyze_frame(image_bytes)
        except Exception as e:
            logger.warning("FaceAnalyzer base64 decode error: %s", e)
            return FaceAnalysisResult()

    @staticmethod
    def _compute_eye_contact_mp(landmarks, img_shape) -> float | None:
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
    def _compute_gaze_horizontal_mp(landmarks) -> float | None:
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
        self._yunet = None
        self._initialized = False
