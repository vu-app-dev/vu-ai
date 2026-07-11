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

logger = logging.getLogger(__name__)

YUNET_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


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
        self._yunet = None
        self._initialized = False
        self._init_error: str | None = None

        if not HAS_CV2:
            self._init_error = "OpenCV not available"
            logger.warning("FaceAnalyzer disabled: %s", self._init_error)
            return

        yunet_path = Path(model_path) if model_path else YUNET_MODEL_PATH
        if not yunet_path.exists():
            self._init_error = f"YuNet model not found at {yunet_path}"
            logger.warning("FaceAnalyzer disabled: %s", self._init_error)
            return

        try:
            self._yunet = cv2.FaceDetectorYN.create(
                str(yunet_path), "", (320, 320),
                score_threshold=0.6, nms_threshold=0.3, top_k=2,
            )
            self._initialized = True
            logger.info("FaceAnalyzer initialized with OpenCV YuNet")
        except Exception as e:
            self._init_error = f"YuNet init failed: {e}"
            logger.warning("FaceAnalyzer disabled: %s", self._init_error)

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def backend(self) -> str | None:
        return "yunet" if self._initialized else None

    def analyze_frame(self, image_bytes: bytes) -> FaceAnalysisResult:
        if not self._initialized:
            return FaceAnalysisResult()

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return FaceAnalysisResult()

            return self._analyze_yunet(img)
        except Exception as e:
            logger.warning("FaceAnalyzer frame error: %s", e)
            return FaceAnalysisResult()

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
        eye_mid_x = (right_eye_x + left_eye_x) / 2.0
        eye_mid_y = (right_eye_y + left_eye_y) / 2.0

        nose_offset_x = (nose_x - eye_mid_x) / eye_dx if eye_dx > 0 else 0.0
        nose_offset_y = (nose_y - eye_mid_y) / bbox_h if bbox_h > 0 else 0.0

        horizontal_gaze = abs(nose_offset_x)
        vertical_gaze = abs(nose_offset_y)

        contact_score = max(0.0, min(100.0, (1.0 - horizontal_gaze * 2.0) * 100.0))
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

    def close(self):
        self._yunet = None
        self._initialized = False
