import numpy as np
import cv2
import base64
import pytest
from unittest.mock import MagicMock, patch

from services.video.face_analyzer import FaceAnalyzer, FaceAnalysisResult


class TestFaceAnalysisResult:
    def test_defaults(self):
        r = FaceAnalysisResult()
        assert r.face_detected is False
        assert r.num_faces == 0
        assert r.eye_contact is None
        assert r.gaze_horizontal is None

    def test_to_dict(self):
        r = FaceAnalysisResult(face_detected=True, num_faces=1, eye_contact=75.0, gaze_horizontal=0.1)
        d = r.to_dict()
        assert d["face_detected"] is True
        assert d["num_faces"] == 1
        assert d["eye_contact"] == 75.0
        assert d["gaze_horizontal"] == 0.1


class TestFaceAnalyzerInit:
    def test_mediapipe_preferred(self):
        fa = FaceAnalyzer()
        assert fa.available is True
        assert fa.backend == "mediapipe"

    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    def test_yunet_fallback(self):
        fa = FaceAnalyzer()
        assert fa.available is True
        assert fa.backend == "yunet"

    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    @patch("services.video.face_analyzer.HAS_CV2", False)
    def test_disabled_when_no_libs(self):
        fa = FaceAnalyzer()
        assert fa.available is False
        assert fa.backend is None


class TestYuNetAnalysis:
    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    def test_no_face_detected(self):
        fa = FaceAnalyzer()
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        result = fa.analyze_frame(buf.tobytes())
        assert result.face_detected is False
        assert result.num_faces == 0

    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    def test_synthetic_face_detected(self):
        fa = FaceAnalyzer()

        mock_faces = np.array([[
            100, 50, 120, 120,
            130, 90,
            170, 90,
            150, 110,
            135, 130,
            165, 130,
            0.95,
        ]], dtype=np.float32)

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, mock_faces)
            result = fa.analyze_frame(buf.tobytes())

        assert result.face_detected is True
        assert result.num_faces == 1
        assert result.eye_contact is not None
        assert 0 <= result.eye_contact <= 100
        assert -1 <= result.gaze_horizontal <= 1

    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    def test_two_faces_detected(self):
        fa = FaceAnalyzer()

        mock_faces = np.array([
            [100, 50, 120, 120, 130, 90, 170, 90, 150, 110, 135, 130, 165, 130, 0.95],
            [200, 50, 120, 120, 230, 90, 270, 90, 250, 110, 235, 130, 265, 130, 0.85],
        ], dtype=np.float32)

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, mock_faces)
            result = fa.analyze_frame(buf.tobytes())

        assert result.face_detected is True
        assert result.num_faces == 2

    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    def test_analyze_base64(self):
        fa = FaceAnalyzer()

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode()

        result = fa.analyze_base64(b64)
        assert isinstance(result, FaceAnalysisResult)
        assert result.face_detected is False

    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    def test_centered_face_high_eye_contact(self):
        fa = FaceAnalyzer()

        cx = 160
        eye_y = 90
        nose_y = 110
        half_eye = 40

        mock_faces = np.array([[
            cx - 60, 50, 120, 120,
            cx - half_eye, eye_y,
            cx + half_eye, eye_y,
            cx, nose_y,
            cx - 15, 130,
            cx + 15, 130,
            0.95,
        ]], dtype=np.float32)

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, mock_faces)
            result = fa.analyze_frame(buf.tobytes())

        assert result.face_detected is True
        assert result.eye_contact > 50.0

    @patch("services.video.face_analyzer.HAS_MEDIAPIPE", False)
    @patch("services.video.face_analyzer.mp", None)
    def test_offset_face_lower_eye_contact(self):
        fa = FaceAnalyzer()

        mock_faces = np.array([[
            100, 50, 120, 120,
            120, 90,
            160, 90,
            200, 110,
            125, 130,
            155, 130,
            0.95,
        ]], dtype=np.float32)

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, mock_faces)
            result = fa.analyze_frame(buf.tobytes())

        assert result.face_detected is True
        assert result.eye_contact < 60.0


class TestFaceAnalyzerEdgeCases:
    def test_invalid_image_bytes(self):
        fa = FaceAnalyzer()
        result = fa.analyze_frame(b"not an image")
        assert result.face_detected is False

    def test_invalid_base64(self):
        fa = FaceAnalyzer()
        result = fa.analyze_base64("!!!invalid!!!")
        assert result.face_detected is False

    def test_close_disables(self):
        fa = FaceAnalyzer()
        assert fa.available is True
        fa.close()
        assert fa.available is False
