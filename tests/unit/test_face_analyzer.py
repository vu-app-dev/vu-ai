import numpy as np
import cv2
import base64
import pytest
from unittest.mock import patch, PropertyMock

from services.video.face_analyzer import FaceAnalyzer, FaceAnalysisResult


class TestFaceAnalysisResult:
    def test_defaults(self):
        r = FaceAnalysisResult()
        assert r.face_detected is False
        assert r.num_faces == 0
        assert r.eye_contact is None
        assert r.gaze_horizontal is None

    def test_with_values(self):
        r = FaceAnalysisResult(face_detected=True, num_faces=1, eye_contact=75.0, gaze_horizontal=0.1)
        assert r.face_detected is True
        assert r.num_faces == 1
        assert r.eye_contact == 75.0
        assert r.gaze_horizontal == 0.1

    def test_to_dict(self):
        r = FaceAnalysisResult(face_detected=True, num_faces=1, eye_contact=75.0, gaze_horizontal=0.1)
        d = r.to_dict()
        assert d["face_detected"] is True
        assert d["num_faces"] == 1
        assert d["eye_contact"] == 75.0
        assert d["gaze_horizontal"] == 0.1

    def test_to_dict_defaults(self):
        d = FaceAnalysisResult().to_dict()
        assert d["face_detected"] is False
        assert d["eye_contact"] is None
        assert d["gaze_horizontal"] is None


class TestFaceAnalyzerInit:
    def test_init_with_yunet_model(self):
        fa = FaceAnalyzer()
        assert fa.available is True
        assert fa.backend == "yunet"

    def test_disabled_when_model_missing(self):
        fa = FaceAnalyzer(model_path="/nonexistent/model.onnx")
        assert fa.available is False
        assert fa.backend is None

    @patch("services.video.face_analyzer.HAS_CV2", False)
    def test_disabled_when_no_opencv(self):
        fa = FaceAnalyzer()
        assert fa.available is False
        assert fa.backend is None

    def test_close_disables(self):
        fa = FaceAnalyzer()
        assert fa.available is True
        fa.close()
        assert fa.available is False
        assert fa.backend is None


def _make_yunet_face(
    right_eye_x, right_eye_y,
    left_eye_x, left_eye_y,
    nose_x, nose_y,
    right_mouth_x=0, right_mouth_y=0,
    left_mouth_x=0, left_mouth_y=0,
    bbox_x=0, bbox_y=0, bbox_w=120, bbox_h=120,
    confidence=0.95,
):
    return np.array([[
        bbox_x, bbox_y, bbox_w, bbox_h,
        right_eye_x, right_eye_y,
        left_eye_x, left_eye_y,
        nose_x, nose_y,
        right_mouth_x, right_mouth_y,
        left_mouth_x, left_mouth_y,
        confidence,
    ]], dtype=np.float32)


def _make_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestYuNetNoFaceDetected:
    def test_black_image_no_face(self):
        fa = FaceAnalyzer()
        result = fa.analyze_frame(_make_image())
        assert result.face_detected is False
        assert result.num_faces == 0
        assert result.eye_contact is None
        assert result.gaze_horizontal is None


class TestYuNetFaceDetection:
    def test_single_face(self):
        fa = FaceAnalyzer()
        face = _make_yunet_face(
            right_eye_x=130, right_eye_y=90,
            left_eye_x=170, left_eye_y=90,
            nose_x=150, nose_y=110,
            bbox_x=100, bbox_y=50, bbox_w=120, bbox_h=120,
        )
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, face)
            result = fa.analyze_frame(_make_image())

        assert result.face_detected is True
        assert result.num_faces == 1
        assert result.eye_contact is not None
        assert result.gaze_horizontal is not None

    def test_two_faces(self):
        fa = FaceAnalyzer()
        faces = np.vstack([
            _make_yunet_face(130, 90, 170, 90, 150, 110, bbox_x=100),
            _make_yunet_face(230, 90, 270, 90, 250, 110, bbox_x=200),
        ])
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, faces)
            result = fa.analyze_frame(_make_image())

        assert result.face_detected is True
        assert result.num_faces == 2

    def test_yunet_returns_none(self):
        fa = FaceAnalyzer()
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, None)
            result = fa.analyze_frame(_make_image())

        assert result.face_detected is False
        assert result.num_faces == 0

    def test_yunet_returns_empty(self):
        fa = FaceAnalyzer()
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, np.array([]))
            result = fa.analyze_frame(_make_image())

        assert result.face_detected is False


class TestEyeContactFormula:
    def _score(self, right_eye_x, left_eye_x, nose_x, nose_y=110, eye_y=90):
        fa = FaceAnalyzer()
        face = _make_yunet_face(
            right_eye_x=right_eye_x, right_eye_y=eye_y,
            left_eye_x=left_eye_x, left_eye_y=eye_y,
            nose_x=nose_x, nose_y=nose_y,
            bbox_x=right_eye_x - 30, bbox_w=left_eye_x - right_eye_x + 60,
        )
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, face)
            result = fa.analyze_frame(_make_image())
        return result.eye_contact

    def test_perfectly_centered_high_score(self):
        score = self._score(right_eye_x=140, left_eye_x=180, nose_x=160)
        assert score is not None
        assert score >= 80.0

    def test_slightly_offset_moderate_score(self):
        score = self._score(right_eye_x=140, left_eye_x=180, nose_x=165)
        assert score is not None
        assert 40 < score < 90

    def test_significantly_offset_low_score(self):
        score = self._score(right_eye_x=140, left_eye_x=180, nose_x=180)
        assert score is not None
        assert score < 50

    def test_extreme_offset_near_zero(self):
        score = self._score(right_eye_x=140, left_eye_x=180, nose_x=200)
        assert score is not None
        assert score <= 10

    def test_nose_centered_but_low_gives_vertical_penalty(self):
        score_normal = self._score(140, 180, 160, nose_y=110, eye_y=90)
        score_low = self._score(140, 180, 160, nose_y=200, eye_y=90)
        assert score_normal > score_low

    def test_score_clamped_0_to_100(self):
        score = self._score(140, 180, 160)
        assert 0 <= score <= 100

    def test_zero_eye_distance_returns_safe_value(self):
        fa = FaceAnalyzer()
        face = _make_yunet_face(
            right_eye_x=150, right_eye_y=90,
            left_eye_x=150, left_eye_y=90,
            nose_x=150, nose_y=110,
            bbox_x=100, bbox_w=100,
        )
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, face)
            result = fa.analyze_frame(_make_image())
        assert result.face_detected is True
        assert result.eye_contact is not None
        assert 0 <= result.eye_contact <= 100


class TestGazeHorizontal:
    def _gaze(self, right_eye_x, left_eye_x, nose_x, bbox_x=100, bbox_w=120):
        fa = FaceAnalyzer()
        face = _make_yunet_face(
            right_eye_x=right_eye_x, right_eye_y=90,
            left_eye_x=left_eye_x, left_eye_y=90,
            nose_x=nose_x, nose_y=110,
            bbox_x=bbox_x, bbox_w=bbox_w,
        )
        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, face)
            result = fa.analyze_frame(_make_image())
        return result.gaze_horizontal

    def test_centered_nose_near_zero(self):
        gaze = self._gaze(130, 170, 160, bbox_x=100, bbox_w=120)
        assert gaze is not None
        assert abs(gaze) < 0.3

    def test_nose_shifted_right_positive(self):
        gaze = self._gaze(130, 170, 190, bbox_x=100, bbox_w=120)
        assert gaze is not None
        assert gaze > 0

    def test_nose_shifted_left_negative(self):
        gaze = self._gaze(130, 170, 130, bbox_x=100, bbox_w=120)
        assert gaze is not None
        assert gaze < 0

    def test_gaze_clamped_minus1_to_1(self):
        gaze = self._gaze(130, 170, 300, bbox_x=100, bbox_w=120)
        assert -1.0 <= gaze <= 1.0


class TestAnalyzeBase64:
    def test_valid_base64_image(self):
        fa = FaceAnalyzer()
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode()

        result = fa.analyze_base64(b64)
        assert isinstance(result, FaceAnalysisResult)
        assert result.face_detected is False

    def test_invalid_base64_returns_default(self):
        fa = FaceAnalyzer()
        result = fa.analyze_base64("!!!invalid!!!")
        assert result.face_detected is False
        assert result.eye_contact is None

    def test_base64_with_mocked_face(self):
        fa = FaceAnalyzer()
        face = _make_yunet_face(130, 90, 170, 90, 150, 110, bbox_x=100, bbox_w=120)

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode()

        with patch.object(fa, "_yunet") as mock_yunet:
            mock_yunet.detect.return_value = (None, face)
            result = fa.analyze_base64(b64)

        assert result.face_detected is True
        assert result.num_faces == 1


class TestEdgeCases:
    def test_invalid_image_bytes(self):
        fa = FaceAnalyzer()
        result = fa.analyze_frame(b"not an image")
        assert result.face_detected is False

    def test_empty_bytes(self):
        fa = FaceAnalyzer()
        result = fa.analyze_frame(b"")
        assert result.face_detected is False

    def test_disabled_analyzer_returns_default(self):
        fa = FaceAnalyzer(model_path="/nonexistent/model.onnx")
        assert fa.available is False
        result = fa.analyze_frame(_make_image())
        assert result.face_detected is False

    def test_disabled_analyzer_base64_returns_default(self):
        fa = FaceAnalyzer(model_path="/nonexistent/model.onnx")
        result = fa.analyze_base64("dGVzdA==")
        assert result.face_detected is False
