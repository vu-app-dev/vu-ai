import pytest

from services.video.face_analyzer import FaceAnalyzer, FaceAnalysisResult
from services.scoring.video_scorer import VideoScorer, VideoFrameResult
from models.scoring import VideoScores


class TestFaceAnalysisResult:
    def test_default_values(self):
        result = FaceAnalysisResult()
        assert result.face_detected is False
        assert result.num_faces == 0
        assert result.eye_contact is None
        assert result.gaze_horizontal is None

    def test_to_dict(self):
        result = FaceAnalysisResult(
            face_detected=True, num_faces=1, eye_contact=85.5, gaze_horizontal=0.02
        )
        d = result.to_dict()
        assert d["face_detected"] is True
        assert d["num_faces"] == 1
        assert d["eye_contact"] == 85.5
        assert d["gaze_horizontal"] == 0.02

    def test_to_dict_defaults(self):
        result = FaceAnalysisResult()
        d = result.to_dict()
        assert d["face_detected"] is False
        assert d["eye_contact"] is None


class TestFaceAnalyzer:
    def test_unavailable_when_model_missing(self):
        analyzer = FaceAnalyzer(model_path="/nonexistent/model.onnx")
        assert analyzer.available is False

    def test_analyze_base64_invalid_data_returns_default(self):
        analyzer = FaceAnalyzer(model_path="/nonexistent/model.onnx")
        result = analyzer.analyze_base64("not-valid-base64")
        assert result.face_detected is False
        assert result.eye_contact is None

    def test_analyze_frame_invalid_bytes_returns_default(self):
        analyzer = FaceAnalyzer(model_path="/nonexistent/model.onnx")
        result = analyzer.analyze_frame(b"not-an-image")
        assert result.face_detected is False


class TestVideoFrameResult:
    def test_default_values(self):
        result = VideoFrameResult()
        assert result.face_detected is False
        assert result.num_faces == 0
        assert result.eye_contact is None
        assert result.gaze_horizontal is None


class TestVideoScorer:
    def test_compute_session_scores_empty(self):
        scorer = VideoScorer()
        result = scorer.compute_session_scores([])
        assert result.eyeContact is None

    def test_compute_session_scores_no_face(self):
        scorer = VideoScorer()
        frames = [
            VideoFrameResult(face_detected=False, num_faces=0),
            VideoFrameResult(face_detected=False, num_faces=0),
        ]
        result = scorer.compute_session_scores(frames)
        assert result.eyeContact is None

    def test_compute_session_scores_with_eye_contact(self):
        scorer = VideoScorer()
        frames = [
            VideoFrameResult(face_detected=True, num_faces=1, eye_contact=80.0, gaze_horizontal=0.01),
            VideoFrameResult(face_detected=True, num_faces=1, eye_contact=85.0, gaze_horizontal=-0.02),
            VideoFrameResult(face_detected=True, num_faces=1, eye_contact=90.0, gaze_horizontal=0.0),
        ]
        result = scorer.compute_session_scores(frames)
        assert result.eyeContact is not None
        assert 80.0 <= result.eyeContact <= 90.0

    def test_compute_session_scores_mixed_frames(self):
        scorer = VideoScorer()
        frames = [
            VideoFrameResult(face_detected=True, num_faces=1, eye_contact=80.0),
            VideoFrameResult(face_detected=False, num_faces=0, eye_contact=None),
        ]
        result = scorer.compute_session_scores(frames)
        assert result.eyeContact == 80.0

    def test_compute_cheat_metrics_empty(self):
        scorer = VideoScorer()
        metrics = scorer.compute_cheat_metrics([])
        assert metrics["noFacePct"] is None
        assert metrics["multipleFacePct"] is None

    def test_compute_cheat_metrics_no_face(self):
        scorer = VideoScorer()
        frames = [
            VideoFrameResult(face_detected=False, num_faces=0),
            VideoFrameResult(face_detected=False, num_faces=0),
            VideoFrameResult(face_detected=False, num_faces=0),
        ]
        metrics = scorer.compute_cheat_metrics(frames)
        assert metrics["noFacePct"] == 100.0
        assert metrics["multipleFacePct"] == 0.0

    def test_compute_cheat_metrics_face_present(self):
        scorer = VideoScorer()
        frames = [
            VideoFrameResult(face_detected=True, num_faces=1, gaze_horizontal=0.01),
            VideoFrameResult(face_detected=True, num_faces=1, gaze_horizontal=0.02),
            VideoFrameResult(face_detected=False, num_faces=0, gaze_horizontal=None),
        ]
        metrics = scorer.compute_cheat_metrics(frames)
        assert metrics["noFacePct"] == 33.3
        assert metrics["multipleFacePct"] == 0.0
        assert metrics["gazeAwayPct"] is not None

    def test_compute_cheat_metrics_multiple_faces(self):
        scorer = VideoScorer()
        frames = [
            VideoFrameResult(face_detected=True, num_faces=2),
            VideoFrameResult(face_detected=True, num_faces=1),
        ]
        metrics = scorer.compute_cheat_metrics(frames)
        assert metrics["multipleFacePct"] == 50.0

    def test_compute_cheat_metrics_gaze_away(self):
        scorer = VideoScorer()
        frames = [
            VideoFrameResult(face_detected=True, num_faces=1, gaze_horizontal=0.5),
            VideoFrameResult(face_detected=True, num_faces=1, gaze_horizontal=-0.6),
            VideoFrameResult(face_detected=True, num_faces=1, gaze_horizontal=0.01),
        ]
        metrics = scorer.compute_cheat_metrics(frames)
        assert metrics["gazeAwayPct"] is not None
        assert metrics["gazeAwayPct"] > 0