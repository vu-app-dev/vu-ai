import pytest
from pydantic import ValidationError

from models.interview import (
    StartSessionRequest,
    StartSessionResponse,
    Question,
    CheatClassification,
    CheatEvidence,
    WSAnswerMessage,
    WSVideoFrameMessage,
    WSTabSwitchMessage,
    WSEndSessionMessage,
    WSIntroMessage,
    WSQuestionMessage,
    WSAcknowledgementMessage,
    WSCheatWarningMessage,
    WSAnalysisUpdateMessage,
    WSSessionEndMessage,
    WSErrorMessage,
)
from models.cv import CvAnalyzeRequest, CvAnalyzeResponse
from models.scoring import (
    TranscriptScores,
    AudioScores,
    VideoScores,
    ScoreWeights,
    LLMAdjustment,
    PerformanceResult,
    SCORE_RUBRIC,
    describe_score,
)


class TestStartSessionRequest:
    def test_valid_request(self):
        req = StartSessionRequest(
            mockId="mock-123", candidateId="cand-456", cvUrl="https://example.com/cv.pdf"
        )
        assert req.mockId == "mock-123"
        assert req.candidateId == "cand-456"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            StartSessionRequest()

    def test_missing_mock_id_defaults_to_empty(self):
        req = StartSessionRequest(candidateId="c1", cvUrl="https://...")
        assert req.mockId == ""
        assert req.mocks is None

    def test_missing_cv_url_defaults_to_empty(self):
        req = StartSessionRequest(mockId="m1", candidateId="c1")
        assert req.cvUrl == ""


class TestQuestion:
    def test_valid_question(self):
        q = Question(id="q1", text="Explain React", difficulty="MEDIUM", order=1)
        assert q.id == "q1"
        assert q.speechType == "question"

    def test_follow_up_speech_type(self):
        q = Question(id="q2", text="Can you elaborate?", difficulty="EASY", order=2, speechType="follow_up")
        assert q.speechType == "follow_up"

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            Question(id="q1", text="Test", difficulty="HARDER", order=1)

    def test_invalid_speech_type(self):
        with pytest.raises(ValidationError):
            Question(id="q1", text="Test", difficulty="MEDIUM", order=1, speechType="announcement")

    def test_active_dimensions_default_none(self):
        q = Question(id="q1", text="Test", difficulty="MEDIUM", order=1)
        assert q.activeDimensions is None

    def test_active_dimensions_with_values(self):
        dims = ["technical", "communication", "structuredThinking"]
        q = Question(id="q1", text="Test", difficulty="MEDIUM", order=1, activeDimensions=dims)
        assert q.activeDimensions == dims
        assert len(q.activeDimensions) == 3

    def test_active_dimensions_empty_list(self):
        q = Question(id="q1", text="Test", difficulty="MEDIUM", order=1, activeDimensions=[])
        assert q.activeDimensions == []


class TestCheatClassification:
    def test_clean(self):
        c = CheatClassification(level="Clean")
        assert c.level == "Clean"
        assert c.evidence.tabSwitches == 0

    def test_flagged_with_evidence(self):
        c = CheatClassification(
            level="Flagged",
            evidence=CheatEvidence(tabSwitches=4, noFacePct=25.0, gazeAwayPct=45.0),
        )
        assert c.level == "Flagged"
        assert c.evidence.tabSwitches == 4
        assert c.evidence.noFacePct == 25.0

    def test_invalid_level(self):
        with pytest.raises(ValidationError):
            CheatClassification(level="Invalid")


class TestWSMessages:
    def test_answer_message(self):
        msg = WSAnswerMessage(
            sessionId="s1",
            questionId="q1",
            transcript="I built a React app...",
            durationSeconds=120,
            startedAt="2025-01-15T10:30:00Z",
            endedAt="2025-01-15T10:32:00Z",
        )
        assert msg.type == "answer"
        assert msg.questionId == "q1"
        assert msg.messageId is not None
        assert msg.timestamp is not None

    def test_answer_requires_question_id(self):
        with pytest.raises(ValidationError):
            WSAnswerMessage(
                sessionId="s1",
                transcript="...",
                durationSeconds=120,
                startedAt="2025-01-15T10:30:00Z",
                endedAt="2025-01-15T10:32:00Z",
            )

    def test_video_frame_message(self):
        msg = WSVideoFrameMessage(
            sessionId="s1", image="<base64>", frameNumber=42
        )
        assert msg.type == "video_frame"
        assert msg.frameNumber == 42

    def test_tab_switch_message(self):
        msg = WSTabSwitchMessage(sessionId="s1", totalCount=3)
        assert msg.type == "tab_switch"
        assert msg.totalCount == 3

    def test_end_session_message(self):
        msg = WSEndSessionMessage(sessionId="s1")
        assert msg.type == "end_session"

    def test_intro_message(self):
        msg = WSIntroMessage(sessionId="s1", text="Hello!")
        assert msg.type == "intro"
        assert msg.text == "Hello!"

    def test_question_message(self):
        msg = WSQuestionMessage(
            sessionId="s1", id="q1", text="Explain X", difficulty="MEDIUM", order=1
        )
        assert msg.type == "question"

    def test_acknowledgement_message(self):
        msg = WSAcknowledgementMessage(sessionId="s1", text="Good point!")
        assert msg.type == "acknowledgement"

    def test_cheat_warning_message(self):
        msg = WSCheatWarningMessage(
            sessionId="s1", level="Flagged", reason="No face detected", evidenceSignals=["no_face_pct: 22%"]
        )
        assert msg.type == "cheat_warning"
        assert msg.level == "Flagged"

    def test_analysis_update_message(self):
        msg = WSAnalysisUpdateMessage(sessionId="s1", eyeContactScore=72.5)
        assert msg.type == "analysis_update"

    def test_session_end_message(self):
        cheat = CheatClassification(level="Clean")
        msg = WSSessionEndMessage(
            sessionId="s1", reason="completed", cheat="Clean", cheatEvidence=CheatEvidence(tabSwitches=2)
        )
        assert msg.type == "session_end"
        assert msg.reason == "completed"

    def test_error_message(self):
        msg = WSErrorMessage(
            sessionId="s1", code="RATE_LIMITED", message="Too many requests", retryable=True
        )
        assert msg.type == "error"
        assert msg.retryable is True
        assert msg.code == "RATE_LIMITED"


class TestEvaluateAnswerResponse:
    def test_valid_response(self):
        from services.scoring.transcript_scorer import EvaluateAnswerResponse
        resp = EvaluateAnswerResponse(
            scores={"communication": 80, "technical": 75},
            overallComment="Solid answer",
            feedback="Good explanation",
            strengths=["Clear structure"],
            areasToImprove=["Add examples"],
            nextAction="next_question",
        )
        assert resp.nextAction == "next_question"

    def test_follow_up_with_question(self):
        from services.scoring.transcript_scorer import EvaluateAnswerResponse
        resp = EvaluateAnswerResponse(
            scores={"communication": 70},
            overallComment="Decent",
            feedback="Consider elaborating",
            nextAction="follow_up",
            followUpQuestion={"id": "f1", "text": "Can you elaborate?", "difficulty": "MEDIUM", "order": 2},
        )
        assert resp.followUpQuestion is not None

    def test_clarify_action(self):
        from services.scoring.transcript_scorer import EvaluateAnswerResponse
        resp = EvaluateAnswerResponse(
            scores={"communication": 0},
            overallComment="",
            feedback="What do you mean by hooks?",
            nextAction="clarify",
        )
        assert resp.nextAction == "clarify"


class TestCvModels:
    def test_analyze_request(self):
        req = CvAnalyzeRequest(
            cvUrl="https://example.com/cv.pdf",
            jobContext={"title": "React Developer", "technologies": ["React"]},
        )
        assert req.cvUrl == "https://example.com/cv.pdf"

    def test_analyze_request_minimal(self):
        req = CvAnalyzeRequest(cvUrl="https://example.com/cv.pdf")
        assert req.jobContext == {}

    def test_analyze_response(self):
        resp = CvAnalyzeResponse(skills=["React", "TypeScript"], summary="5 year dev", score=85.0)
        assert resp.score == 85.0

    def test_analyze_response_null_score(self):
        resp = CvAnalyzeResponse(skills=[], summary="", score=None)
        assert resp.score is None


class TestScoringModels:
    def test_transcript_scores_clamped(self):
        scores = TranscriptScores(
            communication=80, problemSolving=70, technical=75,
            structuredThinking=72,
        )
        assert 0 <= scores.communication <= 100

    def test_transcript_scores_over_100_clamped(self):
        scores = TranscriptScores(
            communication=150, problemSolving=-10, technical=75,
            structuredThinking=72,
        )
        assert scores.communication == 100.0
        assert scores.problemSolving == 0.0

    def test_audio_scores_with_none(self):
        scores = AudioScores(confidence=None)
        assert scores.confidence is None

    def test_video_scores_null(self):
        scores = VideoScores(eyeContact=None)
        assert scores.eyeContact is None

    def test_video_scores_clamped(self):
        scores = VideoScores(eyeContact=150)
        assert scores.eyeContact == 100.0

    def test_score_weights_sum_to_100(self):
        weights = ScoreWeights()
        assert weights.total() == 100.0

    def test_score_weights_normalize_without_confidence(self):
        weights = ScoreWeights()
        normalized = weights.normalize_without(["confidence"])
        assert abs(normalized.total() - 100.0) < 0.1
        assert normalized.confidence == 0.0
        assert normalized.technical > weights.technical

    def test_score_weights_normalize_without_multiple(self):
        weights = ScoreWeights()
        normalized = weights.normalize_without(["confidence", "structuredThinking"])
        assert abs(normalized.total() - 100.0) < 0.1
        assert normalized.confidence == 0.0
        assert normalized.structuredThinking == 0.0

    def test_llm_adjustment_clamped(self):
        adj = LLMAdjustment(adjustment=15, reason="Great insight", confidence="low")
        assert adj.adjustment == 10.0

    def test_llm_adjustment_negative_clamped(self):
        adj = LLMAdjustment(adjustment=-20, reason="Weak", confidence="medium")
        assert adj.adjustment == -10.0

    def test_llm_adjustment_valid_range(self):
        adj = LLMAdjustment(adjustment=5, reason="Slightly above average", confidence="high")
        assert adj.adjustment == 5.0

    def test_performance_result_score_clamped(self):
        result = PerformanceResult(score=150, cheat=CheatClassification(level="Clean"))
        assert result.score == 100.0

    def test_performance_result_with_null_scores(self):
        result = PerformanceResult(
            score=75,
            communication=80,
            technical=78,
            confidence=None,
            cheat=CheatClassification(level="Clean"),
        )
        assert result.confidence is None

    def test_cheat_classification(self):
        clean = CheatClassification(level="Clean")
        assert clean.level == "Clean"

        flagged = CheatClassification(level="Flagged", evidence=CheatEvidence(tabSwitches=4))
        assert flagged.evidence.tabSwitches == 4

    def test_score_rubric(self):
        assert "Poor" in describe_score(20)
        assert "Below Average" in describe_score(40)
        assert "Acceptable" in describe_score(50)
        assert "Good" in describe_score(70)
        assert "Excellent" in describe_score(90)

    def test_describe_score_boundaries(self):
        assert describe_score(0).startswith("Poor")
        assert describe_score(20).startswith("Poor")
        assert describe_score(21).startswith("Below Average")
        assert describe_score(40).startswith("Below Average")
        assert describe_score(41).startswith("Acceptable")
        assert describe_score(60).startswith("Acceptable")
        assert describe_score(61).startswith("Good")
        assert describe_score(80).startswith("Good")
        assert describe_score(81).startswith("Excellent")
        assert describe_score(100).startswith("Excellent")