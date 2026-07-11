import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from models.interview import CheatEvidence
from models.scoring import AudioScores, PerformanceResult, TranscriptScores
from services.interview.session_manager import Session, SessionManager, MockState


class TestCreateSession:
    def test_create_session_returns_session_with_id_and_token(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        assert session.id is not None
        assert session.token is not None
        assert len(session.id) > 0
        assert len(session.token) > 0

    def test_create_session_stores_mock_data(self):
        mgr = SessionManager()
        mock_data = {"type": "TECHNICAL", "difficulty": "MEDIUM", "estimatedTimeInMinutes": 30}
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com", mock_data=mock_data
        )
        assert session.mockData == mock_data
        assert session.mockId == "m1"
        assert session.candidateId == "c1"
        assert session.cvUrl == "https://cv.example.com"

    def test_create_session_sets_time_limit(self):
        mgr = SessionManager()
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com",
            mock_data={"estimatedTimeInMinutes": 30}
        )
        assert session.timeLimitSeconds == 30 * 60 + 60

    def test_create_session_default_time_limit_when_no_mock_data(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        assert session.timeLimitSeconds == 60 * 60 + 60


class TestGetSession:
    def test_get_session_returns_session(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        retrieved = mgr.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    def test_get_session_unknown_id_returns_none(self):
        mgr = SessionManager()
        assert mgr.get_session("unknown") is None

    def test_get_session_expired_returns_none(self):
        mgr = SessionManager(session_timeout_seconds=1)
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        time.sleep(1.5)
        result = mgr.get_session(session.id)
        assert result is None


class TestValidateToken:
    def test_validate_token_correct(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        assert mgr.validate_token(session.id, session.token) is True

    def test_validate_token_wrong_token(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        assert mgr.validate_token(session.id, "wrong-token") is False

    def test_validate_token_wrong_session_id(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        assert mgr.validate_token("wrong-id", session.token) is False

    def test_validate_token_expired_session(self):
        mgr = SessionManager(session_timeout_seconds=1)
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        time.sleep(1.5)
        assert mgr.validate_token(session.id, session.token) is False


class TestAddAnswer:
    def test_add_answer(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.add_answer(session.id, "q1", "I think React is great", 60, "2024-01-01T00:00:00", "2024-01-01T00:01:00")
        retrieved = mgr.get_session(session.id)
        assert len(retrieved.answers) == 1
        assert retrieved.answers[0].questionId == "q1"
        assert retrieved.answers[0].transcript == "I think React is great"

    def test_add_answer_unknown_session_raises(self):
        mgr = SessionManager()
        with pytest.raises(ValueError):
            mgr.add_answer("unknown", "q1", "text", 60, "t1", "t2")


class TestCompleteIntro:
    def test_complete_intro_stores_context_without_answer(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.complete_intro(session.id, "I am a frontend engineer with React experience.")
        assert session.introCompleted is True
        assert "frontend engineer" in session.candidateIntroTranscript
        assert session.answers == []


class TestAddTabSwitch:
    def test_add_tab_switch(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.add_tab_switch(session.id, 3)
        retrieved = mgr.get_session(session.id)
        assert retrieved.tabSwitches == 3

    def test_add_tab_switch_keeps_max(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.add_tab_switch(session.id, 3)
        mgr.add_tab_switch(session.id, 5)
        retrieved = mgr.get_session(session.id)
        assert retrieved.tabSwitches == 5

    def test_add_tab_switch_ignores_expired_session(self):
        mgr = SessionManager(session_timeout_seconds=1)
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        time.sleep(1.5)
        mgr.add_tab_switch(session.id, 3)


class TestCompleteQuestion:
    @pytest.mark.asyncio
    async def test_complete_question(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.add_answer(session.id, "q1", "transcript", 60, "t1", "t2")
        await mgr.complete_question(
            session.id, "q1",
            ai_feedback="Good answer",
            score=75.0,
            strengths=["clarity"],
            areas_to_improve=["depth"],
        )
        retrieved = mgr.get_session(session.id)
        assert retrieved.answers[0].aiFeedback == "Good answer"
        assert retrieved.answers[0].score == 75.0
        assert retrieved.answers[0].strengths == ["clarity"]
        assert retrieved.answers[0].areasToImprove == ["depth"]

    @pytest.mark.asyncio
    async def test_complete_question_persists_to_backend(self):
        from unittest.mock import AsyncMock
        mock_bc = AsyncMock()
        mock_bc.create_question = AsyncMock(return_value=True)
        mgr = SessionManager(backend_client=mock_bc)
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.add_answer(session.id, "q1", "transcript", 60, "t1", "t2")
        await mgr.complete_question(
            session.id, "q1",
            ai_feedback="Good",
            score=80.0,
        )
        await asyncio.sleep(0)
        mock_bc.create_question.assert_called_once()


class TestEndSession:
    @pytest.mark.asyncio
    async def test_end_session_returns_performance(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")

        long_transcript = " ".join(["React"] * 130 + ["hooks", "allow", "state", "management", "in", "functional", "components"])
        mgr.add_answer(session.id, "q1", long_transcript, 60, "t1", "t2")

        answer = session.answers[0]
        answer.aiFeedback = "Good"
        answer.score = 80.0
        answer.transcriptScores = TranscriptScores(
            communication=80.0, problemSolving=80.0, technical=80.0,
            clarityOfExplanation=60.0, structuredThinking=80.0,
        )

        result = await mgr.end_session(session.id)
        assert isinstance(result, PerformanceResult)
        assert result.score > 0
        assert result.communication == 80.0
        assert result.technical == 80.0

    @pytest.mark.asyncio
    async def test_end_session_with_no_scores(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        result = await mgr.end_session(session.id)
        assert isinstance(result, PerformanceResult)
        assert result.score == 0.0
        assert result.cheat.level == "Clean"

    @pytest.mark.asyncio
    async def test_end_session_classifies_cheat(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.add_tab_switch(session.id, 5)
        result = await mgr.end_session(session.id)
        assert result.cheat.level == "Flagged"
        assert result.cheat.evidence.tabSwitches == 5

    @pytest.mark.asyncio
    async def test_end_session_clean_cheat(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        result = await mgr.end_session(session.id)
        assert result.cheat.level == "Clean"
        assert result.cheat.evidence.tabSwitches == 0

    @pytest.mark.asyncio
    async def test_end_session_critical_cheat(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        mgr.add_tab_switch(session.id, 7)
        result = await mgr.end_session(session.id)
        assert result.cheat.level == "Critical"

    @pytest.mark.asyncio
    async def test_end_session_unknown_raises(self):
        mgr = SessionManager()
        with pytest.raises(ValueError):
            await mgr.end_session("unknown")

    @pytest.mark.asyncio
    async def test_end_session_removes_from_manager(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        await mgr.end_session(session.id)
        assert mgr.get_session(session.id) is None

    @pytest.mark.asyncio
    async def test_end_session_persists_to_backend(self):
        from unittest.mock import AsyncMock
        mock_bc = AsyncMock()
        mock_bc.create_performance = AsyncMock(return_value=True)
        mgr = SessionManager(backend_client=mock_bc)
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        with patch("services.scoring.score_aggregator.ScoreAggregator.adjust_with_llm", new=AsyncMock(return_value=None)), \
             patch("services.scoring.score_aggregator.ScoreAggregator.generate_summary", new=AsyncMock(return_value=None)):
            await mgr.end_session(session.id)
        mock_bc.create_performance.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_session_persistence_preserves_null_scores(self):
        from unittest.mock import AsyncMock
        mock_bc = AsyncMock()
        mock_bc.create_performance = AsyncMock(return_value=True)
        mgr = SessionManager(backend_client=mock_bc)
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")

        with patch("services.scoring.score_aggregator.ScoreAggregator.adjust_with_llm", new=AsyncMock(return_value=None)), \
             patch("services.scoring.score_aggregator.ScoreAggregator.generate_summary", new=AsyncMock(return_value=None)):
            await mgr.end_session(session.id)

        payload = mock_bc.create_performance.call_args.kwargs["data"]
        assert payload["eyeContact"] is None
        assert payload["communication"] is None


class TestCleanupExpired:
    def test_cleanup_removes_expired_sessions(self):
        mgr = SessionManager(session_timeout_seconds=1)
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        time.sleep(1.5)
        mgr._cleanup_expired()
        assert mgr.get_session(session.id) is None


class TestServerEnforcedTimeLimit:
    def test_time_limit_exceeded(self):
        mgr = SessionManager(session_timeout_seconds=100)
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com",
            mock_data={"estimatedTimeInMinutes": 0}
        )
        session.timeLimitSeconds = 1
        session.createdAt = time.time() - 2
        result = mgr.get_session(session.id)
        assert result is None

    def test_time_limit_not_exceeded(self):
        mgr = SessionManager(session_timeout_seconds=100)
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com",
            mock_data={"estimatedTimeInMinutes": 60}
        )
        result = mgr.get_session(session.id)
        assert result is not None


class TestMultiMockSession:
    def test_create_session_with_mocks_list(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"type": "TECHNICAL", "estimatedTimeInMinutes": 15}},
            {"mockId": "m2", "mockData": {"type": "BEHAVIORAL", "estimatedTimeInMinutes": 15}},
            {"mockId": "m3", "mockData": {"type": "CODING", "estimatedTimeInMinutes": 15}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        assert len(session.mocks) == 3
        assert session.currentMockIndex == 0
        assert session.mockId == "m1"
        assert session.mockData["type"] == "TECHNICAL"
        assert session.mocks[0].timeLimitSeconds == 15 * 60
        assert session.mocks[1].timeLimitSeconds == 15 * 60
        assert session.mocks[2].timeLimitSeconds == 15 * 60

    def test_multi_mock_total_time_limit(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"estimatedTimeInMinutes": 15}},
            {"mockId": "m2", "mockData": {"estimatedTimeInMinutes": 30}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        expected_total = (15 * 60) + (30 * 60) + 60
        assert session.timeLimitSeconds == expected_total

    def test_transition_to_next_mock(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"type": "TECHNICAL"}},
            {"mockId": "m2", "mockData": {"type": "BEHAVIORAL"}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        assert session.currentMockIndex == 0
        has_next = mgr.transition_to_next_mock(session.id)
        assert has_next is True
        assert session.currentMockIndex == 1
        assert session.mockId == "m2"
        assert session.mockData["type"] == "BEHAVIORAL"

    def test_transition_returns_false_when_no_more_mocks(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"type": "TECHNICAL"}},
            {"mockId": "m2", "mockData": {"type": "BEHAVIORAL"}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        mgr.transition_to_next_mock(session.id)
        has_next = mgr.transition_to_next_mock(session.id)
        assert has_next is False

    def test_get_current_mock(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"type": "TECHNICAL"}},
            {"mockId": "m2", "mockData": {"type": "BEHAVIORAL"}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        mock = mgr.get_current_mock(session.id)
        assert mock is not None
        assert mock.mockId == "m1"
        mgr.transition_to_next_mock(session.id)
        mock = mgr.get_current_mock(session.id)
        assert mock is not None
        assert mock.mockId == "m2"

    def test_answers_stored_per_mock(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"type": "TECHNICAL"}},
            {"mockId": "m2", "mockData": {"type": "BEHAVIORAL"}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        mgr.add_answer(session.id, "q1", "answer 1", 60, "t1", "t2")
        assert len(session.mocks[0].answers) == 1
        assert session.mocks[0].answers[0].mockIndex == 0
        mgr.transition_to_next_mock(session.id)
        mgr.add_answer(session.id, "q2", "answer 2", 60, "t3", "t4")
        assert len(session.mocks[1].answers) == 1
        assert session.mocks[1].answers[0].mockIndex == 1
        all_answers = session.answers
        assert len(all_answers) == 2

    def test_questions_asked_per_mock(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"type": "TECHNICAL"}},
            {"mockId": "m2", "mockData": {"type": "BEHAVIORAL"}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        session.questionsAsked = [{"id": "q1", "text": "tech question"}]
        assert len(session.mocks[0].questionsAsked) == 1
        mgr.transition_to_next_mock(session.id)
        session.questionsAsked = [{"id": "q1", "text": "behavioral question"}]
        assert len(session.mocks[1].questionsAsked) == 1
        assert session.mocks[0].questionsAsked[0]["text"] == "tech question"

    @pytest.mark.asyncio
    async def test_end_session_aggregates_across_mocks(self):
        mgr = SessionManager()
        mocks = [
            {"mockId": "m1", "mockData": {"type": "TECHNICAL", "estimatedTimeInMinutes": 15}},
            {"mockId": "m2", "mockData": {"type": "BEHAVIORAL", "estimatedTimeInMinutes": 15}},
        ]
        session = mgr.create_session(
            candidate_id="c1",
            cv_url="https://cv.example.com",
            mocks=mocks,
        )
        mgr.add_answer(session.id, "q1", " ".join(["React"] * 130), 60, "t1", "t2")
        answer1 = session.mocks[0].answers[0]
        answer1.aiFeedback = "Good"
        answer1.score = 80.0
        answer1.transcriptScores = TranscriptScores(
            communication=80.0, problemSolving=80.0, technical=80.0,
            clarityOfExplanation=60.0, structuredThinking=80.0,
        )
        mgr.transition_to_next_mock(session.id)
        mgr.add_answer(session.id, "q2", " ".join(["leadership"] * 130), 60, "t3", "t4")
        answer2 = session.mocks[1].answers[0]
        answer2.aiFeedback = "Good behavioral answer"
        answer2.score = 85.0
        answer2.transcriptScores = TranscriptScores(
            communication=80.0, problemSolving=60.0, technical=60.0,
            clarityOfExplanation=80.0, structuredThinking=80.0,
        )
        result = await mgr.end_session(session.id)
        assert result.score > 0
        assert result.communication > 0
        assert len(session.answers) == 2

    @pytest.mark.asyncio
    async def test_end_session_per_dimension_averaging(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")

        mgr.add_answer(session.id, "q1", " ".join(["word"] * 130), 60, "t1", "t2")
        answer1 = session.mocks[0].answers[0]
        answer1.aiFeedback = "Good"
        answer1.score = 80.0
        answer1.transcriptScores = TranscriptScores(
            communication=80.0, problemSolving=None, technical=80.0,
            clarityOfExplanation=60.0, structuredThinking=None,
        )
        answer1.activeDimensions = ["technical", "communication", "clarityOfExplanation"]

        mgr.add_answer(session.id, "q2", " ".join(["word"] * 130), 60, "t3", "t4")
        answer2 = session.mocks[0].answers[1]
        answer2.aiFeedback = "Good"
        answer2.score = 80.0
        answer2.transcriptScores = TranscriptScores(
            communication=60.0, problemSolving=80.0, technical=100.0,
            clarityOfExplanation=80.0, structuredThinking=80.0,
        )
        answer2.activeDimensions = None  # all dimensions active

        with patch("services.scoring.score_aggregator.ScoreAggregator.adjust_with_llm", new=AsyncMock(return_value=None)), \
             patch("services.scoring.score_aggregator.ScoreAggregator.generate_summary", new=AsyncMock(return_value=None)):
            result = await mgr.end_session(session.id)
        assert result.communication == 70.0  # (80+60)/2
        assert result.technical == 90.0  # (80+100)/2
        assert result.clarityOfExplanation == 70.0  # (60+80)/2
        assert result.problemSolving == 80.0  # only q2 tested it
        assert result.structuredThinking == 80.0  # only q2


class TestSpeakerDiarizationInEndSession:
    @pytest.mark.asyncio
    async def test_end_session_no_audio_buffer(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        result = await mgr.end_session(session.id)
        assert result.cheat.evidence.speakerCount is None
        assert result.cheat.evidence.secondSpeakerPct is None
        assert result.cheat.level == "Clean"
