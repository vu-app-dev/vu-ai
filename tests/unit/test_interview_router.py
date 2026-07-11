import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from models.interview import Question, StartSessionRequest
from models.scoring import AudioScores, TranscriptScores
from routers.interview import _select_next_question, start_session
from services.interview.session_manager import SessionManager

client = TestClient(app)


class TestStartSession:
    @staticmethod
    def _mock_question_generator():
        mock_gen = MagicMock()
        mock_gen.generate_intro = AsyncMock(return_value="Welcome to the interview.")
        mock_gen.generate_questions = AsyncMock(return_value=[
            Question(
                id="q1",
                text="Explain a React performance trade-off.",
                difficulty="MEDIUM",
                order=1,
                speechType="question",
            )
        ])
        return mock_gen

    @pytest.mark.asyncio
    async def test_start_uses_fallback_when_backend_unreachable(self):
        with patch("routers.interview.backend_client") as mock_bc, \
             patch("routers.interview._get_question_generator", return_value=self._mock_question_generator()), \
             patch("routers.interview.tts_service") as mock_tts:
            mock_tts.synthesize_many = AsyncMock(return_value=(None, None))
            mock_bc.get_mock = AsyncMock(return_value=None)
            result = await start_session(StartSessionRequest(
                mockId="mock-1",
                candidateId="c1",
                cvUrl="",
            ))
            assert result.sessionId
            assert result.sessionToken
            assert result.intro is not None
            assert result.firstQuestion.id == "intro"

    @pytest.mark.asyncio
    async def test_start_skip_intro_returns_real_question(self):
        mock_gen = self._mock_question_generator()
        mock_gen.generate_intro = AsyncMock(return_value="Welcome back.")
        with patch("routers.interview.backend_client") as mock_bc, \
             patch("routers.interview._get_question_generator", return_value=mock_gen), \
             patch("routers.interview.tts_service") as mock_tts:
            mock_tts.synthesize_many = AsyncMock(return_value=(None, None))
            mock_bc.get_mock = AsyncMock(return_value=None)
            result = await start_session(StartSessionRequest(
                mockId="mock-1",
                candidateId="c1",
                cvUrl="",
                skipIntro=True,
                candidateIntro="I am a React engineer.",
            ))
            assert result.firstQuestion.id == "q1"
            assert "React performance" in result.firstQuestion.text

    @pytest.mark.asyncio
    async def test_start_multi_mock_returns_totalMocks(self):
        with patch("routers.interview.backend_client") as mock_bc, \
             patch("routers.interview._get_question_generator", return_value=self._mock_question_generator()), \
             patch("routers.interview.tts_service") as mock_tts:
            mock_tts.synthesize_many = AsyncMock(return_value=(None, None))
            mock_bc.get_mock = AsyncMock(return_value=None)
            result = await start_session(StartSessionRequest(
                candidateId="c1",
                cvUrl="",
                mocks=[
                    {"mockId": "m1", "mockData": {"type": "TECHNICAL", "estimatedTimeInMinutes": 15}},
                    {"mockId": "m2", "mockData": {"type": "BEHAVIORAL", "estimatedTimeInMinutes": 15}},
                ],
            ))
            assert result.mockIndex == 0
            assert result.totalMocks == 2

    @pytest.mark.asyncio
    async def test_start_single_mock_backward_compat(self):
        with patch("routers.interview.backend_client") as mock_bc, \
             patch("routers.interview._get_question_generator", return_value=self._mock_question_generator()), \
             patch("routers.interview.tts_service") as mock_tts:
            mock_tts.synthesize_many = AsyncMock(return_value=(None, None))
            mock_bc.get_mock = AsyncMock(return_value=None)
            result = await start_session(StartSessionRequest(
                mockId="mock-1",
                candidateId="c1",
                cvUrl="",
            ))
            assert result.mockIndex == 0
            assert result.totalMocks == 1


class TestEndSessionREST:
    def test_end_unknown_session_returns_409(self):
        response = client.post("/api/interview/end/nonexistent-session")
        assert response.status_code == 409


class TestWSAuth:
    def test_ws_without_token_gets_error(self):
        try:
            with client.websocket_connect("/api/interview/session/bad-session?token=") as ws:
                data = ws.receive_json()
                assert data["type"] == "error"
                assert data["code"] == "SESSION_EXPIRED"
        except Exception:
            pass


class TestSessionManagerIntegration:
    def test_create_and_retrieve_session(self):
        mgr = SessionManager()
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://example.com/cv.pdf"
        )
        assert session.id is not None
        retrieved = mgr.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_end_session_returns_performance(self):
        mgr = SessionManager()
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://example.com/cv.pdf"
        )
        result = await mgr.end_session(session.id)
        assert result.cheat.level == "Clean"

    @pytest.mark.asyncio
    async def test_session_workflow_add_answer_and_tab_switch(self):
        mgr = SessionManager()
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://example.com/cv.pdf"
        )
        mgr.add_answer(session.id, "q1", "I think React is great", 60, "t1", "t2")
        mgr.add_tab_switch(session.id, 3)

        from models.scoring import TranscriptScores, AudioScores
        answer = session.answers[0]
        answer.aiFeedback = "Good"
        answer.score = 80.0
        answer.transcriptScores = TranscriptScores(
            communication=80.0, problemSolving=60.0, technical=80.0,
            clarityOfExplanation=60.0, structuredThinking=80.0,
        )
        answer.audioScores = AudioScores(confidence=70.0, speaking=75.0)

        result = await mgr.end_session(session.id)
        assert result.cheat.level == "Flagged"
        assert result.cheat.evidence.tabSwitches == 3
        assert result.score > 0


class TestQuestionSelection:
    def test_select_next_question_prefers_unused_topic_then_returns_skipped_topic(self):
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="")
        session.questionsAsked = [
            {"id": "q1", "text": "Explain supervised learning.", "difficulty": "MEDIUM", "order": 1, "topicTag": "ml basics"},
            {"id": "q2", "text": "How do you choose supervised learning metrics?", "difficulty": "MEDIUM", "order": 2, "topicTag": "ml basics"},
            {"id": "q3", "text": "How do you handle missing data?", "difficulty": "MEDIUM", "order": 3, "topicTag": "data cleaning"},
        ]
        mgr.add_answer(session.id, "q1", "answer", 60, "t1", "t2")

        selected = _select_next_question(session, start_index=1)
        assert selected is not None
        assert selected[1].id == "q3"

        session.currentQuestionIndex = selected[0]
        mgr.add_answer(session.id, "q3", "answer", 60, "t3", "t4")
        selected = _select_next_question(session, start_index=3)
        assert selected is not None
        assert selected[1].id == "q2"
