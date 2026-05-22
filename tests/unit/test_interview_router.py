import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from services.interview.session_manager import SessionManager

client = TestClient(app)


class TestStartSession:
    def test_start_uses_fallback_when_backend_unreachable(self):
        with patch("routers.interview.backend_client") as mock_bc:
            mock_bc.get_mock = AsyncMock(return_value=None)
            response = client.post("/api/interview/start", json={
                "mockId": "mock-1",
                "candidateId": "c1",
                "cvUrl": "https://example.com/cv.pdf",
            })
            assert response.status_code == 200
            data = response.json()
            assert "sessionId" in data
            assert "sessionToken" in data
            assert data["intro"] is not None


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

    def test_end_session_returns_performance(self):
        mgr = SessionManager()
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://example.com/cv.pdf"
        )
        result = mgr.end_session(session.id)
        assert result.cheat.level == "Clean"

    def test_session_workflow_add_answer_and_tab_switch(self):
        mgr = SessionManager()
        session = mgr.create_session(
            mock_id="m1", candidate_id="c1", cv_url="https://example.com/cv.pdf"
        )
        mgr.add_answer(session.id, "q1", "I think React is great", 60, "t1", "t2")
        mgr.add_tab_switch(session.id, 3)
        mgr.complete_question(session.id, "q1", ai_feedback="Good", score=75.0)

        result = mgr.end_session(session.id)
        assert result.cheat.level == "Flagged"
        assert result.cheat.evidence.tabSwitches == 3
        assert result.score == 75.0