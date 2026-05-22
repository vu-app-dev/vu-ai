import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from models.interview import CheatClassification, CheatEvidence
from models.scoring import PerformanceResult

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TIMEOUT_SECONDS = 120
TIME_LIMIT_BUFFER_SECONDS = 60


@dataclass
class Answer:
    questionId: str
    transcript: str
    durationSeconds: int
    startedAt: str
    endedAt: str
    aiFeedback: str = ""
    score: float = 0.0
    strengths: list[str] = field(default_factory=list)
    areasToImprove: list[str] = field(default_factory=list)


@dataclass
class Session:
    id: str
    token: str
    mockId: str
    candidateId: str
    cvUrl: str
    mockData: dict[str, Any] = field(default_factory=dict)
    createdAt: float = field(default_factory=time.time)
    lastActivityAt: float = field(default_factory=time.time)
    answers: list[Answer] = field(default_factory=list)
    tabSwitches: int = 0
    videoFrames: list[dict] = field(default_factory=list)
    questionsAsked: list[dict] = field(default_factory=list)
    currentQuestionIndex: int = 0
    status: str = "active"
    timeLimitSeconds: float | None = None

    def touch(self):
        self.lastActivityAt = time.time()


class SessionManager:
    def __init__(
        self,
        session_timeout_seconds: int = DEFAULT_SESSION_TIMEOUT_SECONDS,
        time_limit_buffer_seconds: int = TIME_LIMIT_BUFFER_SECONDS,
        backend_client=None,
    ):
        self._sessions: dict[str, Session] = {}
        self._tokens: dict[str, str] = {}
        self._session_timeout = session_timeout_seconds
        self._time_limit_buffer = time_limit_buffer_seconds
        self._backend_client = backend_client

    def create_session(
        self,
        mock_id: str,
        candidate_id: str,
        cv_url: str,
        mock_data: dict[str, Any] | None = None,
    ) -> Session:
        self._cleanup_expired()
        session_id = secrets.token_urlsafe(16)
        token = secrets.token_urlsafe(32)
        mock_data = mock_data or {}
        estimated_minutes = mock_data.get("estimatedTimeInMinutes", 60)
        time_limit_seconds = estimated_minutes * 60 + self._time_limit_buffer

        session = Session(
            id=session_id,
            token=token,
            mockId=mock_id,
            candidateId=candidate_id,
            cvUrl=cv_url,
            mockData=mock_data,
            timeLimitSeconds=time_limit_seconds,
        )
        self._sessions[session_id] = session
        self._tokens[session_id] = token
        logger.info("Created session %s for candidate %s", session_id, candidate_id)
        return session

    def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._is_expired(session):
            self._remove_session(session_id)
            return None
        if self._is_time_limit_exceeded(session):
            logger.info("Session %s exceeded time limit", session_id)
            self._remove_session(session_id)
            return None
        session.touch()
        return session

    def validate_token(self, session_id: str, token: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        if self._is_expired(session) or self._is_time_limit_exceeded(session):
            return False
        return self._tokens.get(session_id) == token

    def add_answer(
        self,
        session_id: str,
        question_id: str,
        transcript: str,
        duration_seconds: int,
        started_at: str,
        ended_at: str,
    ) -> None:
        session = self._get_active_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")
        session.answers.append(Answer(
            questionId=question_id,
            transcript=transcript,
            durationSeconds=duration_seconds,
            startedAt=started_at,
            endedAt=ended_at,
        ))
        session.touch()

    def add_video_frame(self, session_id: str, frame_data: dict) -> None:
        session = self._get_active_session(session_id)
        if session is None:
            logger.warning("Video frame for expired/unknown session %s", session_id)
            return
        session.videoFrames.append(frame_data)
        session.touch()

    def add_tab_switch(self, session_id: str, count: int) -> None:
        session = self._get_active_session(session_id)
        if session is None:
            logger.warning("Tab switch for expired/unknown session %s", session_id)
            return
        session.tabSwitches = max(session.tabSwitches, count)
        session.touch()

    async def complete_question(
        self,
        session_id: str,
        question_id: str,
        ai_feedback: str,
        score: float,
        strengths: list[str] | None = None,
        areas_to_improve: list[str] | None = None,
    ) -> None:
        session = self._get_active_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")
        for answer in session.answers:
            if answer.questionId == question_id and not answer.aiFeedback:
                answer.aiFeedback = ai_feedback
                answer.score = score
                answer.strengths = strengths or []
                answer.areasToImprove = areas_to_improve or []
                break
        session.touch()

        if self._backend_client:
            try:
                await self._backend_client.create_question(
                    session.candidateId,
                    data={
                        "question": question_id,
                        "answer": next(
                            (a.transcript for a in session.answers if a.questionId == question_id), ""
                        ),
                        "aiFeedback": ai_feedback,
                        "score": score,
                        "strengths": strengths or [],
                        "areasToImprove": areas_to_improve or [],
                        "durationInMinutes": round(
                            next(
                                (a.durationSeconds for a in session.answers if a.questionId == question_id), 0
                            ) / 60, 2
                        ),
                    },
                    idempotency_key=f"{session_id}-{question_id}-1",
                )
            except Exception as e:
                logger.warning("Failed to persist question to backend: %s", e)

    async def end_session(self, session_id: str) -> PerformanceResult:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.status = "completed"
        cheat_evidence = CheatEvidence(
            tabSwitches=session.tabSwitches,
        )
        cheat = CheatClassification(
            level=self._classify_cheat(session.tabSwitches),
            evidence=cheat_evidence,
        )
        total_score = 0.0
        if session.answers:
            scored = [a for a in session.answers if a.aiFeedback]
            if scored:
                total_score = sum(a.score for a in scored) / len(scored)
        result = PerformanceResult(
            score=total_score,
            cheat=cheat,
        )
        logger.info("Ended session %s, score=%.1f, cheat=%s", session_id, total_score, cheat.level)

        if self._backend_client:
            try:
                await self._backend_client.create_performance(
                    session.candidateId,
                    data=result.model_dump(),
                    idempotency_key=f"{session_id}-performance",
                )
            except Exception as e:
                logger.warning("Failed to persist performance to backend: %s", e)

        self._remove_session(session_id)
        return result

    def _get_active_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._is_expired(session) or self._is_time_limit_exceeded(session):
            self._remove_session(session_id)
            return None
        session.touch()
        return session

    def _is_expired(self, session: Session) -> bool:
        return (time.time() - session.lastActivityAt) > self._session_timeout

    def _is_time_limit_exceeded(self, session: Session) -> bool:
        if session.timeLimitSeconds is None:
            return False
        return (time.time() - session.createdAt) > session.timeLimitSeconds

    def _remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._tokens.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if (now - session.lastActivityAt) > self._session_timeout
            or (session.timeLimitSeconds and (now - session.createdAt) > session.timeLimitSeconds)
        ]
        for sid in expired:
            self._remove_session(sid)
            logger.info("Cleaned up expired session %s", sid)

    @staticmethod
    def _classify_cheat(tab_switches: int) -> str:
        if tab_switches >= 6:
            return "Critical"
        if tab_switches >= 3:
            return "Flagged"
        return "Clean"