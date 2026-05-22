import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query

from clients.backend_client import backend_client
from models.interview import (
    Question,
    CheatClassification,
    StartSessionRequest,
    StartSessionResponse,
    WSErrorMessage,
    WSIntroMessage,
    WSQuestionMessage,
    WSAcknowledgementMessage,
    WSCheatWarningMessage,
    WSAnalysisUpdateMessage,
    WSSessionEndMessage,
)
from services.interview.cheat_detector import CheatDetector
from services.interview.question_generator import QuestionGenerator
from services.interview.session_manager import SessionManager
from services.scoring.score_aggregator import ScoreAggregator
from services.scoring.transcript_scorer import TranscriptScorer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/interview", tags=["interview"])

session_manager = SessionManager()
cheat_detector = CheatDetector()


def _get_question_generator() -> QuestionGenerator:
    return QuestionGenerator()


def _get_transcript_scorer() -> TranscriptScorer:
    return TranscriptScorer()


def _get_score_aggregator() -> ScoreAggregator:
    return ScoreAggregator()


FALLBACK_MOCK_DATA = {
    "type": "TECHNICAL",
    "difficulty": "MEDIUM",
    "technologies": ["React", "Node.js", "JavaScript"],
    "topics": ["frontend", "APIs", "system design"],
    "estimatedTimeInMinutes": 30,
    "questions": [
        {"title": "React Hooks", "description": "Explain how useEffect cleanup works and when it runs", "order": 1, "difficulty": "MEDIUM"},
        {"title": "REST vs GraphQL", "description": "Compare REST and GraphQL for building APIs", "order": 2, "difficulty": "MEDIUM"},
        {"title": "System Design", "description": "How would you design a real-time chat application?", "order": 3, "difficulty": "HARD"},
        {"title": "JavaScript Closures", "description": "Explain closures and give a practical example", "order": 4, "difficulty": "EASY"},
        {"title": "Performance Optimization", "description": "What techniques would you use to optimize a slow React app?", "order": 5, "difficulty": "HARD"},
    ],
}


@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    mock_data = await backend_client.get_mock(request.mockId)
    if mock_data is None:
        logger.info("Backend unreachable or mock not found, using fallback mock data")
        mock_data = FALLBACK_MOCK_DATA

    session = session_manager.create_session(
        mock_id=request.mockId,
        candidate_id=request.candidateId,
        cv_url=request.cvUrl,
        mock_data=mock_data,
    )

    question_gen = _get_question_generator()

    try:
        intro_text = await question_gen.generate_intro(
            mock_type=mock_data.get("type", "TECHNICAL"),
            technologies=mock_data.get("technologies", []),
            topics=mock_data.get("topics", []),
            estimated_time=mock_data.get("estimatedTimeInMinutes", 30),
            difficulty=mock_data.get("difficulty", "MEDIUM"),
        )
    except Exception:
        intro_text = QuestionGenerator._fallback_intro(
            mock_data.get("type", "TECHNICAL"),
            mock_data.get("technologies", []),
        )

    try:
        questions = await question_gen.generate_questions(
            mock_data=mock_data,
            cv_skills=[],
        )
    except Exception:
        questions = QuestionGenerator._fallback_questions(
            mock_data.get("questions", []),
            mock_data.get("type", "TECHNICAL"),
            mock_data.get("difficulty", "MEDIUM"),
        )

    if not questions:
        questions = QuestionGenerator._fallback_questions(
            [], mock_data.get("type", "TECHNICAL"), mock_data.get("difficulty", "MEDIUM"),
        )

    first_question = questions[0] if questions else Question(
        id="q1",
        text="Tell me about your experience.",
        difficulty="MEDIUM",
        order=1,
        speechType="question",
    )
    session.questionsAsked = [q.model_dump() for q in questions]

    return StartSessionResponse(
        sessionId=session.id,
        sessionToken=session.token,
        intro=intro_text,
        firstQuestion=first_question,
        cvAnalysis=None,
    )


@router.post("/end/{session_id}")
async def end_session_via_rest(session_id: str):
    session = session_manager._sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=409, detail="Session already ended or not found")

    if session.status == "completed":
        raise HTTPException(status_code=409, detail="Session already ended")

    result = session_manager.end_session(session_id)

    try:
        await backend_client.create_performance(
            session.candidateId,
            data=result.model_dump(),
            idempotency_key=f"{session_id}-performance",
        )
    except Exception as e:
        logger.warning("Failed to persist performance to backend: %s", e)

    return {
        "performance": result.model_dump(),
        "cheat": result.cheat.level,
    }


@router.websocket("/session/{session_id}")
async def interview_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(default=None),
):
    await websocket.accept()

    if not token or not session_manager.validate_token(session_id, token):
        await websocket.send_json(WSErrorMessage(
            sessionId=session_id,
            code="SESSION_EXPIRED",
            message="Invalid or expired session token",
            retryable=False,
        ).model_dump())
        await websocket.close()
        return

    session = session_manager.get_session(session_id)
    if session is None:
        await websocket.send_json(WSErrorMessage(
            sessionId=session_id,
            code="SESSION_EXPIRED",
            message="Session not found or expired",
            retryable=False,
        ).model_dump())
        await websocket.close()
        return

    intro_msg = WSIntroMessage(
        sessionId=session_id,
        text="Welcome to your interview! Let's get started.",
        speechType="intro",
    )
    await websocket.send_json(intro_msg.model_dump())

    if session.questionsAsked:
        first_q_data = session.questionsAsked[0]
        first_q = Question(**first_q_data) if isinstance(first_q_data, dict) else first_q_data
        question_msg = WSQuestionMessage(
            sessionId=session_id,
            id=first_q.id,
            text=first_q.text,
            difficulty=first_q.difficulty,
            order=first_q.order,
            speechType=first_q.speechType,
        )
        await websocket.send_json(question_msg.model_dump())

    transcript_scorer = _get_transcript_scorer()
    score_aggregator = _get_score_aggregator()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "answer":
                await _handle_answer(
                    websocket, session_id, data, session,
                    transcript_scorer, score_aggregator,
                )

            elif msg_type == "video_frame":
                await _handle_video_frame(websocket, session_id, data, session)

            elif msg_type == "tab_switch":
                await _handle_tab_switch(websocket, session_id, data, session)

            elif msg_type == "end_session":
                await _handle_end_session(websocket, session_id, session)
                break

            else:
                await websocket.send_json(WSErrorMessage(
                    sessionId=session_id,
                    code="INVALID_MESSAGE",
                    message=f"Unknown message type: {msg_type}",
                    retryable=True,
                ).model_dump())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as e:
        logger.error("WebSocket error for session %s: %s", session_id, e)
        try:
            await websocket.send_json(WSErrorMessage(
                sessionId=session_id,
                code="PROCESSING_ERROR",
                message="An internal error occurred",
                retryable=False,
            ).model_dump())
        except Exception:
            pass


async def _handle_answer(
    websocket: WebSocket,
    session_id: str,
    data: dict,
    session,
    transcript_scorer: TranscriptScorer,
    score_aggregator: ScoreAggregator,
):
    question_id = data.get("questionId", "")
    transcript = data.get("transcript", "")
    duration_seconds = data.get("durationSeconds", 0)
    started_at = data.get("startedAt", "")
    ended_at = data.get("endedAt", "")

    try:
        session_manager.add_answer(
            session_id, question_id, transcript, duration_seconds, started_at, ended_at
        )
    except ValueError:
        await websocket.send_json(WSErrorMessage(
            sessionId=session_id,
            code="SESSION_EXPIRED",
            message="Session has expired",
            retryable=False,
        ).model_dump())
        return

    mock_type = session.mockData.get("type", "TECHNICAL")
    difficulty = session.mockData.get("difficulty", "MEDIUM")

    try:
        scores = await transcript_scorer.score(
            question=transcript[:200],
            transcript=transcript,
            mock_type=mock_type,
            difficulty=difficulty,
        )
    except Exception:
        from models.scoring import TranscriptScores
        scores = TranscriptScores()

    feedback_text = "Good response! Let's continue."

    acknowledgement = WSAcknowledgementMessage(
        sessionId=session_id,
        text=feedback_text,
        speechType="feedback",
    )
    await websocket.send_json(acknowledgement.model_dump())

    session_manager.complete_question(
        session_id, question_id,
        ai_feedback=feedback_text,
        score=score_aggregator.compute_weighted_average(scores),
        strengths=[],
        areas_to_improve=[],
    )

    current_idx = session.currentQuestionIndex
    next_idx = current_idx + 1

    if next_idx < len(session.questionsAsked):
        session.currentQuestionIndex = next_idx
        next_q_data = session.questionsAsked[next_idx]
        nq = Question(**next_q_data) if isinstance(next_q_data, dict) else next_q_data
        next_question_msg = WSQuestionMessage(
            sessionId=session_id,
            id=next_q.id,
            text=next_q.text,
            difficulty=next_q.difficulty,
            order=next_q.order,
            speechType=next_q.speechType,
        )
        await websocket.send_json(next_question_msg.model_dump())
    else:
        session.currentQuestionIndex = next_idx


async def _handle_video_frame(
    websocket: WebSocket,
    session_id: str,
    data: dict,
    session,
):
    frame_data = {
        "frameNumber": data.get("frameNumber", 0),
        "timestamp": data.get("timestamp", ""),
    }
    session_manager.add_video_frame(session_id, frame_data)

    analysis = WSAnalysisUpdateMessage(
        sessionId=session_id,
        eyeContactScore=None,
    )
    await websocket.send_json(analysis.model_dump())


async def _handle_tab_switch(
    websocket: WebSocket,
    session_id: str,
    data: dict,
    session,
):
    total_count = data.get("totalCount", 0)
    session_manager.add_tab_switch(session_id, total_count)

    classification = cheat_detector.classify(tab_count=total_count)

    if classification.level in ("Flagged", "Critical"):
        warning = WSCheatWarningMessage(
            sessionId=session_id,
            level=classification.level,
            reason=f"Tab switch detected ({total_count} switches)",
            evidenceSignals=[f"tab_switches: {total_count}"],
        )
        await websocket.send_json(warning.model_dump())


async def _handle_end_session(
    websocket: WebSocket,
    session_id: str,
    session,
):
    result = session_manager.end_session(session_id)

    try:
        await backend_client.create_performance(
            session.candidateId,
            data=result.model_dump(),
            idempotency_key=f"{session_id}-performance",
        )
    except Exception as e:
        logger.warning("Failed to persist performance to backend: %s", e)

    end_msg = WSSessionEndMessage(
        sessionId=session_id,
        reason="completed",
        performance=result.model_dump(),
        cheat=result.cheat.level,
        cheatEvidence=result.cheat.evidence.model_dump(),
    )
    await websocket.send_json(end_msg.model_dump())