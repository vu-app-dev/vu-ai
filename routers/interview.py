import asyncio
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
from models.scoring import TranscriptScores
from services.interview.cheat_detector import CheatDetector
from services.interview.question_generator import QuestionGenerator
from services.interview.session_manager import SessionManager
from services.scoring.audio_scorer import AudioScorer
from services.scoring.score_aggregator import ScoreAggregator
from services.scoring.transcript_scorer import TranscriptScorer, EvaluateAnswerResponse
from services.video.face_analyzer import FaceAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/interview", tags=["interview"])

MAX_FOLLOWUPS_TOTAL = 3

session_manager = SessionManager(backend_client=backend_client)
cheat_detector = CheatDetector()
face_analyzer = FaceAnalyzer()


def _question_dict(q_data) -> dict:
    if isinstance(q_data, dict):
        return q_data
    if hasattr(q_data, "model_dump"):
        return q_data.model_dump()
    return {}


def _count_followups(questions: list) -> int:
    return sum(1 for q in questions if _question_dict(q).get("speechType") == "follow_up")


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
    mock_data = request.mockData
    if not mock_data:
        mock_data = await backend_client.get_mock(request.mockId)
    if not mock_data:
        logger.info("No mock data provided and backend unreachable, using fallback")
        mock_data = FALLBACK_MOCK_DATA

    mock_type = mock_data.get("type", "TECHNICAL").upper()
    difficulty = mock_data.get("difficulty", "MEDIUM").upper()

    session = session_manager.create_session(
        mock_id=request.mockId,
        candidate_id=request.candidateId,
        cv_url=request.cvUrl,
        mock_data=mock_data,
    )

    # Store normalized values back so downstream code sees uppercase
    mock_data["type"] = mock_type
    mock_data["difficulty"] = difficulty

    question_gen = _get_question_generator()

    try:
        intro_text = await question_gen.generate_intro(
            mock_type=mock_type,
            technologies=mock_data.get("technologies", []),
            topics=mock_data.get("topics", []),
            estimated_time=mock_data.get("estimatedTimeInMinutes", 30),
            difficulty=difficulty,
        )
    except Exception:
        intro_text = QuestionGenerator._fallback_intro(
            mock_type,
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

    result = await session_manager.end_session(session_id)

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

    # Intro and first question were already sent via REST /start response.
    # WS is for ongoing interaction only (answer handling, follow-ups, video, etc.)

    transcript_scorer = _get_transcript_scorer()
    score_aggregator = _get_score_aggregator()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "answer":
                ended = await _handle_answer(
                    websocket, session_id, data, session,
                    transcript_scorer, score_aggregator,
                )
                if ended:
                    break

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

    # Find the actual question text + its index from the session
    question_text = transcript[:200] if transcript else "No question context"
    answered_idx = None
    for i, q_data in enumerate(session.questionsAsked):
        q = _question_dict(q_data)
        if q.get("id") == question_id:
            question_text = q.get("text", question_text)
            answered_idx = i
            break

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
        return False

    mock_type = session.mockData.get("type", "TECHNICAL")
    difficulty = session.mockData.get("difficulty", "MEDIUM")

    try:
        scores, eval_response = await transcript_scorer.evaluate(
            question=question_text,
            transcript=transcript,
            mock_type=mock_type,
            difficulty=difficulty,
            order=session.currentQuestionIndex + 1,
            duration_seconds=duration_seconds,
        )
    except Exception:
        scores = TranscriptScores()
        eval_response = None

    word_count = len(transcript.split()) if transcript else 0
    filler_count = AudioScorer.count_fillers(transcript) if transcript else 0
    audio_scores = AudioScorer().score(
        word_count=word_count,
        duration_seconds=float(duration_seconds),
        filler_count=filler_count,
    )

    question_score = score_aggregator.compute_weighted_average(scores, audio_scores)

    if eval_response:
        feedback_text = eval_response.feedback or "Good response! Let's continue."
        strengths = eval_response.strengths or []
        areas_to_improve = eval_response.areasToImprove or []
        next_action = eval_response.nextAction or "next_question"
        follow_up = eval_response.followUpQuestion
    else:
        logger.warning("LLM evaluation returned None for session %s, question %s", session_id, question_id)
        feedback_text = "Thanks for your response. Let's move on."
        strengths = []
        areas_to_improve = []
        next_action = "next_question"
        follow_up = None

    if next_action == "clarify":
        clarification = WSAcknowledgementMessage(
            sessionId=session_id,
            text=feedback_text,
            speechType="feedback",
        )
        await websocket.send_json(clarification.model_dump())

        clarified_q = Question(
            id=f"{question_id}_c1",
            text=feedback_text,
            difficulty=difficulty,
            order=(answered_idx + 1) if answered_idx is not None else (session.currentQuestionIndex + 1),
            speechType="question",
        )
        clarify_msg = WSQuestionMessage(
            sessionId=session_id,
            id=clarified_q.id,
            text=clarified_q.text,
            difficulty=clarified_q.difficulty,
            order=clarified_q.order,
            speechType="question",
        )
        await websocket.send_json(clarify_msg.model_dump())
        return False

    await session_manager.complete_question(
        session_id, question_id,
        ai_feedback=feedback_text,
        score=question_score,
        strengths=strengths,
        areas_to_improve=areas_to_improve,
    )

    current_answer = next((a for a in session.answers if a.questionId == question_id), None)
    if current_answer:
        current_answer.transcriptScores = scores
        current_answer.audioScores = audio_scores

    acknowledgement = WSAcknowledgementMessage(
        sessionId=session_id,
        text=feedback_text,
        speechType="feedback",
    )
    await websocket.send_json(acknowledgement.model_dump())

    if next_action == "follow_up" and follow_up is not None:
        fu_text = follow_up.get("text", "")
        already_asked = any(
            _question_dict(q).get("text", "") == fu_text for q in session.questionsAsked
        )
        already_has_fu = any(
            _question_dict(q).get("id", "").startswith(f"{question_id}_f") for q in session.questionsAsked
        )
        too_many = _count_followups(session.questionsAsked) >= MAX_FOLLOWUPS_TOTAL
        if fu_text and not already_asked and not already_has_fu and not too_many:
            try:
                fu_idx = (answered_idx + 1) if answered_idx is not None else (session.currentQuestionIndex + 1)
                follow_up_question = Question(
                    id=follow_up.get("id", f"{question_id}_f1"),
                    text=fu_text,
                    difficulty=follow_up.get("difficulty", difficulty),
                    order=fu_idx + 1,
                    speechType="follow_up",
                )
                session.questionsAsked.insert(fu_idx, follow_up_question.model_dump())
                session.currentQuestionIndex = fu_idx
                question_msg = WSQuestionMessage(
                    sessionId=session_id,
                    id=follow_up_question.id,
                    text=follow_up_question.text,
                    difficulty=follow_up_question.difficulty,
                    order=follow_up_question.order,
                    speechType="follow_up",
                )
                await websocket.send_json(question_msg.model_dump())
                return False
            except Exception as e:
                logger.warning("Follow-up question handling failed: %s", e)

    current_idx = answered_idx if answered_idx is not None else session.currentQuestionIndex
    next_idx = current_idx + 1

    if next_idx < len(session.questionsAsked):
        session.currentQuestionIndex = next_idx
        next_q_data = session.questionsAsked[next_idx]
        nq = Question(**next_q_data) if isinstance(next_q_data, dict) else next_q_data
        next_question_msg = WSQuestionMessage(
            sessionId=session_id,
            id=nq.id,
            text=nq.text,
            difficulty=nq.difficulty,
            order=nq.order,
            speechType=nq.speechType,
        )
        await websocket.send_json(next_question_msg.model_dump())
    else:
        session.currentQuestionIndex = next_idx
        await _handle_end_session(websocket, session_id, session)
        return True

    return False


async def _handle_video_frame(
    websocket: WebSocket,
    session_id: str,
    data: dict,
    session,
):
    image_b64 = data.get("image", "")
    frame_number = data.get("frameNumber", 0)
    timestamp = data.get("timestamp", "")

    frame_result = {"face_detected": False, "num_faces": 0, "eye_contact": None, "gaze_horizontal": None}

    if image_b64 and face_analyzer.available:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(face_analyzer.analyze_base64, image_b64),
                timeout=5.0,
            )
            frame_result = result.to_dict()
        except asyncio.TimeoutError:
            logger.warning("Face analysis timed out for frame %s", frame_number)
        except Exception as e:
            logger.warning("Face analysis failed for frame %s: %s", frame_number, e)

    session_manager.add_video_frame(session_id, frame_result)

    analysis = WSAnalysisUpdateMessage(
        sessionId=session_id,
        eyeContactScore=frame_result.get("eye_contact"),
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
    if session.status == "completed":
        return

    result = await session_manager.end_session(session_id)

    end_msg = WSSessionEndMessage(
        sessionId=session_id,
        reason="completed",
        performance=result.model_dump(),
        cheat=result.cheat.level,
        cheatEvidence=result.cheat.evidence.model_dump(),
    )
    await websocket.send_json(end_msg.model_dump())