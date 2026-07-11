import logging
from typing import Any

from models.scoring import TranscriptScores
from prompts import format_prompt
from services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class EvaluateAnswerResponse(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)
    overallComment: str = ""
    feedback: str = ""
    strengths: list[str] = Field(default_factory=list)
    areasToImprove: list[str] = Field(default_factory=list)
    nextAction: str = "next_question"
    followUpQuestion: dict | None = None


def _bars_to_100(score: float) -> float:
    if score <= 0:
        return 0.0
    clamped = max(1, min(5, score))
    return clamped * 20.0


class TranscriptScorer:
    def __init__(self, llm: LLMService | None = None):
        self._llm = llm or LLMService()

    async def evaluate(
        self,
        question: str,
        transcript: str,
        mock_type: str = "TECHNICAL",
        difficulty: str = "MEDIUM",
        order: int = 1,
        cv_skills: list[str] | None = None,
        duration_seconds: int = 60,
        mock_number: int = 1,
        total_mocks: int = 1,
        asked_questions: list[str] | None = None,
        conversation_history: str = "",
        active_dimensions: list[str] | None = None,
        candidate_intro: str = "",
        remaining_seconds: int | None = None,
        total_questions: int | str = "5-8",
    ) -> tuple[TranscriptScores, EvaluateAnswerResponse | None]:
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript received, returning zero scores")
            return TranscriptScores(), None

        asked_list = "None yet — this is the first question."
        if asked_questions:
            asked_list = "\n".join(f"- {q}" for q in asked_questions)

        try:
            prompt = format_prompt(
                "evaluate_answer",
                question=question,
                transcript=transcript,
                duration_seconds=duration_seconds,
                mock_type=mock_type,
                difficulty=difficulty,
                order=order,
                question_number=order,
                total_questions=total_questions,
                mock_number=mock_number,
                total_mocks=total_mocks,
                asked_questions=asked_list,
                conversation_history=conversation_history,
                active_dimensions=active_dimensions,
                candidate_intro=candidate_intro or "No self-introduction provided.",
                remaining_time_seconds=(
                    str(int(remaining_seconds)) if remaining_seconds is not None else "unknown"
                ),
            )

            response = await self._llm.generate_json(prompt, EvaluateAnswerResponse)

            if response and response.scores:
                all_dims = [
                    "communication", "problemSolving", "technical",
                    "clarityOfExplanation", "structuredThinking",
                ]
                score_vals = {}
                for dim in all_dims:
                    if active_dimensions is not None and dim not in active_dimensions:
                        score_vals[dim] = 0.0
                    else:
                        score_vals[dim] = _bars_to_100(response.scores.get(dim, 0.0))

                scores = TranscriptScores(**score_vals)
                if response.nextAction == "clarify":
                    scores = TranscriptScores()
                return scores, response

        except Exception as e:
            logger.warning("LLM transcript scoring failed: %s", e)

        return TranscriptScores(), None

    async def score(
        self,
        question: str,
        transcript: str,
        mock_type: str = "TECHNICAL",
        difficulty: str = "MEDIUM",
        order: int = 1,
        cv_skills: list[str] | None = None,
        duration_seconds: int = 60,
        mock_number: int = 1,
        total_mocks: int = 1,
        asked_questions: list[str] | None = None,
        conversation_history: str = "",
        active_dimensions: list[str] | None = None,
        candidate_intro: str = "",
        remaining_seconds: int | None = None,
        total_questions: int | str = "5-8",
    ) -> TranscriptScores:
        scores, _ = await self.evaluate(
            question=question,
            transcript=transcript,
            mock_type=mock_type,
            difficulty=difficulty,
            order=order,
            cv_skills=cv_skills,
            duration_seconds=duration_seconds,
            mock_number=mock_number,
            total_mocks=total_mocks,
            asked_questions=asked_questions,
            conversation_history=conversation_history,
            active_dimensions=active_dimensions,
            candidate_intro=candidate_intro,
            remaining_seconds=remaining_seconds,
            total_questions=total_questions,
        )
        return scores
