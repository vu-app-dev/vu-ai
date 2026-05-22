import logging
from typing import Any

from models.scoring import TranscriptScores
from prompts import format_prompt
from services.llm.gemini_service import GeminiService

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class _EvaluateAnswerResponse(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)
    overallComment: str = ""
    feedback: str = ""
    strengths: list[str] = Field(default_factory=list)
    areasToImprove: list[str] = Field(default_factory=list)
    nextAction: str = "next_question"


class TranscriptScorer:
    def __init__(self, gemini_service: GeminiService | None = None):
        self._gemini = gemini_service or GeminiService()

    async def score(
        self,
        question: str,
        transcript: str,
        mock_type: str = "TECHNICAL",
        difficulty: str = "MEDIUM",
        order: int = 1,
        cv_skills: list[str] | None = None,
        duration_seconds: int = 60,
    ) -> TranscriptScores:
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript received, returning zero scores")
            return TranscriptScores()

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
                total_questions="5-8",
            )

            response = await self._gemini.generate_json(prompt, _EvaluateAnswerResponse)

            if response and response.scores:
                return TranscriptScores(
                    communication=response.scores.get("communication", 0.0),
                    problemSolving=response.scores.get("problemSolving", 0.0),
                    technical=response.scores.get("technical", 0.0),
                    clarityOfExplanation=response.scores.get("clarityOfExplanation", 0.0),
                    structuredThinking=response.scores.get("structuredThinking", 0.0),
                    askingClarifications=response.scores.get("askingClarifications", 0.0),
                )

        except Exception as e:
            logger.warning("LLM transcript scoring failed: %s", e)

        return TranscriptScores()