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
    ) -> tuple[TranscriptScores, EvaluateAnswerResponse | None]:
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript received, returning zero scores")
            return TranscriptScores(), None

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

            response = await self._llm.generate_json(prompt, EvaluateAnswerResponse)

            if response and response.scores:
                scores = TranscriptScores(
                    communication=response.scores.get("communication", 0.0),
                    problemSolving=response.scores.get("problemSolving", 0.0),
                    technical=response.scores.get("technical", 0.0),
                    clarityOfExplanation=response.scores.get("clarityOfExplanation", 0.0),
                    structuredThinking=response.scores.get("structuredThinking", 0.0),
                    askingClarifications=response.scores.get("askingClarifications", 0.0),
                )
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
    ) -> TranscriptScores:
        scores, _ = await self.evaluate(
            question=question,
            transcript=transcript,
            mock_type=mock_type,
            difficulty=difficulty,
            order=order,
            cv_skills=cv_skills,
            duration_seconds=duration_seconds,
        )
        return scores