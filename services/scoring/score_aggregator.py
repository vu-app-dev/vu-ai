import logging
from typing import Optional

from models.scoring import (
    AudioScores,
    LLMAdjustment,
    PerformanceResult,
    ScoreWeights,
    TranscriptScores,
)
from prompts import format_prompt
from services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = ScoreWeights()


class ScoreAggregator:
    def __init__(self, weights: ScoreWeights | None = None, llm: LLMService | None = None):
        self._weights = weights or ScoreWeights()
        self._llm = llm or LLMService()

    def compute_weighted_average(
        self,
        transcript: TranscriptScores,
        audio: AudioScores | None = None,
    ) -> float:
        exclude_fields = []
        transcript_fields = (
            "communication", "problemSolving", "technical",
            "structuredThinking",
        )
        for field_name in transcript_fields:
            if getattr(transcript, field_name) is None:
                exclude_fields.append(field_name)
        if audio is None or audio.confidence is None:
            exclude_fields.append("confidence")

        weights = self._weights
        if exclude_fields:
            weights = weights.normalize_without(exclude_fields)

        values = {}
        for field_name in weights.__class__.model_fields:
            weight = getattr(weights, field_name)
            if weight == 0:
                continue
            if field_name in transcript_fields:
                values[field_name] = getattr(transcript, field_name)
            elif field_name == "confidence":
                values[field_name] = audio.confidence if audio and audio.confidence is not None else 0

        if not values:
            return 0.0

        total_weight = sum(
            getattr(weights, f) for f in weights.__class__.model_fields
            if getattr(weights, f) > 0
        )
        if total_weight == 0:
            return 0.0

        weighted_sum = sum((values.get(f) or 0) * getattr(weights, f) for f in values)
        return round(weighted_sum / total_weight, 2)

    async def adjust_with_llm(
        self,
        weighted_avg: float,
        question_results: str,
        mock_type: str = "TECHNICAL",
        duration_minutes: int = 30,
        questions_answered: int = 5,
    ) -> LLMAdjustment | None:
        try:
            prompt = format_prompt(
                "adjust_score",
                weighted_score=weighted_avg,
                question_results=question_results,
                mock_type=mock_type,
                duration_minutes=duration_minutes,
                questions_answered=questions_answered,
            )

            response = await self._llm.generate_json(prompt, LLMAdjustment)

            if response:
                return LLMAdjustment(
                    adjustment=max(-10.0, min(10.0, response.adjustment)),
                    reason=response.reason,
                    confidence=response.confidence,
                )

        except Exception as e:
            logger.warning("LLM score adjustment failed: %s", e)

        return None

    def compute_performance(
        self,
        transcript: TranscriptScores,
        audio: AudioScores | None = None,
        llm_adjustment: LLMAdjustment | None = None,
    ) -> float:
        avg = self.compute_weighted_average(transcript, audio)
        if llm_adjustment:
            avg += llm_adjustment.adjustment
        return max(0.0, min(100.0, avg))

    async def generate_summary(
        self,
        weighted_avg: float,
        question_results: str,
        mock_type: str = "TECHNICAL",
        duration_minutes: int = 30,
        questions_answered: int = 5,
    ) -> str | None:
        try:
            prompt = format_prompt(
                "generate_summary",
                weighted_score=weighted_avg,
                question_results=question_results,
                mock_type=mock_type,
                duration_minutes=duration_minutes,
                questions_answered=questions_answered,
            )

            from pydantic import BaseModel

            class _SummaryResponse(BaseModel):
                summary: str = ""

            response = await self._llm.generate_json(prompt, _SummaryResponse)

            if response and response.summary:
                return response.summary

        except Exception as e:
            logger.warning("LLM summary generation failed: %s", e)

        return None
