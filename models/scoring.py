from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from models.interview import CheatClassification, CheatEvidence


class TranscriptScores(BaseModel):
    communication: float = 0.0
    problemSolving: float = 0.0
    technical: float = 0.0
    clarityOfExplanation: float = 0.0
    structuredThinking: float = 0.0

    @field_validator("*", mode="after")
    @classmethod
    def clamp_0_100(cls, v: float) -> float:
        return max(0.0, min(100.0, v))


class AudioScores(BaseModel):
    confidence: Optional[float] = None
    speaking: Optional[float] = None

    @field_validator("confidence", "speaking", mode="after")
    @classmethod
    def clamp_optional_0_100(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return max(0.0, min(100.0, v))
        return v


class VideoScores(BaseModel):
    eyeContact: Optional[float] = None

    @field_validator("eyeContact", mode="after")
    @classmethod
    def clamp_optional_0_100(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return max(0.0, min(100.0, v))
        return v


class ScoreWeights(BaseModel):
    technical: float = 22.0
    communication: float = 17.0
    problemSolving: float = 17.0
    clarityOfExplanation: float = 12.0
    structuredThinking: float = 12.0
    confidence: float = 8.0
    speaking: float = 4.0
    eyeContact: float = 8.0

    @field_validator("*", mode="after")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v

    def total(self) -> float:
        return sum(getattr(self, f) for f in self.__class__.model_fields)

    def normalize_without(self, exclude_fields: list[str]) -> "ScoreWeights":
        active_values = {k: getattr(self, k) for k in self.__class__.model_fields if k not in exclude_fields}
        total = sum(active_values.values())
        if total == 0:
            return self
        normalized = {k: v * (100.0 / total) for k, v in active_values.items()}
        for k in exclude_fields:
            normalized[k] = 0.0
        return ScoreWeights(**normalized)


class LLMAdjustment(BaseModel):
    adjustment: float
    reason: str
    confidence: Literal["low", "medium", "high"] = "low"

    @field_validator("adjustment", mode="after")
    @classmethod
    def clamp_adjustment(cls, v: float) -> float:
        return max(-10.0, min(10.0, v))


class PerformanceResult(BaseModel):
    score: float
    communication: Optional[float] = None
    problemSolving: Optional[float] = None
    technical: Optional[float] = None
    clarityOfExplanation: Optional[float] = None
    structuredThinking: Optional[float] = None
    confidence: Optional[float] = None
    speaking: Optional[float] = None
    eyeContact: Optional[float] = None
    cheat: CheatClassification = CheatClassification()
    llmAdjustment: Optional[LLMAdjustment] = None
    overallSummary: Optional[str] = None

    @field_validator("score", mode="after")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(100.0, v))


SCORE_RUBRIC = {
    (0, 20): "Poor — Fundamentally lacking in this area",
    (21, 40): "Below Average — Attempted but with major gaps",
    (41, 60): "Acceptable — Covers basics but has notable gaps",
    (61, 80): "Good — Solid demonstration with minor weaknesses",
    (81, 100): "Excellent — Exceptional, thorough, and insightful",
}


def describe_score(score: float) -> str:
    for (lo, hi), label in SCORE_RUBRIC.items():
        if lo <= score <= hi:
            return label
    return "Unknown"