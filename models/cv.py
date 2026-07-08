from typing import Optional

from pydantic import BaseModel, Field


class CvAnalyzeRequest(BaseModel):
    cvUrl: str
    jobContext: dict = Field(default_factory=dict)
    candidateId: Optional[str] = None


class CvAnalyzeResponse(BaseModel):
    skills: list[str] = Field(default_factory=list)
    summary: str = ""
    score: Optional[float] = None