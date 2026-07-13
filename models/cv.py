from typing import Optional

from pydantic import BaseModel, Field


class CvAnalyzeRequest(BaseModel):
    cvUrl: str
    jobContext: dict = Field(default_factory=dict)
    candidateId: Optional[str] = None


class CvDimensions(BaseModel):
    skillsMatch: int = Field(ge=1, le=5)
    experienceDepth: int = Field(ge=1, le=5)
    educationFit: int = Field(ge=1, le=5)
    projectRelevance: int = Field(ge=1, le=5)


CV_DIMENSION_WEIGHTS = {
    "skillsMatch": 40,
    "experienceDepth": 25,
    "projectRelevance": 20,
    "educationFit": 15,
}


class CvAnalyzeResponse(BaseModel):
    skills: list[str] = Field(default_factory=list)
    summary: str = ""
    dimensions: Optional[CvDimensions] = None
    score: Optional[float] = None

    def compute_score(self) -> Optional[float]:
        if self.dimensions is None:
            return None
        weighted = sum(
            getattr(self.dimensions, dim) * weight
            for dim, weight in CV_DIMENSION_WEIGHTS.items()
        )
        return round(weighted / 100 * 20, 1)
