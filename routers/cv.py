import asyncio
import logging

from fastapi import APIRouter, HTTPException

from clients.backend_client import backend_client
from models.cv import CvAnalyzeRequest, CvAnalyzeResponse
from services.cv.cv_analyzer import CvAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cv", tags=["cv"])

cv_analyzer = CvAnalyzer()

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


async def _persist_cv_analysis(candidate_id: str, result: CvAnalyzeResponse):
    try:
        data = {
            "skills": result.skills,
            "summary": result.summary,
            "score": result.score if result.score is not None else 0,
        }
        idempotency_key = f"cv-analysis-{candidate_id}"
        await backend_client.create_cv_analysis(candidate_id, data, idempotency_key)
    except Exception as e:
        logger.error("Failed to persist CV analysis for candidate %s: %s", candidate_id, e)


@router.post("/analyze", response_model=CvAnalyzeResponse)
async def analyze_cv(request: CvAnalyzeRequest):
    extension = cv_analyzer._get_extension(request.cvUrl)
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    result = await cv_analyzer.analyze(request.cvUrl, request.jobContext)

    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Failed to analyze CV. The file may be empty, encrypted, or too large.",
        )

    if request.candidateId:
        asyncio.create_task(_persist_cv_analysis(request.candidateId, result))

    return result