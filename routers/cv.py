import logging

from fastapi import APIRouter, HTTPException

from models.cv import CvAnalyzeRequest, CvAnalyzeResponse
from services.cv.cv_analyzer import CvAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cv", tags=["cv"])

cv_analyzer = CvAnalyzer()

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


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

    return result