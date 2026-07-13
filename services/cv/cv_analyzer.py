import io
import logging
import tempfile
from typing import Any, Optional

import httpx
import pdfplumber
from docx import Document

from config import settings
from models.cv import CvAnalyzeResponse
from prompts import format_prompt
from services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = settings.CV_MAX_FILE_SIZE_BYTES
DOWNLOAD_TIMEOUT = settings.CV_DOWNLOAD_TIMEOUT_SECONDS
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class CvAnalyzer:
    def __init__(self, llm: LLMService | None = None):
        self._llm = llm or LLMService()

    async def analyze(
        self,
        cv_url: str,
        job_context: dict[str, Any] | None = None,
    ) -> CvAnalyzeResponse | None:
        extension = self._get_extension(cv_url)
        if extension not in SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported CV file type: %s", extension)
            return None

        content = await self._download(cv_url)
        if content is None:
            return None

        if len(content) > MAX_FILE_SIZE:
            logger.warning("CV file too large: %d bytes", len(content))
            return None

        text = self._extract_text(content, extension)
        if not text or not text.strip():
            logger.warning("No text extracted from CV")
            return CvAnalyzeResponse(skills=[], summary="", score=None)

        return await self._analyze_with_llm(text, job_context or {})

    def _get_extension(self, url: str) -> str:
        path = url.split("?")[0].split("#")[0].lower()
        for ext in SUPPORTED_EXTENSIONS:
            if path.endswith(ext):
                return ext
        return ""

    async def _download(self, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except httpx.RequestError as e:
            logger.error("Failed to download CV from %s: %s", url, e)
            return None
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error downloading CV from %s: %s", url, e)
            return None

    def _extract_text(self, content: bytes, extension: str) -> str | None:
        try:
            if extension == ".pdf":
                return self._extract_pdf(content)
            elif extension == ".docx":
                return self._extract_docx(content)
        except Exception as e:
            logger.error("Failed to extract text from %s: %s", extension, e)
            return None
        return None

    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages)

    @staticmethod
    def _extract_docx(content: bytes) -> str:
        doc = Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    async def _analyze_with_llm(
        self, cv_text: str, job_context: dict[str, Any]
    ) -> CvAnalyzeResponse:
        try:
            job_context_str = ", ".join(
                f"{k}: {v}" for k, v in job_context.items()
            ) if job_context else "General software engineering role"

            prompt = format_prompt(
                "analyze_cv",
                cv_text=cv_text[:8000],
                job_context=job_context_str,
            )

            response = await self._llm.generate_json(prompt, CvAnalyzeResponse)

            if response:
                response.score = response.compute_score()
                return response

        except Exception as e:
            logger.warning("LLM CV analysis failed: %s", e)

        return CvAnalyzeResponse(skills=[], summary="", score=None)