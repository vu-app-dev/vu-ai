import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.cv import CvAnalyzeResponse
from services.cv.cv_analyzer import CvAnalyzer


class TestGetExtension:
    def test_pdf_extension(self):
        result = CvAnalyzer()._get_extension("https://example.com/resume.pdf")
        assert result == ".pdf"

    def test_docx_extension(self):
        result = CvAnalyzer()._get_extension("https://example.com/resume.docx")
        assert result == ".docx"

    def test_unsupported_extension(self):
        result = CvAnalyzer()._get_extension("https://example.com/resume.txt")
        assert result == ""

    def test_extension_with_query_params(self):
        result = CvAnalyzer()._get_extension("https://example.com/resume.pdf?token=abc")
        assert result == ".pdf"

    def test_extension_with_fragment(self):
        result = CvAnalyzer()._get_extension("https://example.com/resume.docx#page=1")
        assert result == ".docx"


class TestExtractPdf:
    def test_extract_pdf_handles_invalid_content(self):
        result = CvAnalyzer()._extract_text(b"not a real pdf", ".pdf")
        assert result is None

    def test_extract_docx_text(self):
        from docx import Document
        doc = Document()
        doc.add_paragraph("John Doe - Software Engineer")
        doc.add_paragraph("Skills: Python, React, Node.js")
        buf = io.BytesIO()
        doc.save(buf)
        content = buf.getvalue()

        result = CvAnalyzer._extract_docx(content)
        assert "John Doe" in result
        assert "Python" in result


class TestAnalyzeUnsupported:
    @pytest.mark.asyncio
    async def test_unsupported_file_type_returns_none(self):
        analyzer = CvAnalyzer()
        result = await analyzer.analyze("https://example.com/resume.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_url_extension(self):
        analyzer = CvAnalyzer()
        result = await analyzer.analyze("https://example.com/file")
        assert result is None


class TestAnalyzeWithLLM:
    @pytest.mark.asyncio
    async def test_analyze_cv_llm_success(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=CvAnalyzeResponse(
            skills=["Python", "React", "Node.js"],
            summary="Experienced full-stack developer with 5 years of experience.",
            score=82.0,
        ))
        analyzer = CvAnalyzer(llm=mock_llm)
        mock_content = b"dummy pdf content"

        with patch.object(analyzer, "_download", new_callable=AsyncMock) as mock_dl, \
             patch.object(analyzer, "_extract_text", return_value="John Doe, Software Engineer"):
            mock_dl.return_value = mock_content
            result = await analyzer.analyze(
                "https://example.com/resume.pdf",
                job_context={"role": "Senior Developer"},
            )
            assert result is not None
            assert "Python" in result.skills
            assert result.score == 82.0

    @pytest.mark.asyncio
    async def test_analyze_cv_llm_failure_returns_null_scores(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(side_effect=Exception("LLM error"))
        analyzer = CvAnalyzer(llm=mock_llm)

        with patch.object(analyzer, "_download", new_callable=AsyncMock) as mock_dl, \
             patch.object(analyzer, "_extract_text", return_value="John Doe, Software Engineer"):
            mock_dl.return_value = b"dummy content"
            result = await analyzer.analyze("https://example.com/resume.pdf")
            assert result is not None
            assert result.score is None

    @pytest.mark.asyncio
    async def test_analyze_cv_download_failure(self):
        analyzer = CvAnalyzer()
        with patch.object(analyzer, "_download", new_callable=AsyncMock, return_value=None):
            result = await analyzer.analyze("https://unreachable.example.com/resume.pdf")
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_cv_empty_extraction(self):
        mock_llm = AsyncMock()
        analyzer = CvAnalyzer(llm=mock_llm)

        with patch.object(analyzer, "_download", new_callable=AsyncMock, return_value=b"content"), \
             patch.object(analyzer, "_extract_text", return_value="   "):
            result = await analyzer.analyze("https://example.com/resume.pdf")
            assert result is not None
            assert result.skills == []