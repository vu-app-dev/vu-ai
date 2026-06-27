import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from services.llm.llm_service import LLMService, _RateLimiter


class SampleResponse(BaseModel):
    score: int
    feedback: str


def test_rate_limiter_min_interval():
    limiter = _RateLimiter(rpm=60)
    assert limiter._min_interval == 1.0

    limiter2 = _RateLimiter(rpm=30)
    assert limiter2._min_interval == 2.0


class TestProviderSelection:
    def test_gemini_provider_selection(self):
        with patch("services.llm.llm_service._genai_module", MagicMock()) as mock_genai, \
             patch("services.llm.llm_service._genai_types", MagicMock()):
            service = LLMService(provider="gemini")
            assert service._impl.__class__.__name__ == "_Gemini"
            mock_genai.Client.assert_called_once()

    def test_groq_provider_selection(self):
        with patch("services.llm.llm_service._GroqClient", MagicMock()) as mock_groq:
            service = LLMService(provider="groq")
            assert service._impl.__class__.__name__ == "_Groq"
            mock_groq.assert_called_once()


class TestGeminiGenerate:
    @pytest.fixture
    def service(self):
        with patch("services.llm.llm_service._genai_module", MagicMock()) as mock_genai, \
             patch("services.llm.llm_service._genai_types", MagicMock()):
            svc = LLMService(provider="gemini")
        return svc

    @pytest.mark.asyncio
    async def test_generate_returns_text(self, service):
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini!"
        service._impl._client.models.generate_content.return_value = mock_response
        result = await service.generate("Say hello")
        assert result == "Hello from Gemini!"

    @pytest.mark.asyncio
    async def test_generate_returns_none_on_empty(self, service):
        mock_response = MagicMock()
        mock_response.text = None
        service._impl._client.models.generate_content.return_value = mock_response
        result = await service.generate("Say hello")
        assert result is None


class TestGeminiGenerateJSON:
    @pytest.fixture
    def service(self):
        with patch("services.llm.llm_service._genai_module", MagicMock()) as mock_genai, \
             patch("services.llm.llm_service._genai_types", MagicMock()):
            svc = LLMService(provider="gemini")
        return svc

    @pytest.mark.asyncio
    async def test_generate_json_returns_valid_model(self, service):
        mock_response = MagicMock()
        mock_response.text = json.dumps({"score": 85, "feedback": "Good answer"})
        service._impl._client.models.generate_content.return_value = mock_response
        result = await service.generate_json("Evaluate...", response_model=SampleResponse)
        assert isinstance(result, SampleResponse)
        assert result.score == 85

    @pytest.mark.asyncio
    async def test_generate_json_returns_none_on_invalid(self, service):
        mock_response = MagicMock()
        mock_response.text = "always invalid {{{{"
        service._impl._client.models.generate_content.return_value = mock_response
        result = await service.generate_json("Evaluate...", response_model=SampleResponse)
        assert result is None


class TestGroqGenerate:
    @pytest.fixture
    def service(self):
        with patch("services.llm.llm_service._GroqClient", MagicMock()):
            svc = LLMService(provider="groq")
        return svc

    @pytest.mark.asyncio
    async def test_groq_generate_returns_text(self, service):
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Groq!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        service._impl._client.chat.completions.create.return_value = mock_response
        result = await service.generate("Say hello")
        assert result == "Hello from Groq!"

    @pytest.mark.asyncio
    async def test_groq_generate_returns_none_on_empty(self, service):
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        service._impl._client.chat.completions.create.return_value = mock_response
        result = await service.generate("Say hello")
        assert result is None


class TestGroqGenerateJSON:
    @pytest.fixture
    def service(self):
        with patch("services.llm.llm_service._GroqClient", MagicMock()):
            svc = LLMService(provider="groq")
        return svc

    @pytest.mark.asyncio
    async def test_groq_generate_json_returns_valid_model(self, service):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"score": 90, "feedback": "Great answer"})
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        service._impl._client.chat.completions.create.return_value = mock_response
        result = await service.generate_json("Evaluate...", response_model=SampleResponse)
        assert isinstance(result, SampleResponse)
        assert result.score == 90