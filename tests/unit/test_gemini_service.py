import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from services.llm.gemini_service import GeminiService, _RateLimiter


class SampleResponse(BaseModel):
    score: int
    feedback: str


@pytest.fixture
def mock_client():
    with patch("services.llm.gemini_service.genai.Client") as MockClient:
        client_instance = MagicMock()
        MockClient.return_value = client_instance
        yield client_instance


@pytest.fixture
def service(mock_client):
    return GeminiService(api_key="test-key", rpm=1000)


def test_load_prompt(tmp_path):
    template = tmp_path / "test_template.txt"
    template.write_text("Hello {name}, your role is {role}.")
    with patch("services.llm.gemini_service.PROMPTS_DIR", tmp_path):
        result = GeminiService.load_prompt("test_template", name="World", role="interviewer")
    assert "Hello World" in result
    assert "interviewer" in result


def test_load_prompt_missing_file():
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        GeminiService.load_prompt("nonexistent_template")


@pytest.mark.asyncio
async def test_generate_returns_text(service, mock_client):
    mock_response = MagicMock()
    mock_response.text = "Hello, I am an AI interviewer."
    mock_client.models.generate_content.return_value = mock_response

    result = await service.generate("Say hello")
    assert result == "Hello, I am an AI interviewer."


@pytest.mark.asyncio
async def test_generate_returns_none_on_empty(service, mock_client):
    mock_response = MagicMock()
    mock_response.text = None
    mock_client.models.generate_content.return_value = mock_response

    result = await service.generate("Say hello")
    assert result is None


@pytest.mark.asyncio
async def test_generate_json_returns_valid_model(service, mock_client):
    mock_response = MagicMock()
    mock_response.text = json.dumps({"score": 85, "feedback": "Good answer"})
    mock_client.models.generate_content.return_value = mock_response

    result = await service.generate_json("Evaluate this answer...", response_model=SampleResponse)
    assert isinstance(result, SampleResponse)
    assert result.score == 85
    assert result.feedback == "Good answer"


@pytest.mark.asyncio
async def test_generate_json_retries_on_malformed_json(service, mock_client):
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            mock_resp = MagicMock()
            mock_resp.text = "not valid json {"
            return mock_resp
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({"score": 70, "feedback": "Decent"})
        return mock_resp

    mock_client.models.generate_content.side_effect = side_effect

    result = await service.generate_json("Evaluate...", response_model=SampleResponse)
    assert result is not None
    assert result.score == 70


@pytest.mark.asyncio
async def test_generate_json_returns_none_after_exhausting_retries(service, mock_client):
    mock_response = MagicMock()
    mock_response.text = "always invalid {{{{"
    mock_client.models.generate_content.return_value = mock_response

    result = await service.generate_json("Evaluate...", response_model=SampleResponse)
    assert result is None


@pytest.mark.asyncio
async def test_generate_retries_on_exception(service, mock_client):
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("API error")
        mock_resp = MagicMock()
        mock_resp.text = "Success after retry"
        return mock_resp

    mock_client.models.generate_content.side_effect = side_effect

    with patch("services.llm.gemini_service.asyncio.sleep", new_callable=AsyncMock):
        result = await service.generate("Try this")
    assert result == "Success after retry"


def test_rate_limiter_min_interval():
    limiter = _RateLimiter(rpm=60)
    assert limiter._min_interval == 1.0  # 60/60 = 1 second

    limiter2 = _RateLimiter(rpm=30)
    assert limiter2._min_interval == 2.0  # 60/30 = 2 seconds