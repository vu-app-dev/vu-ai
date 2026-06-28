import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.tts.tts_service import TTSService


class TestTTSServiceAvailability:
    def test_unavailable_without_api_key(self):
        svc = TTSService(api_key="")
        assert svc.available is False

    def test_available_with_api_key(self):
        svc = TTSService(api_key="fake-key")
        assert svc.available is True


class TestTTSServiceSynthesize:
    @pytest.mark.asyncio
    async def test_returns_none_when_unavailable(self):
        svc = TTSService(api_key="")
        result = await svc.synthesize("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_text(self):
        svc = TTSService(api_key="fake-key")
        result = await svc.synthesize("")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_none_text(self):
        svc = TTSService(api_key="fake-key")
        result = await svc.synthesize(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_base64_audio_on_success(self):
        svc = TTSService(api_key="fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"audioContent": "dGVzdA=="}
        mock_response.raise_for_status = MagicMock()

        with patch.object(svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await svc.synthesize("Hello interview!")
        assert result == "dGVzdA=="

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_audio_content(self):
        svc = TTSService(api_key="fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"audioContent": ""}
        mock_response.raise_for_status = MagicMock()

        with patch.object(svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await svc.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        import httpx
        svc = TTSService(api_key="fake-key")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=MagicMock(status_code=400, text="error")
        )

        with patch.object(svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await svc.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self):
        import httpx
        svc = TTSService(api_key="fake-key")

        with patch.object(svc._client, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")):
            result = await svc.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_truncates_oversized_text(self):
        svc = TTSService(api_key="fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"audioContent": "dGVzdA=="}
        mock_response.raise_for_status = MagicMock()

        long_text = "x" * 5000
        with patch.object(svc._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await svc.synthesize(long_text)
            sent_text = mock_post.call_args[1]["json"]["input"]["text"]
            assert len(sent_text) <= 4500


class TestTTSServiceSynthesizeMany:
    @pytest.mark.asyncio
    async def test_returns_none_list_when_unavailable(self):
        svc = TTSService(api_key="")
        result = await svc.synthesize_many(["a", "b", "c"])
        assert result == [None, None, None]

    @pytest.mark.asyncio
    async def test_concurrent_synthesize(self):
        svc = TTSService(api_key="fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"audioContent": "dGVzdA=="}
        mock_response.raise_for_status = MagicMock()

        with patch.object(svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await svc.synthesize_many(["hello", "world"])
        assert len(result) == 2
        assert all(r == "dGVzdA==" for r in result)


class TestFormatAudioDataUri:
    def test_wraps_base64_in_data_uri(self):
        from services.tts.tts_service import format_audio_data_uri
        result = format_audio_data_uri("dGVzdA==")
        assert result == "data:audio/mp3;base64,dGVzdA=="