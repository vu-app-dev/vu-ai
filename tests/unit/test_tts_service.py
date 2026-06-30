import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import io

from services.tts.tts_service import TTSService


class TestTTSServiceAvailability:
    def test_available_when_edge_tts_installed(self):
        svc = TTSService()
        assert svc.available is True


class TestTTSServiceSynthesize:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_text(self):
        svc = TTSService()
        result = await svc.synthesize("")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_none_text(self):
        svc = TTSService()
        result = await svc.synthesize(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_base64_audio_on_success(self):
        svc = TTSService(voice="en-US-AriaNeural")

        fake_audio_chunks = [
            {"type": "audio", "data": b"\xff\xfb" + b"fake_audio" * 100},
            {"type": "audio", "data": b"_more_data_"},
        ]

        mock_communicate = MagicMock()
        async def fake_stream():
            for chunk in fake_audio_chunks:
                yield chunk
        mock_communicate.stream = fake_stream

        with patch("services.tts.tts_service.edge_tts.Communicate", return_value=mock_communicate):
            result = await svc.synthesize("Hello interview!")
        assert result is not None
        import base64
        decoded = base64.b64decode(result)
        assert b"fake_audio" in decoded

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_audio(self):
        svc = TTSService()

        mock_communicate = MagicMock()
        async def fake_stream():
            yield {"type": "metadata", "data": {}}
        mock_communicate.stream = fake_stream

        with patch("services.tts.tts_service.edge_tts.Communicate", return_value=mock_communicate):
            result = await svc.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        svc = TTSService()

        with patch("services.tts.tts_service.edge_tts.Communicate", side_effect=Exception("Network error")):
            result = await svc.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_truncates_oversized_text(self):
        svc = TTSService()

        mock_communicate = MagicMock()
        async def fake_stream():
            yield {"type": "audio", "data": b"audio_bytes"}
        mock_communicate.stream = fake_stream

        long_text = "x" * 5000
        with patch("services.tts.tts_service.edge_tts.Communicate", return_value=mock_communicate) as mock_ctor:
            await svc.synthesize(long_text)
            sent_text = mock_ctor.call_args[1]["text"]
            assert len(sent_text) <= 4500


class TestTTSServiceSynthesizeMany:
    @pytest.mark.asyncio
    async def test_concurrent_synthesize(self):
        svc = TTSService()

        mock_communicate = MagicMock()
        async def fake_stream():
            yield {"type": "audio", "data": b"audio"}
        mock_communicate.stream = fake_stream

        with patch("services.tts.tts_service.edge_tts.Communicate", return_value=mock_communicate):
            result = await svc.synthesize_many(["hello", "world"])
        assert len(result) == 2
        assert all(r is not None for r in result)


class TestFormatAudioDataUri:
    def test_wraps_base64_in_data_uri(self):
        from services.tts.tts_service import format_audio_data_uri
        result = format_audio_data_uri("dGVzdA==")
        assert result == "data:audio/mp3;base64,dGVzdA=="