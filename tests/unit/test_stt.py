import pytest
from unittest.mock import patch, MagicMock
import asyncio

from config import settings
from services.stt.stt_service import STT, Transcription
from services.stt.stt_realtime import RealtimeSTT


class TestSTT:
    @patch("services.stt.stt_service.aai")
    def test_transcribe_success(self, mock_aai):
        mock_transcriber = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.text = "Hello world"
        mock_result.audio_duration = 5.0
        mock_result.words = []
        mock_transcriber.transcribe.return_value = mock_result
        mock_aai.Transcriber.return_value = mock_transcriber

        stt = STT()
        result = stt.transcribe("test.wav")

        assert isinstance(result, Transcription)
        assert result.text == "Hello world"
        assert result.duration == 5.0

    @patch("services.stt.stt_service.aai")
    def test_transcribe_failure_raises(self, mock_aai):
        mock_transcriber = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "error"
        mock_result.error = "Audio file not found"
        mock_transcriber.transcribe.return_value = mock_result
        mock_aai.Transcriber.return_value = mock_transcriber

        stt = STT()
        with pytest.raises(RuntimeError, match="Transcription failed"):
            stt.transcribe("nonexistent.wav")

    @patch("services.stt.stt_service.aai")
    def test_transcribe_sets_api_key(self, mock_aai):
        mock_transcriber = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.text = "test"
        mock_result.audio_duration = 1.0
        mock_result.words = []
        mock_transcriber.transcribe.return_value = mock_result
        mock_aai.Transcriber.return_value = mock_transcriber

        STT()
        assert mock_aai.settings.api_key == settings.ASSEMBLYAI_API_KEY


class TestTranscription:
    def test_dataclass_fields(self):
        t = Transcription(text="hello", words=[{"word": "hello"}], duration=2.5)
        assert t.text == "hello"
        assert t.words == [{"word": "hello"}]
        assert t.duration == 2.5

    def test_empty_transcription(self):
        t = Transcription(text="", words=[], duration=0.0)
        assert t.text == ""
        assert len(t.words) == 0


class TestRealtimeSTT:
    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    @patch("services.stt.stt_realtime.StreamingClient")
    def test_connect_creates_client_and_queue(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        stt = RealtimeSTT()
        loop = asyncio.new_event_loop()
        try:
            queue = stt.connect(loop)
            assert queue is not None
            assert isinstance(queue, asyncio.Queue)
            assert stt.client is mock_client
            mock_client.connect.assert_called_once()
        finally:
            loop.close()

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    @patch("services.stt.stt_realtime.StreamingClient")
    def test_stream_sends_audio(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        stt = RealtimeSTT()
        loop = asyncio.new_event_loop()
        try:
            stt.connect(loop)
            stt.stream("aGVsbG8=")
            mock_client.stream.assert_called_once()
            call_args = mock_client.stream.call_args[0][0]
            assert isinstance(call_args, bytes)
        finally:
            loop.close()

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    @patch("services.stt.stt_realtime.StreamingClient")
    def test_stream_ignores_when_no_client(self, mock_client_cls):
        stt = RealtimeSTT()
        stt.client = None
        stt.stream("aGVsbG8=")
        mock_client_cls.assert_not_called()

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    @patch("services.stt.stt_realtime.StreamingClient")
    def test_close_disconnects(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        stt = RealtimeSTT()
        loop = asyncio.new_event_loop()
        try:
            stt.connect(loop)
            stt.close()
            mock_client.disconnect.assert_called_once_with(terminate=True)
            assert stt.client is None
        finally:
            loop.close()

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    def test_close_handles_already_none(self):
        stt = RealtimeSTT()
        stt.client = None
        stt.close()
        assert stt.client is None

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    @patch("services.stt.stt_realtime.StreamingClient")
    def test_close_handles_disconnect_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.disconnect.side_effect = Exception("Connection closed")
        mock_client_cls.return_value = mock_client

        stt = RealtimeSTT()
        loop = asyncio.new_event_loop()
        try:
            stt.connect(loop)
            stt.close()
            assert stt.client is None
        finally:
            loop.close()

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", False)
    def test_init_fails_without_assemblyai(self):
        with pytest.raises(RuntimeError, match="AssemblyAI streaming v3 is not installed"):
            RealtimeSTT()

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    @patch("services.stt.stt_realtime.StreamingClient")
    def test_on_begin_callback(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        stt = RealtimeSTT()
        loop = asyncio.new_event_loop()
        try:
            queue = stt.connect(loop)

            begin_handler = mock_client.on.call_args_list[0]
            mock_event = MagicMock()
            mock_event.id = "test-session-123"
            begin_handler[0][1](mock_client, mock_event)

            task = asyncio.ensure_future(queue.get(), loop=loop)
            loop.run_until_complete(asyncio.sleep(0.1))
            result = task.result()
            assert result["type"] == "session_begins"
            assert result["session_id"] == "test-session-123"
        finally:
            loop.close()

    @patch("services.stt.stt_realtime.ASSEMBLYAI_AVAILABLE", True)
    @patch("services.stt.stt_realtime.StreamingClient")
    def test_sample_rate_from_settings(self, mock_client_cls):
        MagicMock()
        mock_client_cls.return_value = MagicMock()

        stt = RealtimeSTT()
        assert stt.sample_rate == settings.SAMPLE_RATE