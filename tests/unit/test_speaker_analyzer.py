import pytest
from unittest.mock import MagicMock, patch

from services.scoring.speaker_analyzer import SpeakerAnalyzer, SpeakerAnalysis


class TestSpeakerAnalysis:
    def test_single_speaker(self):
        result = SpeakerAnalysis(speaker_count=1, second_speaker_pct=0.0, utterances_by_speaker={"A": 120.0})
        assert result.speaker_count == 1
        assert result.second_speaker_pct == 0.0

    def test_two_speakers(self):
        result = SpeakerAnalysis(speaker_count=2, second_speaker_pct=25.0, utterances_by_speaker={"A": 90.0, "B": 30.0})
        assert result.speaker_count == 2
        assert result.second_speaker_pct == 25.0


class TestSpeakerAnalyzer:
    @pytest.mark.asyncio
    async def test_returns_none_on_empty_audio(self):
        analyzer = SpeakerAnalyzer()
        result = await analyzer.analyze(b"")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_short_audio(self):
        analyzer = SpeakerAnalyzer()
        result = await analyzer.analyze(b"\x00" * 100)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_api_failure(self):
        analyzer = SpeakerAnalyzer()
        with patch.object(analyzer, "_aai") as mock_aai:
            mock_transcriber = MagicMock()
            mock_aai.Transcriber.return_value = mock_transcriber
            mock_transcriber.transcribe.side_effect = Exception("API down")
            result = await analyzer.analyze(b"\x00" * 64000)
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error_status(self):
        analyzer = SpeakerAnalyzer()
        with patch.object(analyzer, "_aai") as mock_aai:
            mock_transcriber = MagicMock()
            mock_aai.Transcriber.return_value = mock_transcriber
            mock_result = MagicMock()
            mock_result.status = "error"
            mock_result.error = "Invalid audio"
            mock_transcriber.transcribe.return_value = mock_result
            result = await analyzer.analyze(b"\x00" * 64000)
            assert result is None

    @pytest.mark.asyncio
    async def test_no_utterances_returns_single_speaker(self):
        analyzer = SpeakerAnalyzer()
        with patch.object(analyzer, "_aai") as mock_aai:
            mock_transcriber = MagicMock()
            mock_aai.Transcriber.return_value = mock_transcriber
            mock_result = MagicMock()
            mock_result.status = "completed"
            mock_result.utterances = None
            mock_transcriber.transcribe.return_value = mock_result
            result = await analyzer.analyze(b"\x00" * 64000)
            assert result is not None
            assert result.speaker_count == 1
            assert result.second_speaker_pct == 0.0

    @pytest.mark.asyncio
    async def test_two_speakers_detected(self):
        analyzer = SpeakerAnalyzer()
        with patch.object(analyzer, "_aai") as mock_aai:
            mock_transcriber = MagicMock()
            mock_aai.Transcriber.return_value = mock_transcriber

            mock_utt_a = MagicMock()
            mock_utt_a.speaker = "A"
            mock_utt_a.start = 0
            mock_utt_a.end = 90000

            mock_utt_b = MagicMock()
            mock_utt_b.speaker = "B"
            mock_utt_b.start = 90000
            mock_utt_b.end = 120000

            mock_result = MagicMock()
            mock_result.status = "completed"
            mock_result.utterances = [mock_utt_a, mock_utt_b]
            mock_transcriber.transcribe.return_value = mock_result

            result = await analyzer.analyze(b"\x00" * 64000)
            assert result is not None
            assert result.speaker_count == 2
            assert result.second_speaker_pct == 25.0
