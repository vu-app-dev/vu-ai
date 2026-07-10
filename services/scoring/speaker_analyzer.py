import asyncio
import logging
import os
import tempfile
import wave
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SpeakerAnalysis:
    speaker_count: int
    second_speaker_pct: float
    utterances_by_speaker: dict[str, float]


class SpeakerAnalyzer:
    def __init__(self):
        import assemblyai as aai
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self._aai = aai

    async def analyze(self, audio_bytes: bytes, sample_rate: int = 16000) -> SpeakerAnalysis | None:
        if not audio_bytes or len(audio_bytes) < sample_rate * 2:
            logger.warning("[Speaker] Audio too short for diarization")
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                with wave.open(tmp_path, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_bytes)

            config = self._aai.TranscriptionConfig(speaker_labels=True)
            transcriber = self._aai.Transcriber(config=config)

            transcript = await asyncio.to_thread(transcriber.transcribe, tmp_path)

            if transcript.status == "error":
                logger.warning("[Speaker] Diarization failed: %s", transcript.error)
                return None

            if not transcript.utterances:
                logger.info("[Speaker] No utterances returned")
                return SpeakerAnalysis(speaker_count=1, second_speaker_pct=0.0, utterances_by_speaker={})

            speaker_durations: dict[str, float] = {}
            for utt in transcript.utterances:
                speaker = utt.speaker or "unknown"
                duration_ms = (utt.end or 0) - (utt.start or 0)
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration_ms

            total_duration = sum(speaker_durations.values())
            if total_duration == 0:
                return SpeakerAnalysis(speaker_count=1, second_speaker_pct=0.0, utterances_by_speaker={})

            primary = max(speaker_durations, key=speaker_durations.get)
            primary_duration = speaker_durations[primary]
            second_speaker_duration = total_duration - primary_duration
            second_speaker_pct = (second_speaker_duration / total_duration) * 100

            utterances_seconds = {s: d / 1000.0 for s, d in speaker_durations.items()}

            return SpeakerAnalysis(
                speaker_count=len(speaker_durations),
                second_speaker_pct=round(second_speaker_pct, 1),
                utterances_by_speaker=utterances_seconds,
            )

        except Exception as e:
            logger.warning("[Speaker] Diarization error: %s", e)
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
