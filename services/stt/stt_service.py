import logging

import assemblyai as aai
from dataclasses import dataclass
from typing import List, Dict

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Transcription:
    """Container for transcription results."""
    text: str
    words: List[Dict]
    duration: float


class STT:
    """Speech-to-Text service for file/URL transcription."""

    def __init__(self):
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self.config = aai.TranscriptionConfig(
            speech_models=["universal-3-pro", "universal-2"],
            language_detection=True,
        )
        self.transcriber = aai.Transcriber(config=self.config)

    def transcribe(self, audio_source: str) -> Transcription:
        """
        Transcribe audio from file or URL.

        Args:
            audio_source: Path to local file or URL

        Returns:
            Transcription with text, words, and duration

        Raises:
            RuntimeError: If transcription fails
        """
        logger.info("[STT] Transcribing: %s", audio_source[:100])
        transcript = self.transcriber.transcribe(audio_source)

        if transcript.status == "error":
            logger.error("[STT] Transcription failed: %s", transcript.error)
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        logger.info("[STT] Transcription complete: %d chars, %.1fs", len(transcript.text), transcript.audio_duration)
        return Transcription(
            text=transcript.text or "",
            words=[word.dict() for word in transcript.words],
            duration=transcript.audio_duration or 0.0,
        )