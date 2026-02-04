import assemblyai as aai
from dataclasses import dataclass
from typing import List, Dict
from config import settings

@dataclass
class Transcription:
    """
    Container for transcription results.
    @dataclass automatically creates __init__, __repr__, etc.
    """
    text: str
    words: List[Dict]
    duration: float

    
class STT:
    """Speech-to-Text service for transcription"""

    def __init__(self):
        settings.validate()
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY

        self.config = aai.TranscriptionConfig(
            speech_models=["universal-3-pro", "universal-2"],
            language_detection=True,
            language=settings.LANGUAGE
        )

        self.transcriber = aai.Transcriber(config=self.config)

    def transcribe(self, audio_source: str) -> Transcription:
        """
        Transcribe audio from file or URL.
        
        Args:
            audio_source: Path to file OR URL
            
        Returns:
            TranscriptionResult with text and metadata
        """
        transcript = self.transcriber.transcribe(audio_source)    
        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}") 
        return Transcription(
            text=transcript.text,
            words=[word._asdict() for word in transcript.words],
            duration=transcript.duration
        )
