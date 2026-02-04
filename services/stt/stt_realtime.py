import asyncio
import base64
from typing import Optional, Type
from config import settings

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
    TerminationEvent,
)


class RealtimeSTT:
    """Real-time Speech-to-Text service using AssemblyAI Universal Streaming v3"""

    def __init__(self):
        settings.validate()
        self.api_key = settings.ASSEMBLYAI_API_KEY
        self.client: Optional[StreamingClient] = None
        self.transcript_queue: Optional[asyncio.Queue] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def connect(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        """
        Connect to AssemblyAI real-time service.
        
        Args:
            loop: Asyncio event loop (needed for callbacks)
            
        Returns:
            Queue that receives transcripts
        """
        self.loop = loop
        self.transcript_queue = asyncio.Queue()

        self.client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )

        def on_begin(client: Type[StreamingClient], event: BeginEvent):
            print(f"[STT] Session started: {event.id}")

        def on_turn(client: Type[StreamingClient], event: TurnEvent):
            """Called when we receive transcription"""
            if event.transcript:
                msg_type = "final" if event.end_of_turn else "partial"
                asyncio.run_coroutine_threadsafe(
                    self.transcript_queue.put({
                        "type": msg_type,
                        "text": event.transcript
                    }),
                    self.loop
                )

        def on_terminated(client: Type[StreamingClient], event: TerminationEvent):
            print(f"[STT] Session terminated: {event.audio_duration_seconds} seconds processed")

        def on_error(client: Type[StreamingClient], error: StreamingError):
            print(f"[STT Error]: {error}")

        self.client.on(StreamingEvents.Begin, on_begin)
        self.client.on(StreamingEvents.Turn, on_turn)
        self.client.on(StreamingEvents.Termination, on_terminated)
        self.client.on(StreamingEvents.Error, on_error)

        self.client.connect(
            StreamingParameters(
                sample_rate=settings.SampleRate,
                format_turns=True,
            )
        )

        return self.transcript_queue

    def stream(self, audio_base64: str):
        """
        Stream audio data to AssemblyAI.
        
        Args:
            audio_base64: Audio encoded as base64 string
        """
        if self.client:
            audio_bytes = base64.b64decode(audio_base64)
            self.client.stream(audio_bytes)

    def close(self):
        """Disconnect from AssemblyAI real-time service."""
        if self.client:
            try:
                self.client.disconnect(terminate=True)
            except Exception as e:
                print(f"[STT] Error closing: {e}")
            self.client = None
            