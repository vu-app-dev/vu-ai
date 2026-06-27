import asyncio
import base64
import logging
from typing import Optional, Type

from config import settings

logger = logging.getLogger(__name__)

try:
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
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    logger.warning("AssemblyAI streaming v3 not available")


class RealtimeSTT:
    """Real-time Speech-to-Text service using AssemblyAI Universal Streaming v3"""

    def __init__(self):
        settings.validate()
        if not ASSEMBLYAI_AVAILABLE:
            raise RuntimeError("AssemblyAI streaming v3 is not installed")
        self.api_key = settings.ASSEMBLYAI_API_KEY
        self.sample_rate = settings.SAMPLE_RATE
        self.client: Optional[StreamingClient] = None
        self.transcript_queue: Optional[asyncio.Queue] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.session_id: Optional[str] = None

    def connect(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        """
        Connect to AssemblyAI real-time service.

        Args:
            loop: Asyncio event loop (needed for callbacks)

        Returns:
            Queue that receives transcript dicts:
            {"type": "partial"|"final"|"session_begins"|"error", "text": str}
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
            self.session_id = event.id
            logger.info("[STT] Session started: %s", event.id)
            asyncio.run_coroutine_threadsafe(
                self.transcript_queue.put({"type": "session_begins", "session_id": event.id}),
                loop,
            )

        def on_turn(client: Type[StreamingClient], event: TurnEvent):
            if event.transcript:
                msg_type = "final" if event.end_of_turn else "partial"
                asyncio.run_coroutine_threadsafe(
                    self.transcript_queue.put({"type": msg_type, "text": event.transcript}),
                    loop,
                )

        def on_terminated(client: Type[StreamingClient], event: TerminationEvent):
            logger.info("[STT] Session terminated: %.1f seconds processed", event.audio_duration_seconds)

        def on_error(client: Type[StreamingClient], error: StreamingError):
            logger.error("[STT] Error: %s", error)
            asyncio.run_coroutine_threadsafe(
                self.transcript_queue.put({"type": "error", "message": str(error)}),
                loop,
            )

        self.client.on(StreamingEvents.Begin, on_begin)
        self.client.on(StreamingEvents.Turn, on_turn)
        self.client.on(StreamingEvents.Termination, on_terminated)
        self.client.on(StreamingEvents.Error, on_error)

        self.client.connect(
            StreamingParameters(
                sample_rate=self.sample_rate,
                format_turns=True,
            )
        )

        return self.transcript_queue

    def stream(self, audio_base64: str):
        """
        Stream audio data to AssemblyAI.

        Args:
            audio_base64: Audio encoded as base64 string (PCM 16-bit, mono, at sample_rate)
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
                logger.warning("[STT] Error closing: %s", e)
            self.client = None
            self.session_id = None
            