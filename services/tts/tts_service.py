import asyncio
import base64
import io
import logging
from typing import Optional

import edge_tts

from config.settings import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-speech service using edge-tts (Microsoft neural voices).

    No API key, no account, no payment required. Uses Microsoft's free
    Edge browser TTS endpoints directly from Python.

    Returns base64-encoded MP3 audio. If the request fails, returns
    ``None`` — the caller (frontend) should fall back to browser
    SpeechSynthesis.
    """

    def __init__(
        self,
        voice: str = "",
        rate: str = "",
        volume: str = "",
        pitch: str = "",
        proxy: str = "",
    ):
        self._voice = voice or settings.TTS_VOICE
        self._rate = rate or settings.TTS_RATE
        self._volume = volume or settings.TTS_VOLUME
        self._pitch = pitch or settings.TTS_PITCH
        self._proxy = proxy or settings.TTS_PROXY

    @property
    def available(self) -> bool:
        """edge-tts is always available (no API key needed)."""
        try:
            import edge_tts as _  # noqa
            return True
        except ImportError:
            return False

    async def synthesize(self, text: str) -> Optional[str]:
        """Synthesize *text* to base64-encoded MP3. Returns ``None`` on failure."""
        if not text or not self.available:
            return None

        if len(text) > 4500:
            text = text[:4500]

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self._voice,
                rate=self._rate or None,
                volume=self._volume or None,
                pitch=self._pitch or None,
                proxy=self._proxy or None,
            )
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            audio_bytes = buf.read()
            if not audio_bytes:
                logger.warning("edge-tts returned empty audio for text (%d chars)", len(text))
                return None
            return base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.warning("edge-tts synthesize failed: %s", e)
            return None

    async def synthesize_many(self, texts: list[str]) -> list[Optional[str]]:
        """Synthesize multiple texts concurrently."""
        if not self.available:
            return [None] * len(texts)
        tasks = [self.synthesize(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def close(self):
        """No persistent resources to clean up for edge-tts."""
        pass


def format_audio_data_uri(base64_mp3: str) -> str:
    """Wrap raw base64 MP3 in a data URI for <audio src=...>."""
    return f"data:audio/mp3;base64,{base64_mp3}"