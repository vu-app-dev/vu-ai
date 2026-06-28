import asyncio
import base64
import logging
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

_GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


class TTSService:
    """Text-to-speech service using Google Cloud TTS REST API.

    Returns base64-encoded MP3 audio. If the API key is missing or the
    request fails, returns ``None`` — the caller (frontend) should fall
    back to browser SpeechSynthesis.
    """

    def __init__(
        self,
        api_key: str = "",
        voice_language: str = "",
        voice_name: str = "",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        timeout: float = 10.0,
    ):
        self._api_key = api_key or settings.GOOGLE_TTS_API_KEY
        self._voice_language = voice_language or settings.TTS_VOICE_LANGUAGE
        self._voice_name = voice_name or settings.TTS_VOICE_NAME
        self._speaking_rate = speaking_rate if speaking_rate != 1.0 else settings.TTS_SPEAKING_RATE
        self._pitch = pitch if pitch != 0.0 else settings.TTS_PITCH
        self._timeout = timeout if timeout != 10.0 else settings.TTS_TIMEOUT_SECONDS
        self._client = httpx.AsyncClient(timeout=self._timeout) if self._api_key else None

    @property
    def available(self) -> bool:
        """True when Google TTS can be used (API key present)."""
        return bool(self._api_key)

    async def synthesize(self, text: str) -> Optional[str]:
        """Synthesize *text* to base64-encoded MP3. Returns ``None`` on failure."""
        if not text or not self.available:
            return None

        # Google TTS limit is 5000 bytes; trim to avoid 400s
        if len(text) > 4500:
            text = text[:4500]

        try:
            response = await self._client.post(
                _GOOGLE_TTS_URL,
                params={"key": self._api_key},
                json={
                    "input": {"text": text},
                    "voice": {
                        "languageCode": self._voice_language,
                        "name": self._voice_name,
                    },
                    "audioConfig": {
                        "audioEncoding": "MP3",
                        "speakingRate": self._speaking_rate,
                        "pitch": self._pitch,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            audio_content = data.get("audioContent")
            if not audio_content:
                logger.warning("Google TTS returned empty audioContent")
                return None
            return audio_content  # already base64
        except httpx.TimeoutException:
            logger.warning("Google TTS request timed out (%.1fs)", self._timeout)
            return None
        except httpx.HTTPStatusError as e:
            logger.warning("Google TTS HTTP error %d: %s", e.response.status_code, e.response.text[:200])
            return None
        except Exception as e:
            logger.warning("Google TTS unexpected error: %s", e)
            return None

    async def synthesize_many(self, texts: list[str]) -> list[Optional[str]]:
        """Synthesize multiple texts concurrently."""
        if not self.available:
            return [None] * len(texts)
        tasks = [self.synthesize(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def format_audio_data_uri(base64_mp3: str) -> str:
    """Wrap raw base64 MP3 in a data URI for <audio src=...>."""
    return f"data:audio/mp3;base64,{base64_mp3}"