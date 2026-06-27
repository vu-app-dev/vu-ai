import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import settings

try:
    from google import genai as _genai_module
    from google.genai import types as _genai_types
except ImportError:
    _genai_module = None
    _genai_types = None

try:
    from groq import Groq as _GroqClient
except ImportError:
    _GroqClient = None

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _extract_json(content: str) -> str | None:
    """Pull a JSON object out of a raw LLM completion.

    Strips markdown ``` fences and returns the substring from the first `{`
    to the last matching `}`. Returns None if no JSON object can be located.
    """
    if not content:
        return None
    text = re.sub(r"```(?:json)?\s*|\s*```", "", content, flags=re.I).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


class _RateLimiter:
    def __init__(self, rpm: int):
        self._rpm = rpm
        self._min_interval = 60.0 / rpm
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


def _create_gemini_service():
    if _genai_module is None:
        raise ImportError("google-genai is not installed. Run: pip install google-genai")

    model = settings.LLM_MODEL or "gemini-2.0-flash"

    class _Gemini:
        def __init__(self, api_key, model_name, max_retries, rpm):
            self._model = model_name
            self._max_retries = max_retries
            self._client = _genai_module.Client(api_key=api_key)
            self._rate_limiter = _RateLimiter(rpm)
            self._types = _genai_types

        async def generate(self, prompt: str) -> str | None:
            await self._rate_limiter.acquire()
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=self._model,
                        contents=prompt,
                    )
                    if response.text:
                        return response.text
                    logger.warning("Gemini returned empty text (attempt %d/%d)", attempt, self._max_retries)
                except Exception as e:
                    logger.error("Gemini generate failed (attempt %d/%d): %s", attempt, self._max_retries, e)
                    if attempt < self._max_retries:
                        await asyncio.sleep(2 ** attempt)
            return None

        async def generate_json(self, prompt: str, response_model: Type[T]) -> T | None:
            await self._rate_limiter.acquire()
            schema = response_model.model_json_schema()
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=self._model,
                        contents=prompt,
                        config=self._types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=schema,
                        ),
                    )
                    if not response.text:
                        logger.warning("Gemini returned empty JSON (attempt %d/%d)", attempt, self._max_retries)
                        if attempt < self._max_retries:
                            await asyncio.sleep(2 ** attempt)
                        continue
                    try:
                        return response_model.model_validate_json(response.text)
                    except ValidationError as ve:
                        logger.warning("Gemini JSON validation failed (attempt %d/%d): %s", attempt, self._max_retries, ve)
                        if attempt < self._max_retries:
                            await asyncio.sleep(2 ** attempt)
                        continue
                except Exception as e:
                    logger.error("Gemini generate_json failed (attempt %d/%d): %s", attempt, self._max_retries, e)
                    if attempt < self._max_retries:
                        await asyncio.sleep(2 ** attempt)
            logger.error("Gemini generate_json exhausted all %d retries", self._max_retries)
            return None

    return _Gemini(
        api_key=settings.GEMINI_API_KEY,
        model_name=model,
        max_retries=settings.LLM_MAX_RETRIES,
        rpm=settings.LLM_RPM_LIMIT,
    )


def _create_groq_service():
    if _GroqClient is None:
        raise ImportError("groq is not installed. Run: pip install groq")

    model = settings.LLM_MODEL or "llama-3.3-70b-versatile"

    class _Groq:
        def __init__(self, api_key, model_name, max_retries, rpm):
            self._model = model_name
            self._max_retries = max_retries
            self._client = _GroqClient(
                api_key=api_key,
                timeout=30.0,
                max_retries=0,
            )
            self._rate_limiter = _RateLimiter(rpm)

        async def generate(self, prompt: str) -> str | None:
            await self._rate_limiter.acquire()
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = await asyncio.to_thread(
                        self._client.chat.completions.create,
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = response.choices[0].message.content
                    if content:
                        return content
                    logger.warning("Groq returned empty text (attempt %d/%d)", attempt, self._max_retries)
                except Exception as e:
                    logger.error("Groq generate failed (attempt %d/%d): %s", attempt, self._max_retries, e)
                    if attempt < self._max_retries:
                        await asyncio.sleep(2 ** attempt)
            return None

        async def generate_json(self, prompt: str, response_model: Type[T]) -> T | None:
            await self._rate_limiter.acquire()
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = await asyncio.to_thread(
                        self._client.chat.completions.create,
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = response.choices[0].message.content
                    if not content:
                        logger.warning("Groq returned empty JSON (attempt %d/%d)", attempt, self._max_retries)
                        if attempt < self._max_retries:
                            await asyncio.sleep(2 ** attempt)
                        continue
                    json_text = _extract_json(content)
                    if not json_text:
                        logger.warning("Groq produced no parseable JSON (attempt %d/%d)", attempt, self._max_retries)
                        if attempt < self._max_retries:
                            await asyncio.sleep(2 ** attempt)
                        continue
                    try:
                        return response_model.model_validate_json(json_text)
                    except ValidationError as ve:
                        logger.warning("Groq JSON validation failed (attempt %d/%d): %s", attempt, self._max_retries, ve)
                        if attempt < self._max_retries:
                            await asyncio.sleep(2 ** attempt)
                        continue
                except Exception as e:
                    logger.error("Groq generate_json failed (attempt %d/%d): %s", attempt, self._max_retries, e)
                    if attempt < self._max_retries:
                        await asyncio.sleep(2 ** attempt)
            logger.error("Groq generate_json exhausted all %d retries", self._max_retries)
            return None

    return _Groq(
        api_key=settings.GROQ_API_KEY,
        model_name=model,
        max_retries=settings.LLM_MAX_RETRIES,
        rpm=settings.LLM_RPM_LIMIT,
    )


class LLMService:
    """Unified LLM service that delegates to the configured provider."""

    def __init__(self, provider: str | None = None):
        provider = (provider or settings.LLM_PROVIDER).lower()
        if provider == "groq":
            self._impl = _create_groq_service()
            logger.info("LLM provider: Groq (%s)", settings.LLM_MODEL or "llama-3.3-70b-versatile")
        else:
            self._impl = _create_gemini_service()
            logger.info("LLM provider: Gemini (%s)", settings.LLM_MODEL or "gemini-2.0-flash")

    async def generate(self, prompt: str) -> str | None:
        return await self._impl.generate(prompt)

    async def generate_json(self, prompt: str, response_model: Type[T]) -> T | None:
        return await self._impl.generate_json(prompt, response_model)


# Backward-compatible alias
GeminiService = LLMService