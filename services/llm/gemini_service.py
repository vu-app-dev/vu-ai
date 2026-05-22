import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
DEFAULT_MODEL = "gemini-2.0-flash"


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


class GeminiService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = settings.LLM_MAX_RETRIES,
        rpm: int = settings.LLM_RPM_LIMIT,
    ):
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._model = model
        self._max_retries = max_retries
        self._client = genai.Client(api_key=self._api_key)
        self._rate_limiter = _RateLimiter(rpm)

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
                    config=types.GenerateContentConfig(
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

    @staticmethod
    def load_prompt(template_name: str, **kwargs) -> str:
        template_path = PROMPTS_DIR / f"{template_name}.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return template.format(**kwargs)