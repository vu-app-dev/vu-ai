import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import httpx

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
RETRY_INTERVAL = 30.0
MAX_ATTEMPTS = 5
MAX_QUEUE_SIZE = 50


@dataclass
class _RetryEntry:
    url: str
    payload: dict
    idempotency_key: str
    endpoint_label: str
    attempts: int = 0


class BackendClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = MAX_ATTEMPTS,
        retry_interval: float = RETRY_INTERVAL,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        self._base_url = (base_url or settings.BACKEND_URL).rstrip("/")
        self._api_key = api_key or settings.BACKEND_API_KEY
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_interval = retry_interval
        self._max_queue_size = max_queue_size
        self._queue: OrderedDict[str, _RetryEntry] = OrderedDict()
        self._retry_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self._client

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self._api_key,
        }
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key
        return headers

    async def get_mock(self, mock_id: str) -> dict | None:
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self._base_url}/mock/get/{mock_id}",
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Backend GET mock %s failed: %s", mock_id, e)
            return None
        except httpx.RequestError as e:
            logger.error("Backend GET mock %s unreachable: %s", mock_id, e)
            return None

    async def create_performance(
        self, candidate_id: str, data: dict, idempotency_key: str
    ) -> bool:
        return await self._post_with_retry(
            endpoint=f"/candidates/{candidate_id}/performance",
            payload=data,
            idempotency_key=idempotency_key,
            endpoint_label="performance",
        )

    async def create_cv_analysis(
        self, candidate_id: str, data: dict, idempotency_key: str
    ) -> bool:
        return await self._post_with_retry(
            endpoint=f"/candidates/{candidate_id}/cv-analysis",
            payload=data,
            idempotency_key=idempotency_key,
            endpoint_label="cv-analysis",
        )

    async def create_question(
        self, candidate_id: str, data: dict, idempotency_key: str
    ) -> bool:
        return await self._post_with_retry(
            endpoint=f"/candidates/{candidate_id}/questions",
            payload=data,
            idempotency_key=idempotency_key,
            endpoint_label="question",
        )

    async def _post_with_retry(
        self,
        endpoint: str,
        payload: dict,
        idempotency_key: str,
        endpoint_label: str,
    ) -> bool:
        url = f"{self._base_url}{endpoint}"
        for attempt in range(1, self._max_attempts + 1):
            try:
                client = await self._get_client()
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._headers(idempotency_key),
                )
                response.raise_for_status()
                logger.info(
                    "Backend POST %s succeeded (attempt %d, key=%s)",
                    endpoint_label, attempt, idempotency_key,
                )
                return True
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    logger.error(
                        "Backend POST %s failed with client error %d (key=%s): %s",
                        endpoint_label, e.response.status_code, idempotency_key, e,
                    )
                    return False
                logger.warning(
                    "Backend POST %s failed (attempt %d/%d, key=%s): %s",
                    endpoint_label, attempt, self._max_attempts, idempotency_key, e,
                )
                if attempt < self._max_attempts:
                    await asyncio.sleep(min(2 ** attempt, self._retry_interval))

        self._enqueue_retry(url, payload, idempotency_key, endpoint_label)
        return False

    def _enqueue_retry(
        self, url: str, payload: dict, idempotency_key: str, endpoint_label: str
    ):
        if idempotency_key in self._queue:
            logger.warning("Retry already queued for key=%s, skipping", idempotency_key)
            return
        if len(self._queue) >= self._max_queue_size:
            oldest_key, _ = self._queue.popitem(last=False)
            logger.warning("Retry queue full, dropping oldest key=%s", oldest_key)
        entry = _RetryEntry(
            url=url,
            payload=payload,
            idempotency_key=idempotency_key,
            endpoint_label=endpoint_label,
        )
        self._queue[idempotency_key] = entry
        logger.info("Queued retry for %s key=%s", endpoint_label, idempotency_key)

    async def start_retry_worker(self):
        if self._retry_task and not self._retry_task.done():
            return
        self._retry_task = asyncio.create_task(self._retry_loop())

    async def stop_retry_worker(self):
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

    async def _retry_loop(self):
        while True:
            await asyncio.sleep(self._retry_interval)
            if not self._queue:
                continue
            keys_to_process = list(self._queue.keys())
            for key in keys_to_process:
                entry = self._queue.get(key)
                if entry is None:
                    continue
                entry.attempts += 1
                if entry.attempts > self._max_attempts:
                    logger.error(
                        "Retry exhausted for %s key=%s after %d attempts",
                        entry.endpoint_label, key, self._max_attempts,
                    )
                    continue
                try:
                    client = await self._get_client()
                    response = await client.post(
                        entry.url,
                        json=entry.payload,
                        headers=self._headers(entry.idempotency_key),
                    )
                    response.raise_for_status()
                    logger.info(
                        "Retry succeeded for %s key=%s (attempt %d)",
                        entry.endpoint_label, key, entry.attempts,
                    )
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                        logger.error(
                            "Retry failed with client error for %s key=%s: %s",
                            entry.endpoint_label, key, e,
                        )
                        continue
                    logger.warning(
                        "Retry failed for %s key=%s (attempt %d): %s",
                        entry.endpoint_label, key, entry.attempts, e,
                    )
                    self._queue[key] = entry

    async def close(self):
        await self.stop_retry_worker()
        if self._client and not self._client.is_closed:
            await self._client.aclose()


backend_client = BackendClient()