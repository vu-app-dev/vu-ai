import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from clients.backend_client import BackendClient, _RetryEntry

MOCK_API_KEY = "test-api-key"
MOCK_BASE_URL = "http://localhost:3000"


def _make_response(status_code: int = 201, json_data: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "http://localhost:3000/test")
    return httpx.Response(status_code=status_code, request=request, json=json_data or {"id": "test"})


def _make_get_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    request = httpx.Request("GET", "http://localhost:3000/mock/get/test")
    return httpx.Response(status_code=status_code, request=request, json=json_data or {})


class TestHeaders:
    def test_headers_include_api_key(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)
        headers = client._headers()
        assert headers["X-API-Key"] == MOCK_API_KEY
        assert headers["Content-Type"] == "application/json"

    def test_headers_include_idempotency_key(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)
        headers = client._headers(idempotency_key="sess1-q1-att1")
        assert headers["X-Idempotency-Key"] == "sess1-q1-att1"


class TestGetMock:
    @pytest.mark.asyncio
    async def test_get_mock_success(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)
        mock_data = {"id": "mock1", "type": "TECHNICAL"}
        mock_response = _make_get_response(200, mock_data)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.get_mock("mock1")
        assert result == mock_data

    @pytest.mark.asyncio
    async def test_get_mock_not_found(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)
        mock_response = _make_get_response(404, {"message": "Not found"})
        mock_response._status_code = 404
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.get_mock("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_mock_unreachable(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.get_mock("mock1")
        assert result is None


class TestCreatePerformance:
    @pytest.mark.asyncio
    async def test_create_performance_success(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, max_attempts=1)
        mock_response = _make_response(201)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.create_performance(
            "c1", {"score": 75}, idempotency_key="sess1-q1-att1"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_idempotency_key_sent(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, max_attempts=1)
        mock_response = _make_response(201)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        client._client = mock_client

        await client.create_performance("c1", {"score": 85}, idempotency_key="sess1-q1-att1")
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"]["X-Idempotency-Key"] == "sess1-q1-att1"
        assert call_kwargs["headers"]["X-API-Key"] == MOCK_API_KEY

    @pytest.mark.asyncio
    async def test_client_error_returns_false(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, max_attempts=1)
        request = httpx.Request("POST", "http://localhost:3000/candidates/c1/performance")
        error_response = httpx.Response(400, request=request, json={"error": "bad request"})
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(message="400", request=request, response=error_response)
        )
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.create_performance("c1", {"score": 75}, idempotency_key="key1")
        assert result is False


class TestRetryOnBackendUnreachable:
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        client = BackendClient(
            base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY,
            max_attempts=3, retry_interval=0.01,
        )
        call_count = 0
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False

        async def side_effect(url, json=None, headers=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            return _make_response(201)

        mock_client.post = AsyncMock(side_effect=side_effect)
        client._client = mock_client

        with patch("clients.backend_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.create_performance(
                "c1", {"score": 75}, idempotency_key="key1"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_retry_exhausted_queues_entry(self):
        client = BackendClient(
            base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY,
            max_attempts=2, retry_interval=0.01,
        )
        request = httpx.Request("POST", "http://localhost:3000/candidates/c1/performance")
        error_response = httpx.Response(503, request=request, json={"error": "unavailable"})
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(message="503", request=request, response=error_response)
        )
        client._client = mock_client

        with patch("clients.backend_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.create_performance(
                "c1", {"score": 75}, idempotency_key="key1"
            )
            assert result is False
            assert "key1" in client._queue


class TestRetryQueue:
    def test_enqueue_retry(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)
        client._enqueue_retry("http://localhost:3000/test", {"data": 1}, "key1", "test")
        assert "key1" in client._queue
        assert client._queue["key1"].endpoint_label == "test"

    def test_duplicate_key_skipped(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)
        client._enqueue_retry("http://localhost:3000/test", {"data": 1}, "key1", "test")
        client._enqueue_retry("http://localhost:3000/test", {"data": 2}, "key1", "test")
        assert len(client._queue) == 1

    def test_queue_max_size_drops_oldest(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, max_queue_size=2)
        client._enqueue_retry("http://localhost:3000/test1", {"data": 1}, "key1", "test1")
        client._enqueue_retry("http://localhost:3000/test2", {"data": 2}, "key2", "test2")
        client._enqueue_retry("http://localhost:3000/test3", {"data": 3}, "key3", "test3")
        assert len(client._queue) == 2
        assert "key1" not in client._queue

    @pytest.mark.asyncio
    async def test_create_cv_analysis_sends_correct_endpoint(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, max_attempts=1)
        mock_response = _make_response(201)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.create_cv_analysis(
            "c1", {"skills": ["Python"]}, idempotency_key="cv1"
        )
        assert result is True
        call_args = mock_client.post.call_args[0][0]
        assert "/candidates/c1/cv-analysis" in call_args

    @pytest.mark.asyncio
    async def test_create_question_sends_correct_endpoint(self):
        client = BackendClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, max_attempts=1)
        mock_response = _make_response(201)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        client._client = mock_client

        result = await client.create_question(
            "c1", {"question": "What is REST?"}, idempotency_key="q1"
        )
        assert result is True
        call_args = mock_client.post.call_args[0][0]
        assert "/candidates/c1/questions" in call_args