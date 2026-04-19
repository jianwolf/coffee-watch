"""LLM backend abstraction.

Provides a ``LLMBackend`` Protocol so orchestration code depends on an
interface rather than on a switch over ``settings.llm_backend``. Two
implementations live here:

- ``GeminiBackend`` — hosted Gemini via ``google.genai``.
- ``MLXBackend`` — local MLX server (``mlx_lm.server`` / ``mlx_vlm.server``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Optional, Protocol, runtime_checkable

import httpx
from google import genai

from .config import Settings
from .gemini import (
    evaluate_roaster_markdown as evaluate_gemini_roaster_markdown,
    generate_digest_markdown as generate_gemini_digest_markdown,
)
from .mlx_server import MLXServerError, MLXServerManager
from .mlx_text import (
    extract_stream_delta_text,
    flatten_chat_message_content,
    looks_like_reasoning_only,
    sanitize_mlx_text,
)


def backend_label(settings: Settings) -> str:
    return "Gemini" if settings.llm_backend == "gemini" else "MLX"


@runtime_checkable
class LLMBackend(Protocol):
    async def start(self) -> None: ...

    async def aclose(self) -> None: ...

    async def evaluate_roaster_markdown(
        self,
        model: str,
        roaster_name: str,
        prompt: str,
        logger: logging.Logger,
        timeout_s: float,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        ...

    async def generate_digest_markdown(
        self,
        model: str,
        prompt: str,
        logger: logging.Logger,
        timeout_s: float,
        request_name: Optional[str] = None,
    ) -> Optional[str]:
        ...


class GeminiBackend:
    def __init__(
        self,
        settings: Settings,
        api_key: Optional[str],
        logger: logging.Logger,
    ) -> None:
        self._settings = settings
        self._api_key = api_key
        self._logger = logger
        self._client: Optional[genai.Client] = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._client is not None:
            return
        async with self._start_lock:
            if self._client is not None:
                return
            self._client = (
                genai.Client(api_key=self._api_key)
                if self._api_key
                else genai.Client()
            )

    async def aclose(self) -> None:
        self._client = None

    async def evaluate_roaster_markdown(
        self,
        model: str,
        roaster_name: str,
        prompt: str,
        logger: logging.Logger,
        timeout_s: float,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        await self.start()
        assert self._client is not None
        return await evaluate_gemini_roaster_markdown(
            self._client,
            model,
            roaster_name,
            prompt,
            logger,
            timeout_s,
            temperature=self._settings.llm_temperature,
        )

    async def generate_digest_markdown(
        self,
        model: str,
        prompt: str,
        logger: logging.Logger,
        timeout_s: float,
        request_name: Optional[str] = None,
    ) -> Optional[str]:
        await self.start()
        assert self._client is not None
        return await generate_gemini_digest_markdown(
            self._client,
            model,
            prompt,
            logger,
            timeout_s,
            temperature=self._settings.llm_temperature,
            request_name=request_name,
        )


class MLXBackend:
    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger
        self._server: Optional[MLXServerManager] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._start_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(1)

    async def start(self) -> None:
        if self._client is not None:
            return
        async with self._start_lock:
            if self._client is not None:
                return
            log_path = self._settings.log_path.parent / "mlx_server.log"
            self._server = MLXServerManager(
                model=self._settings.mlx_model,
                runtime=self._settings.mlx_runtime,
                host=self._settings.mlx_host,
                port=self._settings.mlx_port,
                startup_timeout=int(self._settings.mlx_startup_timeout_s),
                log_path=log_path,
                trust_remote_code=self._settings.mlx_trust_remote_code,
            )
            await asyncio.to_thread(self._server.start)
            base_url = self._server.base_url.rstrip("/") + "/"
            self._client = httpx.AsyncClient(base_url=base_url)
            await self._log_available_models()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._server is not None:
            await asyncio.to_thread(self._server.stop)
            self._server = None

    async def _log_available_models(self) -> None:
        if self._client is None:
            return
        try:
            response = await self._client.get("models", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self._logger.warning("Failed to inspect MLX models endpoint: %s", exc)
            return
        ids: list[str] = []
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        if ids:
            self._logger.info(
                "MLX models endpoint reports: %s", json.dumps(ids, ensure_ascii=True)
            )
            expected = {self._settings.model, self._settings.digest_model}
            if not expected.intersection(ids):
                self._logger.warning(
                    "Configured local model(s) not advertised by MLX server: %s",
                    ", ".join(sorted(expected)),
                )

    async def evaluate_roaster_markdown(
        self,
        model: str,
        roaster_name: str,
        prompt: str,
        logger: logging.Logger,
        timeout_s: float,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        text = await self._chat_completion(
            model=model,
            prompt=prompt,
            logger=logger,
            timeout_s=timeout_s,
            request_name=roaster_name,
        )
        return text, None

    async def generate_digest_markdown(
        self,
        model: str,
        prompt: str,
        logger: logging.Logger,
        timeout_s: float,
        request_name: Optional[str] = None,
    ) -> Optional[str]:
        return await self._chat_completion(
            model=model,
            prompt=prompt,
            logger=logger,
            timeout_s=timeout_s,
            request_name=request_name or "digest",
        )

    async def _chat_completion(
        self,
        model: str,
        prompt: str,
        logger: logging.Logger,
        timeout_s: float,
        request_name: str,
    ) -> Optional[str]:
        await self.start()
        if self._client is None:
            raise RuntimeError("MLX client is not initialized")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._settings.llm_max_tokens,
            "temperature": self._settings.llm_temperature,
            "stream": bool(self._settings.stream_llm_output),
        }

        async def _request() -> Optional[str]:
            if payload["stream"]:
                return await self._streaming_chat_completion(
                    model=model,
                    payload=payload,
                    logger=logger,
                    timeout_s=timeout_s,
                    request_name=request_name,
                )
            try:
                response = await self._client.post(
                    "chat/completions",
                    json=payload,
                    timeout=None if timeout_s <= 0 else timeout_s,
                )
                response.raise_for_status()
            except httpx.TimeoutException:
                logger.warning(
                    "MLX request timed out for %s after %.1fs", request_name, timeout_s
                )
                return None
            except Exception as exc:
                logger.exception("MLX request failed for %s: %s", request_name, exc)
                return None

            try:
                body = response.json()
            except Exception as exc:
                logger.warning(
                    "MLX response JSON decode failed for %s: %s", request_name, exc
                )
                return None

            raw_text = flatten_chat_message_content(
                (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
            )
            text = sanitize_mlx_text(raw_text)
            usage = body.get("usage")
            if isinstance(usage, dict):
                logger.info(
                    "MLX usage for %s: prompt_tokens=%s output_tokens=%s total_tokens=%s",
                    request_name,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                )
            if text:
                return text
            if looks_like_reasoning_only(raw_text):
                logger.warning(
                    "MLX returned reasoning-only output for %s; treating as empty.",
                    request_name,
                )
            logger.warning("MLX returned no text for %s.", request_name)
            return None

        async with self._request_semaphore:
            return await _request()

    async def _streaming_chat_completion(
        self,
        model: str,
        payload: dict[str, Any],
        logger: logging.Logger,
        timeout_s: float,
        request_name: str,
    ) -> Optional[str]:
        if self._client is None:
            raise RuntimeError("MLX client is not initialized")

        chunks: list[str] = []
        raw_lines: list[str] = []
        header = f"\n[MLX stream start: {request_name} | model={model}]\n"
        sys.stderr.write(header)
        sys.stderr.write("<think>\n")
        sys.stderr.flush()
        try:
            async with self._client.stream(
                "POST",
                "chat/completions",
                json=payload,
                timeout=None if timeout_s <= 0 else timeout_s,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    raw_lines.append(line)
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Ignoring non-JSON MLX stream event for %s: %s",
                            request_name,
                            data,
                        )
                        continue
                    for choice in event.get("choices", []):
                        if not isinstance(choice, dict):
                            continue
                        text = extract_stream_delta_text(choice)
                        if not text:
                            continue
                        chunks.append(text)
                        sys.stderr.write(text)
                        sys.stderr.flush()
        except httpx.TimeoutException:
            logger.warning(
                "MLX streaming request timed out for %s after %.1fs",
                request_name,
                timeout_s,
            )
            return None
        except Exception as exc:
            logger.exception("MLX streaming request failed for %s: %s", request_name, exc)
            return None
        finally:
            sys.stderr.write(f"\n[MLX stream end: {request_name}]\n")
            sys.stderr.flush()

        raw_text = "".join(chunks)
        text = sanitize_mlx_text(raw_text)
        if not text and raw_lines:
            try:
                fallback_body = json.loads("\n".join(raw_lines))
            except json.JSONDecodeError:
                fallback_body = None
            if isinstance(fallback_body, dict):
                fallback_raw = flatten_chat_message_content(
                    (
                        ((fallback_body.get("choices") or [{}])[0]).get("message")
                        or {}
                    ).get("content")
                )
                text = sanitize_mlx_text(fallback_raw)
                raw_text = fallback_raw
        if text:
            return text
        if looks_like_reasoning_only(raw_text):
            logger.warning(
                "MLX streamed reasoning-only output for %s; treating as empty.",
                request_name,
            )
        logger.warning("MLX returned no streamed text for %s.", request_name)
        return None


def create_llm_backend(
    settings: Settings,
    api_key: Optional[str],
    logger: logging.Logger,
) -> LLMBackend:
    if settings.llm_backend == "gemini":
        return GeminiBackend(settings, api_key, logger)
    if settings.llm_backend == "mlx":
        return MLXBackend(settings, logger)
    raise ValueError(f"Unknown llm_backend: {settings.llm_backend!r}")


__all__ = [
    "LLMBackend",
    "GeminiBackend",
    "MLXBackend",
    "MLXServerError",
    "backend_label",
    "create_llm_backend",
]
