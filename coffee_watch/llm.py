from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any, Optional

import httpx
from google import genai

from .config import Settings
from .gemini import (
    evaluate_roaster_markdown as evaluate_gemini_roaster_markdown,
    format_grounding_metadata,
    generate_digest_markdown as generate_gemini_digest_markdown,
)
from .mlx_server import MLXServerError, MLXServerManager

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_CLOSE_TAG_RE = re.compile(r"</think>", re.IGNORECASE)
LEADING_REASONING_LABEL_RE = re.compile(
    r"^\s*(thinking\s*process|thinking|reasoning|analysis)\s*:\s*",
    re.IGNORECASE,
)
ANALYZE_REQUEST_RE = re.compile(
    r"analy[sz]e\s*the\s*request", re.IGNORECASE
)
MLX_MAX_TOKENS = 100_000


def backend_label(settings: Settings) -> str:
    return "Gemini" if settings.llm_backend == "gemini" else "MLX"


def _flatten_chat_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                chunks.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks).strip()


def _extract_stream_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            chunks.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _trim_to_markdown_answer(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if index == 0:
            continue
        if line.lstrip().startswith("#"):
            candidate = "\n".join(lines[index:]).strip()
            if candidate:
                return candidate
    return ""


def _looks_like_reasoning_only(text: str) -> bool:
    compact = re.sub(r"\s+", "", text[:2000]).lower()
    if compact.startswith(
        ("thinkingprocess:", "thinking:", "reasoning:", "analysis:")
    ):
        return True
    return "analyzetherequest" in compact


def _sanitize_mlx_text(text: str) -> str:
    cleaned = text
    if THINK_CLOSE_TAG_RE.search(cleaned):
        cleaned = THINK_CLOSE_TAG_RE.split(cleaned)[-1]
    cleaned = THINK_TAG_RE.sub("", cleaned).strip()
    if not cleaned:
        return ""
    if LEADING_REASONING_LABEL_RE.match(cleaned) or ANALYZE_REQUEST_RE.search(
        cleaned[:500]
    ):
        recovered = _trim_to_markdown_answer(cleaned)
        if recovered:
            cleaned = recovered
        else:
            return ""
    return cleaned.strip()


def _extract_stream_delta_text(choice: dict[str, Any]) -> str:
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        text = _extract_stream_content_text(content)
        if text:
            return text
    message = choice.get("message")
    if isinstance(message, dict):
        return _extract_stream_content_text(message.get("content"))
    return ""


class CoffeeWatchLLM:
    def __init__(self, settings: Settings, api_key: Optional[str], logger) -> None:
        self._settings = settings
        self._api_key = api_key
        self._logger = logger
        self._gemini_client: Optional[genai.Client] = None
        self._mlx_server: Optional[MLXServerManager] = None
        self._mlx_client: Optional[httpx.AsyncClient] = None
        self._start_lock = asyncio.Lock()
        self._started = False
        self._request_semaphore = (
            asyncio.Semaphore(1) if settings.llm_backend == "mlx" else None
        )

    @property
    def backend(self) -> str:
        return self._settings.llm_backend

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self._settings.llm_backend == "gemini":
                self._gemini_client = (
                    genai.Client(api_key=self._api_key)
                    if self._api_key
                    else genai.Client()
                )
            else:
                log_path = self._settings.log_path.parent / "mlx_server.log"
                self._mlx_server = MLXServerManager(
                    model=self._settings.mlx_model,
                    runtime=self._settings.mlx_runtime,
                    host=self._settings.mlx_host,
                    port=self._settings.mlx_port,
                    startup_timeout=int(self._settings.mlx_startup_timeout_s),
                    log_path=log_path,
                    trust_remote_code=self._settings.mlx_trust_remote_code,
                )
                try:
                    await asyncio.to_thread(self._mlx_server.start)
                except MLXServerError:
                    raise
                base_url = self._mlx_server.base_url.rstrip("/") + "/"
                self._mlx_client = httpx.AsyncClient(base_url=base_url)
                await self._log_available_models()
            self._started = True

    async def close(self) -> None:
        if self._mlx_client is not None:
            await self._mlx_client.aclose()
            self._mlx_client = None
        if self._mlx_server is not None:
            await asyncio.to_thread(self._mlx_server.stop)
            self._mlx_server = None
        self._gemini_client = None
        self._started = False

    async def _log_available_models(self) -> None:
        if self._mlx_client is None:
            return
        try:
            response = await self._mlx_client.get("models", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self._logger.warning("Failed to inspect MLX models endpoint: %s", exc)
            return
        ids = []
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
        logger,
        timeout_s: float,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        await self.start()
        if self._settings.llm_backend == "gemini":
            assert self._gemini_client is not None
            return await evaluate_gemini_roaster_markdown(
                self._gemini_client,
                model,
                roaster_name,
                prompt,
                logger,
                timeout_s,
            )
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
        logger,
        timeout_s: float,
        request_name: Optional[str] = None,
    ) -> Optional[str]:
        await self.start()
        if self._settings.llm_backend == "gemini":
            assert self._gemini_client is not None
            return await generate_gemini_digest_markdown(
                self._gemini_client,
                model,
                prompt,
                logger,
                timeout_s,
                request_name=request_name,
            )
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
        logger,
        timeout_s: float,
        request_name: str,
    ) -> Optional[str]:
        if self._mlx_client is None:
            raise RuntimeError("MLX client is not initialized")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MLX_MAX_TOKENS,
            "temperature": 0.2,
            "stream": False,
        }

        if self._settings.stream_llm_output:
            payload["stream"] = True

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
                response = await self._mlx_client.post(
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
                logger.warning("MLX response JSON decode failed for %s: %s", request_name, exc)
                return None

            text = _sanitize_mlx_text(
                _flatten_chat_message_content(
                    (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
                )
            )
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
            raw_text = _flatten_chat_message_content(
                (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
            )
            if _looks_like_reasoning_only(raw_text):
                logger.warning(
                    "MLX returned reasoning-only output for %s; treating as empty.",
                    request_name,
                )
            logger.warning("MLX returned no text for %s.", request_name)
            return None

        if self._request_semaphore is None:
            return await _request()
        async with self._request_semaphore:
            return await _request()

    async def _streaming_chat_completion(
        self,
        model: str,
        payload: dict[str, Any],
        logger,
        timeout_s: float,
        request_name: str,
    ) -> Optional[str]:
        if self._mlx_client is None:
            raise RuntimeError("MLX client is not initialized")

        chunks: list[str] = []
        raw_lines: list[str] = []
        header = f"\n[MLX stream start: {request_name} | model={model}]\n"
        sys.stderr.write(header)
        sys.stderr.write("<think>\n")
        sys.stderr.flush()
        try:
            async with self._mlx_client.stream(
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
                        text = _extract_stream_delta_text(choice)
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
        text = _sanitize_mlx_text(raw_text)
        if not text and raw_lines:
            fallback_body: Optional[dict[str, Any]] = None
            try:
                fallback_body = json.loads("\n".join(raw_lines))
            except json.JSONDecodeError:
                fallback_body = None
            if isinstance(fallback_body, dict):
                raw_text = _flatten_chat_message_content(
                    (
                        ((fallback_body.get("choices") or [{}])[0]).get("message")
                        or {}
                    ).get("content")
                )
                text = _sanitize_mlx_text(
                    _flatten_chat_message_content(
                        (
                            ((fallback_body.get("choices") or [{}])[0]).get("message")
                            or {}
                        ).get("content")
                    )
                )
        if text:
            return text
        if _looks_like_reasoning_only(raw_text):
            logger.warning(
                "MLX streamed reasoning-only output for %s; treating as empty.",
                request_name,
            )
        logger.warning("MLX returned no streamed text for %s.", request_name)
        return None
