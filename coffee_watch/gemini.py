from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel


def model_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()  # type: ignore[attr-defined]
    return model_cls.schema()  # type: ignore[no-any-return]


def validate_model(model_cls: type[BaseModel], data: Any) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # type: ignore[attr-defined]
    return model_cls.parse_obj(data)  # type: ignore[call-arg]


def validate_model_json(model_cls: type[BaseModel], text: str) -> BaseModel:
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(text)  # type: ignore[attr-defined]
    return model_cls.parse_raw(text)  # type: ignore[call-arg]


async def generate_content_async(client: genai.Client, **kwargs: Any) -> Any:
    aio = getattr(client, "aio", None)
    method = getattr(getattr(aio, "models", None), "generate_content", None)
    if method and inspect.iscoroutinefunction(method):
        return await method(**kwargs)
    return await asyncio.to_thread(client.models.generate_content, **kwargs)


async def count_tokens_async(client: genai.Client, **kwargs: Any) -> Any:
    aio = getattr(client, "aio", None)
    method = getattr(getattr(aio, "models", None), "count_tokens", None)
    if method and inspect.iscoroutinefunction(method):
        return await method(**kwargs)
    return await asyncio.to_thread(client.models.count_tokens, **kwargs)


async def await_with_timeout(coro: Any, timeout_s: float) -> Any:
    if timeout_s and timeout_s > 0:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    return await coro


def get_value(obj: Any, *names: str) -> Any:
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def extract_total_tokens(count_response: Any) -> Optional[int]:
    if count_response is None:
        return None
    if isinstance(count_response, int):
        return count_response
    value = get_value(
        count_response,
        "total_tokens",
        "totalTokens",
        "total_token_count",
        "totalTokenCount",
    )
    return value if isinstance(value, int) else None


def extract_usage_metadata(response: Any) -> Optional[dict[str, int]]:
    if response is None:
        return None
    usage = get_value(response, "usage_metadata", "usageMetadata")
    if usage is None:
        return None
    prompt_tokens = get_value(usage, "prompt_token_count", "promptTokenCount")
    output_tokens = get_value(usage, "candidates_token_count", "candidatesTokenCount")
    total_tokens = get_value(usage, "total_token_count", "totalTokenCount")
    data = {
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": output_tokens,
        "total_token_count": total_tokens,
    }
    return {k: v for k, v in data.items() if isinstance(v, int)} or None


def extract_candidate(response: Any) -> Any:
    if response is None:
        return None
    if isinstance(response, dict):
        candidates = response.get("candidates") or []
        return candidates[0] if candidates else None
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    return candidates[0]


def extract_parts_text(parts: Any) -> list[str]:
    if not parts:
        return []
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            value = part.get("text")
        else:
            value = getattr(part, "text", None)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    return chunks


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        text = str(response.get("text") or "").strip()
    else:
        text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    candidate = extract_candidate(response)
    if candidate is None:
        return ""
    if isinstance(candidate, dict):
        if isinstance(candidate.get("text"), str):
            return candidate["text"].strip()
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        chunks = extract_parts_text(parts)
        return "\n".join(chunks).strip()
    if isinstance(getattr(candidate, "text", None), str):
        return getattr(candidate, "text", "").strip()
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content else None
    chunks = extract_parts_text(parts)
    return "\n".join(chunks).strip()


def extract_grounding_metadata(response: Any) -> Optional[dict[str, Any]]:
    try:
        candidate = extract_candidate(response)
        if candidate is None:
            return None
        metadata = getattr(candidate, "grounding_metadata", None) or getattr(
            candidate, "groundingMetadata", None
        )
        if metadata is None:
            if isinstance(candidate, dict):
                metadata = candidate.get("groundingMetadata") or candidate.get(
                    "grounding_metadata"
                )
        if metadata is None:
            return None

        raw_queries = get_value(metadata, "web_search_queries", "webSearchQueries") or []
        queries = [str(item) for item in raw_queries if str(item)]

        raw_chunks = get_value(metadata, "grounding_chunks", "groundingChunks") or []
        chunks: list[dict[str, Any]] = []
        for chunk in raw_chunks:
            web = get_value(chunk, "web")
            if web is None:
                continue
            uri = get_value(web, "uri")
            title = get_value(web, "title") or ""
            if uri or title:
                chunks.append({"web": {"uri": uri, "title": title}})

        raw_supports = get_value(metadata, "grounding_supports", "groundingSupports") or []
        supports: list[dict[str, Any]] = []
        for support in raw_supports:
            segment = get_value(support, "segment")
            segment_obj: dict[str, Any] = {}
            if segment is not None:
                start_index = get_value(segment, "start_index", "startIndex")
                end_index = get_value(segment, "end_index", "endIndex")
                text = get_value(segment, "text")
                if isinstance(start_index, int):
                    segment_obj["startIndex"] = start_index
                if isinstance(end_index, int):
                    segment_obj["endIndex"] = end_index
                if isinstance(text, str) and text:
                    segment_obj["text"] = text
            indices = get_value(
                support, "grounding_chunk_indices", "groundingChunkIndices"
            )
            chunk_indices = (
                [int(idx) for idx in indices if isinstance(idx, (int, float))]
                if isinstance(indices, list)
                else []
            )
            if segment_obj or chunk_indices:
                payload: dict[str, Any] = {}
                if segment_obj:
                    payload["segment"] = segment_obj
                if chunk_indices:
                    payload["groundingChunkIndices"] = chunk_indices
                supports.append(payload)

        result: dict[str, Any] = {}
        if queries:
            result["webSearchQueries"] = queries
        if chunks:
            result["groundingChunks"] = chunks
        if supports:
            result["groundingSupports"] = supports
        return result or None
    except Exception:
        return None


def format_grounding_metadata(grounding: dict[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for key in ("webSearchQueries", "groundingChunks", "groundingSupports"):
        value = grounding.get(key)
        if value:
            payload[key] = value
    if not payload:
        return ""
    return (
        "## Grounding metadata\n\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=True)
        + "\n```\n"
    )


async def evaluate_roaster_markdown(
    client: genai.Client,
    model: str,
    roaster_name: str,
    prompt: str,
    logger,
    timeout_s: float,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        response_mime_type="text/plain",
        temperature=0.2,
    )

    input_tokens: Optional[int] = None
    try:
        count_response = await await_with_timeout(
            count_tokens_async(
                client,
                model=model,
                contents=prompt,
            ),
            timeout_s,
        )
        input_tokens = extract_total_tokens(count_response)
        if input_tokens is not None:
            logger.info("Gemini input tokens for %s: %d", roaster_name, input_tokens)
    except asyncio.TimeoutError:
        logger.warning(
            "Gemini input token count timed out for %s after %.1fs",
            roaster_name,
            timeout_s,
        )
    except Exception as exc:
        logger.warning("Gemini input token count failed for %s: %s", roaster_name, exc)

    try:
        response = await await_with_timeout(
            generate_content_async(
                client,
                model=model,
                contents=prompt,
                config=config,
            ),
            timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Gemini request timed out for %s after %.1fs", roaster_name, timeout_s
        )
        return None, None
    except Exception as exc:
        logger.exception("Gemini request failed for %s: %s", roaster_name, exc)
        return None, None

    text = (getattr(response, "text", None) or "").strip()
    usage = extract_usage_metadata(response)
    if usage:
        logger.info(
            "Gemini usage for %s: prompt_tokens=%s output_tokens=%s total_tokens=%s",
            roaster_name,
            usage.get("prompt_token_count"),
            usage.get("candidates_token_count"),
            usage.get("total_token_count"),
        )
    elif text:
        try:
            count_response = await await_with_timeout(
                count_tokens_async(
                    client,
                    model=model,
                    contents=text,
                ),
                timeout_s,
            )
            output_tokens = extract_total_tokens(count_response)
            if output_tokens is not None:
                logger.info(
                    "Gemini output tokens for %s: %d", roaster_name, output_tokens
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Gemini output token count timed out for %s after %.1fs",
                roaster_name,
                timeout_s,
            )
        except Exception as exc:
            logger.warning(
                "Gemini output token count failed for %s: %s", roaster_name, exc
            )
    if text:
        logger.info("Gemini raw response for %s: %s", roaster_name, text)
        return text, extract_grounding_metadata(response)
    return None, extract_grounding_metadata(response)


async def generate_digest_markdown(
    client: genai.Client,
    model: str,
    prompt: str,
    logger,
    timeout_s: float,
) -> Optional[str]:
    config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0.2,
    )
    try:
        response = await await_with_timeout(
            generate_content_async(
                client,
                model=model,
                contents=prompt,
                config=config,
            ),
            timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("Gemini digest request timed out after %.1fs", timeout_s)
        return None
    except Exception as exc:
        logger.exception("Gemini digest request failed: %s", exc)
        return None

    text = extract_response_text(response).strip()
    usage = extract_usage_metadata(response)
    if usage:
        logger.info(
            "Gemini digest usage: prompt_tokens=%s output_tokens=%s total_tokens=%s",
            usage.get("prompt_token_count"),
            usage.get("candidates_token_count"),
            usage.get("total_token_count"),
        )
    if text:
        logger.info("Gemini digest raw response: %s", text)
        return text
    return None
