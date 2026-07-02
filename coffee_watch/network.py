from __future__ import annotations

import asyncio
import json
import logging
import random
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .config import Settings
from .constants import USER_AGENT
from .http_limits import Host429Gate, PerHostLimiter
from .models import PaginationConfig, ProductCandidate, ProductFieldConfig, RoasterSource
from .parsing import (
    parse_products_html_path,
    parse_products_json,
    parse_products_response,
)
from .reporting import log_products_json_snippet, save_products_json, save_products_json_pretty
from .text_utils import (
    extract_product_jsonld_text,
    extract_product_page_price,
    extract_product_page_size_labels,
    extract_visible_text,
    finalize_visible_text,
)
from .url_utils import (
    build_url_with_params,
    canonicalize_url,
    matches_patterns,
    url_is_denylisted,
)

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
# Cap server-provided Retry-After delays so a hostile or misconfigured server
# cannot stall the run with an arbitrarily large value.
RETRY_AFTER_MAX_S = 60.0


def _refuse_reason(url: str, denylist: set[str] | None) -> str | None:
    """Why ``url`` must not be fetched, or ``None`` if it is allowed.

    Product/page URLs come from remote catalog data and redirects, so every
    fetch — not just the per-roaster entry point — enforces scheme and
    denylist restrictions.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return f"non-HTTP(S) scheme {parsed.scheme!r}"
    if not parsed.netloc:
        return "missing host"
    if url_is_denylisted(url, denylist):
        return "denylisted host"
    return None


async def jitter_sleep(min_s: float, max_s: float) -> None:
    if max_s <= 0:
        return
    low = min(min_s, max_s)
    high = max(min_s, max_s)
    await asyncio.sleep(random.uniform(low, high))


def _parse_retry_after_seconds(value: str) -> float | None:
    """Parse a Retry-After header: either delta-seconds or an HTTP-date."""
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(timezone.utc)).total_seconds()


def _retry_delay(
    attempt: int,
    response: httpx.Response | None,
    settings: Settings,
    jitter_multiplier: float,
) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            seconds = _parse_retry_after_seconds(retry_after)
            if seconds is not None:
                return min(
                    max(seconds, settings.jitter_min_s * jitter_multiplier),
                    RETRY_AFTER_MAX_S,
                )
    base = max(0.5, settings.jitter_min_s * jitter_multiplier)
    cap = max(base, settings.jitter_max_s * jitter_multiplier * 3)
    return min(cap, base * (2**attempt))


async def fetch_text_with_jitter(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
    log: logging.Logger,
    headers: dict[str, str] | None = None,
    limiter: PerHostLimiter | None = None,
    jitter_multiplier: float = 1.0,
    denylist: set[str] | None = None,
    gate: Host429Gate | None = None,
) -> httpx.Response | None:
    refusal = _refuse_reason(url, denylist)
    if refusal:
        log.warning("Refusing to fetch %s: %s", url, refusal)
        return None
    if gate is not None and gate.is_gated(url):
        log.info("Skipping fetch of %s: host bot-gated by consecutive 429s.", url)
        return None

    retry_statuses = {429, 500, 502, 503, 504}
    max_retries = max(0, settings.http_max_retries)
    max_bytes = settings.http_max_response_bytes
    # Some CDNs intermittently serve corrupted compressed bodies; after a
    # decoding failure, retries ask for an uncompressed transfer instead.
    identity_fallback = False
    # Size-cap and redirect refusals are deliberate policy drops, not
    # transient failures; retrying them would just repeat the download.
    dropped_by_policy = False
    # Set when the host's 429 gate trips while this fetch is queued.
    gated_skip = False

    async def _get_capped() -> httpx.Response | None:
        """Stream the response, enforcing the body-size cap.

        The body is buffered incrementally so a huge (or maliciously
        unbounded) response is abandoned at ``max_bytes`` instead of being
        read fully into memory.
        """
        nonlocal dropped_by_policy
        request_headers = headers
        if identity_fallback:
            request_headers = dict(headers or {})
            request_headers["Accept-Encoding"] = "identity"
        async with client.stream("GET", url, headers=request_headers) as response:
            final_refusal = _refuse_reason(str(response.url), denylist)
            if final_refusal:
                log.warning(
                    "Dropping response for %s: redirected to %s (%s)",
                    url,
                    response.url,
                    final_refusal,
                )
                dropped_by_policy = True
                return None
            content_length = response.headers.get("content-length")
            if max_bytes and content_length:
                try:
                    if int(content_length) > max_bytes:
                        log.warning(
                            "Dropping response for %s: content-length %s exceeds %d bytes",
                            url,
                            content_length,
                            max_bytes,
                        )
                        dropped_by_policy = True
                        return None
                except ValueError:
                    pass
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if max_bytes and len(body) > max_bytes:
                    log.warning(
                        "Dropping response for %s: body exceeded %d bytes",
                        url,
                        max_bytes,
                    )
                    dropped_by_policy = True
                    return None
            # aiter_bytes() has already decoded any content-encoding, so the
            # original encoding/length headers no longer describe ``body``.
            # Keeping them would make Response() decode the body a second
            # time and fail on every legitimately compressed response.
            clean_headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.lower() not in {"content-encoding", "content-length"}
            ]
            return httpx.Response(
                status_code=response.status_code,
                headers=clean_headers,
                content=bytes(body),
                request=response.request,
            )

    async def _run() -> httpx.Response | None:
        nonlocal identity_fallback
        log.info("HTTP GET %s", url)
        try:
            # httpx's read timeout is per-chunk, so a tarpit trickling bytes
            # could otherwise hold a slot indefinitely; wait_for bounds the
            # whole request.
            if settings.http_total_timeout_s > 0:
                response = await asyncio.wait_for(
                    _get_capped(), timeout=settings.http_total_timeout_s
                )
            else:
                response = await _get_capped()
        except asyncio.TimeoutError:
            log.warning(
                "HTTP request for %s exceeded total deadline of %.1fs",
                url,
                settings.http_total_timeout_s,
            )
            return None
        except httpx.DecodingError as exc:
            log.warning(
                "Decoding failed for %s (%s); retrying with uncompressed transfer.",
                url,
                exc,
            )
            identity_fallback = True
            return None
        except httpx.HTTPError as exc:
            log.warning("HTTP request failed for %s: %s", url, exc)
            return None
        if response is not None:
            log.info("HTTP %s %s", response.status_code, url)
        return response

    async def _attempt() -> httpx.Response | None:
        nonlocal gated_skip
        jitter_min = settings.jitter_min_s * jitter_multiplier
        jitter_max = settings.jitter_max_s * jitter_multiplier
        if limiter is None:
            if gate is not None and gate.is_gated(url):
                gated_skip = True
                return None
            await jitter_sleep(jitter_min, jitter_max)
            return await _run()
        # Sleep while holding only the host slot: politeness spacing applies
        # per host, and a sleeping task must not burn a global slot.
        async with limiter.host_slot(url):
            # Consult the 429 gate only after acquiring the host slot: with
            # per-host serialization this is the first point where earlier
            # requests' 429s are guaranteed to have been recorded.
            if gate is not None and gate.is_gated(url):
                gated_skip = True
                return None
            await jitter_sleep(jitter_min, jitter_max)
            async with limiter.global_slot():
                return await _run()

    response: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        dropped_by_policy = False
        response = await _attempt()
        if gated_skip:
            log.info(
                "Skipping fetch of %s: host bot-gated by consecutive 429s.", url
            )
            return None
        if dropped_by_policy:
            return None
        if gate is not None and response is not None:
            gate.record(url, response.status_code)
        if response is not None and response.status_code < 400:
            return response
        status = response.status_code if response is not None else None
        should_retry = response is None or status in retry_statuses
        if not should_retry or attempt == max_retries:
            return response
        delay = _retry_delay(attempt, response, settings, jitter_multiplier)
        sleep_for = random.uniform(delay / 2, delay) if delay > 0 else 0.0
        log.warning(
            "Retrying %s in %.2fs after status %s",
            url,
            sleep_for,
            status,
        )
        await asyncio.sleep(sleep_for)
    return response


async def robots_allows(
    client: httpx.AsyncClient,
    products_url: str,
    settings: Settings,
    cache: dict[str, RobotFileParser],
    log: logging.Logger,
    limiter: PerHostLimiter | None = None,
    jitter_multiplier: float = 1.0,
    denylist: set[str] | None = None,
) -> bool:
    refusal = _refuse_reason(products_url, denylist)
    if refusal:
        log.warning("Disallowing fetch of %s: %s", products_url, refusal)
        return False
    parsed = urlsplit(products_url)
    cache_key = f"{parsed.scheme}://{parsed.netloc}"
    if cache_key in cache:
        return cache[cache_key].can_fetch(USER_AGENT, products_url)

    robots_url = urljoin(cache_key, "/robots.txt")
    response = await fetch_text_with_jitter(
        client,
        robots_url,
        settings,
        log,
        limiter=limiter,
        jitter_multiplier=jitter_multiplier,
        denylist=denylist,
    )
    parser = RobotFileParser()
    if response is None or response.status_code >= 500:
        # RFC 9309: an unreachable robots.txt (network error or 5xx) means
        # the crawler must assume complete disallow, not crawl unchecked.
        log.warning(
            "Robots.txt unreachable for %s (%s); treating as disallow-all for this run.",
            cache_key,
            "no response" if response is None else response.status_code,
        )
        parser.parse(["User-agent: *", "Disallow: /"])
        cache[cache_key] = parser
        return False
    if response.status_code >= 400:
        log.info("Robots.txt not found for %s; proceeding with allowed default.", cache_key)
        parser.parse([])
        cache[cache_key] = parser
        return parser.can_fetch(USER_AGENT, products_url)
    parser.parse(response.text.splitlines())
    cache[cache_key] = parser
    return parser.can_fetch(USER_AGENT, products_url)


async def fetch_products_for_roaster(
    http_client: httpx.AsyncClient,
    roaster: RoasterSource,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    assets_dir: Path,
    run_id: str,
    limiter: PerHostLimiter | None,
    log: logging.Logger,
    denylist: set[str] | None = None,
) -> list[ProductCandidate]:
    max_products = roaster.max_products or settings.max_products_per_source
    pagination = roaster.pagination or PaginationConfig(max_pages=1)
    stop_on_empty = pagination.stop_on_empty
    products: list[ProductCandidate] = []
    seen_urls: set[str] = set()
    headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.products_headers,
        log,
        f"{roaster.name} products",
    )

    for page_index in range(pagination.start, pagination.start + pagination.max_pages):
        params = dict(roaster.products_params)
        if pagination.max_pages > 1 or "page" in params:
            params[pagination.page_param] = str(page_index)
        if pagination.page_size_param and pagination.page_size:
            params[pagination.page_size_param] = str(pagination.page_size)

        products_url = build_url_with_params(roaster.base_url, roaster.products_path, params)
        allowed = await robots_allows(
            http_client,
            products_url,
            settings,
            robots_cache,
            log,
            limiter=limiter,
            jitter_multiplier=roaster.jitter_multiplier,
            denylist=denylist,
        )
        if not allowed:
            log.warning(
                "Robots.txt disallows %s for %s; skipping.",
                products_url,
                roaster.name,
            )
            if stop_on_empty:
                break
            continue

        response = await fetch_text_with_jitter(
            http_client,
            products_url,
            settings,
            log,
            headers=headers,
            limiter=limiter,
            jitter_multiplier=roaster.jitter_multiplier,
            denylist=denylist,
        )
        if response is None:
            log.warning("Request failed for %s", products_url)
            if stop_on_empty:
                break
            continue
        if response.status_code >= 400:
            log.warning(
                "Non-200 response %s for %s", response.status_code, products_url
            )
            if stop_on_empty:
                break
            continue

        content_type = response.headers.get("content-type", "")
        remaining = max_products - len(products)
        page_products: list[ProductCandidate] = []
        json_branch_taken = (
            roaster.products_type == "json" or "json" in content_type.lower()
        )
        if json_branch_taken:
            json_text = response.text
            raw_path = save_products_json(
                assets_dir, run_id, roaster, page_index, json_text
            )
            log.info("Saved raw products JSON for %s to %s", roaster.name, raw_path)
            if settings.save_raw_products_json:
                report_raw_path = save_products_json(
                    settings.reports_dir, run_id, roaster, page_index, json_text
                )
                log.info(
                    "Saved raw products JSON copy for %s to %s",
                    roaster.name,
                    report_raw_path,
                )
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as exc:
                log.warning("Failed to parse JSON for %s: %s", roaster.name, exc)
                log_products_json_snippet(
                    log,
                    roaster,
                    products_url,
                    json_text,
                    settings.log_json_max_chars,
                )
            else:
                pretty_path = save_products_json_pretty(
                    assets_dir, run_id, roaster, page_index, data
                )
                if pretty_path:
                    log.info(
                        "Saved pretty products JSON for %s to %s",
                        roaster.name,
                        pretty_path,
                    )
                    if settings.save_pretty_products_json:
                        report_pretty_path = save_products_json_pretty(
                            settings.reports_dir, run_id, roaster, page_index, data
                        )
                        if report_pretty_path:
                            log.info(
                                "Saved pretty products JSON copy for %s to %s",
                                roaster.name,
                                report_pretty_path,
                            )
                        else:
                            log.warning(
                                "Failed to serialize pretty products JSON copy for %s",
                                roaster.name,
                            )
                else:
                    log.warning(
                        "Failed to serialize pretty products JSON for %s",
                        roaster.name,
                    )
                product_fields = roaster.product_fields or ProductFieldConfig()
                page_products = parse_products_json(
                    data,
                    roaster.base_url,
                    roaster,
                    remaining,
                    product_fields,
                    roaster.product_url_template,
                    roaster.json_items_path,
                )
        if not page_products:
            if json_branch_taken:
                # JSON was already attempted above; go straight to HTML to
                # avoid a redundant second json.loads in parse_products_response.
                page_products = parse_products_html_path(
                    response.content,
                    roaster.base_url,
                    roaster,
                    remaining,
                    log,
                )
            else:
                page_products = parse_products_response(
                    response.content,
                    content_type,
                    roaster.base_url,
                    roaster,
                    remaining,
                    log,
                )
        # Drop products already collected from an earlier page: paginated
        # storefronts repeat featured items, and a page param past the end
        # often re-serves the first page. An all-duplicate page therefore
        # also counts as empty for stop_on_empty purposes.
        page_products = [
            product for product in page_products if product.url not in seen_urls
        ]
        seen_urls.update(product.url for product in page_products)
        if not page_products and stop_on_empty:
            break
        products.extend(page_products)
        if len(products) >= max_products:
            break

    return products


async def fetch_product_page_text(
    http_client: httpx.AsyncClient,
    product: ProductCandidate,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    log: logging.Logger,
    headers: dict[str, str],
    limiter: PerHostLimiter | None,
    jitter_multiplier: float = 1.0,
    denylist: set[str] | None = None,
    gate: Host429Gate | None = None,
) -> tuple[str, str, str, tuple[str, ...]]:
    product_allowed = await robots_allows(
        http_client,
        product.url,
        settings,
        robots_cache,
        log,
        limiter=limiter,
        jitter_multiplier=jitter_multiplier,
        denylist=denylist,
    )
    if not product_allowed:
        log.warning(
            "Robots.txt disallows product page %s; skipping page fetch.",
            product.url,
        )
        return "", "", "", ()

    page_response = await fetch_text_with_jitter(
        http_client,
        product.url,
        settings,
        log,
        headers=headers,
        limiter=limiter,
        jitter_multiplier=jitter_multiplier,
        denylist=denylist,
        gate=gate,
    )
    if page_response is None:
        log.warning("Request failed for product page %s", product.url)
        return "", "", "", ()
    if page_response.status_code >= 400:
        log.warning(
            "Non-200 response %s for product page %s",
            page_response.status_code,
            product.url,
        )
        return "", "", "", ()
    html = page_response.text
    # Parse the HTML once; the description, price, and size extractors all
    # work from the same visible-text pass.
    raw_visible = extract_visible_text(html)
    plain_text = finalize_visible_text(raw_visible, 0, remove_boilerplate=False)
    page_text = extract_product_jsonld_text(
        html, settings.page_text_max_chars, page_url=product.url
    )
    if not page_text:
        page_text = finalize_visible_text(raw_visible, settings.page_text_max_chars)
    log.info(
        "Sanitized %s chars of page text for %s", len(page_text), product.url
    )
    http_last_modified = page_response.headers.get("last-modified", "")
    return (
        page_text,
        http_last_modified,
        extract_product_page_price(html, plain_text=plain_text),
        extract_product_page_size_labels(html, plain_text=plain_text),
    )


def _strip_xml_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_xml_text(node: ET.Element | None, tag: str) -> str:
    if node is None:
        return ""
    child = node.find(f".//{{*}}{tag}")
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _normalize_sitemap_loc(loc: str) -> list[str]:
    loc = loc.strip()
    if not loc:
        return []
    canonical = canonicalize_url(loc)
    if not canonical:
        return []
    urls = [canonical]
    parts = urlsplit(canonical)
    if parts.path.endswith("/") and parts.path != "/":
        trimmed = canonical.rstrip("/")
        if trimmed not in urls:
            urls.append(trimmed)
    return urls


def _parse_sitemap_xml(xml_text: str) -> tuple[dict[str, str], list[str]]:
    # Legitimate sitemaps never declare DTDs or entities; refusing them
    # blocks entity-expansion tricks against the stdlib XML parser.
    if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        return {}, []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}, []

    tag = _strip_xml_namespace(root.tag).lower()
    if tag == "urlset":
        lastmods: dict[str, str] = {}
        for url_node in root.findall(".//{*}url"):
            loc = _find_xml_text(url_node, "loc")
            lastmod = _find_xml_text(url_node, "lastmod")
            if not loc or not lastmod:
                continue
            for normalized in _normalize_sitemap_loc(loc):
                lastmods[normalized] = lastmod
        return lastmods, []

    if tag == "sitemapindex":
        sitemaps: list[str] = []
        for sitemap_node in root.findall(".//{*}sitemap"):
            loc = _find_xml_text(sitemap_node, "loc")
            if loc:
                sitemaps.append(loc)
        return {}, sitemaps

    return {}, []


async def fetch_wix_product_sitemap_lastmods(
    http_client: httpx.AsyncClient,
    roaster: RoasterSource,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    log: logging.Logger,
    headers: dict[str, str],
    limiter: PerHostLimiter | None,
    jitter_multiplier: float = 1.0,
    denylist: set[str] | None = None,
) -> dict[str, str]:
    base = roaster.base_url if roaster.base_url.endswith("/") else f"{roaster.base_url}/"
    store_products_sitemap = urljoin(base, "store-products-sitemap.xml")
    index_sitemap = urljoin(base, "sitemap.xml")

    def _filter_lastmods(lastmods: dict[str, str]) -> dict[str, str]:
        if not lastmods:
            return {}
        include = roaster.product_link_patterns
        exclude = roaster.product_link_exclude_patterns
        if not include and not exclude:
            return lastmods
        return {
            url: lastmod
            for url, lastmod in lastmods.items()
            if matches_patterns(url, include, exclude)
        }

    async def _fetch(url: str) -> tuple[dict[str, str], list[str]]:
        allowed = await robots_allows(
            http_client,
            url,
            settings,
            robots_cache,
            log,
            limiter=limiter,
            jitter_multiplier=jitter_multiplier,
            denylist=denylist,
        )
        if not allowed:
            log.warning("Robots.txt disallows sitemap %s; skipping.", url)
            return {}, []
        response = await fetch_text_with_jitter(
            http_client,
            url,
            settings,
            log,
            headers=headers,
            limiter=limiter,
            jitter_multiplier=jitter_multiplier,
            denylist=denylist,
        )
        if response is None:
            log.warning("Request failed for sitemap %s", url)
            return {}, []
        if response.status_code >= 400:
            log.info("Sitemap not available (%s): %s", response.status_code, url)
            return {}, []
        return _parse_sitemap_xml(response.text)

    async def _collect(url: str) -> dict[str, str]:
        queue: deque[str] = deque([url])
        seen: set[str] = set()
        combined: dict[str, str] = {}
        cap = max(1, settings.sitemap_max_pages)
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            lastmods, children = await _fetch(current)
            combined.update(_filter_lastmods(lastmods))
            for child in children:
                if "store-products-sitemap" in child.lower():
                    queue.append(child)
            if len(seen) >= cap:
                break
        return combined

    lastmods = await _collect(store_products_sitemap)
    if lastmods:
        log.info(
            "Loaded %d Wix sitemap lastmod entries from %s",
            len(lastmods),
            store_products_sitemap,
        )
        return lastmods

    lastmods = await _collect(index_sitemap)
    if lastmods:
        log.info(
            "Loaded %d Wix sitemap lastmod entries from %s",
            len(lastmods),
            index_sitemap,
        )
    return lastmods


def merge_headers(
    base_headers: dict[str, str],
    extra_headers: dict[str, str],
    log: logging.Logger,
    context: str,
) -> dict[str, str]:
    headers = dict(base_headers)
    for key, value in extra_headers.items():
        if key.lower() == "user-agent":
            log.warning("Ignoring custom User-Agent for %s", context)
            continue
        headers[key] = value
    return headers
