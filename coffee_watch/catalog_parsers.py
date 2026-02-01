from __future__ import annotations

import hashlib
import logging
import re
from typing import Callable
from urllib.parse import urljoin, urlsplit

from .models import ProductCandidate, RoasterSource
from .text_utils import LinkParser, guess_name_from_url, sanitize_html_to_text
from .url_utils import canonicalize_url

CatalogParser = Callable[
    [str, str, RoasterSource, int, logging.Logger], list[ProductCandidate]
]

_PRICE_RE = re.compile(r"(?:\\$|&#36;|&dollar;)\\s*\\d+(?:\\.\\d{2})?")
_BADGE_RE = re.compile(
    r"\b(New Harvest|New|Harvest|Limited|Sold Out|Out of Stock)\b", re.IGNORECASE
)
_WIX_URL_RE = re.compile(r"(/product-page/[^\"'<>\\s?#]+)")
_TITLE_HOOK_RE = re.compile(
    r'data-hook=["\']product-item-title["\'][^>]*>(.*?)<',
    re.IGNORECASE | re.DOTALL,
)
_BADGE_HOOK_RE = re.compile(
    r'data-hook=["\']product-item-badge["\'][^>]*>(.*?)<',
    re.IGNORECASE | re.DOTALL,
)


def _clean_text(html_fragment: str) -> str:
    return sanitize_html_to_text(html_fragment, 0, remove_boilerplate=False).strip()


def _extract_title(snippet: str, url: str) -> str:
    match = _TITLE_HOOK_RE.search(snippet)
    if match:
        title = _clean_text(match.group(1))
        if title:
            return title
    return guess_name_from_url(url)


def _extract_badge(snippet: str) -> str:
    match = _BADGE_HOOK_RE.search(snippet)
    if match:
        badge = _clean_text(match.group(1))
        if badge:
            return badge
    match = _BADGE_RE.search(snippet)
    return match.group(1).strip() if match else ""


def _extract_price(snippet: str) -> str:
    match = _PRICE_RE.search(snippet)
    if not match:
        return ""
    price = match.group(0).strip()
    return (
        price.replace("&#36;", "$")
        .replace("&dollar;", "$")
        .replace("\\$", "$")
    )


def parse_wix_shop_catalog(
    html: str,
    base_url: str,
    roaster: RoasterSource,
    max_count: int,
    logger: logging.Logger,
) -> list[ProductCandidate]:
    parser = LinkParser()
    parser.feed(html)
    hrefs = [href for href, _ in parser.links if "/product-page/" in href]
    urls: list[str] = []
    seen_raw: set[str] = set()
    for href in hrefs:
        if href in seen_raw:
            continue
        seen_raw.add(href)
        urls.append(canonicalize_url(urljoin(base_url, href)))

    if not urls:
        matches = list(_WIX_URL_RE.finditer(html))
        if not matches:
            logger.info("Wix catalog parser found no product URLs for %s", roaster.name)
            return []
        urls = [canonicalize_url(urljoin(base_url, match.group(1))) for match in matches]

    products: list[ProductCandidate] = []
    seen: set[str] = set()
    exclude_keywords = {kw.strip().lower() for kw in roaster.exclude_title_keywords if kw}
    for url in urls:
        if url in seen:
            continue
        seen.add(url)

        slug = urlsplit(url).path
        snippet_index = html.find(slug)
        if snippet_index == -1:
            snippet_index = html.find(url)
        if snippet_index == -1:
            snippet_index = html.find(url.replace("&", "&amp;"))
        snippet_start = max(0, snippet_index - 300) if snippet_index != -1 else 0
        snippet_end = min(
            len(html),
            snippet_index + 800 if snippet_index != -1 else min(len(html), 1200),
        )
        snippet = html[snippet_start:snippet_end]
        title = _extract_title(snippet, url)
        badge = _extract_badge(snippet)
        price = _extract_price(snippet)

        title_lower = title.lower()
        url_lower = url.lower()
        if exclude_keywords and any(
            keyword in title_lower
            or keyword in url_lower
            or keyword.replace(" ", "-") in url_lower
            for keyword in exclude_keywords
        ):
            continue

        products.append(
            ProductCandidate(
                product_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                name=title,
                url=url,
                source=roaster.name,
                list_price=price,
                list_badge=badge,
                body_html="",
                variants=(),
            )
        )
        if len(products) >= max_count:
            break
    return products


_CATALOG_PARSERS: dict[str, CatalogParser] = {
    "wix_shop": parse_wix_shop_catalog,
}


def parse_catalog_html(
    parser_name: str,
    html: str,
    base_url: str,
    roaster: RoasterSource,
    max_count: int,
    logger: logging.Logger,
) -> list[ProductCandidate]:
    parser = _CATALOG_PARSERS.get(parser_name)
    if not parser:
        logger.warning("Unknown catalog parser %s for %s", parser_name, roaster.name)
        return []
    return parser(html, base_url, roaster, max_count, logger)
