from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlsplit

from .models import VariantInfo


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._current_href = href
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href is not None:
            text = " ".join(part.strip() for part in self._text_parts if part.strip())
            self.links.append((self._current_href, text.strip()))
            self._current_href = None
            self._text_parts = []


class VisibleTextExtractor(HTMLParser):
    _skip_tags = {"script", "style", "head", "noscript", "svg", "canvas"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        elif tag in {"br", "p", "li", "div", "section"} and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = " ".join(data.split())
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        parts: list[str] = []
        for chunk in self._chunks:
            if chunk == "\n":
                parts.append("\n")
            else:
                parts.append(f"{chunk} ")
        return "".join(parts)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?[\d][\d\s().-]{7,}\d)")
JSONLD_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
BOILERPLATE_PHRASES = {
    "cookie",
    "privacy policy",
    "terms of service",
    "newsletter",
    "subscribe",
    "sign up",
    "log in",
    "login",
    "add to cart",
    "cart",
    "checkout",
    "shipping",
    "search",
    "filter",
}


def sanitize_html_to_text(
    html: str, max_chars: int, remove_boilerplate: bool = True
) -> str:
    extractor = VisibleTextExtractor()
    extractor.feed(html)
    raw = extractor.text()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    filtered: list[str] = []
    for line in lines:
        if remove_boilerplate:
            lowered = line.lower()
            if any(phrase in lowered for phrase in BOILERPLATE_PHRASES):
                continue
        filtered.append(line)
    text = " ".join(filtered)
    text = " ".join(text.split())
    text = "".join(ch for ch in text if ch.isprintable())
    text = EMAIL_RE.sub("[redacted email]", text)

    def redact_phone(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 10:
            return "[redacted phone]"
        return match.group(0)

    text = PHONE_RE.sub(redact_phone, text)
    if max_chars <= 0:
        return text
    return text[:max_chars]


def sanitize_prompt_field(value: str, max_chars: int) -> str:
    cleaned = " ".join(value.split())
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    if max_chars <= 0:
        return cleaned
    return cleaned[:max_chars]


def trim_text_at_phrases(text: str, phrases: tuple[str, ...]) -> str:
    if not text or not phrases:
        return text
    lowered = text.lower()
    cut_index = None
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        idx = lowered.find(phrase.lower())
        if idx == -1:
            continue
        cut_index = idx if cut_index is None else min(cut_index, idx)
    if cut_index is None:
        return text
    return text[:cut_index].rstrip()


def _iter_jsonld_objects(html: str) -> list[dict]:
    objects: list[dict] = []
    decoder = json.JSONDecoder()
    for match in JSONLD_SCRIPT_RE.finditer(html):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                data, _ = decoder.raw_decode(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            objects.append(data)
        elif isinstance(data, list):
            objects.extend([item for item in data if isinstance(item, dict)])
    return objects


def _normalize_jsonld_type(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, list):
        return {str(item).lower() for item in value if str(item)}
    return set()


def _find_product_in_jsonld(data: dict) -> Optional[dict]:
    types = _normalize_jsonld_type(data.get("@type"))
    if "product" in types or "productgroup" in types:
        return data
    main_entity = data.get("mainEntity") or data.get("mainEntityOfPage")
    if isinstance(main_entity, dict):
        product = _find_product_in_jsonld(main_entity)
        if product:
            return product
    graph = data.get("@graph")
    if isinstance(graph, list):
        for item in graph:
            if isinstance(item, dict):
                product = _find_product_in_jsonld(item)
                if product:
                    return product
    return None


def _extract_offer_details(offers: object) -> dict[str, str]:
    if isinstance(offers, list):
        for offer in offers:
            details = _extract_offer_details(offer)
            if details:
                return details
        return {}
    if not isinstance(offers, dict):
        return {}
    price = str(offers.get("price") or "").strip()
    currency = str(offers.get("priceCurrency") or "").strip()
    availability = str(offers.get("availability") or "").strip()
    return {
        "price": price,
        "currency": currency,
        "availability": availability,
    }


def extract_product_jsonld_text(html: str, max_chars: int) -> str:
    for obj in _iter_jsonld_objects(html):
        product = _find_product_in_jsonld(obj)
        if not product:
            continue
        parts: list[str] = []
        name = str(product.get("name") or "").strip()
        if name:
            parts.append(f"Name: {name}")
        description = str(product.get("description") or "").strip()
        if description:
            parts.append(f"Description: {description}")
        brand = product.get("brand")
        if isinstance(brand, dict):
            brand = brand.get("name")
        if brand:
            parts.append(f"Brand: {brand}")
        sku = str(product.get("sku") or "").strip()
        if sku:
            parts.append(f"SKU: {sku}")
        category = product.get("category")
        if category:
            parts.append(f"Category: {category}")
        offers = product.get("offers")
        offer_details = _extract_offer_details(offers)
        price = offer_details.get("price", "")
        currency = offer_details.get("currency", "")
        availability = offer_details.get("availability", "")
        if price:
            parts.append(f"Price: {price}{(' ' + currency) if currency else ''}")
        if availability:
            parts.append(f"Availability: {availability}")
        text = " | ".join(parts).strip()
        if not text:
            continue
        return text if max_chars <= 0 else text[:max_chars]
    return ""


def format_variant_lines(variants: tuple[VariantInfo, ...]) -> list[str]:
    if not variants:
        return []
    lines = ["  variants:"]
    for variant in variants:
        parts: list[str] = []
        if variant.title:
            parts.append(variant.title)
        if variant.price:
            parts.append(f"${variant.price}")
        if variant.grams:
            parts.append(f"{variant.grams}g")
        parts.append("available" if variant.available else "unavailable")
        lines.append(f"    - {' | '.join(parts)}")
    return lines


def guess_name_from_url(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    if not path:
        return "Unknown Coffee"
    slug = path.split("/")[-1]
    return slug.replace("-", " ").replace("_", " ").strip().title() or "Unknown Coffee"
