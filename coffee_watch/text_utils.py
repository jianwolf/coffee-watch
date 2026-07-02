from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import urlsplit


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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
    _skip_tags: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "head", "noscript", "svg", "canvas"}
    )

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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


class ButtonTextParser(HTMLParser):
    _void_tags: ClassVar[frozenset[str]] = frozenset(
        {"br", "hr", "img", "input", "meta", "link", "source"}
    )

    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[str] = []
        self._button_depth = 0
        self._button_disabled = False
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "button":
            attrs_dict = dict(attrs)
            style = str(attrs_dict.get("style", "")).lower().replace(" ", "")
            class_tokens = str(attrs_dict.get("class", "")).lower().split()
            self._button_depth += 1
            if self._button_depth == 1:
                self._button_disabled = (
                    "disabled" in attrs_dict
                    or "hidden" in attrs_dict
                    or str(attrs_dict.get("aria-disabled", "")).lower() == "true"
                    or str(attrs_dict.get("aria-hidden", "")).lower() == "true"
                    or "hidden" in class_tokens
                    or "display:none" in style
                    or "visibility:hidden" in style
                )
                self._text_parts = []
        elif self._button_depth > 0 and tag not in self._void_tags:
            self._button_depth += 1

    def handle_endtag(self, tag: str) -> None:
        # handle_starttag never counts void tags, but HTMLParser replays a
        # self-closing form like <br/> as start+end; counting the end tag
        # alone would desync the depth and drop the whole button.
        if tag in self._void_tags:
            return
        if self._button_depth <= 0:
            return
        self._button_depth -= 1
        if tag == "button" and self._button_depth == 0:
            text = " ".join(part.strip() for part in self._text_parts if part.strip())
            if text and not self._button_disabled:
                self.buttons.append(" ".join(text.split()))
            self._button_disabled = False
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._button_depth > 0:
            self._text_parts.append(data)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?[\d][\d\s().-]{7,}\d)")
SIZE_BUTTON_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:g|gram|grams|kg|oz|lb|lbs|pound|pounds)$",
    re.IGNORECASE,
)
SIZE_VALUE_RE = re.compile(
    r"(?<![$A-Za-z0-9])(\d+(?:\.\d+)?)\s*(kg|g|gram|grams|oz|lb|lbs|pound|pounds)\b",
    re.IGNORECASE,
)
PRODUCT_PAGE_PRICE_RE = re.compile(
    r"(?:\$|\\\$|&#36;|&dollar;)\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?",
    re.IGNORECASE,
)
# A price whose nearby preceding text matches these markers is a shipping or
# promo banner ("Free shipping on orders over $50"), not the product price.
PRICE_CONTEXT_BLOCKLIST = (
    "shipping",
    "orders over",
    "order over",
    "orders of",
    "delivery over",
    "spend",
)
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


def extract_visible_text(html: str) -> str:
    """Run the HTML parse once; finish with :func:`finalize_visible_text`.

    Callers that need several text variants of the same page (description,
    price, size labels) should extract once and finalize per variant instead
    of re-parsing the HTML each time.
    """
    extractor = VisibleTextExtractor()
    extractor.feed(html)
    return extractor.text()


def finalize_visible_text(
    raw: str, max_chars: int, remove_boilerplate: bool = True
) -> str:
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
    if not text.isprintable():
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


def sanitize_html_to_text(
    html: str, max_chars: int, remove_boilerplate: bool = True
) -> str:
    return finalize_visible_text(
        extract_visible_text(html), max_chars, remove_boilerplate
    )


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


def extract_size_button_labels(html: str) -> tuple[str, ...]:
    parser = ButtonTextParser()
    parser.feed(html)
    labels: list[str] = []
    seen: set[str] = set()
    for button in parser.buttons:
        if not SIZE_BUTTON_RE.match(button):
            continue
        key = button.lower().replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        labels.append(button)
    return tuple(labels)


def _find_json_array_end(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for idx in range(open_index, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_bag_size_option_labels(html: str) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'"title"\s*:\s*"[^"]*Bag\s+Size[^"]*"', html, re.IGNORECASE):
        selections_idx = html.find('"selections"', match.end())
        if selections_idx == -1:
            continue
        array_start = html.find("[", selections_idx)
        if array_start == -1:
            continue
        array_end = _find_json_array_end(html, array_start)
        if array_end == -1:
            continue
        selections_json = html[array_start : array_end + 1]
        for value_match in re.finditer(r'"value"\s*:\s*"((?:\\.|[^"\\])*)"', selections_json):
            try:
                label = json.loads(f'"{value_match.group(1)}"')
            except json.JSONDecodeError:
                label = value_match.group(1)
            label = " ".join(label.split())
            if not SIZE_VALUE_RE.search(label):
                continue
            key = label.lower().replace(" ", "")
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
    return tuple(labels)


def _clean_price(value: str) -> str:
    return (
        value.strip()
        .replace("&#36;", "$")
        .replace("&dollar;", "$")
        .replace("\\$", "$")
        # Thousands separators would truncate downstream numeric parsing.
        .replace(",", "")
    )


def extract_product_page_price(html: str, plain_text: str | None = None) -> str:
    """Find the first plausible product price in the page's visible text.

    ``plain_text`` lets callers that already hold the sanitized page text
    (``sanitize_html_to_text(html, 0, remove_boilerplate=False)``) skip a
    redundant HTML parse.

    Announcement bars render before the product ("Free shipping on orders
    over $50"), so the first dollar amount on a page is not automatically its
    price; amounts preceded by banner-style wording are skipped.
    """
    text = (
        plain_text
        if plain_text is not None
        else sanitize_html_to_text(html, 0, remove_boilerplate=False)
    )
    for match in PRODUCT_PAGE_PRICE_RE.finditer(text):
        # Banner wording sits immediately before its amount ("orders over
        # $50"); a tight window avoids also blocking the real price that
        # follows a short product title.
        context = text[max(0, match.start() - 20) : match.start()].lower()
        if any(marker in context for marker in PRICE_CONTEXT_BLOCKLIST):
            continue
        return _clean_price(match.group(0))
    return ""


def extract_product_page_size_labels(
    html: str, plain_text: str | None = None
) -> tuple[str, ...]:
    button_labels = extract_size_button_labels(html)
    if button_labels:
        return button_labels

    option_labels = _extract_bag_size_option_labels(html)
    if option_labels:
        return option_labels

    text = (
        plain_text
        if plain_text is not None
        else sanitize_html_to_text(html, 0, remove_boilerplate=False)
    )
    labels: list[str] = []
    seen: set[str] = set()
    for label_match in re.finditer(r"\b(?:Bag\s+Size|Size)\*?\b", text, re.IGNORECASE):
        window = text[label_match.end() : label_match.end() + 100]
        for size_match in SIZE_VALUE_RE.finditer(window):
            label = " ".join(size_match.group(0).split())
            key = label.lower().replace(" ", "")
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
            break
    return tuple(labels)


def grams_from_size_label(value: str) -> int:
    match = SIZE_VALUE_RE.search(value)
    if not match:
        return 0
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "kg":
        grams = amount * 1000
    elif unit in {"lb", "lbs", "pound", "pounds"}:
        grams = amount * 453.59237
    elif unit == "oz":
        grams = amount * 28.349523125
    else:
        grams = amount
    return round(grams)


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


def _find_product_in_jsonld(data: dict) -> dict | None:
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


def _format_jsonld_product(product: dict, max_chars: int) -> str:
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
        return ""
    return text if max_chars <= 0 else text[:max_chars]


def _normalize_url_for_compare(url: str) -> str:
    if not url:
        return ""
    cleaned = url.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.split("#", 1)[0]
    cleaned = cleaned.split("?", 1)[0]
    return cleaned.rstrip("/").lower()


def extract_product_jsonld_text(
    html: str, max_chars: int, page_url: str = ""
) -> str:
    """Extract product description text from JSON-LD ``Product`` blocks.

    When ``page_url`` is provided, prefer the JSON-LD product whose ``url`` /
    ``@id`` matches the page URL (after stripping query/fragment). This avoids
    surfacing unrelated upsell products from breadcrumbs or related-items
    JSON-LD blocks. Falls back to the first valid product otherwise.
    """
    candidates: list[dict] = []
    for obj in _iter_jsonld_objects(html):
        product = _find_product_in_jsonld(obj)
        if product:
            candidates.append(product)
    if not candidates:
        return ""

    target = _normalize_url_for_compare(page_url)
    if target:
        for product in candidates:
            for key in ("url", "@id"):
                if _normalize_url_for_compare(str(product.get(key) or "")) == target:
                    text = _format_jsonld_product(product, max_chars)
                    if text:
                        return text

    for product in candidates:
        text = _format_jsonld_product(product, max_chars)
        if text:
            return text
    return ""


def guess_name_from_url(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    if not path:
        return "Unknown Coffee"
    slug = path.split("/")[-1]
    return slug.replace("-", " ").replace("_", " ").strip().title() or "Unknown Coffee"
