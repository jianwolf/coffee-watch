from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


def normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    parsed = urlsplit(url)
    if not parsed.scheme:
        url = f"https://{url.lstrip('/')}"
        parsed = urlsplit(url)
    normalized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    return normalized or url


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def build_url_with_params(base_url: str, path: str, params: dict[str, str]) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        url = path
    else:
        base = base_url if base_url.endswith("/") else f"{base_url}/"
        url = urljoin(base, path.lstrip("/"))
    if not params:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    for key, value in params.items():
        query[str(key)] = str(value)
    new_query = urlencode(query)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, ""))


def matches_patterns(url: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    def match(pattern: str) -> bool:
        if pattern.startswith("re:"):
            try:
                return re.search(pattern[3:], url) is not None
            except re.error:
                return False
        return pattern in url

    if exclude and any(match(pattern) for pattern in exclude):
        return False
    if not include:
        return True
    return any(match(pattern) for pattern in include)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "roaster"
