"""Shopify collection Atom feed as a fallback catalog source.

A gated store's `/collections/{handle}/products.json` refuses roughly nine
requests in ten (docs/shopify-bot-gating.md), while the Atom feed Shopify
generates for the same collection answers normally. It is a published,
robots-allowed endpoint -- these stores serve `User-agent: * / Allow: /` with no
disallow for it -- so reading it is using a different front door, not working
around a closed one.

What it gives up, measured against a real feed, and why the brief has to say so:

- **No per-variant data.** The summary carries a price *range* ("24.00 -
  180.00") and a variant count, not a title, weight, price or stock flag per
  variant. So no purchasable size, no `$/100g`, and availability is unknown
  rather than confirmed.
- **The newest 25 products only.** The feed is capped and ignores `?page`, so a
  large store is truncated. Entries are ordered by published date descending,
  which means the cap keeps exactly the window a "what is new" question needs
  and drops older standing stock.

It does carry the product id, URL, title, published and updated timestamps, and
the roaster's own description text -- enough for discovery, flavour and
newness, which is most of what the reply leads with.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit, urlunsplit

from .models import ProductCandidate

# Marks a product that came from the feed, so the catalog and the brief can say
# the size and stock were never established.
ATOM_PRICE_SOURCE = "atom_feed"

# Shopify caps the feed and ignores ?page, so there is nothing to paginate.
# Recorded here because it is a limit of theirs, not a choice of ours.
ATOM_MAX_ENTRIES = 25

_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_ID_RE = re.compile(r"<id>\s*(.*?)\s*</id>", re.DOTALL)
_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", re.DOTALL)
_LINK_RE = re.compile(r'<link[^>]*rel="alternate"[^>]*href="([^"]+)"')
_PUBLISHED_RE = re.compile(r"<published>\s*(.*?)\s*</published>", re.DOTALL)
_UPDATED_RE = re.compile(r"<updated>\s*(.*?)\s*</updated>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary[^>]*>(.*?)</summary>", re.DOTALL)
_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_PRODUCT_ID_RE = re.compile(r"/products/(\d+)\s*$")
# "<strong>Price: </strong> 24.00 - 180.00" in Shopify's generated summary.
_PRICE_RE = re.compile(
    r"Price:\s*</strong>\s*([\d,]+\.?\d*)", re.IGNORECASE
)


def atom_url_for(base_url: str, products_path: str) -> str:
    """The feed URL for the collection a roaster's catalog path points at.

    ``/collections/coffee/products.json`` becomes ``/collections/coffee.atom``.
    Returns "" when the path is not a Shopify collection endpoint, so a
    non-Shopify roaster never gets a guessed URL.
    """
    path = urlsplit(products_path or "").path
    if not path.startswith("/collections/"):
        return ""
    collection = path[len("/collections/") :]
    collection = collection.split("/products.json")[0].strip("/")
    if not collection or "/" in collection:
        return ""
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        return ""
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/collections/{collection}.atom", "", "")
    )


def _unescape(value: str) -> str:
    # Titles arrive XML-escaped; the description is CDATA and needs no undoing.
    for entity, char in (
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&apos;", "'"),
        ("&amp;", "&"),  # last, so a double-escaped string is not mangled
    ):
        value = value.replace(entity, char)
    return value


def parse_atom_feed(
    xml: str,
    roaster_name: str,
    log: logging.Logger | None = None,
) -> list[ProductCandidate]:
    """Products from one collection feed, newest first.

    Entry text is roaster-authored and stays untrusted: it goes into
    ``body_html`` and is sanitized downstream exactly like a product page.
    """
    products: list[ProductCandidate] = []
    seen: set[str] = set()
    for entry in _ENTRY_RE.findall(xml or ""):
        link = _LINK_RE.search(entry)
        url = _unescape(link.group(1).strip()) if link else ""
        if not url or url in seen:
            continue
        title = _TITLE_RE.search(entry)
        name = _unescape(title.group(1).strip()) if title else ""
        if not name:
            continue

        raw_id = _ID_RE.search(entry)
        numeric = _PRODUCT_ID_RE.search(raw_id.group(1)) if raw_id else None
        product_id = (
            f"{roaster_name}:{numeric.group(1)}" if numeric else f"{roaster_name}:{url}"
        )

        summary = _SUMMARY_RE.search(entry)
        body = summary.group(1) if summary else ""
        cdata = _CDATA_RE.search(body)
        if cdata:
            body = cdata.group(1)

        price = _PRICE_RE.search(body)
        published = _PUBLISHED_RE.search(entry)
        updated = _UPDATED_RE.search(entry)

        seen.add(url)
        products.append(
            ProductCandidate(
                product_id=product_id,
                name=name,
                url=url,
                source=roaster_name,
                # The lower bound of the range. It is a real price the store
                # quotes, but which size it buys is exactly what the feed does
                # not say, so it is never presented as a size-aware label.
                list_price=price.group(1).replace(",", "") if price else "",
                body_html=body,
                variants=(),
                shopify_published_at=published.group(1).strip() if published else "",
                shopify_updated_at=updated.group(1).strip() if updated else "",
            )
        )
    if log and products:
        log.info(
            "Atom feed for %s: %d product(s), newest %s. No per-variant size, "
            "price or stock, and the feed is capped at the newest %d.",
            roaster_name,
            len(products),
            products[0].shopify_published_at[:10] or "unknown",
            ATOM_MAX_ENTRIES,
        )
    return products
