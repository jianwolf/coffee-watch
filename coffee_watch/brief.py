"""Compact, deterministic projection of a combined catalog.

A combined catalog is evidence, not a reading surface: 371 products with
uncapped ``raw_product_text`` measured 1.3 MB, and 3.3 MB in the worst
observed run. An agent session cannot hold that, so this module renders the
same catalog as a few hundred one-line rows plus a coverage header, and keeps
the raw text behind an explicit ``--detail`` lookup.

Everything here is mechanical: filtering, unit math, set differences against a
previous catalog, and formatting. It ranks nothing and judges nothing -- that
stays in the agent skill, per the analysis boundary in AGENTS.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .text_utils import grams_from_size_label

# A variant at or above this weight is a wholesale bag. When it is the only
# thing left in stock, the home-size bag has sold out; the price label jumps,
# which reads like a price rise unless it is called out.
BULK_GRAMS = 900

# Products in one roaster sharing a single published-at minute in a cluster
# this large are a storewide republish (usually a sale), not a drop. Measured
# baseline across three real catalogs: the largest genuine same-minute cluster
# was 4.
REPUBLISH_CLUSTER = 10

NOTES_MAX_CHARS = 90
DETAIL_TEXT_MAX_CHARS = 6000
# Delta lists are evidence, not an inventory. Past this many items the list is
# noise in a reply, so it is capped with a stated remainder.
DELTA_LIST_MAX = 15
CATALOG_GLOB = "*-catalog.json"
RUN_ID_RE = re.compile(r"(\d{8})-catalog\.json$")


@dataclass(frozen=True)
class Row:
    """One product, projected to the fields a buying decision needs."""

    roaster: str
    title: str
    url: str
    price_label: str
    per_100g: str
    notes: str
    flags: tuple[str, ...]
    grams: int

    def render(self) -> str:
        parts = [
            self.title or "(untitled)",
            self.price_label or "price not captured",
            self.per_100g or "-",
            f"[{' '.join(self.flags)}]" if self.flags else "[]",
            f"notes: {self.notes or '-'}",
            self.url or "(no url)",
        ]
        return "- " + " | ".join(parts)


@dataclass
class Delta:
    """Set differences between two catalogs, keyed by product URL."""

    previous_run_id: str = ""
    listed: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)
    sold_out: list[str] = field(default_factory=list)
    restocked: list[str] = field(default_factory=list)
    price_moved: list[str] = field(default_factory=list)
    # The displayed variant changed tier: the label moves, the price did not.
    size_changed: list[str] = field(default_factory=list)
    # Same roaster and title at a new URL: a re-slug, not a new coffee.
    relisted: list[str] = field(default_factory=list)
    # Roasters that returned nothing in one of the two runs. Their coffees
    # cannot be compared at all: missing now is not a sell-out, and missing
    # before is not a new listing.
    unknown_roasters: list[str] = field(default_factory=list)
    unknown_products: int = 0


def _mapping(value: Any) -> dict[str, Any]:
    """A nested object that may be missing, null, or the wrong type."""
    return value if isinstance(value, dict) else {}


def _products(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in catalog.get("products", []) if isinstance(p, dict)]


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a catalog object")
    return data


def catalog_paths(reports_dir: Path) -> list[Path]:
    """Combined catalogs in the reports dir, oldest run id first."""
    found = [
        path
        for path in reports_dir.glob(CATALOG_GLOB)
        if RUN_ID_RE.search(path.name)
    ]
    return sorted(found, key=lambda path: path.name)


def latest_catalog(reports_dir: Path) -> Path | None:
    paths = catalog_paths(reports_dir)
    return paths[-1] if paths else None


def previous_catalog(reports_dir: Path, current: Path) -> Path | None:
    """The newest catalog older than ``current``, or None in a fresh workspace."""
    current_run = RUN_ID_RE.search(current.name)
    if not current_run:
        return None
    older = [
        path
        for path in catalog_paths(reports_dir)
        if (match := RUN_ID_RE.search(path.name))
        and match.group(1) < current_run.group(1)
    ]
    return older[-1] if older else None


def _price_value(product: dict[str, Any]) -> float:
    raw = str(product.get("price") or "").strip()
    cleaned = re.sub(r"[^0-9.]", "", raw)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _variants(product: dict[str, Any]) -> list[dict[str, Any]]:
    return [v for v in product.get("variants", []) if isinstance(v, dict)]


def selected_grams(product: dict[str, Any]) -> int:
    """Weight of the variant the catalog chose for display pricing.

    The variant title wins. Shopify's `grams` is only trusted where the catalog
    already vouched for it (`price_source=variant_api_grams`, a placeholder
    title with a plausible weight), because it is otherwise unreliable in both
    directions: Proud Mary stamps 380 g on a 250 g bag, and a three-bag bundle
    carries its 900 g total. An unknown weight has to stay unknown -- a wrong
    $/100g is worse than a missing one.
    """
    label = str(product.get("price_variant") or "").strip()
    if not label:
        return 0
    from_title = grams_from_size_label(label)
    if from_title:
        return from_title
    if product.get("price_source") != "variant_api_grams":
        return 0
    for variant in _variants(product):
        grams = variant.get("grams")
        if isinstance(grams, int) and grams > 0:
            return grams
    return 0


def per_100g(product: dict[str, Any]) -> str:
    price = _price_value(product)
    grams = selected_grams(product)
    if price <= 0 or grams <= 0:
        return ""
    return f"${price * 100 / grams:.2f}/100g"


def is_bulk_only(product: dict[str, Any]) -> bool:
    """True when every in-stock variant is a wholesale bag and a home size sold out."""
    variants = _variants(product)
    if not variants:
        return False
    available = [v for v in variants if v.get("available")]
    if not available:
        return False
    if not all(int(v.get("grams") or 0) >= BULK_GRAMS for v in available):
        return False
    return any(
        0 < int(v.get("grams") or 0) < BULK_GRAMS
        for v in variants
        if not v.get("available")
    )


def product_flags(product: dict[str, Any]) -> tuple[str, ...]:
    flags: list[str] = []
    if product.get("is_new"):
        flags.append("new")
    availability = str(product.get("availability") or "")
    if availability == "unavailable":
        flags.append("SOLD-OUT")
    elif availability != "available":
        flags.append("avail?")
    if product.get("storefront_status") == "storefront_unavailable":
        flags.append("page-blocked")
    if is_bulk_only(product):
        flags.append("BULK-ONLY")
    source = str(product.get("price_source") or "")
    if source == "atom_feed":
        # The feed quotes a price range and no stock flag, so neither the
        # size nor the availability on this row was ever established.
        flags.append("ATOM-ONLY-no-size-or-stock")
    elif source == "visible_variant":
        flags.append("size-verified")
    elif source == "variant_api_grams":
        flags.append("size-from-api")
    elif source == "variant":
        flags.append("size-unverified")
    elif not product.get("price_variant"):
        flags.append("size?")
    if product.get("errors"):
        flags.append("scrape-err")
    if not product.get("tasting_notes") and product.get("raw_product_text"):
        flags.append("notes-in-text")
    if mentions_competition(product):
        flags.append("COMP-MENTION")
    variety = prized_variety(product)
    if variety:
        flags.append(f"VARIETY:{variety}")
    return tuple(flags)


def mentions_competition(product: dict[str, Any]) -> bool:
    """Whether the page mentions a competition result anywhere.

    Deliberately a mention and not a finding: the match may belong to the
    farm's history, a neighbouring lot or a subscription blurb. Confirming
    that the placing is this lot's is the agent's job, with --detail.
    """
    haystack = " ".join(
        str(product.get(field) or "")
        for field in ("title", "origin", "process", "raw_product_text")
    )
    return bool(COMPETITION_RE.search(haystack))


# A row whose flavour lives only in the page text printed "notes: -", which
# reads as "no flavour was captured" and has been reported that way in a real
# reply. The brief does not parse the text; it says where the text is.
# A third-party placing is the one quality claim on a product page the seller
# could not have written, so it is worth surfacing mechanically rather than
# leaving the reader to grep for it. This is a *mention*, not a verdict: the
# phrase can belong to a neighbouring lot, the farm's history or a
# subscription blurb, so the flag says "check this" and the agent confirms
# against --detail that the placing belongs to this coffee.
# "in the first place" is an English idiom, not a placing, and it put a
# Catuai with no competition history whatsoever on a shortlist of
# competition coffees. The lookbehind is the whole fix: every other reading
# of "first place" on a coffee page is a result.
COMPETITION_RE = re.compile(
    r"cup of excellence|\bCOE\b|best of panama|\bBOP\b|auction lot|"
    r"(?<!in the )(?:1st|2nd|3rd|first|second|third)[- ]place|national winner",
    re.IGNORECASE,
)

# A variety whose name alone carries cup potential - the second-hardest
# quality signal to fake after a third-party placing, because a roaster
# cannot call a Caturra a Sudan Rume and have it survive a cupping.
#
# Matched against the roaster's own title and variety fields ONLY, never
# raw_product_text: page text carries the site's navigation and related
# products, which made an Ethiopian natural match "Pacamara" and an
# Nicaraguan Maracaturra match "Geisha" when this was first measured over
# the full text. 112 false hits became 35 real ones.
PRIZED_VARIETY_RE = re.compile(
    r"\bgei?sha\b|sudan rume|\bSL-?28\b|\bSL-?34\b|pacamara|wush ?wush|"
    r"eugenioides|laurina|chiroso|pink bourbon|\bmokka\b|\bbourbon pointu\b",
    re.IGNORECASE,
)
PRIZED_VARIETY_FIELDS = ("title", "variety")


def prized_variety(product: dict[str, Any]) -> str:
    """The prized variety this product's own fields name, or "".

    Reads only the fields the roaster assigned to this product, so a Gesha
    in the shop's navigation never lands on someone else's coffee.
    """
    haystack = " ".join(
        str(product.get(field) or "") for field in PRIZED_VARIETY_FIELDS
    )
    match = PRIZED_VARIETY_RE.search(haystack)
    return match.group(0) if match else ""


NOTES_IN_TEXT_HINT = "(in page text, use --detail)"


def _notes(product: dict[str, Any]) -> str:
    notes = product.get("tasting_notes")
    if isinstance(notes, list):
        joined = ", ".join(str(n).strip() for n in notes if str(n).strip())
    else:
        joined = str(notes or "").strip()
    if len(joined) > NOTES_MAX_CHARS:
        joined = joined[: NOTES_MAX_CHARS - 3].rstrip() + "..."
    if not joined and product.get("raw_product_text"):
        return NOTES_IN_TEXT_HINT
    return joined


def build_row(product: dict[str, Any]) -> Row:
    return Row(
        roaster=str(product.get("roaster") or "(unknown roaster)"),
        title=str(product.get("title") or "").strip(),
        url=str(product.get("product_url") or "").strip(),
        price_label=str(product.get("price_label") or "").strip(),
        per_100g=per_100g(product),
        notes=_notes(product),
        flags=product_flags(product),
        grams=selected_grams(product),
    )


def republish_clusters(
    products: list[dict[str, Any]], threshold: int = REPUBLISH_CLUSTER
) -> list[tuple[str, str, int]]:
    """Roasters whose new products share one published-at minute.

    Shopify stamps ``published_at`` on a storewide republish, so a sale can
    mark dozens of long-standing products as new at once. Returns
    ``(roaster, minute, count)`` for each cluster at or above ``threshold``.
    """
    counts: dict[tuple[str, str], int] = {}
    for product in products:
        if not product.get("is_new"):
            continue
        published = str(product.get("shopify_published_at") or "")[:16]
        if not published:
            continue
        key = (str(product.get("roaster") or ""), published)
        counts[key] = counts.get(key, 0) + 1
    clusters = [
        (roaster, minute, count)
        for (roaster, minute), count in counts.items()
        if count >= threshold
    ]
    return sorted(clusters, key=lambda item: (-item[2], item[0]))


def history_is_cold(products: list[dict[str, Any]]) -> bool:
    """True when ``first_seen_at`` carries no history.

    A fresh checkout has no ``logs/seen_products.db``, so every product is
    first seen during this run. The field still looks populated, which is how a
    cold run can confidently report the whole catalog as new since last time.
    """
    stamps = {
        str(product.get("first_seen_at") or "")[:10]
        for product in products
        if product.get("first_seen_at")
    }
    return len(stamps) <= 1


def empty_roasters(catalog: dict[str, Any]) -> set[str]:
    """Roasters that returned no products in this run."""
    out = set()
    for entry in catalog.get("roasters", []):
        if not isinstance(entry, dict):
            continue
        if not _mapping(entry.get("summary")).get("products_found"):
            out.add(str(entry.get("name") or ""))
    return out


def compute_delta(current: dict[str, Any], previous: dict[str, Any]) -> Delta:
    now = {
        str(p.get("product_url")): p for p in _products(current) if p.get("product_url")
    }
    before = {
        str(p.get("product_url")): p for p in _products(previous) if p.get("product_url")
    }
    # Coverage is not symmetric between two runs: bot-gating hits a different
    # set of roasters each time. A roaster missing from either side cannot be
    # diffed in that direction without inventing a sell-out or a product drop.
    blank_now = empty_roasters(current)
    blank_before = empty_roasters(previous)
    delta = Delta(previous_run_id=str(previous.get("run_id") or ""))
    unknown: set[str] = set()

    def label(product: dict[str, Any]) -> str:
        return f"{product.get('roaster')} - {product.get('title')}"

    for url, product in sorted(now.items()):
        was = before.get(url)
        if was is None:
            roaster = str(product.get("roaster") or "")
            if roaster in blank_before:
                # The roaster returned nothing last run, so this coffee is
                # newly visible, not newly listed.
                delta.unknown_products += 1
                unknown.add(roaster)
                continue
            delta.listed.append(f"{label(product)} | {product.get('price_label')} | {url}")
            continue
        now_available = product.get("availability") == "available"
        was_available = was.get("availability") == "available"
        if was_available and not now_available:
            delta.sold_out.append(f"{label(product)} | {url}")
        elif now_available and not was_available:
            delta.restocked.append(
                f"{label(product)} | {product.get('price_label')} | {url}"
            )
        old_price = _price_value(was)
        new_price = _price_value(product)
        if old_price > 0 and new_price > 0 and abs(new_price - old_price) >= 0.01:
            # Only a like-for-like comparison is a price move. When the
            # displayed variant changed tier -- a 40 g sample giving way to a
            # 250 g bag -- the label triples and nothing was repriced.
            if str(was.get("price_variant") or "") == str(
                product.get("price_variant") or ""
            ):
                direction = "up" if new_price > old_price else "down"
                delta.price_moved.append(
                    f"{label(product)} | {was.get('price_label')} -> "
                    f"{product.get('price_label')} ({direction}) | {url}"
                )
            else:
                delta.size_changed.append(
                    f"{label(product)} | {was.get('price_label')} -> "
                    f"{product.get('price_label')} (different size, not a price "
                    f"change) | {url}"
                )
    for url, product in sorted(before.items()):
        if url in now:
            continue
        roaster = str(product.get("roaster") or "")
        if roaster in blank_now:
            # The roaster itself returned nothing, so absence is a coverage
            # gap. Calling these "gone" would invent a mass sell-out.
            delta.unknown_products += 1
            unknown.add(roaster)
            continue
        delta.gone.append(f"{label(product)} | {url}")
    delta.unknown_roasters = sorted(unknown)
    _reconcile_relistings(delta)
    return delta


def _reconcile_relistings(delta: Delta) -> None:
    """Collapse a re-slugged product counted as both gone and newly listed.

    Roasters re-slug a standing coffee at season rollover
    (``/kambarare-washed-process-2026`` becoming ``-2025``). Keyed by URL
    that is one disappearance plus one arrival, which inflates both halves of
    the headline and invents new arrivals that are not new.
    """
    gone_by_label = {item.split(" | ")[0]: item for item in delta.gone}
    if not gone_by_label:
        return
    still_listed = []
    for item in delta.listed:
        name = item.split(" | ")[0]
        if name in gone_by_label:
            delta.relisted.append(f"{item} (same title at a new URL)")
            delta.gone.remove(gone_by_label.pop(name))
            continue
        still_listed.append(item)
    delta.listed = still_listed


def _run_gap_days(current_run_id: str, previous_run_id: str) -> str:
    """How stale the baseline is. A monthly reader reads any delta as 'since
    last time', so an old baseline has to say its age."""
    try:
        now = datetime.strptime(current_run_id, "%Y%m%d")
        before = datetime.strptime(previous_run_id, "%Y%m%d")
    except ValueError:
        return ""
    days = (now - before).days
    return f" ({days} days earlier)" if days else " (same day)"


def _roaster_header(entry: dict[str, Any], rows: int) -> str:
    status = _mapping(entry.get("status"))
    summary = _mapping(entry.get("summary"))
    note = str(status.get("note") or "").strip()
    errors = entry.get("errors")
    error_count = len(errors) if isinstance(errors, list) else 0
    bits = [f"{rows} shown", f"{summary.get('new_products', 0)} new"]
    if note:
        bits.append(f"note: {note}")
    if error_count:
        bits.append(f"errors: {error_count}")
    return f"## {entry.get('name')} ({', '.join(bits)})"


def coverage(catalog: dict[str, Any]) -> list[str]:
    """Lines stating what this run actually collected.

    An unattended run cannot use the exit code as a freshness signal: a
    network-blocked scrape still writes a catalog and exits 0. These lines are
    the evidence the reply has to quote.
    """
    summary = _mapping(catalog.get("summary"))
    roasters = [r for r in catalog.get("roasters", []) if isinstance(r, dict)]
    empty = []
    for entry in roasters:
        if _mapping(entry.get("summary")).get("products_found"):
            continue
        reason = str(_mapping(entry.get("status")).get("note") or "no products parsed")
        empty.append(f"{entry.get('name')} ({reason})")
    total = len(roasters)
    with_products = total - len(empty)
    elapsed = summary.get("elapsed_s")
    took = f", took {elapsed}s" if elapsed else ""
    lines = [
        f"run: {catalog.get('run_id')} generated {catalog.get('generated_at')}{took}",
        f"coverage: {with_products}/{total} roasters returned products; "
        f"{summary.get('products', 0)} products, {summary.get('new_products', 0)} flagged is_new",
    ]
    atom_rows = sum(
        1 for product in _products(catalog) if product.get("price_source") == "atom_feed"
    )
    if atom_rows:
        lines.append(
            f"atom fallback: {atom_rows} row(s) came from a collection Atom feed "
            f"because the catalog API never allowed that roaster. They carry NO "
            f"per-variant size, price or stock, so no $/100g and availability is "
            f"unknown - do not present them as purchasable sizes, and the feed is "
            f"capped at each roaster's newest 25 products"
        )
    comp = sum(
        1
        for product in _products(catalog)
        if product.get("availability") == "available" and mentions_competition(product)
    )
    if comp:
        lines.append(
            f"competition mentions: {comp} available product(s) name a third-party "
            f"placing somewhere on the page (COMP-MENTION). The phrase may belong "
            f"to the farm's history or a neighbouring lot - confirm with --detail "
            f"that it is this coffee's"
        )
    prized = sum(
        1
        for product in _products(catalog)
        if product.get("availability") == "available" and prized_variety(product)
    )
    if prized:
        lines.append(
            f"prized varieties: {prized} available product(s) name one in their own "
            f"title or variety field (VARIETY:<name>); --prized lists them all"
        )
    second = _mapping(summary.get("second_pass"))
    if second.get("retried"):
        lines.append(
            f"second pass: {second.get('recovered', 0)} of {second['retried']} "
            f"rate-limited roaster(s) recovered after a cooldown"
        )
    if empty:
        lines.append(f"EMPTY ({len(empty)}): " + "; ".join(empty))
    failed = summary.get("failed_roasters") or []
    if failed:
        lines.append(f"failed: {', '.join(str(name) for name in failed)}")
    return lines


def render_prized(catalog: dict[str, Any]) -> str:
    """Every buyable coffee whose own fields name a prized variety.

    A roster rather than a shortlist: uncapped, one line each, grouped by
    roaster, and deliberately without tasting notes - the point is that
    nothing rare goes unmentioned, not that each one gets an argument.
    """
    products = [
        product
        for product in _products(catalog)
        if product.get("availability") == "available"
        and product.get("storefront_status") != "storefront_unavailable"
        and prized_variety(product)
    ]
    lines = [
        "# coffee-watch prized varieties",
        f"run: {catalog.get('run_id') or '(unknown)'}",
        f"{len(products)} buyable product(s) whose own title or variety field names "
        "a prized variety. The whole roster, uncapped; sold-out and page-blocked "
        "products are already excluded. No tasting notes here - use --detail for "
        "any you decide to write up.",
    ]
    if not products:
        lines.append("")
        lines.append("none this run")
        return "\n".join(lines) + "\n"
    by_roaster: dict[str, list[Row]] = {}
    for product in products:
        row = build_row(product)
        by_roaster.setdefault(row.roaster, []).append(row)
    for roaster in sorted(by_roaster):
        rows = by_roaster[roaster]
        lines.append("")
        lines.append(f"## {roaster} ({len(rows)})")
        for row in sorted(rows, key=lambda r: r.title):
            lines.append(
                "- "
                + " | ".join(
                    [
                        row.title or "(untitled)",
                        row.price_label or "price not captured",
                        row.per_100g or "-",
                        f"[{' '.join(row.flags)}]" if row.flags else "[]",
                        row.url or "(no url)",
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def render_brief(
    catalog: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
    new_only: bool = False,
    available_only: bool = False,
    roaster_filter: str = "",
) -> str:
    products = _products(catalog)
    lines = ["# coffee-watch brief"]
    lines.extend(coverage(catalog))

    if history_is_cold(products):
        lines.append(
            "history: first_seen_at is this run for every product (no seen-product "
            "database in this workspace) -- do not report 'new since last purchase' "
            "from it"
        )
    clusters = republish_clusters(products)
    if clusters:
        lines.append(
            "republish watch: "
            + "; ".join(f"{r} {c} products at {m}" for r, m, c in clusters)
            + " -- probable storewide republish, not a drop"
        )
    else:
        lines.append("republish watch: no same-minute is_new cluster above threshold")

    if previous is None:
        lines.append(
            "delta: no earlier catalog in this workspace, so sold-out, restock and "
            "price moves are unknown this run; is_new comes from the roaster's own "
            "published_at"
        )
    else:
        delta = compute_delta(catalog, previous)
        age = _run_gap_days(str(catalog.get("run_id") or ""), delta.previous_run_id)
        lines.append(
            f"delta vs {delta.previous_run_id}{age}: {len(delta.listed)} newly "
            f"listed, {len(delta.gone)} gone, {len(delta.sold_out)} sold out, "
            f"{len(delta.restocked)} restocked, {len(delta.price_moved)} price "
            f"moves, {len(delta.relisted)} relisted, {len(delta.size_changed)} "
            f"size swaps"
        )
        if delta.unknown_products:
            lines.append(
                f"delta gaps: {delta.unknown_products} products from "
                f"{len(delta.unknown_roasters)} roaster(s) missing from one of the two "
                f"runs are UNKNOWN - neither gone nor newly listed: "
                f"{', '.join(delta.unknown_roasters)}"
            )
        for title, items in (
            ("newly listed", delta.listed),
            ("gone", delta.gone),
            ("sold out", delta.sold_out),
            ("restocked", delta.restocked),
            ("price moves", delta.price_moved),
            ("relisted (same coffee, new URL)", delta.relisted),
            ("size swaps (displayed variant changed, not the price)", delta.size_changed),
        ):
            if not items:
                continue
            lines.append(f"### {title}")
            lines.extend(f"- {item}" for item in items[:DELTA_LIST_MAX])
            if len(items) > DELTA_LIST_MAX:
                lines.append(f"- ...and {len(items) - DELTA_LIST_MAX} more")

    lines.append("")
    lines.append(
        "# rows: title | size = price | $/100g | flags | notes | url "
        "(empty storefront_status is normal; flags are mechanical, not a verdict)"
    )

    roasters = [r for r in catalog.get("roasters", []) if isinstance(r, dict)]
    order = {str(entry.get("name")): index for index, entry in enumerate(roasters)}
    wanted = roaster_filter.strip().lower()
    grouped: dict[str, list[Row]] = {}
    for product in products:
        if new_only and not product.get("is_new"):
            continue
        if available_only and product.get("availability") != "available":
            continue
        name = str(product.get("roaster") or "")
        if wanted and wanted not in name.lower():
            continue
        grouped.setdefault(name, []).append(build_row(product))

    for entry in sorted(roasters, key=lambda e: order.get(str(e.get("name")), 0)):
        name = str(entry.get("name"))
        rows = grouped.pop(name, [])
        if wanted and wanted not in name.lower():
            continue
        if not rows and (new_only or available_only or wanted):
            continue
        lines.append("")
        lines.append(_roaster_header(entry, len(rows)))
        if not rows:
            lines.append("- (no products this run)")
            continue
        lines.extend(
            row.render()
            for row in sorted(rows, key=lambda r: (not r.flags or "new" not in r.flags, r.title))
        )
    for name, rows in sorted(grouped.items()):
        lines.append("")
        lines.append(f"## {name} ({len(rows)} shown)")
        lines.extend(row.render() for row in sorted(rows, key=lambda r: r.title))
    return "\n".join(lines) + "\n"


def render_detail(catalog: dict[str, Any], needle: str) -> str:
    """Full captured evidence for the few products that reach a shortlist.

    Scraped text is untrusted: it is wrapped so a reader (model or human)
    cannot mistake roaster copy for instructions.
    """
    wanted = needle.strip().lower()
    matches = [
        product
        for product in _products(catalog)
        if wanted in str(product.get("title") or "").lower()
        or wanted in str(product.get("product_url") or "").lower()
        or wanted == str(product.get("product_id") or "").lower()
    ]
    if not matches:
        return f"no product matching {needle!r} in this catalog\n"
    blocks = []
    for product in matches:
        row = build_row(product)
        text = str(product.get("raw_product_text") or "").strip()
        if len(text) > DETAIL_TEXT_MAX_CHARS:
            text = text[:DETAIL_TEXT_MAX_CHARS] + "\n...[truncated]"
        blocks.append(
            "\n".join(
                [
                    f"## {row.roaster} - {row.title}",
                    f"url: {row.url}",
                    f"price: {row.price_label or '-'} ({row.per_100g or 'size unknown'}) "
                    f"source={product.get('price_source') or '-'}",
                    f"flags: {' '.join(row.flags) or '-'}",
                    f"variants: {json.dumps(_variants(product), ensure_ascii=False)}",
                    f"visible on page: {product.get('visible_variant_titles') or '-'}",
                    f"origin: {product.get('origin') or '-'}",
                    f"process: {product.get('process') or '-'}",
                    f"tasting_notes: {row.notes or '-'}",
                    f"published_at: {product.get('shopify_published_at') or '-'} "
                    f"update_date: {product.get('update_date') or '-'} "
                    f"date_source: {product.get('date_source') or '-'}",
                    f"errors: {product.get('errors') or '-'}",
                    "<UNTRUSTED_SCRAPED_TEXT>",
                    text or "(no page text captured)",
                    "</UNTRUSTED_SCRAPED_TEXT>",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m coffee_watch.brief",
        description=(
            "Render a combined catalog as compact rows plus a coverage header. "
            "Reads the newest catalog in the reports dir by default."
        ),
    )
    parser.add_argument("--catalog", type=Path, help="Combined catalog JSON to render")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Where to look for catalogs (default: reports)",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        help="Earlier catalog to diff against (default: the newest older one, if any)",
    )
    parser.add_argument(
        "--no-previous",
        action="store_true",
        help="Skip the delta even when an earlier catalog exists",
    )
    parser.add_argument(
        "--new-only", action="store_true", help="Only products flagged is_new"
    )
    parser.add_argument(
        "--available-only", action="store_true", help="Drop sold-out products"
    )
    parser.add_argument("--roaster", default="", help="Filter to roasters matching this text")
    parser.add_argument(
        "--prized",
        action="store_true",
        help="List every buyable product naming a prized variety, grouped by roaster",
    )
    parser.add_argument(
        "--detail",
        help="Print full captured evidence for products matching this title, URL or id",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.catalog or latest_catalog(args.reports_dir)
    if path is None:
        print(
            f"No combined catalog found in {args.reports_dir}. Run the scraper first.",
            file=sys.stderr,
        )
        return 2
    if not path.is_file():
        print(f"Catalog not found: {path}", file=sys.stderr)
        return 2
    catalog = load_catalog(path)

    if args.detail:
        sys.stdout.write(render_detail(catalog, args.detail))
        return 0

    if args.prized:
        sys.stdout.write(render_prized(catalog))
        return 0

    previous_path = args.previous
    if previous_path is None and not args.no_previous:
        previous_path = previous_catalog(args.reports_dir, path)
    previous = (
        load_catalog(previous_path)
        if previous_path and previous_path.is_file() and not args.no_previous
        else None
    )
    sys.stdout.write(
        render_brief(
            catalog,
            previous=previous,
            new_only=args.new_only,
            available_only=args.available_only,
            roaster_filter=args.roaster,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
