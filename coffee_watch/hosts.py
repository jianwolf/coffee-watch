"""List the hosts a scrape needs, for an agent sandbox's egress allowlist.

Cloud agent environments default to a package-registry allowlist, so every
roaster request is refused with a 403 before it reaches the site and the run
writes an empty catalog. The allowlist lives in the environment, not in this
repo, so the repo's job is to print exactly what to paste there -- and to keep
that list honest when ``config/roasters.json`` changes.

Configured ``base_url`` hosts are not sufficient on their own: one roaster is
configured at a ``myshopify.com`` entry point whose product URLs redirect to a
different domain, so a recent catalog's product URLs are folded in when one is
available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .brief import latest_catalog, load_catalog


def _host(url: str) -> str:
    return urlsplit(url.strip()).netloc.lower().split("@")[-1].split(":")[0]


def _apex(host: str) -> str:
    labels = host.split(".")
    return ".".join(labels[-2:]) if len(labels) > 2 else host


def hosts_from_roasters(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("roasters", []) if isinstance(data, dict) else data
    out = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        host = _host(str(entry.get("base_url") or ""))
        if host:
            out.add(host)
    return out


def hosts_from_catalog(catalog: dict[str, Any]) -> set[str]:
    out = set()
    for product in catalog.get("products", []):
        if not isinstance(product, dict):
            continue
        host = _host(str(product.get("product_url") or ""))
        if host:
            out.add(host)
    return out


def allowlist(hosts: set[str]) -> list[str]:
    """Observed hosts, plus an apex wildcard where the apex is the roaster's own.

    A wildcard is only safe when the observed host IS the roaster's domain, so
    `www.` and apex redirects are covered. A tenant subdomain on a shared
    platform (``shop-name.myshopify.com``) gets no wildcard: ``*.myshopify.com``
    would open the allowlist to every store on that platform.
    """
    entries = set(hosts)
    for host in hosts:
        apex = _apex(host)
        if host in {apex, f"www.{apex}"}:
            entries.add(apex)
            entries.add(f"*.{apex}")
    return sorted(entries)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m coffee_watch.hosts",
        description=(
            "Print the hosts this scrape needs, one per line, for a cloud "
            "environment's allowed-domains field."
        ),
    )
    parser.add_argument(
        "--roasters-path", type=Path, default=Path("config/roasters.json")
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Print only observed hosts, without the apex wildcards",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.roasters_path.is_file():
        print(f"Roaster config not found: {args.roasters_path}", file=sys.stderr)
        return 2
    hosts = hosts_from_roasters(args.roasters_path)
    catalog_path = latest_catalog(args.reports_dir)
    if catalog_path is not None:
        hosts |= hosts_from_catalog(load_catalog(catalog_path))
    for entry in sorted(hosts) if args.exact else allowlist(hosts):
        print(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
