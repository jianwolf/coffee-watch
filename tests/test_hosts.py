"""Tests for the egress allowlist generator.

The allowlist is pasted into a cloud environment by hand, so a missing host
means a silently empty catalog and an over-broad one quietly widens what an
unattended session can reach.
"""

from __future__ import annotations

import json
from pathlib import Path

from coffee_watch import hosts


def _write_roasters(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "roasters.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_hosts_come_from_configured_base_urls(tmp_path: Path):
    path = _write_roasters(
        tmp_path,
        [
            {"name": "A", "base_url": "https://littlewolf.coffee"},
            {"name": "B", "base_url": "https://www.seycoffee.com"},
            "not a dict",
        ],
    )
    assert hosts.hosts_from_roasters(path) == {
        "littlewolf.coffee",
        "www.seycoffee.com",
    }


def test_redirect_targets_are_picked_up_from_a_catalog():
    """One roaster is configured on a myshopify entry point that redirects away."""
    catalog = {
        "products": [
            {"product_url": "https://drinkpassenger.com/products/agaro"},
            {"product_url": ""},
            "not a dict",
        ]
    }
    assert hosts.hosts_from_catalog(catalog) == {"drinkpassenger.com"}


def test_a_roasters_own_domain_gets_an_apex_wildcard():
    entries = hosts.allowlist({"www.olympiacoffee.com", "dayglow.coffee"})
    assert "www.olympiacoffee.com" in entries
    assert "olympiacoffee.com" in entries
    assert "*.olympiacoffee.com" in entries
    assert "*.dayglow.coffee" in entries


def test_a_shared_platform_tenant_gets_no_wildcard():
    """`*.myshopify.com` would open the allowlist to every store on the platform."""
    entries = hosts.allowlist({"passenger-coffee-and-tea.myshopify.com"})
    assert entries == ["passenger-coffee-and-tea.myshopify.com"]
    assert not any(entry.startswith("*.") for entry in entries)


def test_a_regional_subdomain_is_listed_exactly():
    entries = hosts.allowlist({"us.lacabra.com"})
    assert entries == ["us.lacabra.com"]


def test_main_prints_config_and_catalog_hosts(tmp_path: Path, capsys):
    roasters = _write_roasters(
        tmp_path, [{"name": "A", "base_url": "https://passenger.myshopify.com"}]
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "20261018-catalog.json").write_text(
        json.dumps(
            {"products": [{"product_url": "https://drinkpassenger.com/products/x"}]}
        ),
        encoding="utf-8",
    )

    assert (
        hosts.main(
            ["--roasters-path", str(roasters), "--reports-dir", str(reports)]
        )
        == 0
    )
    printed = capsys.readouterr().out.split()
    assert "passenger.myshopify.com" in printed
    assert "drinkpassenger.com" in printed
    assert "*.drinkpassenger.com" in printed


def test_exact_mode_drops_the_wildcards(tmp_path: Path, capsys):
    roasters = _write_roasters(tmp_path, [{"name": "A", "base_url": "https://x.coffee"}])
    assert (
        hosts.main(
            [
                "--roasters-path",
                str(roasters),
                "--reports-dir",
                str(tmp_path / "missing"),
                "--exact",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.split() == ["x.coffee"]


def test_missing_config_is_an_error(tmp_path: Path, capsys):
    assert hosts.main(["--roasters-path", str(tmp_path / "nope.json")]) == 2
    assert "not found" in capsys.readouterr().err


def test_the_real_config_resolves_every_roaster():
    project_root = Path(__file__).resolve().parents[1]
    found = hosts.hosts_from_roasters(project_root / "config" / "roasters.json")
    entries = json.loads(
        (project_root / "config" / "roasters.json").read_text(encoding="utf-8")
    )
    assert len(found) >= len({e["base_url"] for e in entries}) - 1
    assert all(host and " " not in host for host in found)
