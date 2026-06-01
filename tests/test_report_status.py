from __future__ import annotations

from pathlib import Path

from coffee_watch.models import RoasterRunStatus, RoasterSource
from coffee_watch.report_status import (
    STATUS_FAILURE,
    STATUS_SUCCESS,
    collect_resume_targets,
    merge_failed_roaster_names,
    read_status_sidecar,
    write_status_sidecar,
)
from coffee_watch.reporting import make_roaster_catalog_path


def _status(**overrides) -> RoasterRunStatus:
    base = dict(
        roaster="R",
        run_id="20260418",
        status=STATUS_SUCCESS,
        attempts=1,
        catalog_path=None,
        products_found=0,
        new_products=0,
        fetched_pages=0,
        platform="shopify",
        classified_by_source={},
        undated=0,
        outside_window=0,
        grounding_queries=(),
        completed_at="2026-04-18T00:00:00+00:00",
        note="",
    )
    base.update(overrides)
    return RoasterRunStatus(**base)


def test_merge_failed_roaster_names_dedupes_and_strips():
    assert merge_failed_roaster_names(["A", " B"], ["a", "B", "C"]) == [
        "A",
        "B",
        "a",
        "C",
    ]


def test_sidecar_roundtrip(tmp_path: Path):
    status = _status(status=STATUS_FAILURE, note="boom")
    path = write_status_sidecar(tmp_path, status, logger=_NullLogger())
    assert path is not None
    data = read_status_sidecar(tmp_path, "R", "20260418")
    assert data is not None
    assert data["status"] == STATUS_FAILURE
    assert data["note"] == "boom"


def test_collect_resume_targets_retries_missing_catalog_even_with_success_sidecar(
    tmp_path: Path,
):
    status = _status(status=STATUS_SUCCESS, catalog_path="20260418-r.catalog.json")
    write_status_sidecar(tmp_path, status, logger=_NullLogger())
    roaster = RoasterSource(name="R", base_url="https://example.com")

    targets = collect_resume_targets(
        [roaster], tmp_path, "20260418", logger=_NullLogger()
    )

    assert [target.name for target in targets] == ["R"]


def test_collect_resume_targets_skips_success_with_catalog(tmp_path: Path):
    roaster = RoasterSource(name="R", base_url="https://example.com")
    catalog_path = make_roaster_catalog_path(tmp_path, roaster.name, "20260418")
    catalog_path.write_text("{}", encoding="utf-8")
    status = _status(status=STATUS_SUCCESS, catalog_path=catalog_path.name)
    write_status_sidecar(tmp_path, status, logger=_NullLogger())

    targets = collect_resume_targets(
        [roaster], tmp_path, "20260418", logger=_NullLogger()
    )

    assert targets == []


class _NullLogger:
    def warning(self, *args, **kwargs):  # noqa: D401 - test shim
        return None

    def info(self, *args, **kwargs):
        return None
