from __future__ import annotations

from pathlib import Path

from coffee_watch.models import RoasterRunStatus, RoasterSource
from coffee_watch.report_status import (
    EMPTY_REPORT_LINE,
    LEGACY_EMPTY_REPORT_LINE,
    STATUS_FAILURE,
    STATUS_SUCCESS,
    append_failed_roaster_lines_to_digest,
    build_roaster_failure_line,
    collect_resume_targets,
    extract_failed_roasters_from_reports,
    merge_failed_roaster_names,
    read_status_sidecar,
    write_status_sidecar,
)


def _status(**overrides) -> RoasterRunStatus:
    base = dict(
        roaster="R",
        run_id="20260418",
        status=STATUS_SUCCESS,
        attempts=1,
        report_path=None,
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


def test_build_roaster_failure_line():
    assert (
        build_roaster_failure_line("SEY Coffee")
        == "SEY Coffee roaster report generation has failed"
    )


def test_append_failed_roaster_lines_to_empty_digest():
    out = append_failed_roaster_lines_to_digest("", ["A", "B"])
    assert "## Report Generation Failures" in out
    assert "A roaster report" in out
    assert "B roaster report" in out


def test_merge_failed_roaster_names_dedupes_and_strips():
    assert merge_failed_roaster_names(["A", " B"], ["a", "B", "C"]) == [
        "A",
        "B",
        "a",
        "C",
    ]


def test_extract_failed_from_legacy_empty_line():
    text = "\n".join(
        [
            "# Coffee Watch Report",
            "",
            "Roaster: Little Wolf Coffee",
            "Generated: ...",
            "Run: 20260418",
            "",
            LEGACY_EMPTY_REPORT_LINE,
        ]
    )
    failed = extract_failed_roasters_from_reports([("20260418-little-wolf-coffee.md", text)])
    assert failed == ["Little Wolf Coffee"]


def test_extract_failed_from_new_empty_line():
    text = "Roaster: SEY Coffee\n" + EMPTY_REPORT_LINE
    failed = extract_failed_roasters_from_reports([("20260418-sey-coffee.md", text)])
    assert failed == ["SEY Coffee"]


def test_sidecar_roundtrip(tmp_path: Path):
    status = _status(status=STATUS_FAILURE, note="boom")
    path = write_status_sidecar(tmp_path, status, logger=_NullLogger())
    assert path is not None
    data = read_status_sidecar(tmp_path, "R", "20260418")
    assert data is not None
    assert data["status"] == STATUS_FAILURE
    assert data["note"] == "boom"


def test_collect_resume_targets_retries_missing_report_even_with_success_sidecar(
    tmp_path: Path,
):
    status = _status(status=STATUS_SUCCESS, report_path="20260418-r.md")
    write_status_sidecar(tmp_path, status, logger=_NullLogger())
    roaster = RoasterSource(name="R", base_url="https://example.com")

    targets = collect_resume_targets(
        [roaster], tmp_path, "20260418", logger=_NullLogger()
    )

    assert [target.name for target in targets] == ["R"]


class _NullLogger:
    def warning(self, *args, **kwargs):  # noqa: D401 - test shim
        return None

    def info(self, *args, **kwargs):
        return None
