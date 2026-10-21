"""Tests for the Discord notifier.

The notifier is how a cloud run reaches a phone at all, so the cases that
matter are: never exceed Discord's hard limit, never cut a note mid-sentence,
and never put the webhook URL anywhere it could be read later.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "notify_discord",
    Path(__file__).resolve().parents[1] / "scripts" / "notify_discord.py",
)
assert _SPEC and _SPEC.loader
notify = importlib.util.module_from_spec(_SPEC)
sys.modules["notify_discord"] = notify
_SPEC.loader.exec_module(notify)


def test_a_short_note_is_one_message():
    assert notify.split_message("10/20 今天没什么值得专门跑一趟的。") == [
        "10/20 今天没什么值得专门跑一趟的。"
    ]


def test_a_real_length_note_fits_one_message():
    """Measured: the 2026-10-20 note was ~1100 characters.

    The sample is Chinese, with Chinese punctuation, because the notes are and
    this test is about their length. RUF001 flags the fullwidth commas as
    ambiguous; here they are the thing being measured.
    """
    line = "- 第 {} 支咖啡，风味是石榴、菠萝、紫罗兰，250g $36"  # noqa: RUF001
    note = "\n".join(line.format(i) for i in range(20))
    chunks = notify.split_message(note)
    assert len(chunks) == 1 and len(chunks[0]) <= notify.MAX_CHARS


def test_splitting_never_exceeds_the_hard_limit():
    note = "\n".join(f"line {i} " + "x" * 80 for i in range(120))
    chunks = notify.split_message(note)
    assert len(chunks) > 1
    assert all(len(c) <= notify.MAX_CHARS for c in chunks)


def test_splitting_happens_on_line_boundaries():
    note = "\n".join(f"line{i}" for i in range(500))
    for chunk in notify.split_message(note):
        assert not chunk.startswith("ine"), "a line was cut in the middle"


def test_an_overlong_single_line_is_hard_split_rather_than_dropped():
    note = "y" * 5000
    chunks = notify.split_message(note)
    assert all(len(c) <= notify.MAX_CHARS for c in chunks)
    assert sum(len(c) for c in chunks) == 5000, "nothing may be silently lost"


def test_no_webhook_is_its_own_exit_code(monkeypatch, capsys):
    """A run that scouted fine but could not deliver is a distinct failure."""
    monkeypatch.delenv(notify.ENV_VAR, raising=False)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("a note"))
    assert notify.main([]) == 3
    assert "No webhook configured" in capsys.readouterr().err


def test_empty_input_posts_nothing(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("   \n  "))
    assert notify.main(["--webhook", "https://example.invalid/x"]) == 2
    assert "Nothing on stdin" in capsys.readouterr().err


def test_a_failure_never_echoes_the_webhook(monkeypatch, capsys):
    """Logs get pasted into chats; the URL is a credential."""
    secret = "https://discord.com/api/webhooks/123/SUPERSECRETTOKEN"

    def boom(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(notify, "post", boom)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("a note"))
    assert notify.main(["--webhook", secret]) == 1
    captured = capsys.readouterr()
    assert "SUPERSECRETTOKEN" not in captured.err + captured.out


def test_dry_run_sends_nothing(monkeypatch, capsys):
    def boom(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("dry run must not post")

    monkeypatch.setattr(notify, "post", boom)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("a note"))
    assert notify.main(["--dry-run"]) == 0
    assert "a note" in capsys.readouterr().out


def test_the_repo_contains_no_webhook_url():
    """The one thing that must never be committed.

    Matches a real webhook - a long numeric id and a long token - so the
    placeholder in the notifier's own docstring and the fake in this file do
    not trip it, while a genuine URL anywhere in the repo does.
    """
    import re

    real_webhook = re.compile(r"discord\.com/api/webhooks/\d{6,}/[\w-]{40,}")
    root = Path(__file__).resolve().parents[1]
    tracked = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git/" not in str(p)
        and ".venv" not in str(p)
        and "reports/" not in str(p)
        and "logs/" not in str(p)
        and p.suffix in {".py", ".md", ".json", ".sh", ".txt", ".toml", ".yml"}
    ]
    for path in tracked:
        text = path.read_text(encoding="utf-8", errors="replace")
        found = real_webhook.search(text)
        assert not found, f"a real webhook URL is committed in {path}"
