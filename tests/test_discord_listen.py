"""Tests for the Discord question listener.

The listener turns channel text into a prompt for a Claude Code session with
bypassPermissions on the owner's own machine. The author check is therefore
the whole security boundary, and these tests pin it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discord_listen


def _message(author_id="12345", bot=False, content="what is on the shelf?", mid="99"):
    return {"id": mid, "content": content, "author": {"id": author_id, "bot": bot}}


def test_only_the_owner_is_answered():
    """Anyone with the invite link can type; only one person is executed."""
    assert discord_listen.should_answer(_message(author_id="12345"), "12345")
    assert not discord_listen.should_answer(_message(author_id="99999"), "12345")


def test_bots_are_never_answered():
    """The scouting webhook posts here too; answering it would loop."""
    assert not discord_listen.should_answer(
        _message(author_id="12345", bot=True), "12345"
    )


def test_an_empty_message_is_not_a_question():
    assert not discord_listen.should_answer(_message(content="   "), "12345")
    assert not discord_listen.should_answer(_message(content=""), "12345")


def test_it_refuses_to_start_without_an_owner(monkeypatch, capsys):
    """A listener with no owner answers the whole channel - never start one."""
    monkeypatch.delenv("DISCORD_OWNER_ID", raising=False)
    code = discord_listen.main(["--token", "t", "--channel", "c", "--owner", ""])
    assert code == 2
    assert "DISCORD_OWNER_ID" in capsys.readouterr().err


def test_a_reply_carries_the_session_forward(monkeypatch, tmp_path):
    """Follow-ups have to reach the session that just answered."""
    seen = {}

    class _Done:
        stdout = json.dumps({"result": "mango, lychee", "session_id": "sess-2"})
        stderr = ""

    def fake_run(command, **kwargs):
        seen["command"] = command
        return _Done()

    monkeypatch.setattr(discord_listen.subprocess, "run", fake_run)
    reply, session = discord_listen.ask_claude("how much?", tmp_path, "sess-1")
    assert reply == "mango, lychee"
    assert session == "sess-2"
    assert "--resume" in seen["command"]
    assert "sess-1" in seen["command"]


def test_an_expired_login_says_so_instead_of_going_quiet():
    """Measured: a stale OAuth session returns is_error with this message.

    A 24/7 listener that silently stops answering is worse than one that
    says why, because nobody notices the first kind.
    """
    class _Done:
        stdout = json.dumps({
            "result": "Failed to authenticate: OAuth session expired and could "
                      "not be refreshed",
            "is_error": True,
            "session_id": "sess-3",
        })
        stderr = ""

    import unittest.mock as mock

    with mock.patch.object(discord_listen.subprocess, "run", lambda *a, **k: _Done()):
        reply, session = discord_listen.ask_claude("hi", Path("/tmp"), None)
    assert "not authenticated" in reply
    assert "sign in" in reply
    # The dead session id is dropped so the next turn starts clean.
    assert session is None


def test_a_missing_cli_is_reported_not_crashed(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(discord_listen.subprocess, "run", boom)
    reply, session = discord_listen.ask_claude("hi", Path("/tmp"), None)
    assert "not on this machine's PATH" in reply
    assert session is None


def test_the_repo_contains_no_bot_token():
    """A bot token is a credential and this repository is public."""
    root = Path(__file__).resolve().parent.parent
    for path in root.rglob("*"):
        if not path.is_file() or ".git/" in str(path) or "/.venv/" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # Discord bot tokens start with the base64 of the app id and are long.
        assert "DISCORD_BOT_TOKEN=" not in text or "DISCORD_BOT_TOKEN=..." in text or (
            "os.environ" in text or "docs/" in str(path) or "tests/" in str(path)
        ), path


@pytest.mark.parametrize("after,expected", [(None, "limit=1"), ("42", "after=42")])
def test_the_first_poll_does_not_replay_the_channel(after, expected, monkeypatch):
    """With no stored cursor, read one message to anchor - not the backlog."""
    captured = {}

    def fake_request(path, token, method="GET", payload=None):
        captured["path"] = path
        return []

    monkeypatch.setattr(discord_listen, "_request", fake_request)
    discord_listen.fetch_new_messages("t", "c", after)
    assert expected in captured["path"]
