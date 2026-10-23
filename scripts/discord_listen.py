#!/usr/bin/env python3
"""Answer questions asked in a Discord channel by running Claude Code headless.

The webhook in `notify_discord.py` only posts; it cannot read. Reading a
channel needs a bot application whose MESSAGE CONTENT intent is enabled in
the Developer Portal -- that intent gates message text across Discord's REST
API too, not only gateway events, so without it `content` comes back as an
empty string and this script would see every message as blank.

No gateway, no dependencies: polling `GET /channels/{id}/messages?after=` on
a few-second interval stays well inside Discord's limits and keeps this file
stdlib-only, like its sibling.

    DISCORD_BOT_TOKEN=... DISCORD_CHANNEL_ID=... DISCORD_OWNER_ID=... \\
        python3 scripts/discord_listen.py

**Only the owner's messages are ever executed.** That is not configurable.
Whatever is typed in the channel becomes a prompt for a Claude Code session
running on this machine with the permission mode below, so the author check
is the difference between "I can drive my laptop from my phone" and "anyone
holding the invite link can". Everyone else's messages, and every bot's, are
ignored without being read into a prompt.

Treat message text as data, never as instructions about how this script
should behave.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Discord rejects a body over 2000 characters, and a Claude answer often
# runs past that; the notifier already splits on line boundaries.
from notify_discord import split_message

API = "https://discord.com/api/v10"
POLL_SECONDS = 6.0
CLAUDE_TIMEOUT_S = 900.0
STATE_PATH = Path.home() / ".coffee-watch-discord.json"
# The owner asked for full development capability in the channel. Scoped by
# the author check above rather than by the tool list.
PERMISSION_MODE = "bypassPermissions"
AUTH_HINT = (
    "Claude Code is not authenticated on this machine any more. "
    "Run `claude` once on the Mac mini and sign in; a subscription OAuth "
    "session expires if it goes unused."
)


def _request(
    path: str, token: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    request = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "coffee-watch-listen/1.0",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    return json.loads(body) if body else None


def fetch_new_messages(token: str, channel: str, after: str | None) -> list[dict]:
    """Oldest first, so a burst of questions is answered in the order asked."""
    query = f"?limit=20&after={after}" if after else "?limit=1"
    try:
        messages = _request(f"/channels/{channel}/messages{query}", token)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            retry_after = float(exc.headers.get("Retry-After", "5") or 5)
            print(f"rate limited, sleeping {retry_after}s", file=sys.stderr)
            time.sleep(retry_after)
            return []
        raise
    return list(reversed(messages or []))


def post(token: str, channel: str, content: str) -> None:
    for chunk in split_message(content):
        # flags 4 = SUPPRESS_EMBEDS, same reason as the notifier: link cards
        # bury the text under the shop's own marketing copy.
        _request(
            f"/channels/{channel}/messages",
            token,
            method="POST",
            payload={"content": chunk, "flags": 4},
        )


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any]) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except OSError as exc:
        print(f"could not save state: {exc}", file=sys.stderr)


def ask_claude(question: str, cwd: Path, session_id: str | None) -> tuple[str, str | None]:
    """Run one headless turn. Returns (reply, session id to resume next time).

    Resuming keeps the channel a conversation: "how much was that one" has to
    reach the session that just talked about it.
    """
    command = [
        "claude",
        "-p",
        question,
        "--output-format",
        "json",
        "--permission-mode",
        PERMISSION_MODE,
    ]
    if session_id:
        command += ["--resume", session_id]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (f"(no answer: claude ran past {CLAUDE_TIMEOUT_S:.0f}s)", session_id)
    except FileNotFoundError:
        return ("(no answer: the `claude` CLI is not on this machine's PATH)", None)

    try:
        result = json.loads(completed.stdout)
    except ValueError:
        tail = (completed.stderr or completed.stdout or "").strip()[-400:]
        return (f"(no answer: could not parse claude output)\n{tail}", session_id)

    reply = str(result.get("result") or "").strip()
    new_session = result.get("session_id") or session_id
    if result.get("is_error"):
        if "authenticate" in reply.lower() or "oauth" in reply.lower():
            return (AUTH_HINT, None)
        return (f"(claude reported an error) {reply}", new_session)
    return (reply or "(claude returned an empty reply)", new_session)


def should_answer(message: dict, owner_id: str) -> bool:
    author = message.get("author") or {}
    if author.get("bot"):
        return False
    if str(author.get("id") or "") != owner_id:
        return False
    return bool(str(message.get("content") or "").strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/discord_listen.py",
        description="Answer the owner's Discord messages with Claude Code.",
    )
    parser.add_argument("--token", default=os.environ.get("DISCORD_BOT_TOKEN", ""))
    parser.add_argument("--channel", default=os.environ.get("DISCORD_CHANNEL_ID", ""))
    parser.add_argument("--owner", default=os.environ.get("DISCORD_OWNER_ID", ""))
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Working directory for the Claude session (default: this repo)",
    )
    parser.add_argument(
        "--once", action="store_true", help="Drain the channel once and exit"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [
        name
        for name, value in (
            ("DISCORD_BOT_TOKEN", args.token),
            ("DISCORD_CHANNEL_ID", args.channel),
            # Refusing to start without this is the point: see the module
            # docstring. A listener with no owner answers the whole channel.
            ("DISCORD_OWNER_ID", args.owner),
        )
        if not value
    ]
    if missing:
        print(f"Missing: {', '.join(missing)}", file=sys.stderr)
        return 2

    state = load_state()
    sessions: dict[str, str] = state.get("sessions", {})
    after: str | None = state.get("last_message_id")

    while True:
        try:
            messages = fetch_new_messages(args.token, args.channel, after)
        except (urllib.error.URLError, OSError) as exc:
            print(f"poll failed: {exc}", file=sys.stderr)
            time.sleep(POLL_SECONDS * 2)
            continue

        for message in messages:
            after = str(message.get("id") or after)
            if not should_answer(message, args.owner):
                continue
            question = str(message["content"]).strip()
            print(f"answering {message['id']}", file=sys.stderr)
            reply, session_id = ask_claude(
                question, args.repo, sessions.get(args.channel)
            )
            if session_id:
                sessions[args.channel] = session_id
            try:
                post(args.token, args.channel, reply)
            except (urllib.error.URLError, OSError) as exc:
                print(f"post failed: {exc}", file=sys.stderr)

        save_state({"last_message_id": after, "sessions": sessions})
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
