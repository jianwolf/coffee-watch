#!/usr/bin/env python3
"""Post a scouting note to a Discord channel webhook.

A cloud routine has no way to reach a phone on its own: push notifications
belong to Remote Control, which runs on the user's own machine, so a routine's
`PushNotification` call is accepted and delivered nowhere (verified twice).
Outbound HTTP is the one channel a cloud run does have, and a Discord webhook
turns it into a phone notification carrying the whole note rather than a
200-character headline.

The webhook URL is a credential: anyone holding it can post to that channel. It
is read from the environment or an argument and must never be committed -- this
repository is public.

Reads the note on stdin so the caller can hand over exactly what it wrote:

    python3 scripts/notify_discord.py < note.md
    COFFEE_DISCORD_WEBHOOK='https://discord.com/api/webhooks/...' \\
        python3 scripts/notify_discord.py --dry-run < note.md

Stdlib only: the cloud VM installs one dependency and this must not add a
second.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ENV_VAR = "COFFEE_DISCORD_WEBHOOK"
# Discord rejects a message body over 2000 characters outright.
MAX_CHARS = 2000
TIMEOUT_S = 20.0


def split_message(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split on line boundaries so a note is never cut mid-sentence.

    A single line longer than the limit is hard-split, because the alternative
    is dropping it.
    """
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line
    if current.strip():
        chunks.append(current)
    return [chunk for chunk in (c.rstrip() for c in chunks) if chunk]


def post(webhook: str, content: str, timeout: float = TIMEOUT_S) -> None:
    request = urllib.request.Request(
        webhook,
        data=json.dumps({"content": content}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "coffee-watch-notify/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status >= 300:
            raise RuntimeError(f"Discord returned HTTP {response.status}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/notify_discord.py",
        description=(
            "Post stdin to a Discord channel webhook, split to Discord's "
            f"{MAX_CHARS}-character limit."
        ),
    )
    parser.add_argument(
        "--webhook",
        default=os.environ.get(ENV_VAR, ""),
        help=f"Webhook URL (default: ${ENV_VAR}). Never commit this.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent, and send nothing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    note = sys.stdin.read().strip()
    if not note:
        print("Nothing on stdin; not posting.", file=sys.stderr)
        return 2
    chunks = split_message(note)

    if args.dry_run:
        for index, chunk in enumerate(chunks, 1):
            print(f"--- message {index}/{len(chunks)} ({len(chunk)} chars) ---")
            print(chunk)
        return 0

    if not args.webhook:
        # Deliberately a distinct exit code: a run that scouted fine but could
        # not deliver is not the same failure as a run that found nothing.
        print(
            f"No webhook configured. Set ${ENV_VAR} or pass --webhook.",
            file=sys.stderr,
        )
        return 3

    for index, chunk in enumerate(chunks, 1):
        try:
            post(args.webhook, chunk)
        except (urllib.error.URLError, RuntimeError, OSError) as exc:
            # Never echo the URL: it is a credential and logs get shared.
            print(
                f"Failed to post message {index}/{len(chunks)}: {exc}",
                file=sys.stderr,
            )
            return 1
    print(f"Posted {len(chunks)} message(s) to Discord.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
